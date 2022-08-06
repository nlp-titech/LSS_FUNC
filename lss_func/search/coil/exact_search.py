from collections import defaultdict, Counter
from itertools import chain
import logging
import os
import sys
import time
import multiprocessing as mp
from typing import Dict
from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


ZERO_NORM_LIMIT = 1e-24
TOKEN_POOLER = "token"
LOCAL_AVE_POOLER = "ave"
ZERO_NORM_LIMIT = 1e-24

logger = logging.getLogger(__name__)


class LSSSearcher:
    def __init__(
        self,
        model,
        ret_score: Dict,
        score_funcs: list,
        batch_size: int = 128,
        pooler: str = "ave",
        tok_only: bool = True,
        window_size: int = 5,
        bm25_k1: int = 0.82,
        bm25_b: int = 0.65,
        idf: Dict = {},
        doc_len_ave: float = 0.0,
        encode_raw: bool = True,
        doc_max_length: int = None,
        norm: bool = True,
        **kwargs,
    ):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.pooler = pooler
        self.all_results = {}
        self.tok_only = tok_only
        self.window_size = window_size
        self.score_funcs = score_funcs
        self.ret_score = ret_score
        self.idf = idf
        self.doc_len_ave = doc_len_ave
        self.encode_raw = encode_raw
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.doc_max_length = doc_max_length
        self.norm = norm
        self.special_tokens = {
            self.model.tokenizer.pad_token_id,
            self.model.tokenizer.bos_token_id,
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.sep_token_id,
            self.model.tokenizer.cls_token_id,
        }

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: List[int],
        score_function: str,
    ) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.all_results = {score_func: {qid: {} for qid in query_ids} for score_func in self.score_funcs}
        l_queries = [queries[qid] for qid in queries]
        q_tok2qid = defaultdict(list)
        q_tok2rep = defaultdict(list)
        for idx in range(0, len(queries), self.batch_size):
            query_batch = l_queries[idx : idx + self.batch_size]
            query_toks = self.model.tokenizer(
                query_batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            feature_batch = {k: v.to(device) for k, v in query_toks.items()}
            with torch.no_grad():
                if self.encode_raw:
                    _, query_toks_batch = self.model.encode_query_raw(feature_batch)
                else:
                    _, query_toks_batch = self.model.encode_query(feature_batch)

            att_masks = query_toks["attention_mask"].numpy()
            i_query_toks = query_toks["input_ids"].numpy()
            query_toks_batch = query_toks_batch.cpu().numpy()
            query_toks_batch, att_masks, i_query_toks = self._preproc_rep(
                query_toks_batch, att_masks, i_query_toks
            )

            for qi, (i_query_tok, query_reps, att_mask) in enumerate(zip(i_query_toks, query_toks_batch, att_masks)):
                qid = query_ids[qi + idx]
                for i, (qt, rep, am) in enumerate(zip(i_query_tok, query_reps, att_mask)):
                    if qt in self.special_tokens:
                        continue
                    if am == 0:
                        continue
                    q_tok2qid[qt].append(qid)
                    q_tok2rep[qt].append(rep)

        for k in q_tok2rep:
            q_tok2rep[k] = np.vstack(q_tok2rep[k])

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        check_q_tok2qid = set()
        for qids in q_tok2qid.values():
            check_q_tok2qid |= set(qids)

        assert len(corpus_ids) != 0
        assert len(query_ids) == len(check_q_tok2qid)
        set_qids = check_q_tok2qid
        l_corpus = [corpus[cid]["title"] + self.model.sep + corpus[cid]["text"] for cid in corpus_ids]

        itr = range(0, len(l_corpus), self.batch_size)
        
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.batch_size, len(corpus))
            sub_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]

            sub_corpus_toks = self.model.tokenizer(
                l_corpus[corpus_start_idx:corpus_end_idx],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.doc_max_length,
            )
            max_doc_length = sub_corpus_toks["input_ids"].shape[1]
            docs_toks_rep = []
            for ti in range(0, max_doc_length, self.model.max_length):
                input_doc = dict()
                for k in sub_corpus_toks:
                    input_doc[k] = sub_corpus_toks[k][:, ti : ti + self.model.max_length].to(device)
                with torch.no_grad():
                    if self.encode_raw:
                        _, doc_toks_rep = self.model.encode_corpus_raw(input_doc)
                    else:
                        _, doc_toks_rep = self.model.encode_corpus(input_doc)

                docs_toks_rep.append(doc_toks_rep.cpu().numpy())

            # Encode chunk of corpus
            sub_corpus_att_mask = sub_corpus_toks["attention_mask"].numpy()
            i_sub_corpus_toks = sub_corpus_toks["input_ids"].numpy()
            sub_corpus_toks_rep = np.hstack(docs_toks_rep)
            sub_corpus_toks_rep, sub_corpus_att_mask, i_sub_corpus_toks = self._preproc_rep(
                sub_corpus_toks_rep, sub_corpus_att_mask, i_sub_corpus_toks
            )

            d_tok2did = defaultdict(list)
            d_tok2rep = defaultdict(list)
            did2doc_len = dict()
            for di, (doc_toks, doc_reps, s_att_mask) in enumerate(
                zip(i_sub_corpus_toks, sub_corpus_toks_rep, sub_corpus_att_mask)
            ):
                did = sub_corpus_ids[di]
                did2doc_len[did] = np.sum(s_att_mask == 1) - 2
                for i, (dt, rep, am) in enumerate(zip(doc_toks, doc_reps, s_att_mask)):
                    if dt in self.special_tokens:
                        continue
                    if am == 0:
                        continue
                    d_tok2did[dt].append(did)
                    d_tok2rep[dt].append(rep)

            for k in d_tok2rep:
                d_tok2rep[k] = np.vstack(d_tok2rep[k])

            batch_rerank_result = self.calc_score(q_tok2qid, q_tok2rep, d_tok2did, d_tok2rep, did2doc_len)

            threshold = int(top_k * 2)
            for score_func in batch_rerank_result:
                for qid in batch_rerank_result[score_func]:
                    if qid not in self.all_results[score_func]:
                        self.all_results[score_func][qid] = batch_rerank_result[score_func][qid]

                    present_sorted = sorted(self.all_results[score_func][qid].items(), key=lambda x: -x[1])[:threshold]
                    self.all_results[score_func][qid] = {k: v for k, v in present_sorted}
                    for did, score in batch_rerank_result[score_func][qid].items():
                        self.all_results[score_func][qid][did] = score

        return self.all_results

    def calc_score(
        self,
        q_tok2id: Dict,
        q_tok2rep: Dict,
        d_tok2id: Dict,
        d_tok2rep: Dict,
        did2doc_len: Dict,
    ) -> torch.Tensor:


        qids = set(chain.from_iterable(q_tok2id.values()))
        batch_soft_tf = dict()
        batch_tf_qd = dict()
        batch_results = {k: dict() for k in self.score_funcs}
        for k in self.score_funcs:
            for qid in qids:
                batch_results[k][qid] = dict()

        # batch_results = dict()
        for tq in tqdm(q_tok2id):
            qids = q_tok2id[tq]
            tf_qid, tf_q = np.unique(qids, return_counts=True)
            tq_tf_q = {str(qid): tf for qid, tf in zip(tf_qid, tf_q)}
            if tq not in d_tok2id:
                continue
            q_rep = q_tok2rep[tq]
            dids = d_tok2id[tq]
            tf_did, tf_d = np.unique(dids, return_counts=True)
            td_tf_d = {str(did): tf for did, tf in zip(tf_did, tf_d)}
            for qid, qtf in tq_tf_q.items():
                if qid not in batch_tf_qd:
                    batch_tf_qd[qid] = dict()
                for did, dtf in td_tf_d.items():
                    if did not in batch_tf_qd[qid]:
                        batch_tf_qd[qid][did] = dict()

                    batch_tf_qd[qid][did][tq] = (qtf, dtf)

            d_rep = d_tok2rep[tq]
            sims = np.maximum(np.dot(q_rep, d_rep.T), 0.0)
            for qid, q_sim in zip(qids, sims):
                if qid not in batch_soft_tf:
                    batch_soft_tf[qid] = dict()
                for did, sim in zip(dids, q_sim):
                    if did not in batch_soft_tf[qid]:
                        batch_soft_tf[qid][did] = defaultdict(list)
                    batch_soft_tf[qid][did][tq].append(sim)

        for score_func in tqdm(self.score_funcs):
            for qid in batch_soft_tf:
                for did, soft_tf in batch_soft_tf[qid].items():
                    if did not in self.ret_score[qid]:
                        continue

                    doc_len = did2doc_len[did]
                    tfs = batch_tf_qd[qid][did]
                    if qid == did:
                        continue
                    if score_func == "maxsim_qtf":
                        score = self._maxsim_qtf(soft_tf, tfs)
                    elif score_func == "maxsim_idf_qtf":
                        score = self._maxsim_idf_qtf(soft_tf, tfs)
                    elif score_func == "maxsim_bm25_qtf":
                        score = self._maxsim_bm25_qtf(soft_tf, doc_len, tfs)
                    else:
                        raise ValueError
                    
                    batch_results[score_func][qid][did] = score

        return batch_results


    def _preproc_rep(self, reps: np.ndarray, att_mask: np.ndarray, input_tok: np.ndarray):
        att_mask = att_mask
        reps = reps
        if self.pooler == TOKEN_POOLER:
            reps = reps[:, 1:]
        elif self.pooler == LOCAL_AVE_POOLER:
            reps = self._rep_lave(reps, att_mask)
        else:
            raise ValueError(f"{self.pooler} doesn't exist")

        if self.norm:
            reps /= np.linalg.norm(reps, axis=2)[:, :, np.newaxis]
        reps[np.isnan(reps)] = 0.0
        return reps, att_mask[:, 1:], input_tok[:, 1:]

    def _rep_lave(self, reps, att_masks):
        tg_reps = np.zeros_like(reps)
        for b, (rep, att_mask) in enumerate(zip(reps, att_masks)):
            og_rep = rep[att_mask == 1, :][1:-1]
            rep_len = og_rep.shape[0]
            for i in range(rep_len):
                start = i - self.window_size if i - self.window_size > 0 else 0
                end = i + self.window_size
                tg_reps[b, i, :] += np.mean(og_rep[start:end, :], axis=0)

        return tg_reps

    def _maxsim_bm25_qtf(self, soft_tf, doc_len, tfs):
        score = 0
        for t, tf_scores in soft_tf.items():
            qd_tf = tfs[t]
            d_tf = qd_tf[1]
            # tf = len(tf_scores)
            # sumsim = np.sum(tf_scores)
            sumsim = np.sum(np.max(np.array(tf_scores).reshape(qd_tf), axis=1))
            score += (
                self.idf[t]
                * sumsim
                * d_tf
                * (1 + self.bm25_k1)
                / (d_tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / self.doc_len_ave))
            )
        return score

    def _maxsim_qtf(self, soft_tf, tfs):
        score = 0.0
        for t, tf_scores in soft_tf.items():
            qd_tf = tfs[t]
            score += np.sum(np.max(np.array(tf_scores).reshape(qd_tf), axis=1))    

        return score

    def _maxsim_idf_qtf(self, soft_tf, tfs):
        score = 0.0
        for t, tf_scores in soft_tf.items():
            qd_tf = tfs[t]
            score += np.sum(np.max(np.array(tf_scores).reshape(qd_tf), axis=1)) * self.idf[t]

        return score

    def _maxsim_tfidf(self, soft_tf):
        score = 0
        for t, tf_scores in soft_tf.items():
            tf = len(tf_scores)
            maxsim = np.max(tf_scores)
            score += self.idf[t] * maxsim * tf
        return score

