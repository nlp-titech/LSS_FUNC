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
LOCAL_AVE_POOLER = "local_ave"
ZERO_NORM_LIMIT = 1e-24

logger = logging.getLogger(__name__)


def dict_2_float(dd):
    for k in dd:
        dd[k] = dd[k].float()

    return dd


class COLBERTSSearcher:
    def __init__(
        self,
        model,
        ret_score: Dict,
        batch_size: int = 128,
        similarity_metric: str = "l2",
        **kwargs,
    ):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.all_results = {}
        self.ret_score = ret_score
        self.similarity_metric = similarity_metric
        
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
        threshold = top_k * 2
        # query_toks_batches = list()
        # query_id_batches = list()
        self.all_results = {qid: {} for qid in query_ids}
        l_queries = [queries[qid] for qid in queries]
        for idx in range(0, len(queries), self.batch_size):
            query_batch = l_queries[idx : idx + self.batch_size]
            query_id_batch = query_ids[idx: idx + self.batch_size]
            query_toks = self.model.q_tokenizer(
                query_batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            feature_batch = {k: v.to(device) for k, v in query_toks.items()}
            with torch.no_grad():
                query_toks_batch = self.model.query(feature_batch["input_ids"], feature_batch["attention_mask"])
            # query_toks_batches.append(query_toks_batch)
            # query_id_batches.append(query_id_batch)

            logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            corpus_ids = list(corpus.keys())
        
            l_corpus = [corpus[cid]["title"] + self.model.sep + corpus[cid]["text"] for cid in corpus_ids]

            itr = range(0, len(l_corpus), self.batch_size)
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
                corpus_end_idx = min(corpus_start_idx + self.batch_size, len(corpus))
                sub_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]
    
                sub_corpus_toks = self.model.d_tokenizer(
                    l_corpus[corpus_start_idx:corpus_end_idx],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                max_doc_length = sub_corpus_toks["input_ids"].shape[1]
                docs_toks_rep = []
                for ti in range(0, max_doc_length, self.model.max_length):
                    input_doc = dict()
                    for k in sub_corpus_toks:
                        input_doc[k] = sub_corpus_toks[k][:, ti : ti + self.model.max_length].to(device)
                    with torch.no_grad():
                        doc_toks_rep = self.model.doc(input_doc["input_ids"], input_doc["attention_mask"])
                        
                    docs_toks_rep.append(doc_toks_rep)
    
                # Encode chunk of corpus
                sub_corpus_toks_rep = torch.cat(docs_toks_rep, dim=1)
    
                for qid, qt in zip(query_id_batch, query_toks_batch):
                    scores = self.model.score(qt, sub_corpus_toks_rep).cpu().numpy()
                    arg_scores = np.argsort(-scores)[:threshold]
                    present_sorted = sorted(self.all_results[qid].items(), key=lambda x: -x[1])[:threshold]
                    self.all_results[qid] = {k: v for k, v in present_sorted}
                    for s_idx in arg_scores:
                        did = sub_corpus_ids[s_idx]
                        score = scores[s_idx]
                        self.all_results[qid][did] = float(score)
    
        return self.all_results

