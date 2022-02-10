import argparse
import json
import os
import torch

from collections import defaultdict, Counter
from beir.datasets.data_loader import GenericDataLoader
from transformers import AutoTokenizer
from tqdm import tqdm, trange
import numpy as np

try:
    from retriever_ext import scatter as c_scatter
except ImportError:
    raise ImportError("Cannot import scatter module." " Make sure you have compiled the retriever extension.")


def dict_2_float(dd):
    for k in dd:
        dd[k] = dd[k].float()

    return dd


def build_full_tok_rep(dd):
    new_offsets = {}
    curr = 0
    reps = []
    for k in dd:
        rep = dd[k]
        reps.append(rep)

        new_offsets[k] = (curr, curr + len(rep))
        curr += len(rep)

    reps = torch.cat(reps).float()
    return reps, new_offsets


def bm25_stats(corpus, cls_ex_ids, idf, doc_len_ave, tokenizer, sep=" "):
    k1 = 0.9
    b = 0.4
    shard_idx_tf = defaultdict(list)
    shard_idx_doclen = {}

    exid2shard_idx = {}
    for i, cls_ex_id in enumerate(cls_ex_ids):
        exid2shard_idx[cls_ex_id] = i

    for cid in tqdm(corpus.keys()):
        if cid not in exid2shard_idx:
            continue
        text = corpus[cid]["title"] + sep + corpus[cid]["text"]
        input_ids = tokenizer(text)["input_ids"]
        tf_d = Counter(input_ids)
        idx_shard = exid2shard_idx[cid]
        shard_idx_doclen[idx_shard] = len(input_ids)
        for tok, freq in tf_d.items():
            shard_idx_tf[tok].append((idx_shard, freq))

    bm25_shard_idx = defaultdict(list)
    bm25_index = defaultdict(list)
    for tok, infos in shard_idx_tf.items():
        for idx, tf in infos:
            bm25_shard_idx[tok].append(idx)
            bm25_index[tok].append(
                tf * (1 + k1) / (tf + k1 * (1 - b + b * shard_idx_doclen[idx] / doc_len_ave)) * idf[tok]
            )

    for tok in bm25_shard_idx:
        bm25_shard_idx[tok] = torch.tensor(bm25_shard_idx[tok])
        bm25_index[tok] = torch.tensor(bm25_index[tok]).to(torch.float)

    return bm25_shard_idx, bm25_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--doc_shard", required=True)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--save_to", required=True)
    parser.add_argument("--stats_dir", default=None)
    parser.add_argument("--corpus_dir")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--no_cls", action="store_true")
    args = parser.parse_args()

    all_ivl_scatter_maps = torch.load(os.path.join(args.doc_shard, "ivl_scatter_maps.pt"))
    all_shard_scatter_maps = torch.load(os.path.join(args.doc_shard, "shard_scatter_maps.pt"))
    tok_id_2_reps = torch.load(os.path.join(args.doc_shard, "tok_reps.pt"))
    ## doc_cls_reps = torch.load(os.path.join(args.doc_shard, "cls_reps.pt")).float()
    cls_ex_ids = torch.load(os.path.join(args.doc_shard, "cls_ex_ids.pt"))
    # cls_ex_ids = np.load(os.path.join(args.doc_shard, "cls_ex_ids.npy"))
    tok_id_2_reps = dict_2_float(tok_id_2_reps)
    if args.weight_dir is not None:
        weight_path = os.path.join(args.weight_dir, "model.pt")
        if os.path.exists(weight_path):
            model_dict = torch.load(weight_path, map_location="cpu")
            weight = model_dict["vocab_weight.weight"]
        else:
            raise OSError("path doesn't exist")

    print("Search index loaded", flush=True)

    query_tok_reps = torch.load(os.path.join(args.query, "tok_reps.pt")).float()
    all_query_offsets = torch.load(os.path.join(args.query, "offsets.pt"))

    idf_path = os.path.join(args.stats_dir, "idf.json")
    with open(idf_path) as f:
        idf = json.load(f)

    doc_len_path = os.path.join(args.stats_dir, "doc_len_ave.npy")
    doc_len_ave = np.load(doc_len_path)

    corpus, queries, qrels = GenericDataLoader(args.corpus_dir).load(split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    bm25_shard_idx, bm25_index = bm25_stats(corpus, cls_ex_ids, idf, doc_len_ave, tokenizer, sep=" ")

    ## query_cls_reps = torch.load(os.path.join(args.query, "cls_reps.pt")).float()
    print("Query representations loaded", flush=True)

    all_query_match_scores = []
    all_query_inids = []

    shard_name = os.path.split(args.doc_shard)[1]

    batch_size = args.batch_size

    for batch_start in trange(0, len(all_query_offsets), batch_size, desc=shard_name):
        # batch_q_reps = query_cls_reps[batch_start : batch_start + batch_size]
        # match_scores = torch.matmul(batch_q_reps, doc_cls_reps.transpose(0, 1))  # D * b
        # if args.no_cls:
        #     match_scores = torch.zeros_like(match_scores)
        match_scores = torch.zeros((batch_size, len(cls_ex_ids)))
        tok_match_scores = torch.zeros((batch_size, len(cls_ex_ids)))
        bm25_match_scores = torch.zeros((batch_size, len(cls_ex_ids)))

        batched_qtok_offsets = defaultdict(list)
        q_batch_offsets = defaultdict(list)
        for batch_offset, q_offsets in enumerate(all_query_offsets[batch_start : batch_start + batch_size]):
            for q_tok_id, q_tok_offset in q_offsets:
                if q_tok_id not in tok_id_2_reps:
                    continue
                batched_qtok_offsets[q_tok_id].append(q_tok_offset)
                q_batch_offsets[q_tok_id].append(batch_offset)

        batch_qtok_ids = list(batched_qtok_offsets.keys())
        batched_tok_scores = []

        for q_tok_id in batch_qtok_ids:
            q_tok_reps = query_tok_reps[batched_qtok_offsets[q_tok_id]]
            tok_reps = tok_id_2_reps[q_tok_id]
            tok_scores = torch.matmul(q_tok_reps, tok_reps.transpose(0, 1)).relu_()  # Bt * Ds
            batched_tok_scores.append(tok_scores)

        for i, q_tok_id in enumerate(batch_qtok_ids):
            ivl_scatter_map = all_ivl_scatter_maps[q_tok_id]
            shard_scatter_map = all_shard_scatter_maps[q_tok_id]

            bm25_scores = bm25_index[q_tok_id]
            shard_bm25_scatter_map = bm25_shard_idx[q_tok_id]

            tok_scores = batched_tok_scores[i]
            ivl_maxed_scores = torch.empty(len(shard_scatter_map))

            for j in range(tok_scores.size(0)):
                ivl_maxed_scores.zero_()
                c_scatter.scatter_max(tok_scores[j].numpy(), ivl_scatter_map.numpy(), ivl_maxed_scores.numpy())
                boff = q_batch_offsets[q_tok_id][j]
                tok_match_scores[boff].scatter_add_(0, shard_scatter_map, ivl_maxed_scores)
                bm25_match_scores[boff].scatter_add_(0, shard_bm25_scatter_map, bm25_scores)
                match_scores += tok_match_scores * bm25_match_scores

        top_scores, top_iids = match_scores.topk(args.top, dim=1)
        all_query_match_scores.append(top_scores)
        all_query_inids.append(top_iids)

    print("Search Done", flush=True)

    # post processing
    all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
    # all_query_exids = torch.cat([cls_ex_ids[inids] for inids in all_query_inids], dim=0)
    all_query_exids = np.concatenate([cls_ex_ids[inids] for inids in all_query_inids], axis=0)

    torch.save((all_query_match_scores, all_query_exids), args.save_to)


if __name__ == "__main__":
    main()
