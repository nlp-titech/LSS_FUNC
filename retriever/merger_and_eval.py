import os
import torch
import pytrec_eval
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List


def main():
    parser = ArgumentParser()
    parser.add_argument("--score_dir", required=True)
    parser.add_argument("--qrel_path", required=True)
    parser.add_argument("--query_lookup", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num_query", type=int)
    parser.add_argument("--save_ranking_to", required=True)
    parser.add_argument("--eval_result")
    parser.add_argument("--marco_document", action="store_true")
    args = parser.parse_args()

    partitions = os.listdir(args.score_dir)

    qrel = dict()

    with open(args.qrel_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            qid, did, score = line.strip().split("\t")
            if qid not in qrel:
                qrel[qid] = dict()
            qrel[qid][did] = int(score)

    pbar = tqdm(partitions)

    q_lookup: List[str] = list(np.load(args.query_lookup))
    result = dict()

    for part_name in pbar:
        pbar.set_description_str(f"Processing {part_name}")
        scores, indices = torch.load(os.path.join(args.score_dir, part_name))

        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, scores.numpy(), indices):
            if qid not in result:
                result[qid] = dict()

            for s, idx in zip(q_doc_scores, q_doc_indices):
                if idx not in result[qid]:
                    result[qid][idx] = float(s)
                else:
                    result[qid][idx] += float(s)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"recall.100", "ndcg_cut.10"})
    scores = evaluator.evaluate(result)
    ndcg_10 = 0
    recall_100 = 0
    for qid in scores.keys():
        ndcg_10 += scores[qid]["ndcg_cut_10"]
        recall_100 += scores[qid]["recall_100"]

    ndcg_10 /= len(scores)
    recall_100 /= len(scores)

    eval_result = {"NDCG@10": round(ndcg_10, 5), "Recall@100": round(recall_100, 5)}
    print(eval_result)

    with open(args.save_ranking_to, "w") as f:
        json.dump(result, f)

    with open(args.eval_result, "w") as f:
        json.dump(eval_result, f)


if __name__ == "__main__":
    main()
