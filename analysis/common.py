import logging
import os
from typing import Tuple, Dict
from beir.datasets.data_loader import GenericDataLoader


logger = logging.getLogger(__name__)
MIN_DISCOUNT = 1e-3

class QrelDataLoader(GenericDataLoader):
    def load(self, split="test") -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.queries, self.qrels


def weight_add_result(bm25_result, dense_result, all_qids, weight1=0.5):
    weight2 = 1 - weight1
    result = {}
    for qid in all_qids:
        d_result1 = bm25_result.get(qid, None)
        d_result2 = dense_result.get(qid, None)
        if not d_result1 and not d_result2:
            continue
        elif not d_result2:
            if weight1 != 0.0:
                result[qid] = {k: v*weight1 for k, v in d_result1.items()}
            continue
        elif not d_result1:
            if weight2 != 0.0:
                result[qid] = {k: v*weight2 for k, v in d_result2.items()}
            continue
        all_dids = set(list(d_result1.keys())) | set(list(d_result2.keys()))
        result[qid] = {}
        try:
            min_score1 = sorted(d_result1.values())[0] - MIN_DISCOUNT
        except:
            print(qid, d_result1)
            raise ValueError()
        min_score2 = sorted(d_result2.values())[0] - MIN_DISCOUNT
        for did in all_dids:
            d_score1 = d_result1.get(did, min_score1)
            d_score2 = d_result2.get(did, min_score2)
            result[qid][did] = weight1 * d_score1 + weight2 * d_score2
    return result


def weight_add_result_org(qrels, dense_results, bm25_results, weight1=0.5):
    hybrid_results = dict()
    weight2 = 1 - weight1
    # for qid in rerank_results:
    for qid in qrels:
        if qid not in dense_results:
            continue
            
        hybrid_results[qid] = dict([(did, score * weight2) for did, score in dense_results[qid].items()])

        lowest_score = min(dense_results[qid].values())
        try:
            did2scores = bm25_results[qid]
        except KeyError:
            continue
        for did, score in did2scores.items():
            if did in dense_results[qid]:
                hybrid_results[qid][did] = weight1 * score + weight2 * dense_results[qid][did]
            else:
                hybrid_results[qid][did] = weight1 * score + weight2 * lowest_score
                
    return hybrid_results


def weight_add_result_2(weight1, result1: dict, result2: dict, top_k: int = 100) -> dict:
    weight2 = 1 - weight1
    new_result = {}
    min_score = {}
    for qid, d2score in result1.items():
        s_d2score = sorted(d2score.items(), key=lambda x: -x[1])[:top_k]
        new_result[qid] = dict()
        min_score[qid] = s_d2score[-1][0]
        for did, score in s_d2score:
            new_result[qid][did] = score
            
    for qid, d2score in result2.items():
        s_d2score = sorted(d2score.items(), key=lambda x: -x[1])[:top_k]
        if qid not in new_result:
            new_result[qid] = dict(s_d2score)
            continue
        for did, score in s_d2score:
            if did not in new_result[qid]:
                new_result[qid][did] = min_score[qid]
            new_result[qid][did] += score 
    return new_result