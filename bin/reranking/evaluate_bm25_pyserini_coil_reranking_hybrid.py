from typing import List, Dict
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.pyclass import autoclass
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from lss_func.search.coil.exact_search import LSSSearcher
from lss_func.models import coil
from lss_func.arguments import ModelArguments, DataArguments, LSSArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)
from tqdm import tqdm

import os
import logging
import random
import json
from collections import Counter, defaultdict
import numpy as np

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


def calc_idf_and_doclen(corpus, tokenizer, sep):
    doc_lens = []
    df = Counter()
    for cid in tqdm(corpus.keys()):
        text = corpus[cid]["title"] + sep + corpus[cid]["text"]
        input_ids = tokenizer(text)["input_ids"]
        doc_lens.append(len(input_ids))
        df.update(list(set(input_ids)))

    idf = defaultdict(float)
    N = len(corpus)
    for w, v in df.items():
        idf[w] = np.log(N / v)

    doc_len_ave = np.mean(doc_lens)
    return idf, doc_len_ave


def hits_iterator(hits: List[JSimpleSearcherResult]):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


def post_proc_for_one_by_one(bm25_results: Dict[str, Dict[str, float]], rerank_results: Dict[str, Dict[str, float]]):
    for qid in bm25_results:
        q_bm25_result = bm25_results[qid]
        q_rereank_result = rerank_results[qid]
        del_did = set()
        for did in q_rereank_result:
            if did not in q_bm25_result:
                del_did.add(did)

        for did in del_did:
            del q_rereank_result[did]


def debug_score(results, func_name):
    keys = sorted(results.keys())
    with open(f"debug_score_{func_name}.tsv", "w") as f:
        for k in keys:
            scores = sorted(results[k].items(), key=lambda x: -x[1])
            for ss in scores:
                oline = "\t".join([k, str(ss[0]), str(ss[1])])
                print(oline, file=f)


#### /print debug information to stdout
parser = HfArgumentParser((ModelArguments, DataArguments, LSSArguments))
(model_args, data_args, lss_args) = parser.parse_args_into_dataclasses()

k1 = 0.9
b = 0.4
top_k = 100
k_values = [1, 3, 5, 10, 100]

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = data_args.dataset
data_path = os.path.join(data_args.root_dir, dataset)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

searcher = SimpleSearcher(data_args.index)
searcher.set_bm25(k1, b)

print(len(corpus))
logging.info("start retrieval")
results = defaultdict(dict)
for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, top_k, query_generator=None, fields=dict())
    for did, rank, score, _ in hits_iterator(hits):
        results[qid][did] = score

## dense retriever
model = DRES(models.SentenceBERT(model_args.model_name_or_path), batch_size=128)
dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=k_values)

#### Retrieve dense results (format of results is identical to qrels)
dense_rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Reranking top-100 docs using Dense Retriever model
base_model = coil.Coil(model_args.model_name_or_path, model_args)
base_model.eval()
idf, doc_len_ave = calc_idf_and_doclen(corpus, base_model.tokenizer, base_model.sep)


score_functions = lss_args.funcs.strip().split(",")
poolers = model_args.pooler_mode.strip().split(",")

eval_result = defaultdict(dict)
for pooler in poolers:
    model = LSSSearcher(
        base_model,
        results,
        score_functions,
        pooler=pooler,
        batch_size=128,
        idf=idf,
        doc_len_ave=doc_len_ave,
        doc_max_length=lss_args.doc_max_length,
        window_size=model_args.window_size,
        norm=model_args.norm,
    )
    dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=[1, 3, 5, 10, 100])

    #### Retrieve dense results (format of results is identical to qrels)
    logging.info(f"start rerank: pooler-{pooler}")
    all_rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=top_k)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    for score_function in score_functions:
        rerank_results = all_rerank_results[score_function]
        hybrid_results = dict()
        for qid in rerank_results:
            if qid not in hybrid_results:
                hybrid_results[qid] = dense_rerank_results[qid]

            for did, score in rerank_results[qid].items():
                if did in hybrid_results[qid]:
                    hybrid_results[qid][did] += score
                else:
                    hybrid_results[qid][did] = score
        
        ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, hybrid_results, k_values)
        eval_result[score_function][pooler] = ndcg

        #### Print top-k documents retrieved ####
        query_id, ranking_scores = random.choice(list(rerank_results.items()))
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        logging.info("Query : %s\n" % queries[query_id])
        if scores_sorted:
            doc_id = random.choice(scores_sorted)[0]
            # Format: Rank x: ID [Title] Body
            logging.info(
                "Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text"))
            )

with open(data_args.resultpath, "w") as f:
    json.dump(eval_result, f)
