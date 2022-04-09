from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.search.colbert.exact_search import COLBERTSSearcher

import argparse
import pathlib, os
import logging
import random
import json
from collections import defaultdict
from typing import List, Dict
from distutils.util import strtobool

from tqdm import tqdm
from pyserini.pyclass import autoclass
from pyserini.search import SimpleSearcher, JSimpleSearcherResult


def hits_iterator(hits: List[JSimpleSearcherResult]):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--root_dir")
parser.add_argument("--base_model_path_or_name")
parser.add_argument("--checkpoint_path")
parser.add_argument("--index")
parser.add_argument("--resultpath")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--query_maxlen", default=32, type=int)
parser.add_argument("--doc_maxlen", default=512, type=int)
parser.add_argument("--mask_punctuation", default=True, type=strtobool)

args = parser.parse_args()

k1 = 0.9
b = 0.4
top_k = 100
k_values = [1, 3, 5, 10, 100]

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)
data_path = os.path.join(args.root_dir, dataset)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

searcher = SimpleSearcher(args.index)
searcher.set_bm25(k1, b)

logging.info("start retrieval")
results = defaultdict(dict)
for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, top_k, query_generator=None, fields=dict())
    for did, rank, score, _ in hits_iterator(hits):
        results[qid][did] = score


#### Reranking top-100 docs using Dense Retriever model 
base_model = models.ColBERT(args.base_model_path_or_name, 
                            args.checkpoint_path,
                            query_maxlen=args.query_maxlen,
                            doc_maxlen=args.doc_maxlen,
                            mask_punctuation=args.mask_punctuation)
model = COLBERTSSearcher(base_model, results, batch_size=args.batch_size)
dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=k_values)

#### Retrieve dense results (format of results is identical to qrels)
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)

with open(args.resultpath, "w") as f:
    json.dump(ndcg, f)

logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


