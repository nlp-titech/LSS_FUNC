from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from lss_func.models.bm25_weight import BM25Weight
from lss_func.models.sentence_bert import SentenceBERTOUTER

import argparse
import os
import logging
import random
import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict

from tqdm import tqdm
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordWeights, Pooling


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


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--root_dir")
parser.add_argument("--model_path")
parser.add_argument("--index")
parser.add_argument("--resultpath")
parser.add_argument("--sep", default=" ")

args = parser.parse_args()

k1 = 0.9
b = 0.4
top_k = 100
k_values = [1, 3, 5, 10, 100]

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
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

word_embedding_model = Transformer(args.model_path)
tokenizer = word_embedding_model.tokenizer
vocab = tokenizer.get_vocab()
idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, args.sep)
#### Reranking top-100 docs using Dense Retriever model 
word_weights = BM25Weight(vocab=vocab, word_weights=idf, doc_len_ave=doc_len_ave)
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
sbert_model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
sbert_model_for_beir = SentenceBERTOUTER(sbert_model, sep=args.sep)
model = DRES(sbert_model_for_beir, batch_size=128)
dense_retriever = EvaluateRetrieval(model, score_function="cos_sim", k_values=k_values)

#### Retrieve dense results (format of results is identical to qrels)
rerank_results = dense_retriever.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, rerank_results, k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


with open(args.resultpath, "w") as f:
    json.dump(ndcg, f)
