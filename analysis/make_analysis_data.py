import os
import logging
import json
from collections import Counter, defaultdict
from typing import List

from transformers import (
    HfArgumentParser,
)
from tqdm import tqdm
import numpy as np

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from lss_func.search.coil.exact_search import LSSSearcher
from lss_func.models.sentence_bert import SentenceBERT, SentenceBERTOUTER
from lss_func.models.bm25_weight import BM25Weight
from lss_func.models import coil
from lss_func.arguments import ModelArguments, DataArguments, LSSArguments
from dataclasses import dataclass, field

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


def save_result(result_path, result):
    with open(result_path, "w") as f:
        json.dump(result, f)


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
bm25_results = defaultdict(dict)
for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, top_k, query_generator=None, fields=dict())
    for did, rank, score, _ in hits_iterator(hits):
        bm25_results[qid][did] = score

bm25_result_path = os.path.join(data_args.resultpath, "bm25_result.json")
save_result(bm25_result_path, bm25_results)

word_embedding_model = Transformer(model_args.model_name_or_path)
tokenizer = word_embedding_model.tokenizer
vocab = tokenizer.get_vocab()

sep = " "
idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, sep)

# dense retriever
sbert_model = SentenceBERT(model_args.model_name_or_path)
sbert_model.q_model.eval()
dense_model = DRES(sbert_model, batch_size=128)
dense_retriever = EvaluateRetrieval(dense_model, score_function="cos_sim", k_values=k_values)

logging.info("start rerank dense")
dense_rerank_results = dense_retriever.rerank(corpus, queries, bm25_results, top_k=100)

dense_result_path = os.path.join(data_args.resultpath, "dense_result.json")
save_result(dense_result_path, dense_rerank_results)


# weighted dense retriever
word_weights = BM25Weight(vocab=vocab, word_weights=idf, doc_len_ave=doc_len_ave)
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
sbert_model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
sbert_model.eval()
sbert_model_for_beir = SentenceBERTOUTER(sbert_model, sep=sep)
weighted_dense_model = DRES(sbert_model_for_beir, batch_size=128)
weighted_dense_retriever = EvaluateRetrieval(weighted_dense_model, score_function="cos_sim", k_values=k_values)

logging.info("start rerank weighted_dense")
weighted_dense_rerank_results = weighted_dense_retriever.rerank(corpus, queries, bm25_results, top_k=100)
weighted_dense_result_path = os.path.join(data_args.resultpath, "weighted_dense_result.json")
save_result(weighted_dense_result_path, weighted_dense_rerank_results)


#### Reranking top-100 docs using Dense Retriever model
score_functions = ["maxsim_bm25_qtf"]
pooler = "ave"
cbm25_core = coil.Coil(model_args.model_name_or_path, model_args)
cbm25_core.eval()

cbm25_model = LSSSearcher(
    cbm25_core,
    bm25_results,
    score_functions,
    pooler=pooler,
    batch_size=128,
    idf=idf,
    doc_len_ave=doc_len_ave,
    doc_max_length=lss_args.doc_max_length,
    window_size=model_args.window_size,
    encode_raw=model_args.encode_raw,
    norm=lss_args.norm,
)
cbm25_retriever = EvaluateRetrieval(cbm25_model, score_function="cos_sim", k_values=[1, 3, 5, 10, 100])
#### Retrieve dense results (format of results is identical to qrels)
logging.info("start rerank cbm25")
cbm25_rerank_results = cbm25_retriever.rerank(corpus, queries, bm25_results, top_k=top_k)
cbm25_result_path = os.path.join(data_args.resultpath, "cbm25_result.json")
save_result(cbm25_result_path, cbm25_rerank_results)
