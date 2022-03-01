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
from lss_func.arguments import ModelArguments, DataArguments
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
parser = HfArgumentParser((ModelArguments, DataArguments))
(model_args, data_args) = parser.parse_args_into_dataclasses()

k1 = 0.9
# k1 = 0.82
b = 0.4
# b = 0.68
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


cbm25_core = coil.Coil(model_args.model_name_or_path, model_args)
tokenizer = cbm25_core.tokenizer
sep = " "
idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, sep)



#### Reranking top-100 docs using Dense Retriever model
score_functions = ["C-bm25"]
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
    doc_max_length=data_args.doc_max_length,
    # window_size=data_args.window_size,
    encode_raw=model_args.encode_raw,
    norm=data_args.norm,
)
cbm25_retriever = EvaluateRetrieval(cbm25_model, score_function="cos_sim", k_values=[1, 3, 5, 10, 100])
#### Retrieve dense results (format of results is identical to qrels)
logging.info("start rerank cbm25")
cbm25_rerank_results = cbm25_retriever.rerank(corpus, queries, bm25_results, top_k=top_k)


for score_function in score_functions:
    cbm25__results = all_rerank_results[score_function]
    # post_proc_for_one_by_one(results, rerank_results)
    ndcg, _map, recall, precision = cbm25_retriever.evaluate(qrels, rerank_results, k_values)
    eval_result[score_function][pooler] = ndcg

    #### Print top-k documents retrieved ####
    # show_top_k = 10
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
