# Retriever
This Retriever is based on [COIL]([COIL repository](https://github.com/luyug/COIL/tree/main/retriever)).

## Fast Retriver(This section is the same with [COIL](https://github.com/luyug/COIL/tree/main/retriever))
It has come to my attention that `pytorch_scatter` does not scale well to multiple cores. I finally decided to write a C binding. While a pure C/C++ implementatoin 
is typically the best for realworld setups, I hope this hybrid implementation can offer a sense of how much C code can speed up the stack.

To run the fast retriver, first compile the Cython extension. You will need cython and in addtion a c++ compiler for this.
```
cd retriver/retriever_ext
pip install Cython
python setup.py build_ext --inplace
```
In extreme cases where you cannot get access to a compiler, consider use the pure python batched retirever.
## Running Retrieval with C-BM25
To do retrieval about C-BM25, run the following steps,

### prepare idf and doc_len average
To execute BM25, idf and average of doc_len is necessary. Thus, execute following 

```
python create_doc_stat.py \
    --corpus_path /path/to/robust04/corpus.jsonl \
    --tokenizer_path /path/to/tokenizer \
    --output_dir /path/to/robust04/stats
```

`execute/create_doc_stat.sh` contains the codes.
Please execute the bash like following.

```
$ cd execute
$ bash create_doc_stat.sh
```
### encode docs to vecs
Now, we are in `./execute`.
Encode docs to vecs for indexing. 

- set following params in `execute/encode_doc_all.sh`
```
ENCODE_OUT_DIR="/path/to/encode/doc"
DATA_DIR="/path/to/data/"
CKPT_DIR="/path/to/model/"
```

- execute the script
```
$ bash enode_doc_all.sh
```

### encode query to vecs
Encode queries for execution.

- set following params in `execute/encode_query.sh`
```
ENCODE_OUT_DIR="/path/to/encode/doc"
DATA_DIR="/path/to/data/robust04/queries.jsonl"
CKPT_DIR="/path/to/model/"
```

The format of queries.jsonl is the same with BEIR style.

- execute the script
```
$ bash enode_query.sh
```

### reformat query
Reformat encoded queries for execution.

- set  following params in `execute/format_query.sh`
```
ENCODE_OUT_DIR="/path/to/encode/query"
QUERY_DIR="/path/to/index/query"
```
Here, ENCODE_OUT_DIR should be the same with the one in `execute/encode_query.sh`

- execute the script
```
$ bash format_query.sh
```

### build document index shards 
Indexing the encoded docs

- set following params in `execute/sharding_all.sh`
```
ENCODE_OUT_DIR="/path/to/robust04/lss/encode/doc"
INDEX_DIR="/path/to/robust04/lss/index/doc"
n_shard=10
```
Here, ENCODE_OUT_DIR should be the same with the one in `execute/encode_docs_all.sh`
You can change n_shard as your server memory.

- execute the script
```
$ bash sharding_all.sh
```

### execute retrieval
- set following params in `execute/retrieve_cbm25_all.sh`
```
QUERY_DIR="/path/to/index/query"
INDEX_DIR="/path/to/index/doc"
SCORE_DIR="/path/to/score"
```
The QUERY_DIR should be the same with the one in `execute/format_query.sh`
The INDEX_DIR should be the same with the one in `execute/sharding_all.sh`

- execute the script
```
$ bash retrieve_cbm25_all.sh
```

### merge results and evaluate
- set following params in `execute/merge_and_eval.sh`
```
SCORE_DIR="/path/to/score"
QUERY_DIR="/path/to/index/query"
```
The SCORE_DIR should be the same with the one in `execute/retrieve_cbm25_all.sh`
The QUERY_DIR should be the same with the one in `execute/format_query.sh`

- execute the script
```
$ bash merge_and_eval.sh
```


## Running Dense Retrieval 
- execute following
```
python ./evaluate_sbert.py \
  --root_dir $root_dir \
  --dataset trec-robust04-title \
  --model_path $model_path
```

`execute/retrieve_sbert.sh` contains the codes.
Please execute the bash like following

```
$ cd ./execute
$ bash retrieve_sbert.sh <model_path>
```

## Prepareing for measuring speed
Almost all the content in `eval_retrieval_time.ipynb`. 
Before exeute it, please prepare following.

1. Make index file of C-BM25
2. Make index file of Dense Retrieval

The index for C-BM25 is made in `Running Retrieval with C-BM25`.

For indexing the dense retrieval, execute folloing.
```
python encode_dense.py \
   --data_dir /path/to/robust04/ \
   --model_path /path/to/model \
   --output_path /path/to/index-output/dense_index.npy
```

`execute/encode_dense.sh` contains the codes.
Please execute the bash like following

```
$ cd ./execute
$ bash encode_densesh
```
