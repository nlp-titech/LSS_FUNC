# Prepare Reranking Experiment
Scripts for our reranking experiment is mainly in `./reranking`. To retrieve top-k documents, you need to prepare index for BM25 retriever. The way of indexing is written in top README of this repo. Please see it.

Next, you need to prepare trained models on MSMARCO. The scripts training models exist in `./training` and the ways of exection are written in `./execute/train_msmarco_v3_lss.sh` for dense retriever and `./execute/train_splade_max.sh` for splade. Please set parameters in these scripts and execute them.

To train COIL and ColBERT, please use these authers codes. COIL is [here](https://github.com/luyug/COIL) and ColBERT is [here](https://github.com/stanford-futuredata/ColBERT).

The authers of COIL used their own format to train a model. To convert SBERT format to their format, we prepare `<this repo>/util/mk_coil_train_data_from_sbert_msmarco.py`

# Reranking experiments
## C-BM25
The script for C-BM25 is `./rerank/evaluate_bm25_pyserini_coil_reranking.py`. 
To execute C-BM25, you need set following argument: `--tok_dim 768` ,`--enc_raw True`, and `--funcs maxsim_bm25_qtf` (These enc_raw and funcs settings are default.)
The settings are also written in `./execute/execute_rerank_lss_dataset.sh` as the case `model_type=dense`. Please refer it.

If you would like to ablation to funcs, you can also set `--funcs maxsim_qtf` and `--funcs maxsim_idf_qtf`. The former is the case without BM25 weight. The latter is the case with only IDF weight.

For ablation to window-size, please set `--window_size`. 

## HC-BM25
The script for HC-BM25 is `./rerank/evaluate_bm25_pyserini_coil_reranking_hybrid.py`. 
To execute HC-BM25, you need set following argument: `tok_dim=768` ,`enc_raw=True`, and `funcs=maxsim_bm25_qtf` (These enc_raw and funcs settings are default.)
The settings are also written in `./execute/execute_rerank_lss_hybrid_dataset.sh` as the case `model_type=dense`. Please refer it.


## Coil-tok
Ths script for Coil-tok is also `./rerank/evaluate_bm25_pyserini_coil_reranking.py`.
To execxute Coil-tok, you need set following argument: `tok_dim=32` ,`enc_raw=False`, and `funcs=maxsim_qtf`. 
The settings are also written in `./execute/execute_rerank_coil_dataset.sh` as the case `model_type=coil`. Please refer it.

## Dense Retrieval
The script for dense retrieval is `./rerank/evaluate_bm25_pyserini_sbert_reranking.py`.
The settings are  written in `./execute/execute_rerank_sbert_dataset.sh`.

## Weighted Dense Retrieval
The script for dense retrieval is `./rerank/evaluate_bm25_pyserini_sbert_weight_reranking.py`.
The settings are  written in `./execute/execute_rerank_sbert_weight_dataset.sh`.

## H-BM25
The script for dense retrieval is `./rerank/evaluate_bm25_pyserini_sbert_reranking.py`.
You need execute the script with `--hybrid`.
The settings are  written in `./execute/execute_rerank_hybrid_dataset.sh`.

## BM25
The script for BM25 is `./rerank/evaluate_bm25_pyserini.py`.
The settings are written in `./execute/execute_pyserini_bm25_dataset.sh`.

## Cross Encoder
The script for Cross Encoder is `./rerank/evaluate_bm25_pyserini_ce_rerank.py`.
The settings are written in `./execute/execute_rerank_ce_dataset.sh`.

## ColBERT
The script for ColBERT is `evaluate_bm25_pyserini_colbert_reranking.py`.
The settings are written in `execute_rerank_colbert_dataset.sh`.

## SPLADE
The script for SPLADE is `evaluate_bm25_pyserini_splade_reranking.py`
the settings are written in `execute_rerank_splade_dataset.sh`.

# Others
For utility, we prepare `**_all.sh` to execute experiments on all dataset at once.
Note that we did not test these utility codes. 










