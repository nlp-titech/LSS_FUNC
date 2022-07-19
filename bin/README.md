# Prepare Reranking Experiment
Here is scripts for reranking experiment. To retrieve top-k documents, you need to prepare index for BM25 retriever. The way of indexing is written in top README of this repo. Please see it.

Next, you need to prepare trained models on MSMARCO. The scripts training models exist in `./training` and the ways of exection are written in `./execute/train_msmarco_v3_lss.sh` for dense retriever and `./execute/train_splade_max.sh` for splade. Please set parameters in these scripts and execute them.

To train COIL and ColBERT, please use these authers codes. COIL is [here](https://github.com/luyug/COIL) and ColBERT is [here](https://github.com/stanford-futuredata/ColBERT).

# Reranking experiments
続きは、ここのREADMEを描くところから。




