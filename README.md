# LSS_FUNC
Repo for our TOD Journal Paper [](). This code covers learning models using experiments in the paper, executing reranking tasks, and executing retrieval tasks.

# Code structure
The codes for training models and executing reranking tasks are in `bin/execute`. Please check the bash files.

The codes for retrieval task and measuring the speed in `retriever`. In addition, the executer are in `retriever/execute`. Please check the bash files.

## Dependencies
The code has been tested with,
```
pytorch==1.8.1
transformers==4.2.1
datasets==1.1.3
beir
sentence-transformers
pyserini
```

To use the retriever, you need in addition,
```
torch_scatter==2.0.6
```

## Premise
We use BEIR dataset. Please set each dataset under root_dir you set. Please see the way of prepareing dataset [here](https://github.com/beir-cellar/beir/tree/main/examples/dataset).


## Prepare BM25 index for reranking experiment
Please create index for BM25 using pyserini.
`util/index_pyserini.sh` can make it.
execute the bash file like following.
```
$ bash util index_pyserini.sh <root_dir of dataset>
```

## Reranking Experiment
See Readme at `./bin`

## Retrieval Experiment
See Readme at `./retriever`

## Code for Case Study
See Readme at `./analysis`