model_name_or_path=$1
root_dir=$2

# datasets=("arguana" "climate-fever" "cqadupstack" "dbpedia-entity" "fever" \
# "fiqa" "germanquad" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")

datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
"fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
"trec-covid" "trec-robust04-desc" "trec-robust04-title" "webis-touche2020")

# datasets=("fever" "hotpotqa")
# datasets=("trec-covid")

# datasets=("trec-robust04-desc" "trec-robust04-title")

for dataset in ${datasets[@]};
do
  echo $i $dataset
  qsub -g gcb50243 execute_rerank_dataset_512.sh $model_name_or_path $dataset $root_dir
  # qsub -g tga-nlp-titech execute_pyserini.sh $model_name_or_path $dataset $root_dir                                                                          
done
