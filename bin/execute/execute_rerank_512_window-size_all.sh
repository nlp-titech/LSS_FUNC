model_name_or_path=$1
root_dir=$2

# datasets=("arguana" "climate-fever" "cqadupstack" "dbpedia-entity" "fever" \
# "fiqa" "germanquad" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")

datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
"fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
"trec-covid" "trec-robust04-desc" "trec-robust04-title" "webis-touche2020")

window_size=(1 2 3 5 7 9)
# datasets=("fever" "hotpotqa")
# datasets=("quora")

# datasets=("trec-robust04-desc" "trec-robust04-title")

for dataset in ${datasets[@]};
do
  echo $i $dataset
  for ws in ${window_size[@]};
  do
    qsub -g gcb50243 execute_rerank_dataset_512_window-size.sh $model_name_or_path $dataset $root_dir $ws
    # qsub -g tga-nlp-titech execute_pyserini.sh $model_name_or_path $dataset $root_dir                                                                     d
  done
done
