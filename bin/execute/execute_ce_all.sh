model_type=$1
root_dir=$2

# datasets=("arguana" "climate-fever" "cqadupstack" "dbpedia-entity" "fever" \
# "fiqa" "germanquad" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")

datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
"fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
"trec-covid" "trec-robust04-desc" "trec-robust04-title" "webis-touche2020")

# datasets=("fever" "hotpotqa")
# datasets=("quora")

# datasets=("trec-robust04-desc" "trec-robust04-title")

for dataset in ${datasets[@]};
do
  echo $i $dataset
  qsub -g gcb50243 execute_ce_dataset.sh $model_type  $dataset $root_dir
done
