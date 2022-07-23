model_name_or_path=$1
index_root=$2
data_root=$3

datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
"fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
"trec-covid" "trec-robust04-desc" "trec-robust04-title" "webis-touche2020")

for dataset in ${datasets[@]};
do
  echo $i $dataset
  bash execute_rerank_coil_dataset.sh $model_name_or_path $dataset $root_dir $data_root
done
