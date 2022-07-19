root_dir=$1

datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
"fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
"trec-covid" "trec-robust04-desc" "trec-robust04-title" "webis-touche2020")


for dataset in ${datasets[@]};
do
  echo $i $dataset
  bash execute_pyserini_bm25_dataset.sh $dataset $root_dir
done
