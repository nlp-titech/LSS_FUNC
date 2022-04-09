root_dir=$1

# datasets=("arguana" "climate-fever" "cqadupstack" "dbpedia-entity" "fever" \
# "fiqa" "germanquad" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")

# datasets=("arguana" "climate-fever" "dbpedia-entity" "fever" \
# "fiqa" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")
# datasets=("arguana")
datasets=("trec-robust04-desc" "trec-robust04-title")

for dataset in ${datasets[@]};
do
  echo $i $dataset
  qsub -g gcb50243 execute_pyserini.sh $dataset $root_dir
done
