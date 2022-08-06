root_dir=$1

# for filepath in `find $root_dir -name corpus.jsonl`;
# do
#   echo $filepath
#   inputdir=`dirname $filepath`
#   outdir=`dirname $inputdir`
#   outdir=$outdir/index
#   mkdir $outdir
#   python -m pyserini.index -collection JsonCollection \
#                          -generator DefaultLuceneDocumentGenerator \
#                          -threads 1 \
#                          -input $inputdir \
#                          -index $outdir/lucene-index.sep_title.pos+docvectors+raw \
#                          -storePositions -storeDocvectors -storeRaw
# done

python -m pyserini.index -collection JsonCollection \
                          -generator DefaultLuceneDocumentGenerator \
                          -threads 1 \
                          -input $root_dir/scifact/corpus \
                          -index $root_dir/scifact/index/lucene-index.sep_title.pos+docvectors+raw \
                          -storePositions -storeDocvectors -storeRaw