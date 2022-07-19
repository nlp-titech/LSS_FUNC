import argparse
import json
from pathlib import Path

import ir_datasets
from tqdm import tqdm


def main(args):
    # now this scrpit works on only robust04.
    datasets = ir_datasets.load("trec-robust04")
    out_dir = Path(args.out_dir)
    out_corpus_path = out_dir / "corpus.jsonl"
    out_query_path = out_dir / "queries.jsonl"
    out_qrel_dir = out_dir / "qrels"
    out_qrel_dir.mkdir(parents=True, exist_ok=True)
    out_qrel_path = out_qrel_dir / "test.tsv"

    with out_query_path.open(mode="w") as f:
        print(out_query_path)
        for query in tqdm(datasets.queries_iter()):
            oquery = {"_id": query.query_id, "text": query.description, "metadata": {}}
            joquery = json.dumps(oquery)
            print(joquery, file=f)

    with out_qrel_path.open(mode="w") as f:
        header = "\t".join(("query-id", "corpus-id", "score"))
        print(header, file=f)
        print(out_qrel_path)
        for qrel in tqdm(datasets.qrels_iter()):
            oline = "\t".join((qrel.query_id, qrel.doc_id, str(qrel.relevance)))
            print(oline, file=f)

    with out_corpus_path.open(mode="w") as f:
        print(out_corpus_path)
        for doc in tqdm(datasets.docs_iter()):
            odoc = {"_id": doc.doc_id, "title": "", "text": doc.text, "metadata": {}}
            jodoc = json.dumps(odoc)
            print(jodoc, file=f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir")
    
    args = parser.parse_args()
    main(args)