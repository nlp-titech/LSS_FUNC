import argparse
import json
import gzip
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def main(args):
    root_dir = Path(args.root_dir)
    out_file = Path(args.out_dir) / "coil_train_data.jsonl"
    corpus_path = root_dir / "collection.tsv"
    query_path = root_dir / "queries.train.tsv"
    neg_path = root_dir / "msmarco-hard-negatives.jsonl.gz"
    num_negs = args.num_negs
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    corpus = dict()
    print("loading corpus")
    with open(corpus_path) as f:
        for line in f:
            did, passage = line.strip().split("\t")
            corpus[did] = passage

    queries = dict()
    print("loading query")
    with open(query_path) as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    with gzip.open(neg_path, "rt") as fIn:
        with out_file.open(mode="w") as fOut:
            for line in tqdm(fIn):
                data = json.loads(line)

                # Get the positive passage ids
                qid = str(data["qid"])
                pos_pids = data["pos"]

                if len(pos_pids) == 0:  # Skip entries without positives passages
                    continue

                query = queries[qid]
                t_query = tokenizer.encode_plus(query, add_special_tokens=False)["input_ids"]
                for pos_pid in pos_pids:
                    pos_pid = str(pos_pid["pid"])
                    out = dict()
                    pos_doc = corpus[pos_pid]
                    t_pos_doc = tokenizer.encode_plus(pos_doc, add_special_tokens=False)["input_ids"]

                    out["qry"] = {"qid": qid, "query": t_query}
                    out["pos"] = [{"pid": pos_pid, "passage": t_pos_doc}]
                    out["neg"] = []

                    if "bm25" not in data["neg"]:
                        continue

                    neg_pids = data["neg"]["bm25"]
                    negs_added = 0
                    for item in neg_pids:
                        if pos_pid not in neg_pids:
                            nid = item["pid"]
                            negs_added += 1
                            neg_doc = corpus[nid]
                            t_neg_doc = tokenizer.encode_plus(neg_doc, add_special_tokens=False)["input_ids"]
                            out["neg"].append({"pid": nid, "passage": t_neg_doc})

                            if negs_added >= num_negs:
                                break

                    print(json.dumps(out), file=fOut)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir")
    parser.add_argument("--tokenizer")
    parser.add_argument("--num_negs", default=101, type=int)
    parser.add_argument("--out_dir")

    args = parser.parse_args()

    main(args)
