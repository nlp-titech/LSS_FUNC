import argparse
import os
import json
from collections import Counter, defaultdict
from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer


def calc_idf_and_doclen(corpus, tokenizer, sep=" "):
    doc_lens = []
    df = Counter()
    for cid in tqdm(corpus.keys()):
        text = corpus[cid]["title"] + sep + corpus[cid]["text"]
        input_ids = tokenizer(text)["input_ids"]
        doc_lens.append(len(input_ids))
        df.update(list(set(input_ids)))

    idf = defaultdict(float)
    N = len(corpus)
    for w, v in df.items():
        idf[w] = np.log(N / v)

    doc_len_ave = np.mean(doc_lens)
    return idf, doc_len_ave


def main(args):
    corpus = {}
    with open(args.corpus_path) as f:
        for line in f:
            jline = json.loads(line)
            cid = jline["_id"]
            del jline["_id"]
            corpus[cid] = jline

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer)
    idf_path = os.path.join(args.output_dir, "idf.json")
    with open(idf_path, "w") as f:
        json.dump(idf, f)

    doc_len_ave_path = os.path.join(args.output_dir, "doc_len_ave.npy")
    np.save(doc_len_ave_path, doc_len_ave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--output_dir")

    args = parser.parse_args()
    main(args)
