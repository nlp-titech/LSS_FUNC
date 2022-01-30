import argparse
import logging
from pathlib import Path


logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main(args):
    corpus_file = Path(args.corpus_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_num = args.split_num

    corpus = []
    with corpus_file.open(mode="r") as f:
        for line in f:
            corpus.append(line.strip())

    all_num = len(corpus)
    chunk_len = all_num // split_num
    for i_chunk, chunk_start in enumerate(range(0, all_num, chunk_len)):
        chunk_end = chunk_start + chunk_len - 1
        corpus_chunk = corpus[chunk_start:chunk_end]
        chunk_file = out_dir / f"split{i_chunk:02}.json"
        logging.info(f"output {chunk_file}")
        with chunk_file.open(mode="w") as f:
            for line in corpus_chunk:
                print(line, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file")
    parser.add_argument("--out_dir")
    parser.add_argument("--split_num", default=10, type=int)

    args = parser.parse_args()
    main(args)
