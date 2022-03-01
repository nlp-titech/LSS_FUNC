import argparse
import json
from pathlib import Path
from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


class Converter:
    def __init__(self, sep: str = ""):
        self.sep = sep

    def convert(self, in_files, out_root_dir):
        for in_file in in_files:
            logger.info(in_file)
            this_file_name = self._filename(in_file.name)
            this_dir = self._out_dir(out_root_dir, in_file)
            this_dir.mkdir(parents=True, exist_ok=True)
            out_file = this_dir / this_file_name
            self._convert(in_file, out_file)

    def _convert(self, jline, out_file_streamer):
        raise NotImplementedError

    def _out_dir(self, out_root_dir, in_file):
        return out_root_dir.joinpath(in_file.parent.name)

    def _filename(self, filename):
        return filename


class CorpusConverter(Converter):
    def __init__(self, sep_title: bool, sep: str = ""):
        super().__init__(sep)
        self.sep_title = sep_title

    def _convert(self, in_file, out_file):
        with in_file.open(mode="r") as f:
            with out_file.open(mode="w") as g:
                for line in f:
                    outline = dict()
                    jline = json.loads(line)
                    outline["id"] = jline["_id"]
                    if self.sep_title:
                        outline["title"] = jline["title"].strip()
                        outline["contents"] = jline["text"].strip()
                    else:
                        outline["contents"] = (jline["title"] + self.sep + jline["text"]).strip()
                    print(json.dumps(outline), file=g)

    def _out_dir(self, out_root_dir, in_file):
        in_file_root = in_file.parent.name
        return out_root_dir.joinpath(in_file_root, "corpus")


class QueryConverter(Converter):
    def _convert(self, in_file, out_file):
        with in_file.open(mode="r") as f:
            with out_file.open(mode="w") as g:
                for line in f:
                    jline = json.loads(line)
                    outline_id = jline["_id"]
                    query = " ".join(jline["text"].strip().split("\t"))
                    oline = "\t".join((outline_id, query))
                    print(oline, file=g)

    def _out_dir(self, out_root_dir, in_file):
        in_file_root = in_file.parent.name
        return out_root_dir.joinpath(in_file_root, "query")

    def _filename(self, filename):
        return filename.replace(".jsonl", ".tsv")


class QrelConverter(Converter):
    def _convert(self, in_file, out_file):
        with in_file.open(mode="r") as f:
            with out_file.open(mode="w") as g:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    sline = line.strip().split("\t")
                    oline = "\t".join([sline[0], "0", sline[1], sline[2]])
                    print(oline, file=g)

    def _out_dir(self, out_root_dir, in_file):
        in_file_root = in_file.parent.parent.name
        return out_root_dir.joinpath(in_file_root, "qrels")


def main(args):
    in_dir = Path(args.in_dir)
    corpus_files = in_dir.glob("*/corpus.jsonl")
    query_files = in_dir.glob("*/queries.jsonl")
    qrel_files = in_dir.glob("*/qrels/*.tsv")
    out_root_dir = Path(args.out_dir)

    corpus_converter = CorpusConverter(args.sep_title)
    query_converter = QueryConverter()
    qrel_converter = QrelConverter()

    corpus_converter.convert(corpus_files, out_root_dir)
    query_converter.convert(query_files, out_root_dir)
    qrel_converter.convert(qrel_files, out_root_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--sep_title", action="store_true")

    args = parser.parse_args()
    main(args)
