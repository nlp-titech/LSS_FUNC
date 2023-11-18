import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from lss_func.models.coil import Coil
from lss_func.indexing.codecs.residual import ResidualCodec, CodecConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Searcher:
    def __init__(self, model: Coil, index_dir: Path, bm25_index: Dict, n_chunk: int):
        self.model = model
        self.index_dir = index_dir
        self.n_chunk = n_chunk
        self.bm25_index = bm25_index

    def search_query(self, query):
        raise NotImplementedError

    def calc_score(self, codec_index, pid_index, codec, tokens, embeds):
        chunk_scores = defaultdict(float)
        for t, e in zip(tokens, embeds):
            d2scores = self.bm25_index[t]
            c_embs = codec_index[t]
            i_embs = codec.decompress(c_embs)
            logger.info(i_embs)
            scores = torch.mv(i_embs, e.squeeze())
            s_dids = pid_index[t]
            for did, score in zip(s_dids, scores):
                chunk_scores[did] = max(score, chunk_scores[did])
            for did, weight in d2scores.items():
                chunk_scores[did] *= weight
        logging.info(chunk_scores)
        return chunk_scores

    def load_index(self, load_path):
        index_path = load_path / "codec_index.pt"
        with index_path.open(mode="rb") as fB:
            codec_index = pickle.load(fB)
        return codec_index

    def load_pids(self, load_path):
        index_path = load_path / "pid_index.pt"
        with index_path.open(mode="rb") as fB:
            pid_index = pickle.load(fB)
        return pid_index

    def load_codec(self, load_path):
        codec = ResidualCodec.load(load_path)
        return codec


class SearcherForEachChunkCodec(Searcher):
    def search_query(self, query):
        tokens = self.model.tokenizer(query, return_tensors="pt")
        embeds = self.model.encode_query_raw_proc(tokens)
        tokens = tokens["input_ids"][0][1:-1]

        all_scores = defaultdict(float)
        for i in tqdm(range(self.n_chunk)):
            load_path = self.index_dir.joinpath(f"chunk_{i}")
            codec_index = self.load_index(load_path)
            codec = self.load_codec(load_path)
            did_index = self.load_pids(load_path)
            chunk_scores = self.calc_score(codec_index, did_index, codec, tokens, embeds)
            logger.info(chunk_scores)
            for did, score in chunk_scores.items():
                all_scores[did] += score
        return all_scores


class SearcherForEachChunkCodecLoadAll(Searcher):
    def __init__(self, model: Coil, index_dir: Path, bm25_index: Dict, n_chunk: int):
        super().__init__(model, index_dir, bm25_index, n_chunk)
        self.index = [self.load_chunk(i) for i in range(n_chunk)]

    def load_chunk(self, i_chunk: int):
        load_path = self.index_dir.joinpath(f"chunk_{i_chunk}")
        codec_index = self.load_index(load_path)
        codec = self.load_codec(load_path)
        did_index = self.load_pids(load_path)
        indexes = {"codec_index": codec_index, "codec": codec, "did_index": did_index}
        return indexes

    def search_query(self, query):
        tokens = self.model.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            embeds = self.model.encode_query_raw_proc(tokens)
        tokens = tokens["input_ids"][0, 1:-1].tolist()

        all_scores = defaultdict(float)
        for i in range(self.n_chunk):
            codec_index = self.index[i]["codec_index"]
            codec = self.index[i]["codec"]
            did_index = self.index[i]["did_index"]
            chunk_scores = self.calc_score(codec_index, did_index, codec, tokens, embeds)
            for did, score in chunk_scores.items():
                all_scores[did] += score
        return all_scores


class SearcherForAllChunkCodec(Searcher):
    def search_query(self, query):
        tokens = self.model.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            embeds = self.model.encode_query_raw_proc(tokens)
        tokens = tokens["input_ids"][1:-1]

        codec = self.load_codec(self.index_dif)
        all_scores = defaultdict(float)
        for i in range(self.n_chunk):
            chunk_load_path = self.index_dir.joinpath(f"chunk_{i}")
            codec_index = self.load_index(chunk_load_path)
            did_index = self.load_pids(chunk_load_path)
            chunk_scores = self.calc_score(codec_index, did_index, codec, tokens, embeds)
            logging.info(chunk_scores)
            for did, score in chunk_scores.items():
                all_scores[did] += score
        return all_scores
