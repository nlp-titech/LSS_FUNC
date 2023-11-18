import pickle
import os
import queue
import ujson
import threading
from typing import Dict

from contextlib import contextmanager

from .codecs.residual import ResidualCodec


class IndexSaver:
    def __init__(self, index_dir):
        self.index_dir_ = index_dir

    def save_codec(self, codec):
        codec.save(index_path=self.index_dir_)

    def load_codec(self):
        return ResidualCodec.load(index_dir=self.index_dir_)

    def try_load_codec(self):
        try:
            ResidualCodec.load(index_dir=self.index_dir_)
            return True
        except Exception as e:
            return False

    def check_chunk_exists(self, chunk_idx):
        # TODO: Verify that the chunk has the right amount of data?

        doclens_path = os.path.join(self.index_dir_, f"doclens.{chunk_idx}.json")
        if not os.path.exists(doclens_path):
            return False

        metadata_path = os.path.join(self.index_dir_, f"{chunk_idx}.metadata.json")
        if not os.path.exists(metadata_path):
            return False

        path_prefix = os.path.join(self.index_dir_, str(chunk_idx))
        codes_path = f"{path_prefix}.codes.pt"
        if not os.path.exists(codes_path):
            return False

        residuals_path = f"{path_prefix}.residuals.pt"  # f'{path_prefix}.residuals.bn'
        if not os.path.exists(residuals_path):
            return False

        return True

    @contextmanager
    def thread(self):
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(self, codec_index: Dict, chunk_idx):
        # compressed_embs = self.codec.compress(embs)

        # self.saver_queue.put((chunk_idx, offset, compressed_embs, doclens))
        self.saver_queue.put((codec_index, chunk_idx))

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _writer_chunk_to_disk(self, codec_index: Dict, num_chunk: int):
        save_dir = os.path.join(self.index_dir, f"chunk_{num_chunk}")
        save_path = os.path.join(save_dir, "index.pt")
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "wb") as fOut:
            pickle.dump(codec_index, fOut)

    # def _write_chunk_to_disk(self, chunk_idx, offset, compressed_embs, doclens):
    #     path_prefix = os.path.join(self.index_dir_, str(chunk_idx))
    #     compressed_embs.save(path_prefix)

    #     doclens_path = os.path.join(self.index_dir_, f'doclens.{chunk_idx}.json')
    #     with open(doclens_path, 'w') as output_doclens:
    #         ujson.dump(doclens, output_doclens)

    #     metadata_path = os.path.join(self.index_dir_, f'{chunk_idx}.metadata.json')
    #     with open(metadata_path, 'w') as output_metadata:
    #         metadata = {'passage_offset': offset, 'num_passages': len(doclens), 'num_embeddings': len(compressed_embs)}
    #         ujson.dump(metadata, output_metadata)
