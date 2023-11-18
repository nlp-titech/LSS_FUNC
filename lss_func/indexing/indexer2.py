import os
import pickle
from collections import defaultdict
from typing import Dict, List
from dataclasses import dataclass

import datasets
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from .codecs.residual import ResidualCodec, CodecConfig

from lss_func.indexing.saver import IndexSaver
from lss_func.models.coil import Coil


class BeirDocDataset(Dataset):
    columns = ["_id", "title", "text"]

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, p_max_len=512):
        self.nlp_dataset = datasets.load_dataset(
            "json",
            data_files=path_to_json,
        )["train"]
        self.tok = tokenizer
        self.p_max_len = p_max_len

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item) -> [BatchEncoding]:
        pid, title, text = (self.nlp_dataset[item][f] for f in self.columns)
        line = title + " " + text
        encoded_psg = self.tok.encode_plus(
            line,
            max_length=self.p_max_len,
            truncation="only_first",
            return_attention_mask=False,
        )
        return encoded_psg


class CoilSparseIndexer:
    def __init__(self, model_args, train_args, data_args):
        self.model = Coil(model_args.model_name_or_path, model_args)
        self.tokenizer = self.model.tokenizer
        self.device = train_args.device
        self.model.to(self.device)
        self.model.eval()

        self.codec_config = CodecConfig()
        self.dataset = BeirDocDataset(data_args.encode_in_path, self.tokenizer)
        self.n_doc = len(self.dataset)
        self.chunk_size = self.n_doc // data_args.num_index_split + 1
        saver = IndexSaver(data_args.encoded_save_path)
        os.makedirs(data_args.encoded_save_path, exist_ok=True)
        # self.codec_indexser = CoilCodecIndexer(self.codec_config, data_args.encoded_save_path)
        self.codec_indexer = CoilCodecIndexer(self.codec_config, saver)
        self.per_device_eval_batch_size = train_args.per_device_eval_batch_size
        self.dataloader_num_workers = train_args.dataloader_num_workers
        self.p_max_len = data_args.p_max_len

    def indexing(self):
        for i_chunk, i_start in tqdm(enumerate(range(0, self.n_doc, self.chunk_size))):
            i_end = i_start + self.chunk_size - 1
            if i_end > len(self.dataset):
                i_end = len(self.dataset)
            encode_loader = DataLoader(
                [self.dataset[i_] for i_ in range(i_start, i_end)],
                batch_size=self.per_device_eval_batch_size,
                collate_fn=DataCollatorWithPadding(self.tokenizer, max_length=self.p_max_len, padding="max_length"),
                shuffle=False,
                drop_last=False,
                num_workers=self.dataloader_num_workers,
            )
            tok_pid_dict, tok_rep_dict = self._encode_chunk(encode_loader, i_chunk, i_start, i_end)
            if i_chunk == 0:
                codec = self._gen_codec(tok_rep_dict, i_chunk)
            self._indexing_chunk(i_chunk, codec, tok_pid_dict, tok_rep_dict)

    def _gen_codec(self, tok_rep_dict, i_chunk) -> ResidualCodec:
        centroid_embs = self.model.doc_model.model.embeddings.word_embeddings.weight.cpu()
        embs = torch.cat([torch.cat(emb) for emb in tok_rep_dict.values()])
        codec = self.codec_indexer.make_codecs_of_segment(embs, centroid_embs)
        self.codec_indexer._save_codecs(codec, i_chunk)
        return codec

    def _indexing_chunk(self, i_chunk, codec, tok_pid_dict, tok_rep_dict):
        new_index = {}
        chunk_residual = 0.0
        for word, emb in tok_rep_dict.items():
            compressed_embs = codec.compress(torch.cat(emb))
            emb_reconstruct = codec.decompress(compressed_embs)
            chunk_residual += torch.sum(torch.cat(emb) - emb_reconstruct)
            new_index[word] = compressed_embs

        print(f"chunk_residual = {chunk_residual}")

        self.codec_indexer._save_index(new_index, i_chunk)

    def _encode_chunk(self, encode_loader, i_chunk, i_start_point, i_end_point):
        def chunk_dataset(start, end):
            for i in range(start, end):
                data = self.dataset.nlp_dataset[i]
                pid, title, text = (data[f] for f in self.dataset.columns)
                line = title + " " + text
                yield pid, line

        encoded = []
        this_chunk_dataset = chunk_dataset(i_start_point, i_end_point)
        rm_counter = 0
        for batch in tqdm(encode_loader):
            rm_counter += int(torch.sum(batch["attention_mask"] == 0))
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    _, reps = self.model.encode_corpus_raw_proc(batch)
                    encoded.append(reps.cpu())
        all_reps = torch.cat(encoded)
        all_pids = []
        tok_rep_dict = defaultdict(list)
        tok_pid_dict = defaultdict(list)

        for pos, (pid, passage) in enumerate(tqdm(this_chunk_dataset)):
            all_pids.append(pid)
            t_passage = self.tokenizer.encode_plus(passage, max_length=self.p_max_len)["input_ids"][1:-1]
            rep_dict = defaultdict(list)
            for sent_pos, tok_id in enumerate(t_passage):
                if tok_id in self.model.remove_tok:
                    rm_counter += 1
                    continue
                rep_dict[tok_id].append(torch.unsqueeze(all_reps[pos, sent_pos, :], 0))  # skip cls
            for tok_id, tok_rep in rep_dict.items():
                if tok_id in self.model.remove_tok:
                    rm_counter += 1
                    continue
                tok_rep_dict[tok_id].append(torch.cat(tok_rep))
                tok_pid_dict[tok_id].extend([pid for _ in range(len(tok_rep))])

        all_tensor_size = 0
        for k, v in tok_pid_dict.items():
            tensor_size = torch.cat(tok_rep_dict[k], dim=0).shape[0]
            pid_size = len(list(v))
            assert tensor_size == pid_size
            all_tensor_size += tensor_size
        assert len(all_pids) == all_reps.shape[0]
        assert all_tensor_size == all_reps.shape[0] * all_reps.shape[1] - rm_counter
        return tok_pid_dict, tok_rep_dict


class CoilCodecIndexer:
    def __init__(self, config: CodecConfig, saver: IndexSaver):
        self.config = config
        self.saver = saver
        self.use_gpu = config.total_visible_gpus > 0

    def make_codecs_of_segment(self, embs: torch.Tensor, centroid_embs: torch.Tensor):
        # The origin of this part is from https://github.com/stanford-futuredata/ColBERT/blob/7d52265a4de953dd05270437800aba583e2281cd/colbert/indexing/collection_indexer.py#L214-L219
        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroid_embs, embs)

        print(f"avg_residual = {avg_residual}")

        codec = ResidualCodec(
            config=self.config,
            centroids=centroid_embs,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )

        return codec

    def _compute_avg_residual(self, centroids, heldout):
        # The origin of this part is from https://github.com/stanford-futuredata/ColBERT/blob/7d52265a4de953dd05270437800aba583e2281cd/colbert/indexing/collection_indexer.py#L289-L313
        compressor = ResidualCodec(config=self.config, centroids=centroids, avg_residual=None)
        if self.use_gpu:
            heldout = heldout.to("cuda")

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device="cuda" if self.use_gpu else "cpu")
        heldout_reconstruct = compressor.lookup_centroids(
            heldout_reconstruct, out_device="cuda" if self.use_gpu else "cpu"
        )
        heldout_avg_residual = heldout - heldout_reconstruct
        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        np_heldout_avg_residual = heldout_avg_residual.detach().cpu().numpy()

        num_options = 2**self.config.nbits
        quantiles = np.arange(0, num_options) * (1 / num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[1:], quantiles + (0.5 / num_options)

        np_bucket_cutoffs = np.quantile(np_heldout_avg_residual, bucket_cutoffs_quantiles)
        np_bucket_weights = np.quantile(np_heldout_avg_residual, bucket_weights_quantiles)
        bucket_cutoffs = torch.from_numpy(np_bucket_cutoffs)
        bucket_weights = torch.from_numpy(np_bucket_weights)

        print(
            f"#> Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}"
        )
        print(f"#> Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}")

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

    def _save_index(self, codec_index: Dict, num_chunk: int):
        with self.saver.thread():
            save_path = os.path.join(self.saver.index_dir_, f"{num_chunk}_index.pt")
            with open(save_path, "wb") as fOut:
                pickle.dump(codec_index, fOut)

    def _save_codecs(self, codec, num_chunk: int):
        self.saver.save_codec(codec)
