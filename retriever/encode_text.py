# coding=utf-8
# Copyright 2021 COIL authors
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch

from lss_func.arguments import ModelArguments, DataArguments, COILTrainingArguments as TrainingArguments
from lss_func.beir_datasets import BeirDocDataset, BeirQueryDataset
from lss_func.coil import Coil
from transformers import DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)


logger = logging.getLogger(__name__)

GLUE_PORTION = 0.0


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        train_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model params %s", model_args)

    # Set seed
    set_seed(training_args.seed)

    model = Coil(model_args.model_name_or_path, model_args)
    tokenizer = model.d_tokenizer

    if training_args.local_rank > -1:
        raise NotImplementedError("Encoding with multi processes is not implemented.")
    from torch.utils.data import DataLoader

    if data_args.query:
        encode_dataset = BeirQueryDataset(data_args.encode_in_path, tokenizer, p_max_len=data_args.p_max_len)
    else:
        encode_dataset = BeirDocDataset(data_args.encode_in_path, tokenizer, p_max_len=data_args.p_max_len)
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, max_length=data_args.p_max_len, padding="max_length"),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    model.to(training_args.device)
    model.eval()
    for batch in tqdm(encode_loader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                _, reps = model.encode_corpus_raw(batch)
                reps /= torch.norm(reps, dim=2).unsqueeze(-1)
                encoded.append(reps.cpu())
    all_reps = torch.cat(encoded).numpy()
    all_pids = []
    tok_rep_dict = defaultdict(list)
    tok_pid_dict = defaultdict(list)

    target_text_fields = ["title", "text"]
    for pos, entry in enumerate(tqdm(encode_dataset.nlp_dataset)):
        pid = entry["_id"]
        passage = " ".join([entry[field] for field in target_text_fields if field in entry])
        all_pids.append(pid)
        t_passage = model.q_tokenizer(passage, max_length=data_args.p_max_len)["input_ids"][1:-1]
        rep_dict = defaultdict(list)
        for sent_pos, tok_id in enumerate(t_passage):
            if tok_id in model.remove_tok:
                continue
            rep_dict[tok_id].append(all_reps[pos][sent_pos])  # skip cls
        for tok_id, tok_rep in rep_dict.items():
            if tok_id in model.remove_tok:
                continue
            tok_rep_dict[tok_id].extend(tok_rep)
            tok_pid_dict[tok_id].extend([pid for _ in range(len(tok_rep))])
    np.save(os.path.join(data_args.encoded_save_path, f"cls_pids"), np.array(all_pids))
    offset_dict = {}
    tok_all_ids = []
    tok_all_reps = []
    _offset = 0
    for tok_id in tok_pid_dict:
        tok_rep = np.stack(tok_rep_dict[tok_id], axis=0)
        offset_dict[tok_id] = (_offset, tok_rep.shape[0])
        _offset += tok_rep.shape[0]
        tok_all_ids.append(np.array(tok_pid_dict[tok_id]))
        tok_all_reps.append(tok_rep)
    np.save(os.path.join(data_args.encoded_save_path, f"tok_pids"), np.concatenate(tok_all_ids, axis=0))
    np.save(os.path.join(data_args.encoded_save_path, f"tok_reps"), np.concatenate(tok_all_reps, axis=0))
    with open(os.path.join(data_args.encoded_save_path, f"offsets.pkl"), "wb") as pf:
        pickle.dump(offset_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
