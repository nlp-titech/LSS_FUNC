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
import numpy as np
import torch
import argparse

from tqdm import tqdm
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader


logger = logging.getLogger(__name__)

GLUE_PORTION = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--model_path")
    parser.add_argument("--output_path")
    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()
    corpus, queries, qrels = GenericDataLoader(args.data_dir).load(split="test")
    print(type(corpus))
    model = models.SentenceBERT(args.model_path)

    rep_index = []
    for idx in tqdm(range(0, len(corpus), args.batch_size)):
        end_idx = idx + args.batch_size
        batch_corpus = corpus[idx:end_idx]
        with torch.no_grad():
            c_rep = model.encode_corpus(batch_corpus, batch_size=args.batch_size)
        rep_index.append(c_rep.cpu())

    rep_index = torch.cat(rep_index)
    rep_index = rep_index.numpy()
    np.save(args.output_path, rep_index)


if __name__ == "__main__":
    main()
