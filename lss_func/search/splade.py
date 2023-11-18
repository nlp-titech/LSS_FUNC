import os
import logging
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.file_utils import ModelOutput
from tqdm.autonotebook import trange

try:
    import sentence_transformers
    from sentence_transformers.util import batch_to_device
except ImportError:
    print("Import Error: could not load sentence_transformers... proceeding")
logger = logging.getLogger(__name__)


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


@dataclass
class RankTrainOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: torch.FloatTensor = None


class BEIRSpladeModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(self.tokenizer, queries, is_q=True, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        return self.model.encode_sentence_bert(self.tokenizer, sentences, maxlen=self.max_length)


class Splade(nn.Module):
    def __init__(
        self, model_type_or_dir, lambda_d=0.0008, lambda_q=0.0006, **kwargs
    ):
        super().__init__()
        if os.path.exists(os.path.join(model_type_or_dir, "0_MLMTransformer")):
            print("path", model_type_or_dir)
            model_type_or_dir = os.path.join(model_type_or_dir, "0_MLMTransformer")

        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)

        self.loss_func = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(
        self,
        inputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vecs = [self.encode(**input_) for input_ in inputs]
        q_vecs = vecs[0]
        d_vecs = torch.cat(vecs[1:])
        scores = torch.mm(q_vecs, d_vecs.T)
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = self.loss_func(scores, labels)
        loss += self.lambda_d * self.FLOPS(d_vecs)
        loss += self.lambda_q * self.FLOPS(q_vecs)
        return RankTrainOutput(loss=loss, scores=scores)

    def encode(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        vec = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1).values
        
        return vec

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode_sentence_bert(
        self,
        tokenizer,
        sentences: Union[str, List[str], List[int]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        maxlen: int = 512,
        is_q: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == "token_embeddings":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            # features = tokenizer(sentences_batch)
            # print(sentences_batch)
            features = tokenizer(
                sentences_batch,
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="only_first",  # truncates to self.max_length
                max_length=maxlen,
                return_attention_mask=True,
                return_tensors="pt",
            )
            # print(features)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.encode(**features)
                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0 : last_mask_id + 1])
                else:  # Sentence embeddings
                    # embeddings = out_features[output_value]
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings