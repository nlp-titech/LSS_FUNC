import os
import logging
from typing import Dict, List, Union, Tuple, Iterable, Optional

import torch
from torch import Tensor, nn

from transformers import PreTrainedModel
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from lss_func.arguments import ModelArguments


logger = logging.getLogger(__name__)
TOKEN_POOLER = "token"
LOCAL_AVE_POOLER = "ave"


class Coil:
    def __init__(self, model_path: Union[str, Tuple], model_args: ModelArguments, sep: str = " ", **kwargs):
        self.sep = sep

        if isinstance(model_path, str):
            self.q_model = COIL_Core.from_pretrained(model_args, model_path)
            self.doc_model = self.q_model
            self.tokenizer = self.q_model.tokenizer

        elif isinstance(model_path, tuple):
            self.q_model = COIL_Core.from_pretrained(model_args, model_path[0])
            self.doc_model = COIL_Core.from_pretrained(model_args, model_path[1])
            self.q_tokenizer = self.q_model.tokenizer
            self.d_tokenizer = self.doc_model.tokenizer

        self.remove_tok = {
            self.doc_model.tokenizer.cls_token_id,
            self.doc_model.tokenizer.pad_token_id,
            self.doc_model.tokenizer.sep_token_id,
        }

        self.max_length = model_args.max_length

    def to(self, device):
        self.q_model.to(device)
        self.doc_model.to(device)

    def encode_query(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.q_model.encode(**features)
        return cls_rep, tok_rep[:, :, :]  # skip cls rep

    def encode_corpus(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.doc_model.encode(**features)
        return cls_rep, tok_rep[:, :, :]  # skip cls rep

    def encode_query_raw(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.q_model.encode_raw(**features)
        return cls_rep, tok_rep[:, :, :]

    def encode_corpus_raw(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.doc_model.encode_raw(**features)
        return cls_rep, tok_rep[:, :, :]

    def encode_query_raw_proc(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.q_model.encode_raw_proc(**features)
        return cls_rep, tok_rep[:, :, :]

    def encode_corpus_raw_proc(self, features: Dict[str, Tensor]) -> Union[Tensor, Tensor]:
        cls_rep, tok_rep = self.doc_model.encode_raw_proc(**features)
        return cls_rep, tok_rep[:, :, :]


    def eval(self):
        self.q_model.eval()
        self.doc_model.eval()


class COIL_Core(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_args,
    ):
        super().__init__()
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.tok_proj = nn.Linear(768, model_args.token_dim)
        self.cls_proj = nn.Linear(768, model_args.cls_dim)
        self.model_args = model_args
        self.pooler_mode = model_args.pooler_mode
        self.window_size = model_args.window_size

        if model_args.token_norm_after:
            self.ln_tok = nn.LayerNorm(model_args.token_dim)
        if model_args.cls_norm_after:
            self.ln_cls = nn.LayerNorm(model_args.cls_dim)

    @classmethod
    def from_pretrained(cls, model_args: ModelArguments, *args, **kwargs):
        hf_tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        hf_model = AutoModel.from_pretrained(*args, **kwargs)
        model = COIL_Core(hf_model, hf_tokenizer, model_args)
        path = args[0]
        if os.path.exists(os.path.join(path, "model.pt")):
            logger.info("loading extra weights from local files")
            model_dict = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith("model")]
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, "model.pt"))

    def encode(self, **features):
        assert all([x in features for x in ["input_ids", "attention_mask"]])
        model_out: BaseModelOutputWithPooling = self.model(**features, return_dict=True)
        cls_rep = self.cls_proj(model_out.last_hidden_state[:, 0])
        reps = self.tok_proj(model_out.last_hidden_state)
        if self.model_args.cls_norm_after:
            cls_rep = self.ln_cls(cls_rep)
        if self.model_args.token_norm_after:
            reps = self.ln_tok(reps)

        if self.model_args.token_rep_relu:
            reps = torch.relu(reps)

        return cls_rep, reps

    def encode_proc(self, **features):
        cls_rep, reps = self.encode(**features)
        reps = self._preproc_rep(reps, features)
        return cls_rep, reps

    def encode_raw(self, **features):
        assert all([x in features for x in ["input_ids", "attention_mask"]])
        model_out: BaseModelOutputWithPooling = self.model(**features, return_dict=True)
        cls_rep = model_out.last_hidden_state[:, 0]
        reps = model_out.last_hidden_state
        return cls_rep, reps

    def encode_raw_proc(self, **features):
        cls_rep, reps = self.encode_raw(**features)
        reps  = self._preproc_rep(reps, features)
        return cls_rep, reps

    def _preproc_rep(self, reps: Tensor, features: Dict[str, Tensor]):
        att_masks = features["attention_mask"][:, 1:-1]
        reps = reps[:, 1:-1]
        if self.pooler_mode == TOKEN_POOLER:
            pass
        elif self.pooler_mode == LOCAL_AVE_POOLER:
            reps = self._rep_lave(reps, att_masks)
        else:
            raise ValueError(f"{self.pooler} doesn't exist")

        
        reps /= torch.norm(reps, dim=2).unsqueeze(-1)
        reps[torch.isnan(reps)] = 0.0
        return reps

    def _rep_lave(self, reps, att_masks):
        tg_reps = torch.zeros_like(reps)
        for b, (rep, att_mask) in enumerate(zip(reps, att_masks)):
            og_rep = rep[att_mask == 1, :]
            rep_len = og_rep.shape[0]
            for i in range(rep_len):
                start = i - self.window_size if i - self.window_size > 0 else 0
                end = i + self.window_size
                tg_reps[b, i, :] += torch.mean(og_rep[start:end, :], dim=0)

        return tg_reps
