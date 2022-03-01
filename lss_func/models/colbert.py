import datetime
from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict, Tuple
import string
import torch
import torch.nn as nn

from transformers import PreTrainedModel, AutoModel, AutoTokenizer


def print_message(*s, condition=True):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


def load_checkpoint(path, model, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path, "..")

    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]
        new_state_dict[name] = v

    checkpoint['model_state_dict'] = new_state_dict

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint


class ColBERT:
    def __init__(self, base_model_path_or_name: str, checkpoint_path: Union[str, Tuple], sep: str = " ",
                 query_maxlen: int = 32, doc_maxlen: int = 180, similarity_metric: str = 'cosine', **kwargs):
        self.sep = sep
        query_maxlen = query_maxlen
        doc_maxlen = doc_maxlen
        mask_punctuation = kwargs.pop("mask_punctuation", None)
        self.similarity_metric = similarity_metric
        tokenizer = AutoTokenizer.from_pretrained(base_model_path_or_name)
        self.q_tokenizer = partial(tokenizer, max_length=query_maxlen)
        self.d_tokenizer = partial(tokenizer, max_length=doc_maxlen)
            
        if isinstance(checkpoint_path, str):
            base_model = AutoModel.from_pretrained(base_model_path_or_name)
            config = base_model.config
            self.q_model = AutoColBERT(base_model, tokenizer, config, mask_punctuation)
            _ = load_checkpoint(checkpoint_path, self.q_model)
            self.d_model = self.q_model
       
        elif isinstance(checkpoint_path, tuple):
            base_model1 = AutoModel.from_pretrained(base_model_path_or_name)
            config1 = base_model1.config
            base_model2 = AutoModel.from_pretrained(base_model_path_or_name)
            config2 = base_model2.config
            self.q_model = AutoColBERT(base_model1, tokenizer, config1, mask_punctuation)
            self.d_model = AutoColBERT(base_model2, tokenizer, config2, mask_punctuation)
            _ = load_checkpoint(checkpoint_path[0], self.q_model)
            _ = load_checkpoint(checkpoint_path[1], self.d_model)
            
        self.remove_tok = {
            self.d_model.tokenizer.cls_token_id,
            self.d_model.tokenizer.pad_token_id,
            self.d_model.tokenizer.sep_token_id,
        }

        self.max_length = doc_maxlen

    def to(self, device):
        self.q_model.to(device)
        self.d_model.to(device)

    def query(self, input_ids, attention_mask):
        return self.q_model.query(input_ids, attention_mask)

    def doc(self, input_ids, attention_mask):
        return self.d_model.doc(input_ids, attention_mask)

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)


class AutoColBERT(PreTrainedModel):
    def __init__(self, model, tokenizer, config, mask_punctuation, dim=128):
        super(AutoColBERT, self).__init__(config)

        self.dim = dim
        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = tokenizer
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = model
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        device = input_ids.device
        mask = torch.tensor(self.mask(input_ids), device=device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask