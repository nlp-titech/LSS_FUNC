from sentence_transformers.models import WordWeights
from collections import Counter
from typing import List, Dict
from torch import Tensor
import torch


class BM25Weight(WordWeights):
    def __init__(self, vocab: List[str], word_weights: Dict[str, float], doc_len_ave: float, 
                 bm25_k1: float = 0.9, bm25_b: float = 0.4, unknown_word_weight: float = 1):
        super().__init__(vocab, word_weights, unknown_word_weight)
        self.doc_len_ave = doc_len_ave
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def bm25_tf(self, tf, doc_lens):
        nume = tf * (1 + self.bm25_k1)
        denom = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_lens / self.doc_len_ave)
        return nume / denom

    def forward(self, features: Dict[str, Tensor]):
        attention_mask = features['attention_mask']
        token_embeddings = features['token_embeddings']

        input_tfs = []
        for input_tokens, att_mask in zip(features["input_ids"], features["attention_mask"]):
            tf = Counter(input_tokens.tolist())
            input_tfs.append(torch.tensor([tf[t.item()] for t in input_tokens]).unsqueeze(0))
        input_tfs = torch.cat(input_tfs).to(att_mask.device)
        input_tfs *= features["attention_mask"]
        doc_lens = torch.sum(features["attention_mask"], dim=1).unsqueeze(-1)
        tf_weight = self.bm25_tf(input_tfs, doc_lens)

        #Compute a weight value for each token
        token_weights_raw = self.emb_layer(features['input_ids']).squeeze(-1)
        token_weights = tf_weight * token_weights_raw * attention_mask.float()
        token_weights_sum = torch.sum(token_weights, 1)

        #Multiply embedding by token weight value
        token_weights_expanded = token_weights.unsqueeze(-1).expand(token_embeddings.size())
        token_embeddings = token_embeddings * token_weights_expanded

        features.update({'token_embeddings': token_embeddings, 'token_weights_sum': token_weights_sum})
        return features