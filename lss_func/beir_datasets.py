import datasets
from typing import List
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, BatchEncoding


class BeirDocDataset(Dataset):
    columns = ["_id", "title", "text"]

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, p_max_len=128):
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


class BeirQueryDataset(Dataset):
    columns = ["_id", "text"]

    def __getitem__(self, item) -> [BatchEncoding]:
        pid, text = (self.nlp_dataset[item][f] for f in self.columns)
        encoded_psg = self.tok.encode_plus(
            text,
            max_length=self.p_max_len,
            truncation="only_first",
            return_attention_mask=False,
        )
        return encoded_psg
