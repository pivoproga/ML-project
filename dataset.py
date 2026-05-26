
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

def align_labels(
    word_labels: List[str],
    word_ids: List[int | None],
    label2id: Dict[str, int],
) -> List[int]:
    aligned = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev_word_id:
            aligned.append(label2id[word_labels[wid]])
        else:
            aligned.append(-100)
        prev_word_id = wid
    return aligned

class MarketplaceNERDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tokenizer: PreTrainedTokenizerFast,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        label_ids = align_labels(rec["labels"], enc.word_ids(), self.label2id)
        return {
            "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(label_ids,             dtype=torch.long),
        }
