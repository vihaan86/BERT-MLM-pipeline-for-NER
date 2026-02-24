"""
NER data pipeline: OntoNotes (tner/ontonotes5) with tokens + tags.
Aligns word-level BIO tags to BERT subwords (first subword gets tag, rest -100).
"""

import random
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from .data_ontonotes import ONTONOTES_DATASET, get_tokenizer

# tner/ontonotes5: 37 BIO labels (0 = O, 1-36 = entity types)
ONTONOTES_NUM_LABELS = 37
# id -> label name for decoding (from dataset label.json)
ONTONOTES_LABEL_NAMES = [
    "O", "B-CARDINAL", "B-DATE", "I-DATE", "B-PERSON", "I-PERSON", "B-NORP", "B-GPE", "I-GPE",
    "B-LAW", "I-LAW", "B-ORG", "I-ORG", "B-PERCENT", "I-PERCENT", "B-ORDINAL", "B-MONEY", "I-MONEY",
    "B-WORK_OF_ART", "I-WORK_OF_ART", "B-FAC", "B-TIME", "I-CARDINAL", "B-LOC", "B-QUANTITY", "I-QUANTITY",
    "I-NORP", "I-LOC", "B-PRODUCT", "I-TIME", "B-EVENT", "I-EVENT", "I-FAC", "B-LANGUAGE", "I-PRODUCT",
    "I-ORDINAL", "I-LANGUAGE",
]
DEFAULT_DATA_FRACTION = 0.1
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16


def _load_ontonotes_raw(split: str):
    """Load tner/ontonotes5 raw (same as MLM pipeline)."""
    try:
        return load_dataset(ONTONOTES_DATASET, split=split, trust_remote_code=True)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            raise RuntimeError(
                "OntoNotes requires datasets 3.x. Install: pip install 'datasets>=2.14.0,<4.0.0'"
            ) from e
        raise
    except TypeError as e:
        if "_batch_setitems" in str(e) or "takes 2 positional arguments but 3" in str(e):
            raise RuntimeError(
                "Dataset loading failed (Python 3.14). Use Python 3.10--3.12."
            ) from e
        raise


def _align_tags_to_subwords(
    tokenizer: BertTokenizer,
    tokens: List[str],
    tags: List[int],
    max_length: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
) -> tuple[List[int], List[int]]:
    """
    Tokenize words and align BIO tags: first subword of each word gets the word's tag, rest get -100.
    Returns (input_ids, labels) with [CLS] ... [SEP] and padding, labels -100 for [CLS],[SEP], pad, and continuation subwords.
    """
    input_ids = [cls_id]
    labels = [-100]
    for word, tag in zip(tokens, tags):
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        if not subword_ids:
            continue
        input_ids.extend(subword_ids)
        labels.append(tag)
        labels.extend([-100] * (len(subword_ids) - 1))
    input_ids.append(sep_id)
    labels.append(-100)
    # Truncate
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    # Pad
    pad_len = max_length - len(input_ids)
    input_ids = input_ids + [pad_id] * pad_len
    labels = labels + [-100] * pad_len
    return input_ids, labels


def load_ontonotes_ner(
    tokenizer: BertTokenizer,
    split: str = "train",
    data_fraction: float = DEFAULT_DATA_FRACTION,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """
    Load OntoNotes with NER tags; align to subwords.
    Returns list of {"input_ids", "attention_mask", "token_type_ids", "labels"}.
    """
    dataset = _load_ontonotes_raw(split)
    n_total = len(dataset)
    n_use = max(1, int(n_total * data_fraction))
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    indices = indices[:n_use]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    rows = []
    for idx in indices:
        row = dataset[int(idx)]
        tokens = row.get("tokens", [])
        tags = row.get("tags", [])
        if not tokens or len(tokens) != len(tags):
            continue
        input_ids, labels = _align_tags_to_subwords(
            tokenizer, tokens, tags, max_length, cls_id, sep_id, pad_id
        )
        attention_mask = [1 if i != pad_id else 0 for i in input_ids]
        # token_type_ids: all 0 for single segment
        token_type_ids = [0] * len(input_ids)
        rows.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        })
    return rows


class OntoNotesNERDataset(Dataset):
    def __init__(self, rows: List[Dict[str, List[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(r["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(r["labels"], dtype=torch.long),
        }


def get_ner_dataloader(
    data_fraction: float = DEFAULT_DATA_FRACTION,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tokenizer: Optional[BertTokenizer] = None,
    split: str = "train",
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build DataLoader for NER on OntoNotes (tokens + tags, aligned to subwords)."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    rows = load_ontonotes_ner(
        tokenizer=tokenizer,
        split=split,
        data_fraction=data_fraction,
        max_length=max_length,
        seed=seed,
    )
    dataset = OntoNotesNERDataset(rows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
