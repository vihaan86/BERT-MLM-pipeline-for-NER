"""
OntoNotes data pipeline for BERT MLM (no-flags default: 10% data).
Loads tner/ontonotes5 from HuggingFace, tokenizes with BERT tokenizer,
and provides a DataLoader with on-the-fly MLM masking.
"""

import random
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# Default for no-flags run: 10% of train
DEFAULT_DATA_FRACTION = 0.1
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
ONTONOTES_DATASET = "tner/ontonotes5"
BERT_MODEL = "bert-base-uncased"


def get_tokenizer(model_name: str = BERT_MODEL) -> BertTokenizer:
    """Return BERT tokenizer (vocab 30522, pad_id=0, mask_id=103) for use with our BERT config."""
    return BertTokenizer.from_pretrained(model_name)


def _mlm_mask_sequence(
    input_ids: List[int],
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
    seed: Optional[int] = None,
) -> tuple[List[int], List[int]]:
    """
    BERT-style MLM: mask ~15% of non-special, non-pad positions.
    Of those: 80% [MASK], 10% random token, 10% unchanged.
    Returns (masked_input_ids, labels) with labels = -100 at non-masked positions.
    """
    if seed is not None:
        random.seed(seed)
    masked_input_ids = list(input_ids)
    labels = [-100] * len(input_ids)
    special_ids = {0, 101, 102}  # PAD, CLS, SEP
    if pad_token_id not in special_ids:
        special_ids.add(pad_token_id)
    cands = [i for i in range(len(input_ids)) if input_ids[i] not in special_ids]
    if not cands:
        return masked_input_ids, labels
    n_mask = max(1, int(len(cands) * mlm_probability))
    chosen = random.sample(cands, min(n_mask, len(cands)))
    for idx in chosen:
        labels[idx] = input_ids[idx]
        r = random.random()
        if r < 0.8:
            masked_input_ids[idx] = mask_token_id
        elif r < 0.9:
            masked_input_ids[idx] = random.randrange(0, vocab_size)
        # else 10% leave unchanged
    return masked_input_ids, labels


def mlm_mask_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MLM masking to a batch. Only positions with attention_mask==1 can be masked.
    Returns (masked_input_ids, labels); labels are -100 where attention_mask==0 or not chosen for MLM.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100, dtype=torch.long, device=device)
    for b in range(batch_size):
        seq = input_ids[b].tolist()
        # Only consider positions that are real tokens (attention_mask == 1)
        real_positions = [i for i in range(seq_len) if attention_mask[b, i].item() == 1]
        if not real_positions:
            continue
        n_mask = max(1, int(len(real_positions) * mlm_probability))
        chosen = random.sample(real_positions, min(n_mask, len(real_positions)))
        for idx in chosen:
            labels[b, idx] = input_ids[b, idx].item()
            r = random.random()
            if r < 0.8:
                masked_input_ids[b, idx] = mask_token_id
            elif r < 0.9:
                masked_input_ids[b, idx] = random.randrange(0, vocab_size)
    return masked_input_ids, labels


def _load_ontonotes_dataset(split: str):
    """Load tner/ontonotes5. Requires datasets 3.x and Python 3.10--3.12 (pickle/dill break on 3.14)."""
    try:
        return load_dataset(ONTONOTES_DATASET, split=split, trust_remote_code=True)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            raise RuntimeError(
                "OntoNotes (tner/ontonotes5) uses a dataset script, which requires datasets 3.x. "
                "Install with: pip install 'datasets>=2.14.0,<4.0.0'"
            ) from e
        raise
    except TypeError as e:
        if "_batch_setitems" in str(e) or "takes 2 positional arguments but 3" in str(e):
            raise RuntimeError(
                "Dataset loading failed due to Python 3.14 compatibility (pickle/dill in the "
                "datasets library). Use Python 3.10, 3.11, or 3.12 instead. Example:\n"
                "  py -3.12 -m venv .venv\n"
                "  .venv\\Scripts\\activate   (Windows)\n"
                "  pip install -r requirements.txt\n"
                "  python train_mlm.py"
            ) from e
        raise


def load_ontonotes(
    tokenizer: BertTokenizer,
    split: str = "train",
    data_fraction: float = DEFAULT_DATA_FRACTION,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """
    Load OntoNotes (tner/ontonotes5), take a fraction of the split, tokenize.
    Returns list of {"input_ids": [...], "attention_mask": [...], "token_type_ids": [...]}.
    """
    dataset = _load_ontonotes_dataset(split)
    n_total = len(dataset)
    n_use = max(1, int(n_total * data_fraction))
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    indices = indices[:n_use]
    rows = []
    for idx in indices:
        row = dataset[int(idx)]
        tokens = row.get("tokens", [])
        if not tokens:
            continue
        text = " ".join(tokens)
        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
            return_token_type_ids=True,
        )
        rows.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc["token_type_ids"],
        })
    return rows


class OntoNotesMLMDataset(Dataset):
    """Dataset of tokenized OntoNotes sequences (no masking yet; masking in collate)."""

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
        }


def _collate_mlm(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Stack batch and apply on-the-fly MLM masking. Returns batch dict with labels."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    token_type_ids = torch.stack([b["token_type_ids"] for b in batch])
    masked_input_ids, labels = mlm_mask_batch(
        input_ids,
        attention_mask,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        mlm_probability=mlm_probability,
    )
    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def get_ontonotes_dataloader(
    data_fraction: float = DEFAULT_DATA_FRACTION,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tokenizer: Optional[BertTokenizer] = None,
    split: str = "train",
    seed: int = 42,
    num_workers: int = 0,
    mlm_probability: float = 0.15,
) -> DataLoader:
    """
    Build DataLoader for OntoNotes MLM (default: 10% data, no flags).
    Uses tner/ontonotes5, BERT tokenizer, on-the-fly MLM masking in collate.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    rows = load_ontonotes(
        tokenizer=tokenizer,
        split=split,
        data_fraction=data_fraction,
        max_length=max_length,
        seed=seed,
    )
    dataset = OntoNotesMLMDataset(rows)
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return _collate_mlm(
            batch,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            mlm_probability=mlm_probability,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
