# BERT Encoder Boilerplate

Minimal PyTorch implementation of the **BERT encoder** with MLM support: embeddings, stacked encoder (with Add + LayerNorm after each sublayer), and optional Masked Language Model head.

## BERT architecture — component checklist (all present)

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | **Config** | `config.py` | `BERTConfig`: vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, dropouts, max_position_embeddings, type_vocab_size, layer_norm_eps, pad_token_id, head_dim |
| 2 | **Token embeddings** | `embeddings.py` | `nn.Embedding(vocab_size, hidden_size, padding_idx)` |
| 3 | **Position embeddings** | `embeddings.py` | `nn.Embedding(max_position_embeddings, hidden_size)` |
| 4 | **Segment (token type) embeddings** | `embeddings.py` | `nn.Embedding(type_vocab_size, hidden_size)` |
| 5 | **Embedding LayerNorm + dropout** | `embeddings.py` | After sum of three embeddings; `_init_weights` |
| 6 | **Multi-head self-attention** | `attention.py` | Q/K/V linear, scaled dot-product, attention mask, output dense, dropout on probs; `_init_weights` |
| 7 | **Attention residual (Add) + LayerNorm** | `encoder.py` | `hidden_states = hidden_states + attn_output` then `attention_layer_norm` |
| 8 | **Feed-forward (FFN)** | `encoder.py` | `intermediate` linear → GELU → `output` linear; dropout |
| 9 | **FFN residual (Add) + LayerNorm** | `encoder.py` | `hidden_states = hidden_states + dropout(ffn_output)` then `ffn_layer_norm` |
| 10 | **Encoder stack** | `encoder.py` | `BertEncoder`: N × `BertEncoderLayer`; optional `output_attentions` |
| 11 | **Extended attention mask** | `model.py` | `get_extended_attention_mask`: 0/1 → additive mask for padding |
| 12 | **Pooled output [CLS]** | `model.py` | `encoder_outputs[:, 0, :]` |
| 13 | **MLM head** | `mlm.py` | Dense → GELU → LayerNorm → decoder (vocab logits); optional weight tying; `_init_weights` |
| 14 | **MLM loss** | `mlm.py` | CrossEntropyLoss with `ignore_index=-100` over masked positions only |

## OntoNotes MLM pipeline (no-flags default)

- **`train/data_ontonotes.py`** — Loads **tner/ontonotes5** from HuggingFace, uses **10% of train** by default, tokenizes with `bert-base-uncased`, and provides a DataLoader with **on-the-fly BERT-style MLM masking** (15% of tokens; 80% [MASK], 10% random, 10% unchanged).
- **`get_tokenizer()`** — Returns HuggingFace `BertTokenizer` (vocab 30522, pad_id=0, mask_id=103).
- **`get_ontonotes_dataloader(data_fraction=0.1, max_length=128, batch_size=16, ...)`** — Returns a PyTorch DataLoader yielding batches of `input_ids`, `attention_mask`, `token_type_ids`, `labels`.

Install deps then use:

```python
from train import get_tokenizer, get_ontonotes_dataloader

tokenizer = get_tokenizer()
dataloader = get_ontonotes_dataloader()  # 10% data, max_length=128, batch_size=16
batch = next(iter(dataloader))  # input_ids, attention_mask, token_type_ids, labels
```

## Structure

- **`bert/config.py`** — `BERTConfig`: vocab size, hidden size, layers, heads, dropout, etc.
- **`bert/embeddings.py`** — `BERTEmbeddings`: token + position + segment; LayerNorm + dropout.
- **`bert/attention.py`** — `MultiHeadSelfAttention`: scaled dot-product multi-head attention + mask.
- **`bert/encoder.py`** — `BertEncoderLayer` (attention → **Add + LayerNorm** → FFN → **Add + LayerNorm**), `BertEncoder`.
- **`bert/model.py`** — `BERTModel`: encoder only; last hidden state, pooled [CLS], optional attentions.
- **`bert/mlm.py`** — `BertMLMHead`, `BERTForMaskedLM`: MLM head and full model for masked token prediction.

## Install

**Python 3.10, 3.11, or 3.12 is required.** Python 3.14 is not supported: the HuggingFace `datasets` library uses pickle/dill internally and fails on 3.14 when loading OntoNotes.

```bash
pip install -r requirements.txt
```

## Run first

1. **Install** (above).
2. **Run the pipeline check** (downloads OntoNotes on first run):

```bash
python run_first.py
```

This loads the tokenizer and one batch from the OntoNotes dataloader. No other file needs to be "run first"—`bert/` and `train/` are packages you import from your own script or from a training script.

## Train MLM

Run MLM training on OntoNotes:

```bash
# Default: 10% data, 4-layer BERT, ~10 min on GPU
python train_mlm.py

# Full: 100% data, BERT-base (12 layers), ~1–6 h on GPU
python train_mlm.py --full

# Demo: tiny model, 1% data, ~1–2 min
python train_mlm.py --demo
```

Optional args: `--data_fraction`, `--batch_size`, `--max_length`, `--epochs`, `--lr`, `--output_dir` (default `checkpoints`). Checkpoints are saved as `checkpoints/bert_mlm.safetensors` and `checkpoints/bert_mlm_config.json`. Load with `safetensors.torch.load_file()` and the config JSON.

## Evaluate MLM

After training, run validation and example predictions:

```bash
python eval_mlm.py
```

This loads the checkpoint from `checkpoints/`, computes **validation loss** and **accuracy on masked tokens** over a subset of the validation set, and runs **fill-in-the-blank** on example sentences (e.g. "The capital of France is [MASK]."). Options: `--checkpoint_dir`, `--max_batches`, `--examples "Your [MASK] here."`.

## Train NER

Train BERT for Named Entity Recognition (token classification) on OntoNotes (same data, using the `tags` field). Optionally start from your MLM checkpoint:

```bash
# From scratch
python train_ner.py

# From MLM checkpoint (recommended)
python train_ner.py --mlm_checkpoint_dir checkpoints
```

Saves to `checkpoints_ner/` by default. Options: `--data_fraction`, `--batch_size`, `--max_length`, `--epochs`, `--lr`, `--output_dir`.

## Evaluate NER

After training NER:

```bash
python eval_ner.py --checkpoint_dir checkpoints_ner
```

Reports validation loss and **token-level accuracy** (on labeled positions only).

## Usage

**Encoder only:**

```python
from bert import BERTConfig, BERTModel
import torch

config = BERTConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, ...)
model = BERTModel(config)
last_hidden, pooled, _ = model(input_ids=..., attention_mask=..., token_type_ids=...)
```

**MLM (masked word prediction):**

```python
from bert import BERTConfig, BERTForMaskedLM

config = BERTConfig(...)
model = BERTForMaskedLM(config, tie_embeddings=True)
# input_ids: use mask_token_id (e.g. 103) at positions to predict
# labels: same shape; true token id at masked positions, -100 elsewhere
logits, loss, _ = model(input_ids=..., attention_mask=..., labels=...)
# loss is CrossEntropy only over masked positions (ignore_index=-100)
```

**NER (token classification):**

```python
from bert import BERTConfig, BERTForTokenClassification

config = BERTConfig(...)
model = BERTForTokenClassification(config, num_labels=37)
# labels: token-level label ids; use -100 for positions to ignore (e.g. subword tails)
logits, loss, _ = model(input_ids=..., attention_mask=..., labels=...)
```

You can add other heads (e.g. NSP, classification) on top of `BERTModel` outputs.
