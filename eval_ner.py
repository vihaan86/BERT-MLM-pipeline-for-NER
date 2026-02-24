"""
Evaluate BERT NER: validation loss, token-level accuracy, and optional example sentences.
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file

from bert import BERTConfig, BERTForTokenClassification
from train import get_tokenizer
from train.data_ner import ONTONOTES_LABEL_NAMES, ONTONOTES_NUM_LABELS, get_ner_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_ner")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--examples",
        type=str,
        nargs="*",
        default=[],
        help="Example sentences to run NER on (e.g. \"Barack Obama visited Paris.\")",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "bert_ner_config.json")
    ckpt_path = os.path.join(args.checkpoint_dir, "bert_ner.safetensors")
    if not os.path.isfile(config_path) or not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {config_path} and {ckpt_path}")
        return

    with open(config_path) as f:
        cfg = json.load(f)
    num_labels = cfg.pop("num_labels", ONTONOTES_NUM_LABELS)
    config = BERTConfig(**cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTForTokenClassification(config, num_labels=num_labels)
    model.load_state_dict(load_file(ckpt_path))
    model = model.to(device)
    model.eval()

    tokenizer = get_tokenizer()
    val_loader = get_ner_dataloader(
        data_fraction=0.05,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        split="validation",
        seed=123,
    )
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= args.max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()
            num_batches += 1
    n = max(1, num_batches)
    avg_loss = total_loss / n
    acc = total_correct / max(1, total_tokens)
    print(f"Validation (first {num_batches} batches):")
    print(f"  Loss:    {avg_loss:.4f}")
    print(f"  Token accuracy (labeled only): {100 * acc:.2f}%")

    # --- Custom example sentences ---
    if args.examples:
        label_names = ONTONOTES_LABEL_NAMES
        print("\nExample predictions (token -> label):")
        for sent in args.examples:
            if not sent.strip():
                continue
            enc = tokenizer(
                sent,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_offsets_mapping=False,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids, device=device)
            else:
                token_type_ids = token_type_ids.to(device)
            with torch.no_grad():
                logits, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            pred_ids = logits[0].argmax(dim=-1).tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            # Skip [CLS], [SEP], [PAD]
            out = []
            for tok, pid in zip(tokens, pred_ids):
                if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                    continue
                name = label_names[pid] if 0 <= pid < len(label_names) else f"LABEL_{pid}"
                out.append((tok, name))
            print(f"  \"{sent}\"")
            print("    " + "  ".join(f"{t}({n})" for t, n in out))
            print()


if __name__ == "__main__":
    main()
