"""
Evaluate the trained BERT MLM: validation loss, accuracy on masked tokens, and example predictions.
Run after training: python eval_mlm.py [--checkpoint_dir checkpoints]
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file

from bert import BERTConfig, BERTForMaskedLM
from train import get_ontonotes_dataloader, get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Folder with bert_mlm.safetensors and config")
    parser.add_argument("--max_batches", type=int, default=50, help="Max validation batches for loss/accuracy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--examples", type=str, nargs="*", default=[
        "The capital of France is [MASK].",
        "Machine [MASK] is a branch of artificial intelligence.",
    ], help="Example sentences with [MASK] to predict")
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "bert_mlm_config.json")
    ckpt_path = os.path.join(args.checkpoint_dir, "bert_mlm.safetensors")
    if not os.path.isfile(config_path) or not os.path.isfile(ckpt_path):
        print(f"Missing checkpoint. Expected {config_path} and {ckpt_path}")
        return

    with open(config_path) as f:
        cfg = json.load(f)
    config = BERTConfig(**cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTForMaskedLM(config, tie_embeddings=True)
    model.load_state_dict(load_file(ckpt_path))
    model = model.to(device)
    model.eval()

    tokenizer = get_tokenizer()
    mask_id = tokenizer.mask_token_id

    # --- Validation loss and accuracy ---
    print("Loading validation data...")
    val_loader = get_ontonotes_dataloader(
        data_fraction=0.05,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        split="validation",
        seed=123,
    )
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= args.max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += loss.item()
            logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            preds = logits.argmax(dim=-1)
            mask_pos = labels != -100
            total_correct += (preds[mask_pos] == labels[mask_pos]).sum().item()
            total_masked += mask_pos.sum().item()
            num_batches += 1
    n = max(1, num_batches)
    avg_loss = total_loss / n
    acc = total_correct / max(1, total_masked)
    print(f"\nValidation (first {num_batches} batches):")
    print(f"  Loss:     {avg_loss:.4f}")
    print(f"  Accuracy (masked tokens): {100 * acc:.2f}%")

    # --- Example fill-in-the-blank ---
    if args.examples:
        print("\nExample predictions ([MASK] filled):")
        for sent in args.examples:
            if "[MASK]" not in sent:
                continue
            enc = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            input_ids = enc["input_ids"].to(device)
            mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
            if mask_positions.numel() == 0:
                print(f"  No [MASK] in: {sent}")
                continue
            with torch.no_grad():
                logits, _, _ = model(
                    input_ids=input_ids,
                    attention_mask=enc["attention_mask"].to(device),
                    token_type_ids=enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device),
                )
            tokens = input_ids[0].tolist()
            for idx, pos in enumerate(mask_positions):
                pos = pos.item()
                top5 = logits[0, pos].topk(5)
                pred_id = top5.indices[0].item()
                pred_tok = tokenizer.decode([pred_id])
                tokens[pos] = pred_id
                top5_toks = [tokenizer.decode([i]) for i in top5.indices.tolist()]
                if idx == 0:
                    print(f"  \"{sent}\"")
                print(f"    [MASK] -> \"{pred_tok}\"  (top-5: {top5_toks})")
            filled = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"    filled: \"{filled}\"\n")


if __name__ == "__main__":
    main()
