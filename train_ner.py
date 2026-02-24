"""
Train BERT for NER (token classification) on OntoNotes.
Optionally load encoder from MLM checkpoint: --mlm_checkpoint_dir checkpoints
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

from bert import BERTConfig, BERTForTokenClassification
from train import get_tokenizer
from train.data_ner import ONTONOTES_NUM_LABELS, get_ner_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train BERT NER on OntoNotes")
    parser.add_argument("--mlm_checkpoint_dir", type=str, default=None, help="Load BERT encoder from this MLM checkpoint dir")
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="checkpoints_ner")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer()
    dataloader = get_ner_dataloader(
        data_fraction=args.data_fraction,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        split="train",
        seed=args.seed,
    )
    num_batches = len(dataloader)

    if args.mlm_checkpoint_dir:
        config_path = os.path.join(args.mlm_checkpoint_dir, "bert_mlm_config.json")
        mlm_path = os.path.join(args.mlm_checkpoint_dir, "bert_mlm.safetensors")
        if not os.path.isfile(config_path) or not os.path.isfile(mlm_path):
            print(f"MLM checkpoint not found in {args.mlm_checkpoint_dir}; training from scratch.")
            args.mlm_checkpoint_dir = None
    if args.mlm_checkpoint_dir:
        with open(os.path.join(args.mlm_checkpoint_dir, "bert_mlm_config.json")) as f:
            cfg = json.load(f)
        config = BERTConfig(**cfg)
        model = BERTForTokenClassification(config, num_labels=ONTONOTES_NUM_LABELS)
        state = load_file(os.path.join(args.mlm_checkpoint_dir, "bert_mlm.safetensors"))
        bert_state = {k[5:]: v.clone() for k, v in state.items() if k.startswith("bert.")}
        model.bert.load_state_dict(bert_state, strict=False)
        print("Loaded BERT encoder from MLM checkpoint.")
    else:
        config = BERTConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=256,
        )
        model = BERTForTokenClassification(config, num_labels=ONTONOTES_NUM_LABELS)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            _, loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "bert_ner.safetensors")
    config_path = os.path.join(args.output_dir, "bert_ner_config.json")
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state_dict, ckpt_path)
    with open(config_path, "w") as f:
        json.dump(
            {
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "intermediate_size": config.intermediate_size,
                "max_position_embeddings": config.max_position_embeddings,
                "num_labels": ONTONOTES_NUM_LABELS,
            },
            f,
            indent=2,
        )
    print(f"Saved NER checkpoint to {ckpt_path} and {config_path}")


if __name__ == "__main__":
    main()
