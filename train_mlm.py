"""
Train BERT for Masked Language Modeling on OntoNotes.
Default (no flags): 10% data, 4-layer BERT, ~10 min on GPU.
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file

from bert import BERTConfig, BERTForMaskedLM
from train import get_ontonotes_dataloader, get_tokenizer


def get_config(mode: str) -> BERTConfig:
    """mode: 'default' | 'full' | 'demo'"""
    if mode == "full":
        return BERTConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
    if mode == "demo":
        return BERTConfig(
            vocab_size=30522,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=128,
        )
    # default: ~10 min run
    return BERTConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=256,
    )


def main():
    parser = argparse.ArgumentParser(description="Train BERT MLM on OntoNotes")
    parser.add_argument("--full", action="store_true", help="100%% data, BERT-base 12 layers (~1-6h)")
    parser.add_argument("--demo", action="store_true", help="Tiny model + 1%% data (~1-2 min)")
    parser.add_argument("--data_fraction", type=float, default=0.1, help="Fraction of train data (default 0.1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup fraction of total steps")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.demo:
        mode = "demo"
        args.data_fraction = 0.01
        args.epochs = 2
        args.batch_size = 8
        args.max_length = 64
    elif args.full:
        mode = "full"
        args.data_fraction = 1.0
        args.epochs = 10
    else:
        mode = "default"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, mode: {mode}")

    tokenizer = get_tokenizer()
    dataloader = get_ontonotes_dataloader(
        data_fraction=args.data_fraction,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        seed=args.seed,
    )
    num_batches = len(dataloader)
    total_steps = num_batches * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    config = get_config(mode)
    model = BERTForMaskedLM(config, tie_embeddings=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step = 0
    start = time.perf_counter()

    for epoch in range(args.epochs):
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
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / num_batches
        elapsed = time.perf_counter() - start
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "bert_mlm.safetensors")
    config_path = os.path.join(args.output_dir, "bert_mlm_config.json")
    # Clone state_dict so shared tensors (from weight tying) are not duplicated on disk; safetensors rejects shared memory
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
            },
            f,
            indent=2,
        )
    print(f"Saved checkpoint to {ckpt_path} (load with safetensors.torch.load_file) and config to {config_path}")


if __name__ == "__main__":
    main()
