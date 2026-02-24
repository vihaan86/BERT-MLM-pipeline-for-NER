"""
Run this first after installing dependencies.
Checks tokenizer + OntoNotes pipeline (downloads dataset on first run).
"""
from train import get_tokenizer, get_ontonotes_dataloader

def main():
    print("1. Loading tokenizer (bert-base-uncased)...")
    tokenizer = get_tokenizer()
    print(f"   vocab_size={tokenizer.vocab_size}, pad_id={tokenizer.pad_token_id}, mask_id={tokenizer.mask_token_id}")

    print("2. Loading OntoNotes dataloader (10% train, first batch may download dataset)...")
    dataloader = get_ontonotes_dataloader(data_fraction=0.01, batch_size=2, max_length=64)
    batch = next(iter(dataloader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   Masked positions in batch: {(batch['labels'] != -100).sum().item()}")

    print("3. Done. Pipeline is ready.")

if __name__ == "__main__":
    main()
