"""
Train a LoRA adapter on TinyLlama for smart contract vulnerability classification.

Usage:
    python train.py                          # Full training with defaults
    python train.py --max_samples 200        # Quick test run on 200 samples
    python train.py --epochs 5 --lr 3e-4     # Custom hyperparameters
"""

import os
import json
import argparse
import random
from datetime import date

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 2048  # TinyLlama context window

# Label mapping (cleaned)
LABELS = {
    0: "Block number dependency",
    1: "Dangerous delegatecall",
    2: "Ether frozen",
    3: "Ether strict equality",
    4: "Integer overflow",
    5: "Reentrancy",
    6: "Timestamp dependency",
    7: "Unchecked external call",
    8: "Safe",
}

# Reverse mapping: raw CSV label -> integer
RAW_LABEL_TO_ID = {
    "Safe": 8,
    "./Dataset/block number dependency (BN)": 0,
    "./Dataset/dangerous delegatecall (DE)/": 1,
    "./Dataset/ether frozen (EF)": 2,
    "./Dataset/ether strict equality (SE)": 3,
    "./Dataset/integer overflow (OF)/": 4,
    "./Dataset/reentrancy (RE)/": 5,
    "./Dataset/timestamp dependency (TP)/": 6,
    "./Dataset/unchecked external call (UC)": 7,
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for smart contract classification")
    parser.add_argument("--dataset", type=str, default="dataset/dataset_9l_w_v2 (1).csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not set)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for quick tests")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--no_quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max_code_tokens", type=int, default=1500, help="Max tokens for code truncation (rest is for prompt)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def clean_label(raw_label: str) -> int:
    """Convert raw CSV label to integer label."""
    raw_label = raw_label.strip()
    if raw_label in RAW_LABEL_TO_ID:
        return RAW_LABEL_TO_ID[raw_label]
    # Fallback: try partial matching
    for key, val in RAW_LABEL_TO_ID.items():
        if raw_label in key or key in raw_label:
            return val
    raise ValueError(f"Unknown label: '{raw_label}'")


def build_labels_description() -> str:
    """Build the label description string for the prompt."""
    lines = []
    for idx, name in LABELS.items():
        lines.append(f"{idx} = {name}")
    return "\n".join(lines)


def format_prompt(code: str, label_id: int, tokenizer, max_code_tokens: int) -> str:
    """
    Format a training sample as a chat prompt.

    Truncates the code to fit within the context window, keeping the
    beginning of the contract (where imports, inheritance, and key
    function signatures live — most vulnerability patterns are there).
    """
    # Truncate code by tokens to stay within budget
    code_tokens = tokenizer.encode(code, add_special_tokens=False)
    if len(code_tokens) > max_code_tokens:
        code_tokens = code_tokens[:max_code_tokens]
        code = tokenizer.decode(code_tokens, skip_special_tokens=True)

    labels_desc = build_labels_description()

    # Use TinyLlama chat format
    prompt = f"""<|system|>
You are a smart contract security auditor. You analyze Solidity code and classify vulnerabilities.</s>
<|user|>
Analyze this Solidity smart contract and classify its vulnerability type.

Respond with ONLY a single digit (0-8):
{labels_desc}

Solidity contract:
```
{code}
```

Classification (single digit):</s>
<|assistant|>
{label_id}</s>"""

    return prompt


def format_prompt_inference(code: str, tokenizer, max_code_tokens: int) -> str:
    """Format a prompt for inference (no answer)."""
    code_tokens = tokenizer.encode(code, add_special_tokens=False)
    if len(code_tokens) > max_code_tokens:
        code_tokens = code_tokens[:max_code_tokens]
        code = tokenizer.decode(code_tokens, skip_special_tokens=True)

    labels_desc = build_labels_description()

    prompt = f"""<|system|>
You are a smart contract security auditor. You analyze Solidity code and classify vulnerabilities.</s>
<|user|>
Analyze this Solidity smart contract and classify its vulnerability type.

Respond with ONLY a single digit (0-8):
{labels_desc}

Solidity contract:
```
{code}
```

Classification (single digit):</s>
<|assistant|>
"""
    return prompt


def load_and_prepare_data(args, tokenizer):
    """Load CSV, clean labels, split, and format as HF Dataset."""
    print(f"Loading dataset from {args.dataset}...")
    df = pd.read_csv(args.dataset, on_bad_lines="skip")
    print(f"  Raw samples: {len(df)}")

    # Clean labels
    df["label_clean"] = df["label"].apply(clean_label)

    # Drop rows with missing code
    df = df.dropna(subset=["code"])
    print(f"  After cleanup: {len(df)}")

    # Optional: limit dataset size for quick tests
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=SEED)
        print(f"  Limited to: {len(df)} samples")

    # Stratified split BEFORE any preprocessing
    # Fallback to non-stratified if too few samples per class
    min_class_count = df["label_clean"].value_counts().min()
    use_stratify = min_class_count >= 2
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=SEED,
        stratify=df["label_clean"] if use_stratify else None,
    )
    if not use_stratify:
        print("  (!) Too few samples per class for stratified split — using random split")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Print class distribution
    print("\n  Class distribution (train):")
    for label_id in sorted(LABELS.keys()):
        count = (train_df["label_clean"] == label_id).sum()
        print(f"    {label_id} ({LABELS[label_id]}): {count}")

    # Format prompts
    print("\nFormatting prompts...")
    train_texts = [
        format_prompt(row["code"], row["label_clean"], tokenizer, args.max_code_tokens)
        for _, row in train_df.iterrows()
    ]
    val_texts = [
        format_prompt(row["code"], row["label_clean"], tokenizer, args.max_code_tokens)
        for _, row in val_df.iterrows()
    ]

    # Check token lengths
    train_lengths = [len(tokenizer.encode(t)) for t in train_texts]
    print(f"  Train token lengths — mean: {np.mean(train_lengths):.0f}, "
          f"median: {np.median(train_lengths):.0f}, "
          f"max: {np.max(train_lengths)}, "
          f"overflow (>{MAX_SEQ_LENGTH}): {sum(1 for l in train_lengths if l > MAX_SEQ_LENGTH)}")

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return train_dataset, val_dataset, val_df


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    """Load TinyLlama with optional 4-bit quantization and apply LoRA."""
    print(f"\nLoading model: {MODEL_NAME}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config (QLoRA)
    bnb_config = None
    if not args.no_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (QLoRA)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if not args.no_quantize:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(SEED)

    # Load model first (need tokenizer for data prep)
    model, tokenizer = load_model_and_tokenizer(args)

    # Prepare data
    train_dataset, val_dataset, val_df = load_and_prepare_data(args, tokenizer)

    # Output directory
    if args.output_dir is None:
        today = date.today().isoformat()
        run_name = f"tinyllama-lora_run-01_date-{today}"
        args.output_dir = f"models/{run_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        optim="adamw_torch" if args.no_quantize else "paged_adamw_8bit",
        dataset_text_field="text",
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config for reproducibility
    config = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_code_tokens": args.max_code_tokens,
        "quantization": "4bit-nf4" if not args.no_quantize else "none",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "dataset": args.dataset,
        "test_size": args.test_size,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Save validation set info for evaluate.py
    val_df.to_csv(os.path.join(args.output_dir, "val_split.csv"), index=False)
    print(f"Validation split saved for evaluation")

    print("\nTraining complete!")
    print(f"  Model saved to: {args.output_dir}")
    print(f"  Run evaluate.py to test: python evaluate.py --model_dir {args.output_dir}")

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train(args)
