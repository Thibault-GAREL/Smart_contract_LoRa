"""
Evaluate a trained LoRA model on smart contract classification.

Usage:
    python evaluate.py --model_dir models/tinyllama-lora_run-01_date-2026-03-27
    python evaluate.py --model_dir models/... --baseline   # Test base TinyLlama (zero-shot)
    python evaluate.py --model_dir models/... --max_samples 50  # Quick eval on subset
"""

import os
import json
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Import shared config from train.py
from train import (
    LABELS,
    RAW_LABEL_TO_ID,
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    SEED,
    clean_label,
    format_prompt_inference,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved LoRA model")
    parser.add_argument("--dataset", type=str, default=None, help="CSV to evaluate on (default: val_split.csv from model_dir)")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model (zero-shot, no LoRA)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--max_code_tokens", type=int, default=None, help="Override max code tokens (loaded from config if not set)")
    parser.add_argument("--batch_inference", action="store_true", help="Use batch inference (faster but uses more VRAM)")
    parser.add_argument("--no_quantize", action="store_true", help="Disable 4-bit quantization (use for Blackwell GPUs or when bitsandbytes fails)")
    return parser.parse_args()


def load_model(args):
    """Load model — either base (zero-shot) or fine-tuned LoRA."""
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for batch generation

    # Quantization
    bnb_config = None
    if not args.no_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if args.baseline:
        print(f"Loading BASE model (zero-shot, no LoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print(f"Loading FINE-TUNED model from {args.model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(model, args.model_dir)
        # Skip merge_and_unload to save VRAM — PEFT model works fine for inference

    model.eval()
    return model, tokenizer


def extract_prediction(generated_text: str) -> int:
    """
    Extract the predicted digit (0-8) from model output.

    Tries multiple strategies:
    1. First character is a digit 0-8
    2. First digit found anywhere in the response
    3. Fallback: -1 (invalid)
    """
    text = generated_text.strip()

    # Strategy 1: first char is a valid digit
    if text and text[0].isdigit() and int(text[0]) <= 8:
        return int(text[0])

    # Strategy 2: find first digit 0-8
    match = re.search(r"[0-8]", text)
    if match:
        return int(match.group())

    # Strategy 3: no valid prediction
    return -1


def predict_single(model, tokenizer, prompt: str, device: str) -> tuple[int, str]:
    """Run inference on a single sample. Returns (prediction, raw_output)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    prediction = extract_prediction(raw_output)
    return prediction, raw_output


def evaluate(args):
    set_seed(SEED)

    # Load training config for max_code_tokens
    config_path = os.path.join(args.model_dir, "training_config.json")
    max_code_tokens = args.max_code_tokens
    if max_code_tokens is None and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_code_tokens = config.get("max_code_tokens", 1500)
        print(f"Loaded max_code_tokens={max_code_tokens} from training config")
    elif max_code_tokens is None:
        max_code_tokens = 1500
        print(f"Using default max_code_tokens={max_code_tokens}")

    # Load model
    model, tokenizer = load_model(args)
    device = next(model.parameters()).device

    # Load evaluation data
    if args.dataset:
        df = pd.read_csv(args.dataset, on_bad_lines="skip")
        df["label_clean"] = df["label"].apply(clean_label)
    else:
        val_path = os.path.join(args.model_dir, "val_split.csv")
        if os.path.exists(val_path):
            df = pd.read_csv(val_path, on_bad_lines="skip")
            df["label_clean"] = df["label"].apply(clean_label)
            print(f"Loaded validation split from {val_path}")
        else:
            print(f"ERROR: No val_split.csv found in {args.model_dir}")
            print("  Provide --dataset or retrain to generate val_split.csv")
            return

    df = df.dropna(subset=["code"])

    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=SEED)

    print(f"\nEvaluating on {len(df)} samples...")

    # Run predictions
    y_true = []
    y_pred = []
    invalid_count = 0
    raw_outputs = []

    for i, (_, row) in enumerate(df.iterrows()):
        prompt = format_prompt_inference(row["code"], tokenizer, max_code_tokens)
        pred, raw = predict_single(model, tokenizer, prompt, device)

        y_true.append(row["label_clean"])
        y_pred.append(pred)
        raw_outputs.append(raw)

        if pred == -1:
            invalid_count += 1

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}] true={row['label_clean']} ({LABELS[row['label_clean']]}), "
                  f"pred={pred}, raw='{raw.strip()}'")

    # --- Metrics ---
    print(f"\n{'='*60}")
    mode = "BASELINE (zero-shot)" if args.baseline else "FINE-TUNED LoRA"
    print(f"  Results — {mode}")
    print(f"{'='*60}")

    # Filter out invalid predictions for metric calculation
    valid_mask = [p != -1 for p in y_pred]
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

    total = len(y_true)
    valid = len(y_true_valid)
    print(f"\n  Total samples: {total}")
    print(f"  Valid predictions: {valid} ({valid/total*100:.1f}%)")
    print(f"  Invalid (no digit): {invalid_count} ({invalid_count/total*100:.1f}%)")

    if valid == 0:
        print("\n  No valid predictions — cannot compute metrics.")
        return

    # Accuracy
    acc = accuracy_score(y_true_valid, y_pred_valid)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # F1 scores
    f1_macro = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    # Per-class report
    label_names = [LABELS[i] for i in sorted(LABELS.keys())]
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true_valid, y_pred_valid,
        labels=list(range(9)),
        target_names=label_names,
        zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=list(range(9)))
    print("  Confusion Matrix:")
    # Header
    header = "          " + " ".join(f"{i:>4}" for i in range(9))
    print(header)
    for i in range(9):
        row_str = " ".join(f"{cm[i][j]:>4}" for j in range(9))
        abbrev = LABELS[i][:8]
        print(f"  {abbrev:>8} {row_str}")

    # Prediction distribution
    print(f"\n  Prediction distribution:")
    pred_dist = Counter(y_pred)
    for label_id in sorted(LABELS.keys()):
        count = pred_dist.get(label_id, 0)
        print(f"    {label_id} ({LABELS[label_id]}): {count}")
    if -1 in pred_dist:
        print(f"    -1 (invalid): {pred_dist[-1]}")

    # Save results
    results_dir = os.path.join(args.model_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "mode": mode,
        "total_samples": total,
        "valid_predictions": valid,
        "invalid_predictions": invalid_count,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    results_path = os.path.join(results_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved to {results_path}")

    # Save detailed predictions
    pred_df = pd.DataFrame({
        "true_label": y_true,
        "true_name": [LABELS.get(t, "?") for t in y_true],
        "predicted_label": y_pred,
        "predicted_name": [LABELS.get(p, "invalid") for p in y_pred],
        "raw_output": raw_outputs,
        "correct": [t == p for t, p in zip(y_true, y_pred)],
    })
    pred_path = os.path.join(results_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}")

    # Save confusion matrix plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
        ax.set_title(f"Confusion Matrix — {mode}\nAccuracy: {acc:.1%} | F1 macro: {f1_macro:.3f}")
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix plot saved to {plot_path}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
