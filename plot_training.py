"""
Plot training curves from trainer_state.json saved in the model directory.

Usage:
    python plot_training.py --model_dir models/tinyllama-lora_run-01_date-2026-03-28
    python plot_training.py --from_hub Thibault-GAREL/smart-contract-lora
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_log_history(args):
    if args.state_file:
        path = args.state_file
    elif args.from_hub:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=args.from_hub,
            filename="checkpoint-612/trainer_state.json",
            repo_type="model"
        )
    else:
        path = os.path.join(args.model_dir, "checkpoint-612", "trainer_state.json")
        if not os.path.exists(path):
            checkpoints = [d for d in os.listdir(args.model_dir) if d.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.model_dir, checkpoints[-1], "trainer_state.json")

    with open(path) as f:
        state = json.load(f)
    return state["log_history"]


def parse_metrics(log_history):
    train_steps, train_loss, train_acc, train_lr = [], [], [], []
    eval_epochs, eval_loss, eval_acc = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("epoch", 0))
            train_loss.append(entry["loss"])
            train_acc.append(entry.get("mean_token_accuracy", None))
            train_lr.append(entry.get("learning_rate", None))
        elif "eval_loss" in entry:
            eval_epochs.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_mean_token_accuracy", None))

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "train_acc": [a for a in train_acc if a is not None],
        "train_lr": [lr for lr in train_lr if lr is not None],
        "eval_epochs": eval_epochs,
        "eval_loss": eval_loss,
        "eval_acc": [a for a in eval_acc if a is not None],
    }


def plot(metrics, output_path="training_curves.png"):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Training Curves — TinyLlama LoRA (Smart Contract Classification)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # --- Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics["train_steps"], metrics["train_loss"], color="#4C72B0", alpha=0.6, linewidth=1, label="Train loss")
    if metrics["eval_epochs"] and metrics["eval_loss"]:
        ax1.plot(metrics["eval_epochs"], metrics["eval_loss"], "o-", color="#DD8452", linewidth=2, markersize=7, label="Val loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Token Accuracy ---
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics["train_acc"]:
        ax2.plot(metrics["train_steps"][:len(metrics["train_acc"])], metrics["train_acc"],
                 color="#4C72B0", alpha=0.6, linewidth=1, label="Train acc")
    if metrics["eval_acc"]:
        ax2.plot(metrics["eval_epochs"][:len(metrics["eval_acc"])], metrics["eval_acc"],
                 "o-", color="#DD8452", linewidth=2, markersize=7, label="Val acc")
    ax2.set_title("Token Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Learning Rate ---
    ax3 = fig.add_subplot(gs[1, 0])
    if metrics["train_lr"]:
        ax3.plot(metrics["train_steps"][:len(metrics["train_lr"])], metrics["train_lr"],
                 color="#55A868", linewidth=1.5)
    ax3.set_title("Learning Rate (cosine schedule)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")
    ax3.grid(True, alpha=0.3)

    # --- Train vs Val Loss comparison ---
    ax4 = fig.add_subplot(gs[1, 1])
    if metrics["eval_epochs"] and metrics["eval_loss"]:
        # Interpolate train loss at eval epochs
        import numpy as np
        train_loss_interp = np.interp(metrics["eval_epochs"], metrics["train_steps"], metrics["train_loss"])
        ax4.plot(metrics["eval_epochs"], train_loss_interp, "s--", color="#4C72B0", linewidth=2, markersize=7, label="Train loss")
        ax4.plot(metrics["eval_epochs"], metrics["eval_loss"], "o-", color="#DD8452", linewidth=2, markersize=7, label="Val loss")
        ax4.set_title("Train vs Val Loss (per epoch)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Local model directory")
    parser.add_argument("--state_file", type=str, default=None, help="Direct path to trainer_state.json")
    parser.add_argument("--from_hub", type=str, default=None, help="HuggingFace repo id (e.g. Thibault-GAREL/smart-contract-lora)")
    parser.add_argument("--output", type=str, default="training_curves.png")
    args = parser.parse_args()

    if args.model_dir is None and args.from_hub is None and args.state_file is None:
        parser.error("Provide --model_dir, --state_file, or --from_hub")

    log_history = load_log_history(args)
    metrics = parse_metrics(log_history)
    plot(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
