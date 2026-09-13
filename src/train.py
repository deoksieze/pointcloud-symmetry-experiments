"""Train a TinyPointNet on synthetic point clouds.

CLI example:
    python -m src.train --experiment canonical --epochs 20 \
        --num-points 256 --seed 42 --device auto
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import GLOBAL_SEED, PointCloudDataset
from src.model import TinyPointNet

OUTPUTS_DIR = "outputs"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_rotation_policy(experiment: str) -> (bool, bool):
    """Return (train_rotation, test_rotation) for an experiment name."""
    policies = {
        "canonical": (False, False),            # Experiment A
        "rotation_augmented": (True, True),     # Experiment B
        "rotation_shift": (False, True),        # Experiment C
    }
    if experiment not in policies:
        raise ValueError(f"Unknown experiment: {experiment}. "
                         f"Choose from {sorted(policies)}")
    return policies[experiment]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (mean_loss, accuracy) over a dataset without gradients."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def save_metrics(metrics: dict):
    os.makedirs(os.path.join(OUTPUTS_DIR, "metrics"), exist_ok=True)
    exp = metrics["experiment"]
    # JSON
    json_path = os.path.join(OUTPUTS_DIR, "metrics", f"{exp}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # CSV table per epoch
    csv_path = os.path.join(OUTPUTS_DIR, "metrics", f"{exp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "test_loss", "test_acc"])
        for e in range(metrics["config"]["epochs"]):
            writer.writerow([
                e + 1,
                metrics["history"]["train_loss"][e],
                metrics["history"]["train_acc"][e],
                metrics["history"]["test_loss"][e],
                metrics["history"]["test_acc"][e],
            ])
    return csv_path, json_path


def plot_learning_curve(metrics: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = metrics["history"]
    epochs = range(1, metrics["config"]["epochs"] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, h["train_loss"], label="train")
    ax1.plot(epochs, h["test_loss"], label="test")
    ax1.set_title(f"{metrics['experiment']}: loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.legend()
    ax2.plot(epochs, h["train_acc"], label="train")
    ax2.plot(epochs, h["test_acc"], label="test")
    ax2.set_title(f"{metrics['experiment']}: accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train TinyPointNet")
    parser.add_argument("--experiment", default="canonical",
                        choices=["canonical", "rotation_augmented",
                                 "rotation_shift"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device: {device} | experiment: {args.experiment}")

    train_rotation, test_rotation = experiment_rotation_policy(args.experiment)

    train_ds = PointCloudDataset(
        num_samples=args.train_samples, num_points=args.num_points,
        apply_rotation=train_rotation, seed=args.seed,
    )
    test_ds = PointCloudDataset(
        num_samples=args.test_samples, num_points=args.num_points,
        apply_rotation=test_rotation, seed=args.seed + 1000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"train samples: {len(train_ds)} | test samples: {len(test_ds)} | "
          f"train rotated: {train_rotation} | test rotated: {test_rotation}")

    model = TinyPointNet(num_classes=2,
                         embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [],
               "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        test_loss, test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:2d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.3f} | "
              f"test_loss {test_loss:.4f} test_acc {test_acc:.3f}")

    metrics = {
        "experiment": args.experiment,
        "config": {
            "epochs": args.epochs, "num_points": args.num_points,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "batch_size": args.batch_size, "lr": args.lr,
            "embedding_dim": args.embedding_dim, "seed": args.seed,
            "train_rotation": train_rotation, "test_rotation": test_rotation,
            "device": str(device),
        },
        "history": history,
        "final": {
            "test_loss": test_loss, "test_acc": test_acc,
        },
    }

    csv_path, json_path = save_metrics(metrics)
    print(f"saved metrics: {csv_path}")

    os.makedirs(os.path.join(OUTPUTS_DIR, "figures"), exist_ok=True)
    fig_path = os.path.join(OUTPUTS_DIR, "figures",
                            f"learning_curve_{args.experiment}.png")
    plot_learning_curve(metrics, fig_path)
    print(f"saved figure: {fig_path}")

    os.makedirs(os.path.join(OUTPUTS_DIR, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(OUTPUTS_DIR, "checkpoints",
                             f"{args.experiment}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": metrics["config"],
        "test_acc": test_acc,
    }, ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")
    print(f"final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()