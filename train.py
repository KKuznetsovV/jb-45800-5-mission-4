#!/usr/bin/env python3
"""
train.py

Train a small PyTorch MLP to predict Titanic passenger survival.

Baseline version: uses the raw Kaggle columns directly (Pclass, Sex, Age,
SibSp, Parch, Fare, Embarked) with simple median/mode imputation, one
hidden layer, and a short training run.

Usage:
    python train.py --data data/titanic.csv --output titanic_model.pt
    python train.py --data data/titanic.csv --epochs 30 --hidden_sizes 16
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CATEGORICAL_VALUES = {
    "Pclass": [1, 2, 3],
    "Sex": ["female", "male"],
    "Embarked": ["C", "Q", "S"],
}
NUMERIC_COLUMNS = ["Age", "SibSp", "Parch", "Fare"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Titanic survival classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/titanic.csv"))
    parser.add_argument("--output", type=Path, default=Path("titanic_model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--hidden_sizes", type=str, default="16", help="Comma-separated hidden layer sizes.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_raw(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def build_features(
    df: pd.DataFrame,
    age_median: float,
    fare_median: float,
    embarked_mode: str,
) -> pd.DataFrame:
    """Impute missing values and one-hot encode categoricals into a fixed column layout."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    parts = [df[NUMERIC_COLUMNS].astype(float)]
    for column, values in CATEGORICAL_VALUES.items():
        for value in values:
            parts.append((df[column] == value).astype(float).rename(f"{column}_{value}"))
    return pd.concat(parts, axis=1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    raw = load_raw(args.data)
    age_median = float(raw["Age"].median())
    fare_median = float(raw["Fare"].median())
    embarked_mode = str(raw["Embarked"].mode(dropna=True).iloc[0])

    features = build_features(raw, age_median, fare_median, embarked_mode)
    feature_columns = list(features.columns)
    labels = raw["Survived"].astype(float).to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        features.to_numpy(dtype=np.float32),
        labels.astype(np.float32),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels,
    )

    numeric_slice = slice(0, len(NUMERIC_COLUMNS))
    mean = x_train[:, numeric_slice].mean(axis=0)
    std = x_train[:, numeric_slice].std(axis=0)
    std[std == 0] = 1.0
    x_train[:, numeric_slice] = (x_train[:, numeric_slice] - mean) / std
    x_val[:, numeric_slice] = (x_val[:, numeric_slice] - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    hidden_sizes = [int(size) for size in args.hidden_sizes.split(",") if size]
    model = build_model(len(feature_columns), hidden_sizes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_loss = evaluate(model, val_x, val_y, criterion)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    checkpoint = {
        "model_state_dict": best_state,
        "hidden_sizes": hidden_sizes,
        "feature_columns": feature_columns,
        "categorical_values": CATEGORICAL_VALUES,
        "numeric_columns": NUMERIC_COLUMNS,
        "numeric_mean": mean.tolist(),
        "numeric_std": std.tolist(),
        "age_median": age_median,
        "fare_median": fare_median,
        "embarked_mode": embarked_mode,
        "best_val_accuracy": best_val_acc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {args.output}")


def build_model(input_dim: int, hidden_sizes: list[int]) -> nn.Module:
    layers: list[nn.Module] = []
    in_features = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        in_features = size
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(x).squeeze(1)
        loss = criterion(logits, y).item()
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy, loss


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
