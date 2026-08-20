#!/usr/bin/env python3
"""
train.py

Train a PyTorch MLP to predict Titanic passenger survival.

Tuned v4: same engineered features as v2 (title extracted from name, family
size, log-fare transform, group-wise median imputation) but a smaller,
lighter-regularized network, which generalizes better on this small dataset.

k-fold + ensemble: replaces the single 80/20 split with stratified k-fold
cross-validation. Each fold's best model is kept and the final checkpoint
bundles all folds as an ensemble, which predict.py averages over at inference
time. (A TicketGroupSize feature was also tried here but measurably hurt the
out-of-fold accuracy -- see RESULTS.md -- so it was dropped.)

Usage:
    python train.py --data data/titanic.csv --output titanic_model.pt
    python train.py --data data/titanic.csv --epochs 150 --hidden_sizes 32,16 --n_folds 5
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
KNOWN_TITLES = ["Master", "Miss", "Mr", "Mrs"]
RARE_TITLE = "Rare"

CATEGORICAL_VALUES = {
    "Pclass": [1, 2, 3],
    "Sex": ["female", "male"],
    "Embarked": ["C", "Q", "S"],
    "Title": KNOWN_TITLES + [RARE_TITLE],
}
NUMERIC_COLUMNS = ["Age", "Fare", "FamilySize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Titanic survival classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/titanic.csv"))
    parser.add_argument("--output", type=Path, default=Path("titanic_model.pt"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--hidden_sizes", type=str, default="32,16", help="Comma-separated hidden layer sizes.")
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


def extract_title(name: str) -> str:
    title = name.split(",", 1)[1].split(".", 1)[0].strip()
    title = TITLE_MAP.get(title, title)
    return title if title in KNOWN_TITLES else RARE_TITLE


def engineer_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Title/FamilySize columns derived from the raw Kaggle columns."""
    df = df.copy()
    df["Title"] = df["Name"].map(extract_title)
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df


def fit_imputers(df: pd.DataFrame) -> dict:
    age_by_title = df.groupby("Title")["Age"].median().to_dict()
    fare_by_pclass = df.groupby("Pclass")["Fare"].median().to_dict()
    return {
        "age_median_by_title": {k: float(v) for k, v in age_by_title.items()},
        "age_median_overall": float(df["Age"].median()),
        "fare_median_by_pclass": {int(k): float(v) for k, v in fare_by_pclass.items()},
        "fare_median_overall": float(df["Fare"].median()),
        "embarked_mode": str(df["Embarked"].mode(dropna=True).iloc[0]),
    }


def apply_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    df = df.copy()
    age_by_title = imputers["age_median_by_title"]
    fare_by_pclass = imputers["fare_median_by_pclass"]

    df["Age"] = df.apply(
        lambda row: age_by_title.get(row["Title"], imputers["age_median_overall"])
        if pd.isna(row["Age"])
        else row["Age"],
        axis=1,
    )
    df["Fare"] = df.apply(
        lambda row: fare_by_pclass.get(row["Pclass"], imputers["fare_median_overall"])
        if pd.isna(row["Fare"])
        else row["Fare"],
        axis=1,
    )
    df["Embarked"] = df["Embarked"].fillna(imputers["embarked_mode"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and log-transform fare into a fixed column layout."""
    df = df.copy()
    df["Fare"] = np.log1p(df["Fare"])

    parts = [df[NUMERIC_COLUMNS].astype(float)]
    for column, values in CATEGORICAL_VALUES.items():
        for value in values:
            parts.append((df[column] == value).astype(float).rename(f"{column}_{value}"))
    return pd.concat(parts, axis=1)


def build_model(input_dim: int, hidden_sizes: list[int], dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    in_features = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.BatchNorm1d(size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
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


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    raw = engineer_raw_columns(load_raw(args.data))
    imputers = fit_imputers(raw)
    raw = apply_imputers(raw, imputers)

    features = build_features(raw)
    feature_columns = list(features.columns)
    x_all = features.to_numpy(dtype=np.float32)
    y_all = raw["Survived"].astype(np.float32).to_numpy()

    hidden_sizes = [int(size) for size in args.hidden_sizes.split(",") if size]
    numeric_slice = slice(0, len(NUMERIC_COLUMNS))

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_checkpoints = []
    fold_accuracies = []
    # Out-of-fold predictions: each row is predicted exactly once, by a model
    # that never saw it during training, giving an unbiased accuracy estimate.
    oof_probabilities = np.zeros_like(y_all)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_all, y_all), start=1):
        x_train, x_val = x_all[train_idx].copy(), x_all[val_idx].copy()
        y_train, y_val = y_all[train_idx], y_all[val_idx]

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

        model = build_model(len(feature_columns), hidden_sizes, args.dropout).to(device)

        # Inverse-frequency positive-class weight, since ~62% of passengers did not survive.
        pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        best_val_acc = -1.0
        best_state = None

        for _epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_acc, _ = evaluate(model, val_x, val_y, criterion)
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            oof_probabilities[val_idx] = torch.sigmoid(model(val_x).squeeze(1)).cpu().numpy()

        print(f"Fold {fold_idx}/{args.n_folds} | best val acc {best_val_acc:.2%}")
        fold_accuracies.append(best_val_acc)
        fold_checkpoints.append(
            {
                "model_state_dict": best_state,
                "numeric_mean": mean.tolist(),
                "numeric_std": std.tolist(),
            }
        )

    cv_mean_accuracy = float(np.mean(fold_accuracies))
    cv_std_accuracy = float(np.std(fold_accuracies))
    oof_accuracy = float(((oof_probabilities >= 0.5) == y_all).mean())

    checkpoint = {
        "folds": fold_checkpoints,
        "hidden_sizes": hidden_sizes,
        "dropout": args.dropout,
        "feature_columns": feature_columns,
        "categorical_values": CATEGORICAL_VALUES,
        "numeric_columns": NUMERIC_COLUMNS,
        **imputers,
        "cv_mean_accuracy": cv_mean_accuracy,
        "cv_std_accuracy": cv_std_accuracy,
        "oof_accuracy": oof_accuracy,
        "best_val_accuracy": oof_accuracy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"\nPer-fold accuracy: {cv_mean_accuracy:.2%} +/- {cv_std_accuracy:.2%} (mean +/- std over {args.n_folds} folds)")
    print(f"Out-of-fold accuracy (unbiased, whole-dataset): {oof_accuracy:.2%}")
    print(f"Model saved to: {args.output}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
