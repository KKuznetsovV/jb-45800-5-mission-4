#!/usr/bin/env python3
"""
predict.py

Predict Titanic survival for one passenger using a checkpoint from train.py.

The checkpoint holds an ensemble of k-fold models; predictions are averaged
across all folds.

Example (foreign input, not part of the training data):
    python predict.py --model titanic_model.pt --pclass 3 --sex male --age 22 \\
        --sibsp 1 --parch 0 --fare 7.25 --embarked S --name "Doe, Mr. John"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
KNOWN_TITLES = ["Master", "Miss", "Mr", "Mrs"]
RARE_TITLE = "Rare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Titanic passenger survival.")
    parser.add_argument("--model", type=Path, default=Path("titanic_model.pt"))
    parser.add_argument("--pclass", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--sex", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sibsp", type=int, default=0, help="Number of siblings/spouses aboard.")
    parser.add_argument("--parch", type=int, default=0, help="Number of parents/children aboard.")
    parser.add_argument("--fare", type=float, required=True)
    parser.add_argument("--embarked", type=str, default="S", choices=["C", "Q", "S"])
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Full name in Kaggle format, e.g. "Doe, Mr. John". Used to derive the title feature.',
    )
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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


def extract_title(name: str | None, sex: str, age: float) -> str:
    if name and "," in name and "." in name.split(",", 1)[1]:
        title = name.split(",", 1)[1].split(".", 1)[0].strip()
        title = TITLE_MAP.get(title, title)
        return title if title in KNOWN_TITLES else RARE_TITLE

    # No name given: fall back to a reasonable title from sex/age.
    if sex == "male":
        return "Master" if age < 15 else "Mr"
    return "Miss" if age < 15 else "Mrs"


def build_feature_vector(
    args: argparse.Namespace, checkpoint: dict[str, Any], mean: list[float], std: list[float]
) -> torch.Tensor:
    title = extract_title(args.name, args.sex, args.age)
    family_size = args.sibsp + args.parch + 1

    numeric_columns: list[str] = checkpoint["numeric_columns"]
    raw_numeric: dict[str, float] = {
        "Age": args.age,
        "Fare": float(np.log1p(args.fare)),
        "FamilySize": float(family_size),
    }
    values: list[float] = [
        (raw_numeric[column] - mean[i]) / std[i] for i, column in enumerate(numeric_columns)
    ]

    raw_categorical: dict[str, Any] = {
        "Pclass": args.pclass,
        "Sex": args.sex,
        "Embarked": args.embarked,
        "Title": title,
    }
    categorical_values: dict[str, list[Any]] = checkpoint["categorical_values"]
    for column, allowed_values in categorical_values.items():
        actual = raw_categorical[column]
        values.extend(1.0 if actual == value else 0.0 for value in allowed_values)

    return torch.tensor([values], dtype=torch.float32)


def predict(args: argparse.Namespace) -> tuple[float, int]:
    device = choose_device()
    checkpoint = load_checkpoint(args.model, device)

    probabilities: list[float] = []
    for fold in checkpoint["folds"]:
        model = build_model(
            len(checkpoint["feature_columns"]), checkpoint["hidden_sizes"], checkpoint["dropout"]
        ).to(device)
        model.load_state_dict(fold["model_state_dict"])
        model.eval()

        features = build_feature_vector(args, checkpoint, fold["numeric_mean"], fold["numeric_std"]).to(device)
        with torch.inference_mode():
            logit = model(features).squeeze(1)
            probabilities.append(torch.sigmoid(logit).item())

    probability = sum(probabilities) / len(probabilities)
    return probability, int(probability >= 0.5)


def main() -> None:
    args = parse_args()
    probability, predicted_class = predict(args)
    label = "Survived" if predicted_class == 1 else "Did not survive"
    print(f"Survival probability: {probability:.2%}")
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
