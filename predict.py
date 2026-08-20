#!/usr/bin/env python3
"""
predict.py

Predict Titanic survival for one passenger using a checkpoint from train.py.

Example (foreign input, not part of the training data):
    python predict.py --model titanic_model.pt --pclass 3 --sex male --age 22 \\
        --sibsp 1 --parch 0 --fare 7.25 --embarked S
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


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
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(input_dim: int, hidden_sizes: list[int]) -> nn.Module:
    layers: list[nn.Module] = []
    in_features = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        in_features = size
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


def build_feature_vector(args: argparse.Namespace, checkpoint: dict) -> torch.Tensor:
    numeric_columns = checkpoint["numeric_columns"]
    raw_numeric = {
        "Age": args.age,
        "SibSp": float(args.sibsp),
        "Parch": float(args.parch),
        "Fare": args.fare,
    }
    mean = checkpoint["numeric_mean"]
    std = checkpoint["numeric_std"]
    values = [
        (raw_numeric[column] - mean[i]) / std[i] for i, column in enumerate(numeric_columns)
    ]

    raw_categorical = {"Pclass": args.pclass, "Sex": args.sex, "Embarked": args.embarked}
    for column, allowed_values in checkpoint["categorical_values"].items():
        actual = raw_categorical[column]
        values.extend(1.0 if actual == value else 0.0 for value in allowed_values)

    return torch.tensor([values], dtype=torch.float32)


def predict(args: argparse.Namespace) -> tuple[float, int]:
    device = choose_device()
    checkpoint = load_checkpoint(args.model, device)

    model = build_model(len(checkpoint["feature_columns"]), checkpoint["hidden_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = build_feature_vector(args, checkpoint).to(device)
    with torch.inference_mode():
        logit = model(features).squeeze(1)
        probability = torch.sigmoid(logit).item()

    return probability, int(probability >= 0.5)


def main() -> None:
    args = parse_args()
    probability, predicted_class = predict(args)
    label = "Survived" if predicted_class == 1 else "Did not survive"
    print(f"Survival probability: {probability:.2%}")
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
