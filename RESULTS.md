# Model calibration results

Each experiment branch below trains `train.py` on `data/titanic.csv` with an
80/20 stratified train/val split (`--seed 42`) and reports the best validation
accuracy achieved during training.

| Branch | Features | Architecture | Epochs | Best val accuracy |
| --- | --- | --- | --- | --- |
| `experiment/baseline-mlp` | Raw columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), median/mode imputation | 1 hidden layer (16 units) | 20 | 79.89% |
| `experiment/tuned-mlp-v2` | + Title (from name), FamilySize, log(Fare), group-wise median imputation, class-weighted loss | 2 hidden layers (64, 32) + BatchNorm + Dropout(0.3) | 150 | **84.36%** |
| `experiment/tuned-mlp-v3` | + Deck (from Cabin, mostly "unknown") on top of v2 features | 3 hidden layers (128, 64, 32) + BatchNorm + Dropout(0.25) | 200 | 82.68% |

## experiment/baseline-mlp

Command:

```bash
python train.py --data data/titanic.csv --output titanic_model.pt --epochs 20
```

Result: **79.89%** validation accuracy. Val accuracy was still trending up at
epoch 20 and the training loss curve had not plateaued, so the next branch
increases epochs, adds engineered features (title extracted from name, family
size), and widens/deepens the network.

## experiment/tuned-mlp-v2

Command:

```bash
python train.py --data data/titanic.csv --output titanic_model.pt --epochs 150
```

Changes vs. baseline:
- Added `Title` (extracted from the `Name` column: Mr/Mrs/Miss/Master/Rare),
  `FamilySize` (SibSp + Parch + 1), and a `log1p(Fare)` transform.
- Missing `Age`/`Fare` are imputed with the median grouped by `Title`/`Pclass`
  instead of a single global median.
- Deeper network (64 -> 32 hidden units) with `BatchNorm1d` + `Dropout(0.3)`,
  `AdamW` with weight decay, `ReduceLROnPlateau`, and a class-weighted
  `BCEWithLogitsLoss` (~62% of passengers did not survive).
- Epochs increased from 20 to 150 (val accuracy plateaus around epoch 90-110).

Result: **84.36%** validation accuracy, up from 79.89% on the baseline.

## experiment/tuned-mlp-v3

Command:

```bash
python train.py --data data/titanic.csv --output titanic_model.pt --epochs 200
```

Changes vs. v2:
- Added a `Deck` feature (first letter of `Cabin`), one-hot encoded with an
  explicit "unknown" bucket since ~77% of passengers have no cabin recorded.
- Widened/deepened the network to 3 hidden layers (128 -> 64 -> 32) and ran
  for more epochs (200).

Result: **82.68%** validation accuracy — worse than v2. The `Deck` feature is
mostly "unknown" and adds noisy one-hot columns without real signal, and the
larger network overfits the small (891-row) dataset instead of generalizing
better. This confirms bigger/more isn't automatically better for this
dataset size.

## Conclusion

`experiment/tuned-mlp-v2` has the best validation accuracy (84.36%) and was
merged into `main`.
