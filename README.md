# jb-45800-5-mission-4 — Titanic Survival Prediction

Trains a PyTorch neural network to predict passenger survival on the Titanic,
using the [Titanic Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data)
from Kaggle (891 labeled passengers, same data as the classic Kaggle Titanic competition).

The dataset CSV is committed at [data/titanic.csv](data/titanic.csv) so `train.py` and
`predict.py` can run immediately with no external download.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data data/titanic.csv --output titanic_model.pt --epochs 60
```

Prints per-epoch train/val loss and accuracy, and saves the best checkpoint
(model weights + all preprocessing statistics needed for inference) to `--output`.

## Predict (foreign / unseen input)

```bash
python predict.py --model titanic_model.pt --pclass 3 --sex male --age 22 --sibsp 1 --parch 0 --fare 7.25 --embarked S
python predict.py --model titanic_model.pt --name "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" --pclass 1 --sex female --age 38 --sibsp 1 --parch 0 --fare 71.28 --embarked C
```

Prints the predicted survival probability and class for a passenger that was not part of training.

## Model calibration process

Model architecture and training hyperparameters (epochs, hidden layer sizes,
feature engineering) were iterated on across git branches, each documenting its
own validation accuracy in `RESULTS.md`. The best-performing branch was merged
into `main`. See `RESULTS.md` for the full comparison.