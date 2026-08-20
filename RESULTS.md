# Model calibration results

Each experiment branch below trains `train.py` on `data/titanic.csv` with an
80/20 stratified train/val split (`--seed 42`) and reports the best validation
accuracy achieved during training.

| Branch | Features | Architecture | Epochs | Best val accuracy |
| --- | --- | --- | --- | --- |
| `experiment/baseline-mlp` | Raw columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked), median/mode imputation | 1 hidden layer (16 units) | 20 | 79.89% |

## experiment/baseline-mlp

Command:

```bash
python train.py --data data/titanic.csv --output titanic_model.pt --epochs 20
```

Result: **79.89%** validation accuracy. Val accuracy was still trending up at
epoch 20 and the training loss curve had not plateaued, so the next branch
increases epochs, adds engineered features (title extracted from name, family
size), and widens/deepens the network.
