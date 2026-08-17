# gradient-boosting

A gradient boosting framework written from scratch, with NumPy as its only
dependency.

The point of the exercise was to build every piece by hand - the regression
tree and its split search, the loss interface, the boosting loop - and then
check the result against the well known gradient boosting libraries. On three
real datasets a few hundred lines of NumPy land within a fraction of a percent
of XGBoost and LightGBM.

## The implementation

Everything is in `src/gbm`, a little over 500 lines.

- **`tree.py`** is the regression tree. It fits gradients and hessians instead
  of labels: sort each column, walk every candidate threshold, keep the best
  gain. Leaves get `-G / (H + reg_lambda)`. Growth stops on `max_depth`,
  `gamma`, `min_child_weight`, `min_samples_leaf` or `min_samples_split`.

- **`losses.py`** is the only file that knows what the target means. A loss
  hands back the base score to start from, the gradients and hessians for the
  current round, and the transform from raw scores to predictions. Squared
  error, logistic and softmax are implemented.

- **`boosting.py`** is the loop: shrinkage, row sampling per round, column
  sampling per tree. It grows one tree per output, so multiclass needs no
  special case. Softmax asks for K outputs, and the round grows K trees.

## Benchmark

`examples/benchmark.py` runs this implementation against XGBoost and LightGBM on
three real datasets: California Housing for regression, MAGIC Gamma Telescope
for binary classification, Forest Covertype for multiclass.

Every model gets the same split and the same settings: `n_estimators=200`,
`learning_rate=0.1`, `max_depth=6`, `min_child_weight=1.0`, `reg_lambda=1.0`,
`subsample=0.8`, `colsample_bytree=0.8`. XGBoost runs with
`tree_method="exact"`, so it searches the same splits this tree does.

**California Housing** | regression | 15 480 train / 5 160 test, 8 features

| model           | RMSE   | R2     | fit (s) |
| --------------- | ------ | ------ | ------- |
| gbm (this repo) | 0.4688 | 0.8354 | 6.38    |
| xgboost         | 0.4677 | 0.8362 | 0.62    |
| lightgbm        | 0.4669 | 0.8368 | 1.48    |

**MAGIC Gamma Telescope** | binary classification | 14 265 train / 4 755 test, 10 features

| model           | accuracy | ROC AUC | fit (s) |
| --------------- | -------- | ------- | ------- |
| gbm (this repo) | 0.8736   | 0.9298  | 7.81    |
| xgboost         | 0.8761   | 0.9322  | 0.70    |
| lightgbm        | 0.8730   | 0.9297  | 1.04    |

**Forest Covertype** | multiclass over 7 classes | 18 750 train / 6 250 test, 54 features

| model           | accuracy | log loss | fit (s) |
| --------------- | -------- | -------- | ------- |
| gbm (this repo) | 0.8205   | 0.4394   | 143.74  |
| xgboost         | 0.8211   | 0.4425   | 5.56    |
| lightgbm        | 0.8256   | 0.4284   | 5.61    |

## Running it

```sh
git clone https://github.com/djordjijeK/gradient-boosting.git
cd gradient-boosting

python -m venv venv
source venv/bin/activate
pip install -e ".[dev,examples]"

pytest
python examples/benchmark.py
```

The benchmark downloads its datasets the first time you run it and caches them
under `~/scikit_learn_data`. Expect about three minutes, most of it the
Covertype fit.
