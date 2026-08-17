from __future__ import annotations

import time
import warnings

import numpy as np
import xgboost as xgb
import lightgbm as lgb

from datasets import BINARY, LOADERS, REGRESSION, Dataset
from gbm import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")


PARAMS = dict(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1.0,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
)


def models(task: str) -> dict: 
    regression = task == REGRESSION

    return {
        "gbm (this repo)": (GradientBoostingRegressor if regression else GradientBoostingClassifier)(**PARAMS),
        "xgboost": (xgb.XGBRegressor if regression else xgb.XGBClassifier)(tree_method="exact", **PARAMS),
        "lightgbm": (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(
            num_leaves=2 ** PARAMS["max_depth"], min_child_samples=1, verbose=-1, **PARAMS
        ),
    }


def metrics(dataset: Dataset, model) -> dict:
    y = dataset.y_test

    if dataset.task == REGRESSION:
        prediction = model.predict(dataset.x_test)

        return {"RMSE": np.sqrt(mean_squared_error(y, prediction)), "R2": r2_score(y, prediction)}

    probabilities = model.predict_proba(dataset.x_test)
    accuracy = accuracy_score(y, model.predict(dataset.x_test))

    if dataset.task == BINARY:
        return {"accuracy": accuracy, "ROC AUC": roc_auc_score(y, probabilities[:, 1])}

    return {"accuracy": accuracy, "log loss": log_loss(y, probabilities, labels=dataset.classes)}


def report(dataset: Dataset, rows: list) -> None:
    columns = [*rows[0][1], "fit (s)"]

    print(f"\n{dataset.headline()}\n")
    print(f"  {'model':<18}" + "".join(f"{column:>12}" for column in columns))

    for name, scores, seconds in rows:
        print(f"  {name:<18}" + "".join(f"{value:>12.4f}" for value in scores.values()) + f"{seconds:>12.2f}")


def main() -> None:
    for load in LOADERS.values():
        dataset = load()
        rows = []

        for name, model in models(dataset.task).items():
            started = time.perf_counter()
            model.fit(dataset.x_train, dataset.y_train)

            rows.append((name, metrics(dataset, model), time.perf_counter() - started))

        report(dataset, rows)


if __name__ == "__main__":
    main()
