from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gbm.tree import DecisionTree
from gbm.losses import Loss, LogisticLoss, SquaredError


class GradientBoosting:

    _loss_class: type[Loss] | None = None

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 3,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        reg_lambda: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self._tree_params = {
            "max_depth": max_depth,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "reg_lambda": reg_lambda,
        }

        self._loss: Loss | None = None
        self._base_score: np.ndarray | None = None
        self._trees: list[list[DecisionTree]] = []


    def fit(self, x: ArrayLike, y: ArrayLike) -> GradientBoosting:
        if self._loss_class is None:
            raise NotImplementedError(
                "GradientBoosting has no loss; use a subclass that sets _loss_class."
            )

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D array for x, got {x.ndim}D array instead. "
                f"Reshape your data using x.reshape(-1, 1) if it has a single feature."
            )

        if y.ndim != 1:
            raise ValueError(f"Expected 1D array for y, got {y.ndim}D array instead.")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of rows, got "
                f"x: {x.shape[0]}, y: {y.shape[0]}."
            )

        self._loss = self._loss_class()
        self._trees = []

        self._base_score = self._loss.base_score(y)
        raw = np.tile(self._base_score, (y.shape[0], 1))

        for _ in range(self.n_estimators):
            gradients, hessians = self._loss.gradients_and_hessians(y, raw)

            round_trees = []
            for k in range(self._loss.n_outputs):
                tree = DecisionTree(**self._tree_params)
                tree.fit(x, gradients[:, k], hessians[:, k])

                raw[:, k] += self.learning_rate * tree.predict(x)

                round_trees.append(tree)

            self._trees.append(round_trees)

        return self
        

    def _raw_predict(self, x: ArrayLike) -> np.ndarray:
        if self._base_score is None:
            raise ValueError("This estimator is not fitted yet; call fit first.")

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D array for x, got {x.ndim}D array instead. "
                f"Reshape your data using x.reshape(-1, 1) if it has a single feature."
            )

        raw = np.tile(self._base_score, (x.shape[0], 1))

        for round_trees in self._trees:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(x)

        return raw


class GradientBoostingRegressor(GradientBoosting):

    _loss_class = SquaredError

    def predict(self, x: ArrayLike, output_margin: bool = False) -> np.ndarray:
        raw = self._raw_predict(x)

        if output_margin:
            return raw[:, 0]

        return self._loss.output(raw)


class GradientBoostingClassifier(GradientBoosting):

    _loss_class = LogisticLoss

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        raw = self._raw_predict(x)
        positive_class_proba = self._loss.output(raw)

        return np.column_stack([1.0 - positive_class_proba, positive_class_proba])


    def predict(self, x: ArrayLike, output_margin: bool = False) -> np.ndarray:
        if output_margin:
            return self._raw_predict(x)[:, 0]

        positive_class_proba = self.predict_proba(x)[:, 1]

        return (positive_class_proba >= 0.5).astype(int)
