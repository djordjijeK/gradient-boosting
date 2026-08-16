from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gbm.tree import DecisionTree
from gbm.losses import Loss, LogisticLoss, SoftmaxLoss, SquaredError


class GradientBoosting:

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
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

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
        self._feature_indices: list[list[np.ndarray]] = []


    def _make_loss(self, y: np.ndarray) -> Loss:
        """The loss to boost, chosen once the targets have been seen."""
        raise NotImplementedError(
            "GradientBoosting has no loss; use GradientBoostingRegressor or "
            "GradientBoostingClassifier."
        )


    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        """The targets in the coding the loss expects."""
        return y


    def fit(self, x: ArrayLike, y: ArrayLike) -> GradientBoosting:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

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

        for parameter, value in (
            ("subsample", self.subsample),
            ("colsample_bytree", self.colsample_bytree),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{parameter} must be in (0, 1], got {value}.")

        self._loss = self._make_loss(y)
        y = self._encode_target(y)

        self._trees = []
        self._feature_indices = []
        rng = np.random.default_rng(self.random_state)

        self._base_score = self._loss.base_score(y)
        raw = np.tile(self._base_score, (y.shape[0], 1))

        for _ in range(self.n_estimators):
            gradients, hessians = self._loss.gradients_and_hessians(y, raw)

            # One gradient pass per round, so one row sample per round: every
            # tree in the round is fitted to the same subsample of rows.
            row_indices = self._sample(rng, x.shape[0], self.subsample)

            round_trees = []
            round_feature_indices = []
            for k in range(self._loss.n_outputs):
                feature_indices = self._sample(rng, x.shape[1], self.colsample_bytree)

                tree = DecisionTree(**self._tree_params)
                tree.fit(
                    x[row_indices][:, feature_indices],
                    gradients[row_indices, k],
                    hessians[row_indices, k],
                )

                raw[:, k] += self.learning_rate * tree.predict(x[:, feature_indices])

                round_trees.append(tree)
                round_feature_indices.append(feature_indices)

            self._trees.append(round_trees)
            self._feature_indices.append(round_feature_indices)

        return self


    @staticmethod
    def _sample(rng: np.random.Generator, n: int, fraction: float) -> np.ndarray:
        if fraction == 1.0:
            return np.arange(n)

        return rng.choice(n, size=max(1, int(n * fraction)), replace=False)


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

        for round_trees, round_feature_indices in zip(self._trees, self._feature_indices):
            for k, (tree, feature_indices) in enumerate(zip(round_trees, round_feature_indices)):
                raw[:, k] += self.learning_rate * tree.predict(x[:, feature_indices])

        return raw


class GradientBoostingRegressor(GradientBoosting):

    def _make_loss(self, y: np.ndarray) -> Loss:
        return SquaredError()


    def predict(self, x: ArrayLike, output_margin: bool = False) -> np.ndarray:
        raw = self._raw_predict(x)

        if output_margin:
            return raw[:, 0]

        return self._loss.output(raw)


class GradientBoostingClassifier(GradientBoosting):
    """Binary and multiclass, told apart by the labels alone.

    Two classes are boosted as one log-odds score against LogisticLoss, more
    than two as one score per class against SoftmaxLoss. Nothing below the
    choice of loss knows which of the two it is running.
    """

    _classes: np.ndarray | None = None

    def _make_loss(self, y: np.ndarray) -> Loss:
        self._classes = np.unique(y)

        if self._classes.shape[0] < 2:
            raise ValueError(
                f"Classification needs at least two classes, got {self._classes.shape[0]}."
            )

        if self._classes.shape[0] == 2:
            return LogisticLoss()

        return SoftmaxLoss(self._classes.shape[0])


    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        """Labels of any dtype as the class indices 0, 1, ... the losses expect."""
        return np.searchsorted(self._classes, y).astype(float)


    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        return self._loss.output(self._raw_predict(x))


    def predict(self, x: ArrayLike, output_margin: bool = False) -> np.ndarray:
        if output_margin:
            raw = self._raw_predict(x)

            return raw[:, 0] if self._loss.n_outputs == 1 else raw

        return self._classes[np.argmax(self.predict_proba(x), axis=1)]
