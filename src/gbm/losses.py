from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Loss(ABC):
    """Everything the booster knows about the target.

    ``n_outputs`` is how many raw scores a row carries -- the booster grows one
    tree per output per round -- not how many classes there are. Binary logistic
    regression boosts a single score and reports two probabilities.
    """

    n_outputs: int = 1

    @abstractmethod
    def base_score(self, y: ArrayLike) -> np.ndarray:
        """The constant every raw score starts from, one entry per output."""
        raise NotImplementedError


    @abstractmethod
    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """First and second derivatives with respect to raw, both shaped like raw."""
        raise NotImplementedError


    @abstractmethod
    def output(self, raw: ArrayLike) -> np.ndarray:
        """Raw scores in the space the caller asked for: values, or probabilities."""
        raise NotImplementedError


    def _as_raw(self, raw: ArrayLike) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)

        if raw.ndim != 2 or raw.shape[1] != self.n_outputs:
            raise ValueError(
                f"raw must have shape (n, {self.n_outputs}), got shape {raw.shape}"
            )

        return raw


    def _as_targets(self, y: ArrayLike, raw: np.ndarray | None = None) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

        if raw is not None and raw.shape[0] != y.shape[0]:
            raise ValueError(
                f"y and raw must have the same number of rows: "
                f"y has {y.shape[0]}, raw has {raw.shape[0]}"
            )

        return y


class SquaredError(Loss):

    def base_score(self, y: ArrayLike) -> np.ndarray:
        return np.array([np.mean(self._as_targets(y))])


    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raw = self._as_raw(raw)
        y = self._as_targets(y, raw)

        return raw - y[:, None], np.ones_like(raw)


    def output(self, raw: ArrayLike) -> np.ndarray:
        return self._as_raw(raw)[:, 0]


class LogisticLoss(Loss):

    def base_score(self, y: ArrayLike) -> np.ndarray:
        p = np.clip(np.mean(self._as_targets(y)), 1e-15, 1 - 1e-15)

        return np.array([np.log(p / (1 - p))])


    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raw = self._as_raw(raw)
        y = self._as_targets(y, raw)

        p = 1 / (1 + np.exp(-raw))

        return p - y[:, None], p * (1 - p)


    def output(self, raw: ArrayLike) -> np.ndarray:
        p = 1 / (1 + np.exp(-self._as_raw(raw)[:, 0]))

        return np.column_stack([1 - p, p])


class SoftmaxLoss(Loss):
    """Cross-entropy over one raw score per class.

    The generalisation is exact at every step. The base score is the log prior
    of each class, which for two classes is the log odds LogisticLoss uses up to
    a constant shift that softmax ignores. The gradient is again the probability
    minus the target, the target now being a one-hot row.

    Only the hessian has a choice in it. The true second derivative is a K x K
    matrix per row and a tree can carry a diagonal, so the off-diagonal coupling
    between classes is dropped; following XGBoost we boost against twice the
    diagonal, 2 p (1 - p), which halves the leaf step and keeps the trees from
    overshooting on the strength of a curvature they only half know.
    """

    def __init__(self, n_classes: int) -> None:
        if n_classes < 2:
            raise ValueError(f"SoftmaxLoss needs at least two classes, got {n_classes}.")

        self.n_outputs = n_classes


    def base_score(self, y: ArrayLike) -> np.ndarray:
        labels = self._as_labels(y)
        priors = np.bincount(labels, minlength=self.n_outputs) / labels.shape[0]

        return np.log(np.clip(priors, 1e-15, None))


    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raw = self._as_raw(raw)
        labels = self._as_labels(y, raw)

        p = self._probabilities(raw)

        return p - np.eye(self.n_outputs)[labels], 2 * p * (1 - p)


    def output(self, raw: ArrayLike) -> np.ndarray:
        return self._probabilities(self._as_raw(raw))


    def _as_labels(self, y: ArrayLike, raw: np.ndarray | None = None) -> np.ndarray:
        y = self._as_targets(y, raw)
        labels = y.astype(int)

        if not np.array_equal(labels, y) or labels.min() < 0 or labels.max() >= self.n_outputs:
            raise ValueError(
                f"y must hold class indices in [0, {self.n_outputs}), "
                f"got values in [{y.min()}, {y.max()}]"
            )

        return labels


    @staticmethod
    def _probabilities(raw: np.ndarray) -> np.ndarray:
        """Softmax, shifted by the row maximum so the exponential cannot overflow."""
        exponentials = np.exp(raw - raw.max(axis=1, keepdims=True))

        return exponentials / exponentials.sum(axis=1, keepdims=True)
