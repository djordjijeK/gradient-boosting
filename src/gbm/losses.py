from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Loss(ABC):

    n_outputs: int
    
    @abstractmethod
    def base_score(self, y: ArrayLike) -> np.ndarray:
        raise NotImplementedError


    @abstractmethod
    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

 
    @abstractmethod
    def output(self, raw: ArrayLike) -> np.ndarray:
        raise NotImplementedError


class SquaredError(Loss):

    n_outputs = 1


    def base_score(self, y: ArrayLike) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

        return np.array([np.mean(y)])


    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        raw = np.asarray(raw, dtype=float)

        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

        if raw.ndim != 2 or raw.shape[1] != self.n_outputs:
            raise ValueError(
                f"raw must have shape (n, {self.n_outputs}), got shape {raw.shape}"
            )

        if raw.shape[0] != y.shape[0]:
            raise ValueError(
                f"y and raw must have the same number of rows: "
                f"y has {y.shape[0]}, raw has {raw.shape[0]}"
            )

        grad = raw - y[:, None]
        hessian = np.ones_like(raw)

        return grad, hessian


    def output(self, raw: ArrayLike) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)

        if raw.ndim != 2 or raw.shape[1] != self.n_outputs:
            raise ValueError(
                f"raw must have shape (n, {self.n_outputs}), got shape {raw.shape}"
            )

        return raw[:, 0]


class LogisticLoss(Loss):

    n_outputs = 1

    def base_score(self, y: ArrayLike) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        log_odds = np.log(p / (1 - p))

        return np.array([log_odds])


    def gradients_and_hessians(self, y: ArrayLike, raw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        raw = np.asarray(raw, dtype=float)

        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
        
        if raw.ndim != 2 or raw.shape[1] != self.n_outputs:
            raise ValueError(
                f"raw must have shape (n, {self.n_outputs}), got shape {raw.shape}"
            )
        
        if raw.shape[0] != y.shape[0]:
            raise ValueError(
                f"y and raw must have the same number of rows: "
                f"y has {y.shape[0]}, raw has {raw.shape[0]}"
            )
        
        p = 1 / (1 + np.exp(-raw))

        grad = p - y[:, None]
        hessian = p * (1 - p)

        return grad, hessian


    def output(self, raw: ArrayLike) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)

        if raw.ndim != 2 or raw.shape[1] != self.n_outputs:
            raise ValueError(
                f"raw must have shape (n, {self.n_outputs}), got shape {raw.shape}"
            )

        return (1 / (1 + np.exp(-raw)))[:, 0]

