from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike


class DecisionTreeNode:

    def __init__(
        self, 
        value: float, 
        is_leaf: bool = False, 
        best_feature_index: int | None = None, 
        best_threshold: float | None =None
    ):
        self._value = value
        self._is_leaf = is_leaf
        self._best_feature_index = best_feature_index
        self._best_threshold = best_threshold

        self._left_child: DecisionTreeNode | None = None
        self._right_child: DecisionTreeNode | None = None


class DecisionTree:

    def __init__(
        self,
        max_depth: int = 3,
        gamma: float = 0.0,
        min_child_weight: float = 0.0,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        reg_lambda: float = 0.0,
    ):
        self._max_depth = max_depth
        self._gamma = gamma
        self._min_child_weight = min_child_weight
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split
        self._reg_lambda = reg_lambda

        self._root: DecisionTreeNode | None = None


    def fit(self, x: ArrayLike, g: ArrayLike, h: ArrayLike) -> DecisionTree:
        x = np.asarray(x, dtype=float)
        g = np.asarray(g, dtype=float)
        h = np.asarray(h, dtype=float)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D array for x, got {x.ndim}D array instead. "
                f"Reshape your data using x.reshape(-1, 1) if it has a single feature."
            )
        
        if g.ndim != 1:
            raise ValueError(f"Expected 1D array for g, got {g.ndim}D array instead.")
        
        if h.ndim != 1:
            raise ValueError(f"Expected 1D array for h, got {h.ndim}D array instead.")
        
        if not (x.shape[0] == g.shape[0] == h.shape[0]):
            raise ValueError(
                f"x, g, h must have the same number of rows, got "
                f"x: {x.shape[0]}, g: {g.shape[0]}, h: {h.shape[0]}."
            )

        self._root = self._grow_tree(x, g, h)

        return self


    def predict(self, x: ArrayLike) -> np.ndarray:
        if self._root is None:
            raise ValueError("This DecisionTree instance is not fitted yet. Call 'fit' before using 'predict'.")

        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D array for x, got {x.ndim}D array instead. "
                f"Reshape your data using x.reshape(-1, 1) if it has a single feature."
            )

        predictions = np.empty(x.shape[0])
        index = np.arange(x.shape[0])

        self._predict(x, self._root, predictions, index)

        return predictions


    def _predict(self, x: np.ndarray, root: DecisionTreeNode, predictions: np.ndarray, index: np.ndarray):
        if root._is_leaf:
            predictions[index] = root._value
            return

        mask = x[:, root._best_feature_index] <= root._best_threshold

        self._predict(x[mask], root._left_child, predictions, index[mask])
        self._predict(x[~mask], root._right_child, predictions, index[~mask])


    def _grow_tree(self, x: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        value = self._leaf_value(np.sum(g), np.sum(h))

        if depth == self._max_depth or x.shape[0] < self._min_samples_split:
            return DecisionTreeNode(value=value, is_leaf=True)

        best_feature_index, best_threshold, best_gain = self._best_split(x, g, h)

        if best_feature_index is None or best_threshold is None or best_gain is None:
            return DecisionTreeNode(value=value, is_leaf=True)

        root = DecisionTreeNode(value=value, best_feature_index=best_feature_index, best_threshold=best_threshold)

        split_mask = x[:, best_feature_index] <= best_threshold

        root._left_child = self._grow_tree(x[split_mask], g[split_mask], h[split_mask], depth + 1)
        root._right_child = self._grow_tree(x[~split_mask], g[~split_mask], h[~split_mask], depth + 1)

        return root


    def _best_split(self, x: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[int | None, float | None, float | None]:
        best_feature_index, best_threshold, best_gain = None, None, -np.inf

        for feature_index in range(x.shape[1]):
            split_threshold, gain = self._best_split_on_feature(x[:, feature_index], g, h)

            if split_threshold is not None and gain is not None and best_gain < gain:
                best_feature_index, best_threshold, best_gain = feature_index, split_threshold, gain

        return best_feature_index, best_threshold, None if best_gain == -np.inf else best_gain


    def _best_split_on_feature(self, x: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[float, float] | tuple[None, None]:
        order = np.argsort(x)

        x_ordered, g_ordered, h_ordered = x[order], g[order], h[order]

        g_parent, h_parent, n_parent = g_ordered.sum(), h_ordered.sum(), len(x)
        g_left, h_left, n_left = np.cumsum(g_ordered)[:-1], np.cumsum(h_ordered)[:-1], np.arange(1, len(x))
        g_right, h_right, n_right = g_parent - g_left, h_parent - h_left, n_parent - n_left

        valid = x_ordered[:-1] != x_ordered[1:]
        valid &= (n_left >= self._min_samples_leaf) & (n_right >= self._min_samples_leaf)
        valid &= (h_left >= self._min_child_weight) & (h_right >= self._min_child_weight)

        if not valid.any():
            return None, None

        gains = self._gain(g_left, h_left, g_right, h_right)
        gains = np.where(valid, gains, -np.inf)

        max_gain_index = np.argmax(gains)
        if gains[max_gain_index] - self._gamma <= 0:
            return None, None

        return 0.5 * (x_ordered[max_gain_index] + x_ordered[max_gain_index + 1]), gains[max_gain_index] - self._gamma


    def _leaf_value(self, G: float, H: float) -> float:
        return -G / (H + self._reg_lambda)


    def _gain(self, G_L, H_L, G_R, H_R):
        return 0.5 * (G_L**2 / (H_L + self._reg_lambda) + G_R**2 / (H_R + self._reg_lambda) - (G_L + G_R)**2 / (H_L + H_R + self._reg_lambda))
