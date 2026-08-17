from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def load_dataset(filename):
    """Return (X, y) from tests/data/<filename>; last column is the target."""
    table = np.loadtxt(DATA_DIR / filename, delimiter=",", skiprows=1)

    return table[:, :-1], table[:, -1]


def logistic_grad_hessian(y, p):
    return p - y, p * (1.0 - p)


def random_logistic_grad_hessian(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    raw_scores = rng.normal(0.0, 1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-raw_scores))

    return logistic_grad_hessian(y, p)


def random_tree_data(n, n_features, seed):
    X = np.random.default_rng(seed).normal(size=(n, n_features))
    g, h = random_logistic_grad_hessian(n=n, seed=seed)

    return X, g, h


def count_leaves(node):
    if node._is_leaf:
        return 1

    return count_leaves(node._left_child) + count_leaves(node._right_child)


def tree_depth(node):
    if node._is_leaf:
        return 0

    return 1 + max(tree_depth(node._left_child), tree_depth(node._right_child))


def leaf_sizes(node, X):
    sizes = []

    def walk(subtree, rows):
        if subtree._is_leaf:
            sizes.append(len(rows))
            return

        mask = rows[:, subtree._best_feature_index] <= subtree._best_threshold

        walk(subtree._left_child, rows[mask])
        walk(subtree._right_child, rows[~mask])

    walk(node, np.asarray(X, dtype=float))

    return sizes
