from __future__ import annotations

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


def xgboost_margins(X, g, h, max_depth, reg_lambda=0.0, gamma=0.0, min_child_weight=0.0):
    import xgboost as xgb

    def fixed_gradient_objective(preds, dmatrix):
        return g, h

    params = {
        "tree_method": "exact",
        "max_depth": max_depth,
        "eta": 1.0,
        "lambda": reg_lambda,
        "gamma": 2.0 * gamma,
        "min_child_weight": min_child_weight,
        "alpha": 0.0,
        "base_score": 0.0,
        "nthread": 1,
    }
    booster = xgb.train(params, xgb.DMatrix(X), num_boost_round=1, obj=fixed_gradient_objective)
    return booster.predict(xgb.DMatrix(X), output_margin=True)


def lightgbm_margins(X, g, h, max_depth, reg_lambda=0.0, gamma=0.0, min_child_weight=0.0):
    import lightgbm as lgb

    def fixed_gradient_objective(preds, dataset):
        return g, h

    num_leaves_for_full_depth_expansion = 2 ** max_depth

    params = {
        "objective": fixed_gradient_objective,
        "num_leaves": num_leaves_for_full_depth_expansion,
        "max_depth": max_depth,
        "learning_rate": 1.0,
        "lambda_l2": reg_lambda,
        "lambda_l1": 0.0,
        "min_gain_to_split": 2.0 * gamma,
        "min_sum_hessian_in_leaf": min_child_weight,
        "min_data_in_leaf": 1,
        "max_bin": 65535,
        "min_data_in_bin": 1,
        "feature_pre_filter": False,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "boost_from_average": False,
        "verbose": -1,
    }
    dataset = lgb.Dataset(X, label=np.zeros(len(X)), free_raw_data=False, init_score=np.zeros(len(X)))
    booster = lgb.train(params, dataset, num_boost_round=1)

    return booster.predict(X, raw_score=True)


def count_leaves(node):
    if node._is_leaf:
        return 1
    
    return count_leaves(node._left_child) + count_leaves(node._right_child)


def tree_depth(node):
    if node._is_leaf:
        return 0
    
    return 1 + max(tree_depth(node._left_child), tree_depth(node._right_child))


def leaf_sizes(tree, X):
    sizes = []

    def walk(node, rows):
        if node._is_leaf:
            sizes.append(len(rows))
            return
        
        mask = rows[:, node._best_feature_index] <= node._best_threshold

        walk(node._left_child, rows[mask])
        walk(node._right_child, rows[~mask])

    walk(tree._root, np.asarray(X, dtype=float))
    return sizes
