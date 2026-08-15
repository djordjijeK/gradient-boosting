from __future__ import annotations

import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from helpers import load_dataset, random_tree_data


@pytest.fixture
def stump_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g = np.array([1.0, 2.0, -10.0, -11.0])
    h = np.array([1.0, 1.0, 1.0, 1.0])

    return X, g, h


@pytest.fixture
def tree_data():
    return random_tree_data


@pytest.fixture(scope="session")
def regression_data():
    """Diabetes, 200 rows, split 70/30. Returns (X_train, X_test, y_train, y_test)."""
    X, y = load_dataset("diabetes.csv")

    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="session")
def binary_data():
    """Breast cancer, 200 rows, stratified 70/30. Returns (X_train, X_test, y_train, y_test)."""
    X, y = load_dataset("breast_cancer.csv")

    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
