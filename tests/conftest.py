import pytest
import numpy as np

from helpers import load_dataset, random_tree_data
from sklearn.model_selection import train_test_split


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
    X, y = load_dataset("diabetes.csv")

    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="session")
def binary_data():
    X, y = load_dataset("breast_cancer.csv")

    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


@pytest.fixture(scope="session")
def multiclass_data():
    X, y = load_dataset("wine.csv")

    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
