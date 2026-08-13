from __future__ import annotations

import pytest
import numpy as np

from helpers import random_tree_data


@pytest.fixture
def stump_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g = np.array([1.0, 2.0, -10.0, -11.0])
    h = np.array([1.0, 1.0, 1.0, 1.0])

    return X, g, h


@pytest.fixture
def tree_data():
    return random_tree_data
