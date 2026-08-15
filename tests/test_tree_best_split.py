import pytest
import numpy as np

from gbm.tree import DecisionTree
from helpers import logistic_grad_hessian


G, H = logistic_grad_hessian(
    np.array([0, 0, 1, 1], dtype=float),
    np.array([0.2, 0.35, 0.55, 0.9], dtype=float),
)


class TestBestSplit:

    def test_single_feature_returns_midpoint_threshold_and_gain(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        threshold, gain = DecisionTree()._best_split_on_feature(x=x, g=G, h=H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.8384707287933094)


    def test_single_feature_reg_lambda_shrinks_the_gain(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        threshold, gain = DecisionTree(reg_lambda=0.5)._best_split_on_feature(x=x, g=G, h=H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.35101955013664077)


    def test_single_feature_identical_values_find_no_split(self):
        x = np.array([3, 3, 3, 3], dtype=float)

        assert DecisionTree()._best_split_on_feature(x=x, g=G, h=H) == (None, None)


    def test_single_feature_min_samples_leaf_blocks_every_candidate(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        assert DecisionTree(min_samples_leaf=3)._best_split_on_feature(x=x, g=G, h=H) == (None, None)


    def test_single_feature_min_child_weight_blocks_every_candidate(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        assert DecisionTree(min_child_weight=0.5)._best_split_on_feature(x=x, g=G, h=H) == (None, None)


    def test_single_feature_gamma_above_the_best_gain_blocks_the_split(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        assert DecisionTree(gamma=1.0)._best_split_on_feature(x=x, g=G, h=H) == (None, None)


    def test_single_feature_gamma_below_the_best_gain_is_charged_once(self):
        x = np.array([1, 2, 3, 4], dtype=float)

        threshold, gain = DecisionTree(gamma=0.3)._best_split_on_feature(x=x, g=G, h=H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.8384707287933094 - 0.3)


    def test_multiple_features_pick_the_higher_gain_feature(self):
        X = np.array([[1, 1], [2, 3], [3, 2], [4, 4]], dtype=float)

        feature, threshold, gain = DecisionTree()._best_split(x=X, g=G, h=H)

        assert feature == 0
        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.8384707287933094)


    def test_multiple_features_skip_a_constant_feature(self):
        X = np.array([[7, 1], [7, 3], [7, 2], [7, 4]], dtype=float)

        feature, threshold, gain = DecisionTree()._best_split(x=X, g=G, h=H)

        assert feature == 1
        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.17511231341481087)


    def test_multiple_features_break_ties_toward_the_lower_index(self):
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=float)

        feature, threshold, gain = DecisionTree()._best_split(x=X, g=G, h=H)

        assert feature == 0
        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.8384707287933094)


    def test_multiple_features_all_constant_find_no_split(self):
        X = np.array([[7, 9], [7, 9], [7, 9], [7, 9]], dtype=float)

        assert DecisionTree()._best_split(x=X, g=G, h=H) == (None, None, None)
