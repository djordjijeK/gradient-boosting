from __future__ import annotations

import pytest
import numpy as np

from gbm import DecisionTree
from helpers import count_leaves, leaf_sizes, tree_depth


class TestFitPredictContract:

    def test_predict_before_fit_raises_rather_than_crashing(self):
        with pytest.raises(ValueError, match="not fitted"):
            DecisionTree().predict(np.array([[1.0], [2.0]]))


    def test_predict_with_one_dimensional_input_raises(self, stump_data):
        X, g, h = stump_data
        tree = DecisionTree(max_depth=1).fit(X, g, h)

        with pytest.raises(ValueError, match="2D array"):
            tree.predict(np.array([1.0, 2.0, 3.0]))


    def test_fit_with_one_dimensional_x_raises(self, stump_data):
        _, g, h = stump_data

        with pytest.raises(ValueError, match="2D array"):
            DecisionTree().fit(np.array([1.0, 2.0, 3.0, 4.0]), g, h)


    def test_fit_with_mismatched_row_counts_raises(self, stump_data):
        X, g, h = stump_data

        with pytest.raises(ValueError, match="same number of rows"):
            DecisionTree().fit(X, g[:3], h)


    def test_fit_accepts_python_lists_not_only_arrays(self):
        tree = DecisionTree(max_depth=1).fit(
            [[1.0], [2.0], [3.0], [4.0]],
            [1.0, 2.0, -10.0, -11.0],
            [1.0, 1.0, 1.0, 1.0],
        )

        assert np.allclose(tree.predict([[1.0], [4.0]]), [-1.5, 10.5])


class TestLeafValues:

    def test_stump_leaf_values_equal_negative_g_over_h(self, stump_data):
        X, g, h = stump_data
        predictions = DecisionTree(max_depth=1).fit(X, g, h).predict(X)
        
        assert np.allclose(predictions, [-1.5, -1.5, 10.5, 10.5])


    def test_max_depth_zero_predicts_the_root_value_everywhere(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        predictions = DecisionTree(max_depth=0).fit(X, g, h).predict(X)
        expected = -g.sum() / h.sum()

        assert np.allclose(predictions, expected)


    def test_reg_lambda_appears_in_the_leaf_denominator(self, stump_data):
        X, g, h = stump_data
        predictions = DecisionTree(max_depth=1, reg_lambda=2.0).fit(X, g, h).predict(X)

        
        assert np.allclose(predictions, [-3.0 / 4.0, -3.0 / 4.0, 21.0 / 4.0, 21.0 / 4.0])


    def test_reg_lambda_shrinks_every_leaf_toward_zero(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        unregularized = DecisionTree(max_depth=3).fit(X, g, h).predict(X)
        regularized = DecisionTree(max_depth=3, reg_lambda=10.0).fit(X, g, h).predict(X)

        assert np.all(np.abs(regularized) < np.abs(unregularized))


    def test_leaf_value_is_the_hessian_weighted_not_the_plain_mean(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        g = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([0.1, 0.2, 4.0, 8.0])
        predictions = DecisionTree(max_depth=0).fit(X, g, h).predict(X)

        assert np.allclose(predictions, -g.sum() / h.sum())
        assert not np.allclose(predictions, -np.mean(g))


class TestStoppingRules:

    def test_max_depth_bounds_the_realized_depth(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)

        for depth in (1, 2, 3, 4):
            tree = DecisionTree(max_depth=depth).fit(X, g, h)
            assert tree_depth(tree._root) <= depth
            assert count_leaves(tree._root) <= 2 ** depth


    def test_min_samples_split_above_row_count_yields_a_single_leaf(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        tree = DecisionTree(max_depth=3, min_samples_split=41).fit(X, g, h)

        assert count_leaves(tree._root) == 1


    def test_min_samples_split_at_the_row_count_still_splits(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        tree = DecisionTree(max_depth=1, min_samples_split=40).fit(X, g, h)

        assert count_leaves(tree._root) == 2


    def test_min_samples_leaf_bounds_the_smallest_leaf(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=4, min_samples_leaf=20).fit(X, g, h)

        assert min(leaf_sizes(tree, X)) >= 20


    def test_min_child_weight_blocks_splits_as_it_rises(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)

        loose = DecisionTree(max_depth=2, reg_lambda=1.0, min_child_weight=0.0).fit(X, g, h)
        tight = DecisionTree(max_depth=2, reg_lambda=1.0, min_child_weight=4.0).fit(X, g, h)
        blocking = DecisionTree(max_depth=2, reg_lambda=1.0, min_child_weight=6.0).fit(X, g, h)

        assert count_leaves(loose._root) == 4
        assert count_leaves(tight._root) == 2
        assert count_leaves(blocking._root) == 1


    def test_gamma_blocks_splits_as_it_rises(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        loose = DecisionTree(max_depth=2, reg_lambda=1.0, gamma=0.0).fit(X, g, h)
        tight = DecisionTree(max_depth=2, reg_lambda=1.0, gamma=2.5).fit(X, g, h)
        blocking = DecisionTree(max_depth=2, reg_lambda=1.0, gamma=4.0).fit(X, g, h)

        assert count_leaves(loose._root) == 4
        assert count_leaves(tight._root) == 2
        assert count_leaves(blocking._root) == 1
        
        assert np.allclose(blocking.predict(X), -g.sum() / (h.sum() + 1.0))


    def test_constant_features_cannot_be_split(self):
        X = np.ones((10, 2))
        g, h = np.arange(10.0), np.full(10, 0.5)
        tree = DecisionTree(max_depth=3).fit(X, g, h)

        assert count_leaves(tree._root) == 1
        assert np.allclose(tree.predict(X), -g.sum() / h.sum())


class TestRouting:

    def test_unseen_rows_route_by_threshold_not_by_position(self, stump_data):
        X, g, h = stump_data
        tree = DecisionTree(max_depth=1).fit(X, g, h)
        
        assert np.allclose(tree.predict([[2.49], [2.51]]), [-1.5, 10.5])
        assert np.allclose(tree.predict([[-1e9], [1e9]]), [-1.5, 10.5])


    def test_a_row_exactly_on_the_threshold_goes_left(self, stump_data):
        X, g, h = stump_data
        tree = DecisionTree(max_depth=1).fit(X, g, h)

        assert np.allclose(tree.predict([[2.5]]), [-1.5])


    def test_predicting_a_single_row_matches_predicting_the_batch(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=4).fit(X, g, h)
        batch = tree.predict(X)

        for i in (0, 7, 51, 199):
            assert np.allclose(tree.predict(X[i:i + 1]), batch[i])


    def test_rows_all_routing_to_one_side_do_not_break_recursion(self, stump_data):
        X, g, h = stump_data
        tree = DecisionTree(max_depth=1).fit(X, g, h)

        assert np.allclose(tree.predict([[1.0], [2.0]]), [-1.5, -1.5])
        assert np.allclose(tree.predict([[3.0], [4.0]]), [10.5, 10.5])


    def test_row_order_does_not_change_predictions(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=3).fit(X, g, h)
        order = np.random.default_rng(1).permutation(len(X))
        
        assert np.allclose(tree.predict(X)[order], tree.predict(X[order]))
