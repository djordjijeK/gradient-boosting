import pytest
import numpy as np

from gbm.tree import DecisionTree
from helpers import count_leaves, leaf_sizes, logistic_grad_hessian, tree_depth


G, H = logistic_grad_hessian(np.array([0, 0, 1, 1], dtype=float), np.array([0.2, 0.35, 0.55, 0.9], dtype=float))
BEST_GAIN = 0.8384707287933094
ASCENDING = np.array([1.0, 2.0, 3.0, 4.0])


class TestFitContract:

    @pytest.mark.parametrize(
        ("x", "g", "h", "message"),
        [
            (np.zeros(4), G, H, "2D array"),
            (np.zeros((4, 1)), G[:, None], H, "1D array for g"),
            (np.zeros((4, 1)), G, H[:, None], "1D array for h"),
            (np.zeros((4, 1)), G[:3], H, "same number of rows"),
        ],
    )
    def test_malformed_input_raises(self, x, g, h, message):
        with pytest.raises(ValueError, match=message):
            DecisionTree().fit(x, g, h)


    def test_fit_accepts_python_lists_not_only_arrays(self):
        tree = DecisionTree(max_depth=1).fit(
            [[1.0], [2.0], [3.0], [4.0]],
            [1.0, 2.0, -10.0, -11.0],
            [1.0, 1.0, 1.0, 1.0],
        )

        assert np.allclose(tree.predict([[1.0], [4.0]]), [-1.5, 10.5])


    def test_refitting_replaces_the_previous_tree(self, stump_data, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        refitted = DecisionTree(max_depth=2).fit(*stump_data).fit(X, g, h)

        assert np.allclose(refitted.predict(X), DecisionTree(max_depth=2).fit(X, g, h).predict(X))


class TestPredictContract:

    def test_predict_before_fit_raises_rather_than_crashing(self):
        with pytest.raises(ValueError, match="not fitted"):
            DecisionTree().predict(np.array([[1.0], [2.0]]))


    def test_predict_with_one_dimensional_input_raises(self, stump_data):
        tree = DecisionTree(max_depth=1).fit(*stump_data)

        with pytest.raises(ValueError, match="2D array"):
            tree.predict(np.array([1.0, 2.0, 3.0]))


    def test_predicting_a_single_row_matches_predicting_the_batch(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=4).fit(X, g, h)
        batch = tree.predict(X)

        for i in (0, 7, 51, 199):
            assert np.allclose(tree.predict(X[i:i + 1]), batch[i])


    def test_row_order_does_not_change_predictions(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=3).fit(X, g, h)
        order = np.random.default_rng(1).permutation(len(X))

        assert np.allclose(tree.predict(X)[order], tree.predict(X[order]))


class TestSplitSearch:

    def test_the_threshold_is_the_midpoint_between_the_straddling_values(self):
        threshold, gain = DecisionTree()._best_split_on_feature(ASCENDING, G, H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(BEST_GAIN)


    def test_reg_lambda_shrinks_the_gain(self):
        threshold, gain = DecisionTree(reg_lambda=0.5)._best_split_on_feature(ASCENDING, G, H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(0.35101955013664077)


    def test_gamma_below_the_best_gain_is_charged_once(self):
        threshold, gain = DecisionTree(gamma=0.3)._best_split_on_feature(ASCENDING, G, H)

        assert threshold == pytest.approx(2.5)
        assert gain == pytest.approx(BEST_GAIN - 0.3)


    @pytest.mark.parametrize(
        ("x", "parameters"),
        [
            (np.full(4, 7.0), {}),                      # no two distinct values to split between
            (ASCENDING, {"min_samples_leaf": 3}),
            (ASCENDING, {"min_child_weight": 0.5}),
            (ASCENDING, {"gamma": 1.0}),                # gamma above the best gain
        ],
    )
    def test_a_blocked_feature_offers_no_split(self, x, parameters):
        assert DecisionTree(**parameters)._best_split_on_feature(x, G, H) == (None, None)


    def test_the_higher_gain_feature_wins(self):
        X = np.array([[1, 1], [2, 3], [3, 2], [4, 4]], dtype=float)
        feature, threshold, gain = DecisionTree()._best_split(X, G, H)

        assert (feature, threshold) == (0, pytest.approx(2.5))
        assert gain == pytest.approx(BEST_GAIN)


    def test_a_constant_feature_is_skipped(self):
        X = np.array([[7, 1], [7, 3], [7, 2], [7, 4]], dtype=float)
        feature, threshold, gain = DecisionTree()._best_split(X, G, H)

        assert (feature, threshold) == (1, pytest.approx(2.5))
        assert gain == pytest.approx(0.17511231341481087)


    def test_ties_break_toward_the_lower_feature_index(self):
        X = np.tile(ASCENDING[:, None], (1, 3))
        feature, threshold, gain = DecisionTree()._best_split(X, G, H)

        assert (feature, threshold) == (0, pytest.approx(2.5))
        assert gain == pytest.approx(BEST_GAIN)


class TestLeafValues:

    def test_a_leaf_holds_negative_g_over_h(self, stump_data):
        X, g, h = stump_data

        assert np.allclose(DecisionTree(max_depth=1).fit(X, g, h).predict(X), [-1.5, -1.5, 10.5, 10.5])


    def test_reg_lambda_appears_in_the_leaf_denominator(self, stump_data):
        X, g, h = stump_data
        predictions = DecisionTree(max_depth=1, reg_lambda=2.0).fit(X, g, h).predict(X)

        assert np.allclose(predictions, [-3.0 / 4.0, -3.0 / 4.0, 21.0 / 4.0, 21.0 / 4.0])


    def test_the_leaf_is_hessian_weighted_not_a_plain_mean(self):
        X = ASCENDING[:, None]
        g = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([0.1, 0.2, 4.0, 8.0])
        predictions = DecisionTree(max_depth=0).fit(X, g, h).predict(X)

        assert np.allclose(predictions, -g.sum() / h.sum())
        assert not np.allclose(predictions, -np.mean(g))


class TestGrowth:

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_max_depth_bounds_the_realized_depth(self, tree_data, depth):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=depth).fit(X, g, h)

        assert tree_depth(tree._root) <= depth
        assert count_leaves(tree._root) <= 2 ** depth


    def test_min_samples_split_above_the_row_count_yields_a_single_leaf(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)

        assert count_leaves(DecisionTree(max_depth=3, min_samples_split=41).fit(X, g, h)._root) == 1


    def test_min_samples_split_at_the_row_count_still_splits(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)

        assert count_leaves(DecisionTree(max_depth=1, min_samples_split=40).fit(X, g, h)._root) == 2


    def test_min_samples_leaf_bounds_the_smallest_leaf(self, tree_data):
        X, g, h = tree_data(n=200, n_features=10, seed=7)
        tree = DecisionTree(max_depth=4, min_samples_leaf=20).fit(X, g, h)

        assert min(leaf_sizes(tree._root, X)) >= 20


    @pytest.mark.parametrize(
        ("parameter", "loose", "tight", "blocking"),
        [("min_child_weight", 0.0, 4.0, 6.0), ("gamma", 0.0, 2.5, 4.0)],
    )
    def test_a_rising_penalty_prunes_the_tree_back(self, tree_data, parameter, loose, tight, blocking):
        X, g, h = tree_data(n=40, n_features=3, seed=42)

        def leaves(value):
            tree = DecisionTree(max_depth=2, reg_lambda=1.0, **{parameter: value}).fit(X, g, h)

            return count_leaves(tree._root)

        assert (leaves(loose), leaves(tight), leaves(blocking)) == (4, 2, 1)


    def test_a_penalty_that_blocks_every_split_leaves_the_root_value(self, tree_data):
        X, g, h = tree_data(n=40, n_features=3, seed=42)
        tree = DecisionTree(max_depth=2, reg_lambda=1.0, gamma=4.0).fit(X, g, h)

        assert np.allclose(tree.predict(X), -g.sum() / (h.sum() + 1.0))


class TestRouting:

    def test_unseen_rows_route_by_threshold_not_by_position(self, stump_data):
        tree = DecisionTree(max_depth=1).fit(*stump_data)

        assert np.allclose(tree.predict([[2.49], [2.51]]), [-1.5, 10.5])
        assert np.allclose(tree.predict([[-1e9], [1e9]]), [-1.5, 10.5])


    def test_a_row_exactly_on_the_threshold_goes_left(self, stump_data):
        tree = DecisionTree(max_depth=1).fit(*stump_data)

        assert np.allclose(tree.predict([[2.5]]), [-1.5])


    def test_rows_all_routing_to_one_side_do_not_break_recursion(self, stump_data):
        tree = DecisionTree(max_depth=1).fit(*stump_data)

        assert np.allclose(tree.predict([[1.0], [2.0]]), [-1.5, -1.5])
        assert np.allclose(tree.predict([[3.0], [4.0]]), [10.5, 10.5])
        assert tree.predict(np.empty((0, 1))).shape == (0,)
