"""Stochastic row and column sampling from the third build increment."""
from types import SimpleNamespace

import numpy as np
import pytest

import gbm.boosting as boosting_module
from gbm.boosting import GradientBoostingRegressor


class RecordingTree:
    """A zero-output tree that records exactly what the booster gives it."""

    instances = []

    def __init__(self, **_):
        self.fit_x = None
        self.fit_g = None
        self.fit_h = None
        self.predict_inputs = []
        self._root = SimpleNamespace(_is_leaf=True)
        type(self).instances.append(self)

    def fit(self, x, g, h):
        self.fit_x = np.asarray(x).copy()
        self.fit_g = np.asarray(g).copy()
        self.fit_h = np.asarray(h).copy()
        return self

    def predict(self, x):
        x = np.asarray(x)
        self.predict_inputs.append(x.copy())
        return np.zeros(x.shape[0])


@pytest.fixture
def recording_tree(monkeypatch):
    RecordingTree.instances = []
    monkeypatch.setattr(boosting_module, "DecisionTree", RecordingTree)
    return RecordingTree


def original_columns(sampled, full):
    """Map columns in ``sampled`` back to their positions in ``full``."""
    matches = []
    for column in sampled.T:
        found = [
            j for j in range(full.shape[1])
            if np.array_equal(np.sort(column), np.sort(full[:, j]))
        ]
        assert len(found) == 1
        matches.append(found[0])
    return tuple(matches)


class TestSampling:

    def test_full_sampling_preserves_the_legacy_row_and_column_order(
        self, recording_tree
    ):
        x = np.arange(30.0).reshape(6, 5)
        y = np.arange(6.0)

        GradientBoostingRegressor(n_estimators=3).fit(x, y)

        assert all(np.array_equal(tree.fit_x, x) for tree in recording_tree.instances)

    def test_subsample_draws_the_requested_rows_without_replacement_each_round(
        self, recording_tree
    ):
        x = np.arange(12.0)[:, None]
        y = np.linspace(-1.0, 1.0, len(x))

        GradientBoostingRegressor(
            n_estimators=4,
            subsample=0.5,
            random_state=7,
            min_child_weight=0.0,
        ).fit(x, y)

        row_draws = [
            tuple(sorted(tree.fit_x[:, 0].astype(int)))
            for tree in recording_tree.instances
        ]

        assert all(len(draw) == 6 for draw in row_draws)
        assert all(len(set(draw)) == 6 for draw in row_draws)
        assert len(set(row_draws)) > 1

    def test_random_state_reproduces_the_sampling_plan(self, recording_tree):
        x = np.arange(60.0).reshape(12, 5)
        y = np.linspace(-1.0, 1.0, len(x))

        def plan(seed):
            recording_tree.instances = []
            GradientBoostingRegressor(
                n_estimators=4,
                subsample=0.5,
                colsample_bytree=0.6,
                random_state=seed,
                min_child_weight=0.0,
            ).fit(x, y)
            return tuple(tuple(tree.fit_x.ravel()) for tree in recording_tree.instances)

        assert plan(19) == plan(19)
        assert plan(19) != plan(20)

    def test_colsample_uses_the_same_columns_when_the_tree_predicts(
        self, recording_tree
    ):
        rows = np.arange(10.0)[:, None]
        x = rows + 100.0 * np.arange(6.0)[None, :]
        y = np.linspace(-1.0, 1.0, len(x))

        model = GradientBoostingRegressor(
            n_estimators=4,
            colsample_bytree=0.5,
            random_state=11,
            min_child_weight=0.0,
        ).fit(x, y)

        plans = [original_columns(tree.fit_x, x) for tree in recording_tree.instances]
        assert all(len(plan) == 3 and len(set(plan)) == 3 for plan in plans)
        assert len(set(plans)) > 1

        model.predict(x[::-1])
        for tree, columns in zip(recording_tree.instances, plans):
            assert np.array_equal(tree.predict_inputs[-1], x[::-1][:, columns])

    @pytest.mark.parametrize(
        ("parameter", "value"),
        [("subsample", 0.0), ("subsample", 1.01),
         ("colsample_bytree", 0.0), ("colsample_bytree", 1.01)],
    )
    def test_sampling_fraction_outside_zero_open_one_closed_raises(
        self, parameter, value
    ):
        x = np.arange(8.0)[:, None]
        y = np.arange(8.0)

        with pytest.raises(ValueError, match=parameter):
            GradientBoostingRegressor(**{parameter: value}).fit(x, y)
