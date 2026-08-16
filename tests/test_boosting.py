"""Boosting behaviour the oracle-agreement suite cannot prove on its own.

Matching XGBoost row-for-row already establishes that base scores, gradients,
hessians, shrinkage and the loop are correct. What is left is the arithmetic
verified by hand, the contract, and the shape of what predict returns.
"""
import numpy as np
import pytest

from gbm.boosting import GradientBoostingClassifier, GradientBoostingRegressor


X = np.array([[1.0], [2.0], [3.0], [4.0]])
Y = np.array([1.0, 2.0, 10.0, 11.0])
LABELS = np.array([0.0, 0.0, 1.0, 1.0])

UNREGULARIZED = dict(max_depth=1, reg_lambda=0.0, min_child_weight=0.0, gamma=0.0)


class TestBoostingLoop:

    def test_one_round_at_full_learning_rate_matches_the_hand_computation(self):
        # base = mean(y) = 6.0, so g = raw - y = [5, 4, -4, -5] and h = 1.
        # The only split is at 2.5; leaves are -G/H = -9/2 and +9/2.
        # raw becomes 6 + [-4.5, -4.5, 4.5, 4.5].
        model = GradientBoostingRegressor(
            n_estimators=1, learning_rate=1.0, **UNREGULARIZED
        ).fit(X, Y)

        assert model.predict(X) == pytest.approx([1.5, 1.5, 10.5, 10.5])


    def test_a_tenth_of_the_learning_rate_moves_a_tenth_as_far(self):
        base = 6.0
        full = GradientBoostingRegressor(n_estimators=1, learning_rate=1.0, **UNREGULARIZED)
        tenth = GradientBoostingRegressor(n_estimators=1, learning_rate=0.1, **UNREGULARIZED)

        assert tenth.fit(X, Y).predict(X) - base == pytest.approx(
            0.1 * (full.fit(X, Y).predict(X) - base)
        )


    def test_more_rounds_reduce_training_error(self, regression_data):
        X_train, _, y_train, _ = regression_data

        errors = [
            float(np.sqrt(np.mean((
                GradientBoostingRegressor(n_estimators=n).fit(X_train, y_train)
                .predict(X_train) - y_train
            ) ** 2)))
            for n in (1, 5, 25)
        ]

        assert errors[0] > errors[1] > errors[2]


    def test_refitting_replaces_trees_rather_than_accumulating(self):
        model = GradientBoostingRegressor(n_estimators=3)

        model.fit(X, Y)
        model.fit(X, Y)

        assert len(model._trees) == 3


class TestPredictContract:

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match="not fitted"):
            GradientBoostingRegressor().predict(X)


    def test_regressor_predict_returns_one_value_per_row(self, regression_data):
        X_train, X_test, y_train, _ = regression_data

        model = GradientBoostingRegressor(n_estimators=5).fit(X_train, y_train)

        assert model.predict(X_test).shape == (len(X_test),)


    def test_classifier_probabilities_are_two_columns_summing_to_one(self, binary_data):
        X_train, X_test, y_train, _ = binary_data

        proba = GradientBoostingClassifier(n_estimators=5).fit(
            X_train, y_train
        ).predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        assert proba.sum(axis=1) == pytest.approx(np.ones(len(X_test)))


    def test_classifier_predict_agrees_with_the_probability_it_reports(self, binary_data):
        X_train, X_test, y_train, _ = binary_data
        model = GradientBoostingClassifier(n_estimators=10).fit(X_train, y_train)

        assert np.array_equal(
            model.predict(X_test), (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        )


    def test_output_margin_returns_log_odds_not_probabilities(self):
        model = GradientBoostingClassifier(
            n_estimators=25, learning_rate=0.5, **UNREGULARIZED
        ).fit(X, LABELS)

        margins = model.predict(X, output_margin=True)

        assert margins.min() < 0.0
        assert margins.max() > 1.0


class TestMulticlass:
    """Three or more labels: one raw score, and so one tree, per class per round."""

    def test_the_label_count_alone_selects_the_softmax_loss(self, multiclass_data):
        X_train, _, y_train, _ = multiclass_data

        model = GradientBoostingClassifier(n_estimators=2).fit(X_train, y_train)

        assert model._loss.n_outputs == 3
        assert all(len(round_trees) == 3 for round_trees in model._trees)


    def test_two_labels_still_boost_a_single_logistic_score(self, binary_data):
        X_train, _, y_train, _ = binary_data

        model = GradientBoostingClassifier(n_estimators=2).fit(X_train, y_train)

        assert model._loss.n_outputs == 1
        assert all(len(round_trees) == 1 for round_trees in model._trees)


    def test_probabilities_are_one_column_per_class_summing_to_one(self, multiclass_data):
        X_train, X_test, y_train, _ = multiclass_data

        proba = GradientBoostingClassifier(n_estimators=5).fit(
            X_train, y_train
        ).predict_proba(X_test)

        assert proba.shape == (len(X_test), 3)
        assert proba.sum(axis=1) == pytest.approx(np.ones(len(X_test)))


    def test_predict_returns_the_most_probable_class(self, multiclass_data):
        X_train, X_test, y_train, _ = multiclass_data
        model = GradientBoostingClassifier(n_estimators=10).fit(X_train, y_train)

        assert np.array_equal(
            model.predict(X_test), np.argmax(model.predict_proba(X_test), axis=1)
        )


    def test_output_margin_returns_one_score_per_class(self, multiclass_data):
        X_train, X_test, y_train, _ = multiclass_data
        model = GradientBoostingClassifier(n_estimators=10).fit(X_train, y_train)

        assert model.predict(X_test, output_margin=True).shape == (len(X_test), 3)


    def test_labels_come_back_as_they_went_in(self, multiclass_data):
        X_train, X_test, y_train, _ = multiclass_data
        labels = np.array(["b", "a", "c"])[y_train.astype(int)]

        predictions = GradientBoostingClassifier(n_estimators=5).fit(
            X_train, labels
        ).predict(X_test)

        assert set(predictions) <= {"a", "b", "c"}


    def test_more_rounds_reduce_the_training_error(self, multiclass_data):
        X_train, _, y_train, _ = multiclass_data

        errors = [
            float(-np.mean(np.log(
                GradientBoostingClassifier(n_estimators=n).fit(X_train, y_train)
                .predict_proba(X_train)[np.arange(len(y_train)), y_train.astype(int)]
            )))
            for n in (1, 5, 25)
        ]

        assert errors[0] > errors[1] > errors[2]


    def test_a_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            GradientBoostingClassifier().fit(X, np.zeros(len(X)))
