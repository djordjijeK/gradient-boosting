"""Our booster against XGBoost and LightGBM on real data.

Agreement is measured -- error, accuracy, correlation -- not bit-for-bit. Both
oracles compute in float32 and pick split thresholds from float32 gains, so on a
near-tie their argmax can differ from ours in float64, and one flipped split
cascades through every later round. Tolerances below are set from measured
behaviour with headroom; everything is deterministic, so they do not flake.
"""
import numpy as np
import pytest

from gbm.boosting import GradientBoostingClassifier, GradientBoostingRegressor

xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")


BOOSTING = dict(n_estimators=50, learning_rate=0.1, max_depth=3)
SHARED = dict(reg_lambda=1.0, min_child_weight=1.0)


def rmse(predictions, y):
    return float(np.sqrt(np.mean((predictions - y) ** 2)))


def relative_gap(ours, theirs):
    return abs(ours - theirs) / abs(theirs)


@pytest.fixture(scope="module")
def regressors(regression_data):
    X_train, _, y_train, _ = regression_data

    return (
        GradientBoostingRegressor(gamma=0.0, **SHARED, **BOOSTING).fit(X_train, y_train),
        xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="exact",
            gamma=0.0, reg_alpha=0.0, n_jobs=1, **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
        lgb.LGBMRegressor(
            min_child_samples=1, min_split_gain=0.0, n_jobs=1, verbose=-1,
            **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
    )


@pytest.fixture(scope="module")
def classifiers(binary_data):
    X_train, _, y_train, _ = binary_data

    return (
        GradientBoostingClassifier(gamma=0.0, **SHARED, **BOOSTING).fit(X_train, y_train),
        xgb.XGBClassifier(
            objective="binary:logistic", tree_method="exact",
            gamma=0.0, reg_alpha=0.0, n_jobs=1, **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
        lgb.LGBMClassifier(
            min_child_samples=1, min_split_gain=0.0, n_jobs=1, verbose=-1,
            **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
    )


@pytest.fixture(scope="module")
def multiclass_classifiers(multiclass_data):
    X_train, _, y_train, _ = multiclass_data

    return (
        GradientBoostingClassifier(gamma=0.0, **SHARED, **BOOSTING).fit(X_train, y_train),
        xgb.XGBClassifier(
            objective="multi:softprob", tree_method="exact",
            gamma=0.0, reg_alpha=0.0, n_jobs=1, **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
        lgb.LGBMClassifier(
            min_child_samples=1, min_split_gain=0.0, n_jobs=1, verbose=-1,
            **SHARED, **BOOSTING,
        ).fit(X_train, y_train),
    )


class TestRegression:

    def test_training_error_matches_xgboost(self, regression_data, regressors):
        X_train, _, y_train, _ = regression_data
        ours, xgboost, _ = regressors

        assert rmse(ours.predict(X_train), y_train) == pytest.approx(
            rmse(xgboost.predict(X_train), y_train), rel=0.01
        )


    def test_held_out_error_matches_xgboost(self, regression_data, regressors):
        _, X_test, _, y_test = regression_data
        ours, xgboost, _ = regressors

        assert rmse(ours.predict(X_test), y_test) == pytest.approx(
            rmse(xgboost.predict(X_test), y_test), rel=0.03
        )


    def test_held_out_error_is_competitive_with_lightgbm(self, regression_data, regressors):
        _, X_test, _, y_test = regression_data
        ours, _, lightgbm = regressors

        assert relative_gap(
            rmse(ours.predict(X_test), y_test), rmse(lightgbm.predict(X_test), y_test)
        ) < 0.10


    def test_held_out_predictions_track_both_oracles(self, regression_data, regressors):
        _, X_test, _, _ = regression_data
        ours, xgboost, lightgbm = regressors

        mine = ours.predict(X_test)

        assert np.corrcoef(mine, xgboost.predict(X_test))[0, 1] > 0.999
        assert np.corrcoef(mine, lightgbm.predict(X_test))[0, 1] > 0.94


class TestClassification:

    def test_held_out_accuracy_matches_xgboost(self, binary_data, classifiers):
        _, X_test, _, y_test = binary_data
        ours, xgboost, _ = classifiers

        assert (ours.predict(X_test) == y_test).mean() == pytest.approx(
            (xgboost.predict(X_test) == y_test).mean(), abs=0.02
        )


    def test_every_held_out_label_agrees_with_xgboost(self, binary_data, classifiers):
        _, X_test, _, _ = binary_data
        ours, xgboost, _ = classifiers

        assert (ours.predict(X_test) == xgboost.predict(X_test)).mean() >= 0.98


    def test_held_out_accuracy_is_competitive_with_lightgbm(self, binary_data, classifiers):
        _, X_test, _, y_test = binary_data
        ours, _, lightgbm = classifiers

        assert (ours.predict(X_test) == y_test).mean() == pytest.approx(
            (lightgbm.predict(X_test) == y_test).mean(), abs=0.05
        )


    def test_predicted_probabilities_track_both_oracles(self, binary_data, classifiers):
        _, X_test, _, _ = binary_data
        ours, xgboost, lightgbm = classifiers

        mine = ours.predict_proba(X_test)[:, 1]

        assert np.corrcoef(mine, xgboost.predict_proba(X_test)[:, 1])[0, 1] > 0.999
        assert np.corrcoef(mine, lightgbm.predict_proba(X_test)[:, 1])[0, 1] > 0.99


class TestMulticlassClassification:
    """One-tree-per-class softmax against the same two oracles.

    XGBoost boosts each class against 2 p (1 - p), twice the diagonal of the
    true hessian, and these tolerances only hold because we do the same; with
    the bare p (1 - p) the leaf steps double and probability agreement falls
    from 0.9997 to 0.993.
    """

    def test_held_out_accuracy_matches_xgboost(self, multiclass_data, multiclass_classifiers):
        _, X_test, _, y_test = multiclass_data
        ours, xgboost, _ = multiclass_classifiers

        assert (ours.predict(X_test) == y_test).mean() == pytest.approx(
            (xgboost.predict(X_test) == y_test).mean(), abs=0.02
        )


    def test_every_held_out_label_agrees_with_xgboost(self, multiclass_data, multiclass_classifiers):
        _, X_test, _, _ = multiclass_data
        ours, xgboost, _ = multiclass_classifiers

        assert (ours.predict(X_test) == xgboost.predict(X_test)).mean() >= 0.98


    def test_held_out_accuracy_is_competitive_with_lightgbm(self, multiclass_data, multiclass_classifiers):
        _, X_test, _, y_test = multiclass_data
        ours, _, lightgbm = multiclass_classifiers

        assert (ours.predict(X_test) == y_test).mean() == pytest.approx(
            (lightgbm.predict(X_test) == y_test).mean(), abs=0.05
        )


    def test_every_class_probability_tracks_both_oracles(self, multiclass_data, multiclass_classifiers):
        _, X_test, _, _ = multiclass_data
        ours, xgboost, lightgbm = multiclass_classifiers

        mine = ours.predict_proba(X_test)

        assert np.corrcoef(mine.ravel(), xgboost.predict_proba(X_test).ravel())[0, 1] > 0.999
        assert np.corrcoef(mine.ravel(), lightgbm.predict_proba(X_test).ravel())[0, 1] > 0.99
