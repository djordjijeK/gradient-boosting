"""The two losses, checked against the derivations they came from.

Everything here is arithmetic that can be verified by hand. Whether the losses
drive a correct booster is settled by test_oracle_agreement.py.
"""
import numpy as np
import pytest

from gbm.losses import LogisticLoss, SquaredError


Y = np.array([1.0, 3.0, 8.0])
RAW = np.array([[2.0], [5.0], [6.0]])

LABELS = np.array([0.0, 1.0, 1.0])
SCORES = np.array([[0.0], [1.0], [-1.0]])

SIGMOID_OF_ONE = 0.7310585786300049
SIGMOID_OF_MINUS_ONE = 0.2689414213699951


class TestSquaredError:

    def test_base_score_is_the_mean_of_y(self):
        assert SquaredError().base_score(Y) == pytest.approx([4.0])


    def test_gradient_is_the_residual_and_hessian_is_one(self):
        gradient, hessian = SquaredError().gradients_and_hessians(Y, RAW)

        assert gradient.ravel() == pytest.approx([1.0, 2.0, -2.0])
        assert hessian.ravel() == pytest.approx([1.0, 1.0, 1.0])


    def test_output_returns_the_raw_score_unchanged(self):
        assert SquaredError().output(RAW) == pytest.approx([2.0, 5.0, 6.0])


class TestLogisticLoss:

    def test_base_score_is_the_log_odds_of_the_base_rate(self):
        assert LogisticLoss().base_score(LABELS) == pytest.approx([np.log(2.0)])


    def test_base_score_with_a_pure_label_stays_finite(self):
        assert np.isfinite(LogisticLoss().base_score(np.ones(4))).all()


    def test_gradient_is_p_minus_y_and_hessian_is_p_times_one_minus_p(self):
        gradient, hessian = LogisticLoss().gradients_and_hessians(LABELS, SCORES)

        assert gradient.ravel() == pytest.approx(
            [0.5, SIGMOID_OF_ONE - 1.0, SIGMOID_OF_MINUS_ONE - 1.0]
        )
        assert hessian.ravel() == pytest.approx([
            0.25,
            SIGMOID_OF_ONE * (1.0 - SIGMOID_OF_ONE),
            SIGMOID_OF_MINUS_ONE * (1.0 - SIGMOID_OF_MINUS_ONE),
        ])


    def test_output_maps_raw_scores_to_probabilities(self):
        assert LogisticLoss().output(SCORES) == pytest.approx(
            [0.5, SIGMOID_OF_ONE, SIGMOID_OF_MINUS_ONE]
        )


@pytest.mark.parametrize("loss", [SquaredError(), LogisticLoss()], ids=["squared", "logistic"])
class TestSharedContract:

    def test_base_score_has_one_entry_per_output(self, loss):
        assert loss.base_score(LABELS).shape == (loss.n_outputs,)


    def test_gradients_and_hessians_keep_the_shape_of_raw(self, loss):
        gradient, hessian = loss.gradients_and_hessians(LABELS, SCORES)

        assert gradient.shape == SCORES.shape
        assert hessian.shape == SCORES.shape


    def test_one_dimensional_raw_is_rejected(self, loss):
        with pytest.raises(ValueError, match="must have shape"):
            loss.gradients_and_hessians(LABELS, SCORES.ravel())
