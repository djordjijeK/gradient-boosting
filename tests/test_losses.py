import pytest
import numpy as np

from gbm.losses import LogisticLoss, SoftmaxLoss, SquaredError


Y = np.array([1.0, 3.0, 8.0])
RAW = np.array([[2.0], [5.0], [6.0]])

LABELS = np.array([0.0, 1.0, 1.0])
SCORES = np.array([[0.0], [1.0], [-1.0]])

CLASS_LABELS = np.array([0.0, 1.0, 2.0])
CLASS_SCORES = np.array([[0.0, 0.0, 0.0], [0.0, np.log(2.0), np.log(3.0)], [0.0, 0.0, 0.0]])

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


    def test_output_maps_raw_scores_to_the_probability_of_both_classes(self):
        assert LogisticLoss().output(SCORES) == pytest.approx(
            np.column_stack([
                [0.5, 1.0 - SIGMOID_OF_ONE, 1.0 - SIGMOID_OF_MINUS_ONE],
                [0.5, SIGMOID_OF_ONE, SIGMOID_OF_MINUS_ONE],
            ])
        )


class TestSoftmaxLoss:

    def test_base_score_is_the_log_prior_of_each_class(self):
        # Priors are 1/2, 1/4, 1/4.
        base = SoftmaxLoss(3).base_score(np.array([0.0, 0.0, 1.0, 2.0]))

        assert base == pytest.approx(np.log([0.5, 0.25, 0.25]))


    def test_base_score_generalises_the_log_odds_of_the_binary_case(self):
        base = SoftmaxLoss(2).base_score(LABELS)

        assert base[1] - base[0] == pytest.approx(np.log(2.0))


    def test_output_is_the_softmax_of_each_row(self):
        probabilities = SoftmaxLoss(3).output(CLASS_SCORES)

        assert probabilities[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        assert probabilities[1] == pytest.approx([1 / 6, 2 / 6, 3 / 6])


    def test_output_ignores_a_constant_shift_along_a_row(self):
        assert SoftmaxLoss(3).output(CLASS_SCORES + 1000.0) == pytest.approx(
            SoftmaxLoss(3).output(CLASS_SCORES)
        )


    def test_two_class_softmax_agrees_with_the_logistic_it_generalises(self):
        raw = np.column_stack([np.zeros(len(SCORES)), SCORES[:, 0]])

        assert SoftmaxLoss(2).output(raw) == pytest.approx(LogisticLoss().output(SCORES))


    def test_gradient_is_p_minus_the_one_hot_target(self):
        gradient, _ = SoftmaxLoss(3).gradients_and_hessians(CLASS_LABELS, CLASS_SCORES)

        assert gradient[0] == pytest.approx([1 / 3 - 1.0, 1 / 3, 1 / 3])
        assert gradient[1] == pytest.approx([1 / 6, 2 / 6 - 1.0, 3 / 6])


    def test_hessian_is_twice_the_diagonal_of_the_true_second_derivative(self):
        _, hessian = SoftmaxLoss(3).gradients_and_hessians(CLASS_LABELS, CLASS_SCORES)

        assert hessian[0] == pytest.approx([4 / 9, 4 / 9, 4 / 9])
        assert hessian[1] == pytest.approx(2 * np.array([1 / 6, 2 / 6, 3 / 6]) * np.array([5 / 6, 4 / 6, 3 / 6]))


    def test_gradients_sum_to_zero_across_the_classes_of_a_row(self):
        gradient, _ = SoftmaxLoss(3).gradients_and_hessians(CLASS_LABELS, CLASS_SCORES)

        assert gradient.sum(axis=1) == pytest.approx(np.zeros(len(CLASS_LABELS)))


    @pytest.mark.parametrize("label", [-1.0, 3.0, 0.5])
    def test_a_target_that_is_not_a_class_index_is_rejected(self, label):
        with pytest.raises(ValueError, match="class indices"):
            SoftmaxLoss(3).base_score(np.array([0.0, 1.0, label]))


    def test_fewer_than_two_classes_is_rejected(self):
        with pytest.raises(ValueError, match="at least two classes"):
            SoftmaxLoss(1)
