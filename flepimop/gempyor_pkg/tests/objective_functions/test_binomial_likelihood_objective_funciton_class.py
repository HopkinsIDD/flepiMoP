import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.objective_functions import BinomialLoglikelihood


@pytest.mark.parametrize("n", [0, 1, 50], ids=["zero_n", "one_n", "large_n"])
def test_binomial_loglikelihood_init_valid(n: int) -> None:
    dist = BinomialLoglikelihood(n=n)
    assert dist.n == n
    assert dist.distribution == "binomial"


@pytest.mark.parametrize("invalid_n", [-1, -10], ids=["-1", "-10"])
def test_binomial_loglikelihood_init_invalid_n(invalid_n: int) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialLoglikelihood(n=invalid_n)


@pytest.mark.parametrize(
    "gt_data, model_data",
    [
        (np.array([0, 5, 10, 15, 20]), np.array([0, 0.25, 0.5, 0.75, 1.0])),
        (np.array([1]), np.array([0.5])),
        (np.array([]), np.array([])),
    ],
    ids=["original_case", "single_value", "empty_arrays"],
)
def test_binomial_loglikelihood_error_metric_calculation_valid(gt_data, model_data) -> None:
    n_trials = 20
    dist = BinomialLoglikelihood(n=n_trials)

    result = dist.error_metric_calculation(gt_data=gt_data, model_data=model_data)
    expected = scipy.stats.binom.logpmf(k=gt_data, n=n_trials, p=model_data)

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "invalid_model_data",
    [
        (np.array([-0.5, 0.25, 0.5, 0.75, 1.5])),
        (np.array([1.1, 0.9])),
        (np.array([-0.1])),
    ],
    ids=["p_below_and_above_range", "p_above_1", "p_below_0"],
)
def test_binomial_loglikelihood_error_metric_calculation_invalid_p_raises_error(
    invalid_model_data,
) -> None:
    n_trials = 20
    dist = BinomialLoglikelihood(n=n_trials)
    gt_data = np.zeros_like(invalid_model_data)

    with pytest.raises(
        ValueError, match="probabilities in `model_data` must be in the range"
    ):
        dist.error_metric_calculation(gt_data=gt_data, model_data=invalid_model_data)
