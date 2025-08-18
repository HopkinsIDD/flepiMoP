import numpy as np
import pytest

from gempyor.likelihoods import FixedLoglikelihood


@pytest.mark.parametrize(
    "value", [123.45, -100.0, 0.0], ids=["positive", "negative", "zero"]
)
def test_fixed_loglikelihood_init_valid(value: float) -> None:
    dist = FixedLoglikelihood(value=value)
    assert dist.value == value
    assert dist.distribution == "fixed"


@pytest.mark.parametrize(
    "value, gt_data, expected",
    [
        (
            10.0,
            np.array([10.0, 5.0, 10.0, 20.0]),
            np.array([0.0, -np.inf, 0.0, -np.inf]),
        ),
        (
            -5.0,
            np.array([-5.0, -5.0, -5.0]),
            np.array([0.0, 0.0, 0.0]),
        ),
        (
            0.0,
            np.array([1.0, 2.0, 3.0]),
            np.array([-np.inf, -np.inf, -np.inf]),
        ),
    ],
    ids=["mixed_match", "all_match", "none_match"],
)
def test_fixed_loglikelihood_calculation(
    value: float, gt_data: np.ndarray, expected: np.ndarray
) -> None:
    dist = FixedLoglikelihood(value=value)
    model_data = np.random.rand(*gt_data.shape) * 100
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    assert np.array_equal(result, expected)
