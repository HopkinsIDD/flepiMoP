import numpy as np
import pytest

from gempyor.objective_functions import FixedLoglikelihood


def test_fixed_loglikelihood_init_valid() -> None:
    dist = FixedLoglikelihood()
    assert dist.distribution == "fixed"


@pytest.mark.parametrize(
    "gt_data, model_data, expected",
    [
        (
            np.array([10.0, 5.0, 10.0, 20.0]),
            np.array([10.0, 99.0, 10.0, 98.0]),
            np.array([0.0, -np.inf, 0.0, -np.inf]),
        ),
        (
            np.array([-5.0, -5.0, -5.0]),
            np.array([-5.0, -5.0, -5.0]),
            np.array([0.0, 0.0, 0.0]),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([-np.inf, -np.inf, -np.inf]),
        ),
    ],
    ids=["mixed_match", "all_match", "none_match"],
)
def test_fixed_loglikelihood_error_metric_calculation(
    gt_data: np.ndarray, model_data: np.ndarray, expected: np.ndarray
) -> None:
    dist = FixedLoglikelihood()
    result = dist.error_metric_calculation(gt_data=gt_data, model_data=model_data)
    assert np.array_equal(result, expected)
