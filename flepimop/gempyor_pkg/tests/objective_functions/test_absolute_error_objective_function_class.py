import numpy as np
import pytest

from gempyor.objective_functions import AbsoluteError


def test_absolute_error_init() -> None:
    dist = AbsoluteError()
    assert dist.distribution == "absolute_error"


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.parametrize(
    "gt_data, model_data",
    [
        (np.array([1, 2, 3, 4]), np.array([2, 2, 5, 3])),
        (np.array([1.5, 2.5, np.nan]), np.array([0.5, 4.0, 10.0])),
        (np.array([-10, 0, 10]), np.array([-5, 5, 5])),
        (np.array([5, 10]), np.array([5, 10])),
    ],
    ids=["integers", "floats_and_nan", "mixed_sign", "zero_error"],
)
def test_absolute_error_error_metric_calculation(
    gt_data: np.ndarray, model_data: np.ndarray
) -> None:
    dist = AbsoluteError()
    result = dist.error_metric_calculation(gt_data, model_data)
    absolute_error = np.abs(gt_data - model_data)
    total_absolute_error = np.nansum(absolute_error)
    expected_value = -np.log(total_absolute_error)
    expected_array = np.full(gt_data.shape, expected_value)
    assert np.allclose(result, expected_array)
