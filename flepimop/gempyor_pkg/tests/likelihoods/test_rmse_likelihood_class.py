import numpy as np
import pytest

from gempyor.likelihoods import RMSELoglikelihood


def test_rmse_loglikelihood_init() -> None:
    dist = RMSELoglikelihood()
    assert dist.distribution == "rmse"


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.parametrize(
    "gt_data, model_data",
    [
        (np.array([1, 2, 6]), np.array([3, 4, 4])),
        (np.array([1, 5, np.nan]), np.array([3, 3, 10])),
        (np.array([5, 10]), np.array([5, 10])),
    ],
    ids=["basic_case", "with_nan", "zero_error"],
)
def test_rmse_loglikelihood_calculation(
    gt_data: np.ndarray, model_data: np.ndarray
) -> None:
    dist = RMSELoglikelihood()
    result = dist.loglikelihood(gt_data, model_data)
    squared_error = (gt_data - model_data) ** 2
    mean_squared_error = np.nanmean(squared_error)
    rmse = np.sqrt(mean_squared_error)
    expected_value = -np.log(rmse)
    expected_array = np.full(gt_data.shape, expected_value)
    assert np.allclose(result, expected_array)
