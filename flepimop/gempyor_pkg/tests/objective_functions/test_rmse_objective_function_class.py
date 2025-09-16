import numpy as np
import pytest

from gempyor.objective_functions import RMSE


def test_rmse_init() -> None:
    objfunc = RMSE()
    assert objfunc.distribution == "rmse"


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
def test_absolute_error_error_metric_calculation_raises_notimplemented_error(
    gt_data: np.ndarray, model_data: np.ndarray
) -> None:
    objfunc = RMSE()
    with pytest.raises(NotImplementedError):
        objfunc.error_metric_calculation(gt_data=gt_data, model_data=model_data)
