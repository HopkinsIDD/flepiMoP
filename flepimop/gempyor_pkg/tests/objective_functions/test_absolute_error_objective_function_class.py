import numpy as np
import pytest

from gempyor.objective_functions import AbsoluteError


def test_absolute_error_init() -> None:
    objfunc = AbsoluteError()
    assert objfunc.distribution == "absolute_error"


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
def test_absolute_error_error_metric_calculation_raises_notimplemented_error(
    gt_data: np.ndarray, model_data: np.ndarray
) -> None:
    objfunc = AbsoluteError()
    with pytest.raises(NotImplementedError):
        objfunc.error_metric_calculation(gt_data=gt_data, model_data=model_data)
