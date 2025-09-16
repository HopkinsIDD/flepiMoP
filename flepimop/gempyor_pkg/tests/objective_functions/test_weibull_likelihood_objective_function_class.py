import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.objective_functions import WeibullLoglikelihood


@pytest.mark.parametrize(
    "shape", [0.5, 1.0, 2.0], ids=["shape_lt_1", "shape_eq_1", "shape_gt_1"]
)
def test_weibull_loglikelihood_init_valid(shape: float) -> None:
    dist = WeibullLoglikelihood(shape=shape)
    assert dist.shape == shape
    assert dist.distribution == "weibull"


@pytest.mark.parametrize(
    "invalid_shape", [0.0, -1.0, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_weibull_loglikelihood_init_invalid_shape(invalid_shape: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        WeibullLoglikelihood(shape=invalid_shape)


@pytest.mark.parametrize(
    "shape, gt_data, model_data",
    [
        (
            0.5,
            np.array([0.1, 1.0, 5.0, 10.0]),
            np.array([1, 2, 6, 12]),
        ),
        (
            1.0,
            np.array([1, 2, 3]),
            np.array([2, 2, 4]),
        ),
        (
            2.0,
            np.array([5, 10, 15]),
            np.array([6, 12, 18]),
        ),
    ],
    ids=["shape_lt_1", "shape_eq_1", "shape_gt_1"],
)
def test_weibull_loglikelihood_error_metric_calculation(
    shape: float,
    gt_data: np.ndarray,
    model_data: np.ndarray,
) -> None:
    dist = WeibullLoglikelihood(shape=shape)
    result = dist.error_metric_calculation(gt_data, model_data)
    expected_scale = model_data / scipy.special.gamma(1 + 1 / shape)
    expected = scipy.stats.weibull_min.logpdf(x=gt_data, c=shape, scale=expected_scale)

    assert np.allclose(result, expected, atol=1e-5)
