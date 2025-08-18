import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import WeibullLoglikelihood


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
    "shape", [0.5, 1.0, 2.0], ids=["shape_lt_1", "shape_eq_1", "shape_gt_1"]
)
def test_weibull_loglikelihood_calculation(shape: float) -> None:
    dist = WeibullLoglikelihood(shape=shape)
    gt_data = np.array([0.1, 1.0, 5.0, 10.0])
    model_data = np.array([1, 2, 6, 12])
    result = dist.loglikelihood(gt_data, model_data)
    expected = scipy.stats.weibull_min.logpdf(x=gt_data, c=shape, scale=model_data)
    assert np.allclose(result, expected)
