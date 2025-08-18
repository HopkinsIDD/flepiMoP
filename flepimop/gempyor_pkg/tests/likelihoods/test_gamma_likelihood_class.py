import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import GammaLoglikelihood


@pytest.mark.parametrize(
    "shape", [0.1, 1.0, 9.0], ids=["small_shape", "unit_shape", "large_shape"]
)
def test_gamma_loglikelihood_init_valid(shape: float) -> None:
    dist = GammaLoglikelihood(shape=shape)
    assert dist.shape == shape
    assert dist.distribution == "gamma"


@pytest.mark.parametrize(
    "invalid_shape", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_gamma_loglikelihood_init_invalid_shape(invalid_shape: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        GammaLoglikelihood(shape=invalid_shape)


@pytest.mark.parametrize(
    "shape", [0.5, 2.0, 5.0], ids=["shape_0.5", "shape_2.0", "shape_5.0"]
)
def test_gamma_loglikelihood_calculation(shape: float) -> None:
    dist = GammaLoglikelihood(shape=shape)
    gt_data = np.array([1, 5, 10, 20])
    model_data = np.array([2, 6, 9, 18])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = scipy.stats.gamma.logpdf(x=gt_data, a=shape, scale=model_data)
    assert np.allclose(result, expected)
