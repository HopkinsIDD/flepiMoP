import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import NormalLoglikelihood


@pytest.mark.parametrize(
    "sigma", [0.1, 1.0, 50.5], ids=["small_sigma", "unit_sigma", "large_sigma"]
)
def test_normal_loglikelihood_init_valid(sigma: float) -> None:
    dist = NormalLoglikelihood(sigma=sigma)
    assert dist.sigma == sigma
    assert dist.distribution == "norm"


@pytest.mark.parametrize(
    "invalid_sigma",
    [0.0, -0.5, -50.5],
    ids=["zero_sigma", "small_neg_sigma", "large_neg_sigma"],
)
def test_normal_loglikelihood_init_invalid_sigma(invalid_sigma: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        NormalLoglikelihood(sigma=invalid_sigma)


@pytest.mark.parametrize(
    "sigma", [0.5, 1.0, 10.0], ids=["sigma_0.5", "sigma_1.0", "sigma_10.0"]
)
def test_normal_loglikelihood_calculation(sigma: float) -> None:
    dist = NormalLoglikelihood(sigma=sigma)
    gt_data = np.array([10, 15, 20, 25])
    model_data = np.array([11, 14, 22, 25])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = scipy.stats.norm.logpdf(x=gt_data, loc=model_data, scale=sigma)
    assert np.allclose(result, expected)
