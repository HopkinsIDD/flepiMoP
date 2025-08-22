import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import TruncatedNormalLoglikelihood


@pytest.mark.parametrize(
    "sd, a, b",
    [
        (1.0, -2.0, 2.0),
        (10.0, 0.0, 20.0),
        (0.5, -5.0, -4.0),
    ],
    ids=["standard", "wide_range", "shifted_negative"],
)
def test_truncatednormal_loglikelihood_init_valid(sd: float, a: float, b: float) -> None:
    dist = TruncatedNormalLoglikelihood(sd=sd, a=a, b=b)
    assert dist.sd == sd
    assert dist.a == a
    assert dist.b == b
    assert dist.distribution == "truncnorm"


@pytest.mark.parametrize(
    "invalid_sd", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_truncatednormal_loglikelihood_init_invalid_sd(invalid_sd: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        TruncatedNormalLoglikelihood(sd=invalid_sd, a=0.0, b=1.0)


@pytest.mark.parametrize(
    "sd, a, b",
    [
        (2.0, 0.0, 10.0),
        (5.0, -20.0, 0.0),
    ],
    ids=["positive_bounds", "negative_bounds"],
)
def test_truncatednormal_loglikelihood_calculation(sd: float, a: float, b: float) -> None:
    dist = TruncatedNormalLoglikelihood(sd=sd, a=a, b=b)
    gt_data = np.array([1, 3, 5, 7, 9])
    model_data = np.array([2, 2, 6, 6, 8])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    a_prime = (a - model_data) / sd
    b_prime = (b - model_data) / sd
    expected = scipy.stats.truncnorm.logpdf(
        x=gt_data, a=a_prime, b=b_prime, loc=model_data, scale=sd
    )
    assert np.allclose(result, expected)
