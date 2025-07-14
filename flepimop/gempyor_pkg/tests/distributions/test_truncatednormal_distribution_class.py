import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import TruncatedNormalDistribution


@pytest.mark.parametrize(
    "mean, sd, a, b",
    [
        (0.0, 1.0, -2.0, 2.0),
        (10.0, 2.0, 5.0, 15.0),
        (-5.0, 0.5, -6.0, -4.0),
        (0.0, 10.0, -1.0, 1.0),
    ],
    ids=["standard", "shifted_positive", "shifted_negative", "wide_sd"],
)
def test_truncated_normal_distribution_init_valid(
    mean: float, sd: float, a: float, b: float
) -> None:
    dist = TruncatedNormalDistribution(mean=mean, sd=sd, a=a, b=b)
    assert dist.mean == mean
    assert dist.sd == sd
    assert dist.a == a
    assert dist.b == b
    assert dist.distribution == "truncnorm"


@pytest.mark.parametrize(
    "invalid_sd", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_truncated_normal_distribution_init_invalid_sd(invalid_sd: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        TruncatedNormalDistribution(mean=0.0, sd=invalid_sd, a=0.0, b=1.0)


@pytest.mark.parametrize(
    "mean, sd, a, b",
    [
        (5.0, 2.0, 0.0, 10.0),
        (0.0, 1.0, -1.0, 1.0),
        (100.0, 10.0, 90.0, 110.0),
    ],
    ids=["mean_in_range", "std_normal_truncated", "high_mean"],
)
def test_truncated_normal_distribution_sample_range(
    mean: float, sd: float, a: float, b: float
) -> None:
    dist = TruncatedNormalDistribution(mean=mean, sd=sd, a=a, b=b)
    sample = dist.sample(size=(10, 10))
    assert np.all((sample >= a) & (sample <= b))
