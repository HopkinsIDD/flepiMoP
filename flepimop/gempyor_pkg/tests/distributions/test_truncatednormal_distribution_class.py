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
    "size, expected_shape",
    [
        ((3, 5), (3, 5)),
        (10, (10,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_truncated_normal_distribution_sample_properties(
    size, expected_shape, use_rng
) -> None:
    a, b = 0.0, 10.0
    dist = TruncatedNormalDistribution(mean=5.0, sd=2.0, a=a, b=b)
    kwargs = {"size": size}
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample >= a)
    assert np.all(sample <= b)
