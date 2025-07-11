import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import BinomialDistribution


@pytest.mark.parametrize(
    "n, p",
    [
        (10, 0.5),
        (0, 0.5),
        (100, 0.0),
        (100, 1.0),
        (50, 0.25),
    ],
    ids=[
        "standard_case",
        "zero_trials",
        "zero_probability",
        "full_probability",
        "standard_case_2",
    ],
)
def test_binomial_distribution_init_valid(n: int, p: float) -> None:
    dist = BinomialDistribution(n=n, p=p)
    assert dist.n == n
    assert dist.p == p
    assert dist.distribution == "binomial"


@pytest.mark.parametrize(
    "invalid_n",
    [-1, -10],
    ids=["-1", "-10"],
)
def test_binomial_distribution_init_invalid_n(invalid_n: int) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialDistribution(n=invalid_n, p=0.5)


@pytest.mark.parametrize(
    "invalid_p",
    [-0.1, -10.0],
    ids=["small_negative", "large_negative"],
)
def test_binomial_distribution_init_invalid_p_below_zero(invalid_p: float) -> None:
    """Test that initialization fails for probability 'p' less than 0."""
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialDistribution(n=10, p=invalid_p)


@pytest.mark.parametrize(
    "invalid_p",
    [1.1, 25.0],
    ids=["small_positive", "large_positive"],
)
def test_binomial_distribution_init_invalid_p_above_one(invalid_p: float) -> None:
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        BinomialDistribution(n=10, p=invalid_p)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((4, 4), (4, 4)),
        (30, (30,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
def test_binomial_distribution_sample_properties(size, expected_shape) -> None:
    n = 20
    dist = BinomialDistribution(n=n, p=0.5)
    sample = dist.sample(size=size)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.int64
    assert np.all(sample >= 0)
    assert np.all(sample <= n)
