import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import BetaDistribution


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (2.0, 5.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (100.0, 25.0),
    ],
    ids=[
        "standard_case",
        "small_values",
        "uniform_case",
        "large_values",
    ],
)
def test_beta_distribution_init_valid(alpha: float, beta: float) -> None:
    dist = BetaDistribution(alpha=alpha, beta=beta)
    assert dist.alpha == alpha
    assert dist.beta == beta
    assert dist.distribution == "beta"


@pytest.mark.parametrize(
    "invalid_alpha",
    [0.0, -0.1, -10.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_beta_distribution_init_invalid_alpha(invalid_alpha: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        BetaDistribution(alpha=invalid_alpha, beta=1.0)


@pytest.mark.parametrize(
    "invalid_beta",
    [0.0, -1.0, -50.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_beta_distribution_init_invalid_beta(invalid_beta: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        BetaDistribution(alpha=1.0, beta=invalid_beta)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((3, 3), (3, 3)),
        (5, (5,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
def test_beta_distribution_sample_properties(size, expected_shape) -> None:
    dist = BetaDistribution(alpha=2.0, beta=5.0)
    sample = dist.sample(size=size)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample >= 0) and np.all(sample <= 1)
