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
    "alpha, beta",
    [
        (2.0, 5.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ],
    ids=["alpha_gt_1", "uniform_case", "u_shaped_case"],
)
def test_beta_distribution_sample_range(alpha: float, beta: float) -> None:
    dist = BetaDistribution(alpha=alpha, beta=beta)
    sample = dist.sample(size=(10, 10))
    assert np.all((sample >= 0.0) & (sample <= 1.0))
