import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import PoissonDistribution


@pytest.mark.parametrize(
    "valid_lam",
    [
        (0.0),
        (1.0),
        (25.5),
    ],
    ids=["zero_rate", "unit_rate", "positive_rate"],
)
def test_poisson_distribution_init_valid(valid_lam: float) -> None:
    dist = PoissonDistribution(lam=valid_lam)
    assert dist.lam == valid_lam
    assert dist.distribution == "poisson"


@pytest.mark.parametrize(
    "invalid_lam", [-0.1, -10.0], ids=["small_negative", "large_negative"]
)
def test_poisson_distribution_init_invalid_lam(invalid_lam: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        PoissonDistribution(lam=invalid_lam)


@pytest.mark.parametrize(
    "lam",
    [0.0, 1.0, 50.5],
    ids=["zero_rate", "one_rate", "high_float_rate"],
)
def test_poisson_samples_are_non_negative(lam: float) -> None:
    dist = PoissonDistribution(lam=lam)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample >= 0)

