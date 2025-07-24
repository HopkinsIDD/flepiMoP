import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import PoissonDistribution


@pytest.mark.parametrize(
    "valid_lam",
    [
        (1.0),
        (25.5),
    ],
    ids=["unit_rate", "positive_rate"],
)
def test_poisson_distribution_init_valid(valid_lam: float) -> None:
    dist = PoissonDistribution(lam=valid_lam)
    assert dist.lam == valid_lam
    assert dist.distribution == "poisson"
    assert dist.allow_edge_cases is False


def test_poisson_distribution_init_fails_on_edge_case_false_by_default() -> None:
    with pytest.raises(
        ValidationError,
        match="Input for `lam` cannot be zero when `allow_edge_cases` is `False`.",
    ):
        PoissonDistribution(lam=0.0)


def test_poisson_distribution_init_succeeds_on_edge_case_true() -> None:
    dist = PoissonDistribution(lam=0.0, allow_edge_cases=True)
    assert dist.lam == 0.0
    assert dist.allow_edge_cases is True


@pytest.mark.parametrize(
    "invalid_lam",
    [
        -5.0,
        -50.5,
    ],
    ids=["small_negative_lam", "large_negative_lam"],
)
def test_poisson_distribution_init_fails_on_negative_lam(invalid_lam: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        PoissonDistribution(lam=invalid_lam)


@pytest.mark.parametrize(
    "lam, allow_edges",
    [
        (1.0, False),
        (50.5, False),
        (0.0, True),
    ],
    ids=["unit_rate", "high_float_rate", "zero_rate_edge_case"],
)
def test_poisson_samples_are_non_negative(lam: float, allow_edges: bool) -> None:
    dist = PoissonDistribution(lam=lam, allow_edge_cases=allow_edges)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample >= 0)
