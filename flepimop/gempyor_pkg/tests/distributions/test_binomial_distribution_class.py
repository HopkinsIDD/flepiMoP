import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import BinomialDistribution


@pytest.mark.parametrize(
    "n, p",
    [
        (10, 0.5),
        (50, 0.25),
        (1, 0.99),
    ],
    ids=[
        "standard_case_1",
        "standard_case_2",
        "min_valid_n_for_non_edge_case",
    ],
)
def test_binomial_distribution_init_valid(n: int, p: float) -> None:
    dist = BinomialDistribution(n=n, p=p)
    assert dist.n == n
    assert dist.p == p
    assert dist.distribution == "binomial"
    assert dist.allow_edge_cases is False


@pytest.mark.parametrize(
    "n, p",
    [(0, 0.5), (10, 0.0), (10, 1.0)],
    ids=["n_is_zero", "p_is_zero", "p_is_one"],
)
def test_binomial_distribution_init_valid_edge_cases(n: int, p: float) -> None:
    dist = BinomialDistribution(n=n, p=p, allow_edge_cases=True)
    assert dist.n == n
    assert dist.p == p
    assert dist.allow_edge_cases is True


def test_binomial_distribution_init_invalid_n_default() -> None:
    with pytest.raises(
        ValidationError,
        match="Input for `n` cannot be zero when `allow_edge_cases` is `False`.",
    ):
        BinomialDistribution(n=0, p=0.5)


@pytest.mark.parametrize(
    "invalid_n",
    [-1, -10],
    ids=["-1", "-10"],
)
def test_binomial_distribution_init_invalid_n_negative(invalid_n: int) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialDistribution(n=invalid_n, p=0.5)


@pytest.mark.parametrize(
    "invalid_p",
    [0.0, 1.0],
    ids=["zero", "one"],
)
def test_binomial_distribution_init_invalid_p_default(invalid_p: float) -> None:
    with pytest.raises(
        ValidationError,
        match="Input for `p` cannot be 0 or 1 when `allow_edge_cases` is `False`.",
    ):
        BinomialDistribution(n=10, p=invalid_p)


@pytest.mark.parametrize(
    "invalid_p",
    [-0.1, -10.0],
    ids=["small_negative", "large_negative"],
)
def test_binomial_distribution_init_invalid_p_below_range(invalid_p: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialDistribution(n=10, p=invalid_p)


@pytest.mark.parametrize(
    "invalid_p",
    [1.1, 25.0],
    ids=["small_gt_one", "large_gt_one"],
)
def test_binomial_distribution_init_invalid_p_above_range(invalid_p: float) -> None:
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        BinomialDistribution(n=10, p=invalid_p)


@pytest.mark.parametrize(
    "n, p",
    [
        (10, 0.5),
        (100, 0.1),
        (20, 0.99),
    ],
    ids=["n_10_p_0.5", "n_100_p_0.1", "n_20_p_0.99"],
)
def test_binomial_sample_properties(n: int, p: float) -> None:
    dist = BinomialDistribution(n=n, p=p)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample >= 0)
    assert np.all(sample <= n)


@pytest.mark.parametrize(
    "n, p, expected_val",
    [
        (10, 1.0, 10),
        (10, 0.0, 0),
        (0, 0.5, 0),
    ],
    ids=["p_is_one", "p_is_zero", "n_is_zero"],
)
def test_binomial_sample_edge_cases(n: int, p: float, expected_val: int) -> None:
    dist = BinomialDistribution(n=n, p=p, allow_edge_cases=True)
    sample = dist.sample(size=10)
    assert np.all(sample == expected_val)
