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
    assert dist.allow_edge_cases is False


@pytest.mark.parametrize(
    "invalid_sd", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_truncated_normal_distribution_init_invalid_sd(invalid_sd: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        TruncatedNormalDistribution(mean=0.0, sd=invalid_sd, a=0.0, b=1.0)


@pytest.mark.parametrize(
    "a, b",
    [
        (5.0, 5.0),
        (10.0, 5.0),
    ],
    ids=["b_equals_a", "b_less_than_a"],
)
def test_truncated_normal_init_fails_when_b_not_gt_a(a: float, b: float) -> None:
    with pytest.raises(ValidationError, match="must be > to lower bound `a`"):
        TruncatedNormalDistribution(mean=5.0, sd=2.0, a=a, b=b, allow_edge_cases=False)


def test_truncated_normal_init_succeeds_on_edge_case_true() -> None:
    dist = TruncatedNormalDistribution(
        mean=5.0, sd=2.0, a=5.0, b=5.0, allow_edge_cases=True
    )
    assert dist.a == 5.0
    assert dist.b == 5.0
    assert dist.allow_edge_cases is True


def test_truncated_normal_init_fails_when_b_lt_a_with_edge_cases() -> None:
    with pytest.raises(ValidationError, match="must be >= to lower bound `a`"):
        TruncatedNormalDistribution(mean=5.0, sd=2.0, a=10.0, b=5.0, allow_edge_cases=True)


@pytest.mark.parametrize("mean", [0.0, 100.0], ids=["mean_zero", "mean_high"])
@pytest.mark.parametrize("sd", [1.0, 10.0], ids=["sd_one", "sd_high"])
@pytest.mark.parametrize(
    ("a", "b"), [(-2.0, 2.0), (90.0, 110.0)], ids=["bounds_centered", "bounds_high"]
)
def test_truncated_normal_distribution_sample_range(
    mean: float, sd: float, a: float, b: float
) -> None:
    dist = TruncatedNormalDistribution(mean=mean, sd=sd, a=a, b=b)
    sample = dist.sample(size=(10, 10))
    assert np.all((sample >= a) & (sample <= b))


def test_truncated_normal_distribution_sample_range_edge_case() -> None:
    dist = TruncatedNormalDistribution(
        mean=5.0, sd=2.0, a=7.0, b=7.0, allow_edge_cases=True
    )
    sample = dist.sample(size=(10, 10))
    assert np.all(sample == 7.0)
