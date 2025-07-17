import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import NormalDistribution


@pytest.mark.parametrize("mu", [-100.0, 0.0, 50.5], ids=["mu_neg", "mu_zero", "mu_pos"])
@pytest.mark.parametrize(
    "sigma", [10.0, 1.0, 0.5], ids=["sigma_ten", "sigma_one", "sigma_pointfive"]
)
def test_normal_distribution_init(mu: float, sigma: float) -> None:
    dist = NormalDistribution(mu=mu, sigma=sigma)
    assert dist.mu == mu
    assert dist.sigma == sigma
    assert dist.distribution == "norm"


@pytest.mark.parametrize(
    "invalid_sigma", [0.0, -0.5, -50.5], ids=["sigma_zero", "sigma_neg", "sigma_larger_neg"]
)
def test_normal_distribution_sample_raises_error_for_invalid_sigma(
    invalid_sigma: float,
) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        NormalDistribution(mu=10.0, sigma=invalid_sigma)
