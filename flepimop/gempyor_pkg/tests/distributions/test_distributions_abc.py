import numpy as np
import pytest
from gempyor.distributions import (
    BetaDistribution,
    BinomialDistribution,
    # No FixedDistribution; does not use RNGs
    GammaDistribution,
    LognormalDistribution,
    NormalDistribution,
    PoissonDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    WeibullDistribution,
)


@pytest.mark.parametrize(
    "dist",
    [
        BetaDistribution(alpha=2.0, beta=5.0),
        BinomialDistribution(n=10, p=0.5),
        GammaDistribution(shape=2.0, scale=1.5),
        LognormalDistribution(meanlog=0.0, sdlog=1.0),
        NormalDistribution(mu=2.3, sigma=4.5),
        PoissonDistribution(lam=3.0),
        TruncatedNormalDistribution(mean=5.0, sd=2.0, a=0.0, b=10.0),
        UniformDistribution(low=-0.5, high=1.5),
        WeibullDistribution(shape=2.5, scale=5.0),
    ],
    ids=[
        "Beta",
        "Binomial",
        "Gamma",
        "Lognormal",
        "Normal",
        "Poisson",
        "TruncatedNormal",
        "Uniform",
        "Weibull",
    ],
)
def test_distribution_abc_rng_behavior(dist) -> None:
    rng1 = np.random.default_rng(seed=42)
    sample1 = dist.sample(rng=rng1)
    rng2 = np.random.default_rng(seed=42)
    sample2 = dist.sample(rng=rng2)
    np.testing.assert_array_equal(sample1, sample2)

    sample3_default = dist.sample()
    sample4_default = dist.sample()
    assert not np.array_equal(sample3_default, sample4_default)
