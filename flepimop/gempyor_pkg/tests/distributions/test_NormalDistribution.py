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
    with pytest.raises(ValidationError, match="Input should be greater than 0") as e:
        NormalDistribution(mu=10.0, sigma=invalid_sigma)


def test_normal_distribution_sample_rng_reproducibility() -> None:
    dist = NormalDistribution(mu=10.0, sigma=1.5)

    rng1 = np.random.default_rng(seed=100)
    sample1 = dist.sample(size=(2, 2), rng=rng1)

    rng2 = np.random.default_rng(seed=100)
    sample2 = dist.sample(size=(2, 2), rng=rng2)

    assert np.array_equal(sample1, sample2)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((3, 2), (3, 2)),
        (10, (10,)),
        ((3, 4), (3, 4)),
    ],
    ids=["tuple_size1", "integer_size", "tuple_size2"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_normal_distribution_sample_properties(size, expected_shape, use_rng) -> None:
    dist = NormalDistribution(mu=0.0, sigma=1.0)
    kwargs = {}
    kwargs["size"] = size
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
