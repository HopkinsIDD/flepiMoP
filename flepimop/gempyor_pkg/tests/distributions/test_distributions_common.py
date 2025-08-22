import numpy as np
import pytest
from gempyor.distributions import DistributionABC
from pydantic import PrivateAttr


class DummyDistribution(DistributionABC):
    """A simple dummy implementation for testing DistributionABC logic."""

    distribution: str = "dummy"

    _lower_bound: float = PrivateAttr(default=0.0)
    _upper_bound: float = PrivateAttr(default=1.0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """A fake sampling implementation."""
        return rng.random(size)


def test_reproducible_sampling_with_seeded_rng() -> None:
    dist = DummyDistribution()
    rng1 = np.random.default_rng(seed=42)
    sample1 = dist.sample(rng=rng1)
    rng2 = np.random.default_rng(seed=42)
    sample2 = dist.sample(rng=rng2)
    assert np.array_equal(sample1, sample2)


def test_stochastic_sampling_with_default_rng() -> None:
    dist = DummyDistribution()
    sample1 = dist.sample(size=10)
    sample2 = dist.sample(size=10)
    assert not np.array_equal(sample1, sample2)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        (15, (15,)),
        ((2, 5), (2, 5)),
        ((2, 3, 2), (2, 3, 2)),
    ],
    ids=["integer_size", "2d_tuple_size", "3d_tuple_size"],
)
def test_distribution_abc_sample_properties(size, expected_shape) -> None:
    dist = DummyDistribution()
    sample = dist.sample(size=size)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert np.issubdtype(sample.dtype, np.number)


def test_distribution_abc_callable() -> None:
    dist = DummyDistribution()
    sample_callable_result = dist()
    assert isinstance(
        sample_callable_result, float
    )  # b/c rng.random always returns a float
