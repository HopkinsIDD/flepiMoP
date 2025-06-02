"""Representations of distributions used for modifiers, likelihoods, etc."""

__all__: tuple[str, ...] = (
    "DistributionABC",
    "FixedDistribution",
    "NormalDistribution",
    "Distribution",
)


from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from pydantic import BaseModel, Field


class DistributionABC(ABC, BaseModel):
    """Base class for distributions used in modifiers, likelihoods, etc."""

    distribution: str

    @abstractmethod
    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64 | np.int64]:
        """Sample from the distribution."""
        raise NotImplementedError


class FixedDistribution(DistributionABC):
    """
    Represents a fixed distribution that always returns the same value.

    Examples:
        >>> from gempyor.distributions import FixedDistribution
        >>> dist = FixedDistribution(value=1.23)
        >>> dist.sample()
        array([1.23])
        >>> dist.sample(size=(3, 5))
        array([[1.23, 1.23, 1.23, 1.23, 1.23],
               [1.23, 1.23, 1.23, 1.23, 1.23],
               [1.23, 1.23, 1.23, 1.23, 1.23]])
    """

    distribution: Literal["fixed"] = "fixed"
    value: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the fixed distribution."""
        return np.full(size, self.value)


class NormalDistribution(DistributionABC):
    """
    Represents a normal distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import NormalDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = NormalDistribution(mu=2.3, sigma=4.5)
        >>> dist
        NormalDistribution(distribution='norm', mu=2.3, sigma=4.5)
        >>> dist.sample(rng=rng)
        array([3.67122686])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[-2.37992848,  5.67703038,  6.53254122, -6.47965835, -3.55980778],
               [ 2.87528181,  0.87690833,  2.22439479, -1.53869767,  6.25729089],
               [ 5.80006371,  2.59713814,  7.37258543,  4.40379204, -1.56681608]])
    """

    distribution: Literal["norm"] = "norm"
    mu: float
    sigma: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the normal distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=self.mu, scale=self.sigma, size=size)


Distribution = Annotated[
    FixedDistribution | NormalDistribution, Field(discriminator="distribution")
]
