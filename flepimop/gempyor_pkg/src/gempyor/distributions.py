"""Representations of distributions used for modifiers, likelihoods, etc."""

__all__: tuple[str, ...] = (
    "Distribution",
    "DistributionABC",
    "FixedDistribution",
    "NormalDistribution",
    "UniformDistribution",
)


from abc import ABC, abstractmethod
from math import isclose
from typing import Annotated, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator


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


class UniformDistribution(DistributionABC):
    """
    Represents a uniform distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import UniformDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = UniformDistribution(low=-0.5, high=1.5)
        >>> dist
        UniformDistribution(distribution='uniform', low=-0.5, high=1.5)
        >>> dist.sample(rng=rng)
        array([1.0479121])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[ 0.37775688,  1.21719584,  0.89473606, -0.3116453 ,  1.4512447 ],
               [ 1.0222794 ,  1.07212861, -0.24377273,  0.40077188,  0.24159605],
               [ 1.35352998,  0.78773024,  1.14552323,  0.3868284 , -0.04552256]])
    """

    distribution: Literal["uniform"] = "uniform"
    low: float
    high: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the uniform distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(low=self.low, high=self.high, size=size)

    @model_validator(mode="after")
    def _ensure_high_greater_than_low(self) -> "UniformDistribution":
        """Ensure that high is greater than low."""
        if self.high < self.low or isclose(self.high, self.low):
            raise ValueError(
                f"The 'high' value, {self.high}, must be "
                f"greater than the 'low' value, {self.low}."
            )
        return self


Distribution = Annotated[
    FixedDistribution | NormalDistribution | UniformDistribution,
    Field(discriminator="distribution"),
]
