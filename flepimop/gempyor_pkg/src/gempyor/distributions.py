"""Representations of distributions used for modifiers, likelihoods, etc."""

__all__: tuple[str, ...] = (
    "BinomialDistribution",
    "Distribution",
    "DistributionABC",
    "FixedDistribution",
    "GammaDistribution",
    "LognormalDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
    "WeibullDistribution",
)


from abc import ABC, abstractmethod
from math import isclose
from typing import Annotated, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator
from scipy.stats import truncnorm


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
    sigma: float = Field(..., gt=0)

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


class LognormalDistribution(DistributionABC):
    """
    Represents a Lognormal distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import LognormalDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = LognormalDistribution(meanlog=0.0, sdlog=1.0)
        >>> dist
        LognormalDistribution(distribution='lognorm', meanlog=0.0, sdlog=1.0)
        >>> dist.sample(rng=rng)
        array([1.35624124])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[0.3534603 , 2.11795541, 2.56142749, 0.14212687, 0.27193845],
               [1.13637163, 0.72888261, 0.98333919, 0.42611589, 2.40944872],
               [2.17666075, 1.06825951, 3.08712799, 1.59601411, 0.42346159]])
    """

    distribution: Literal["lognorm"] = "lognorm"
    meanlog: float
    sdlog: float = Field(..., gt=0)

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the Lognormal distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.lognormal(mean=self.meanlog, sigma=self.sdlog, size=size)


class TruncatedNormalDistribution(DistributionABC):
    # pylint: disable=line-too-long
    """
    Represents a truncated normal distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import TruncatedNormalDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = TruncatedNormalDistribution(mean=1.0, sd=1.0, a=0.0, b=10.0)
        >>> dist
        TruncatedNormalDistribution(distribution='truncnorm', mean=1.0, sd=1.0, a=0.0, b=10.0)
        >>> dist.sample(rng=rng)
        array([1.87722989])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[1.07000038, 2.18016199, 1.66002835, 0.28689654, 3.04332767],
               [1.83818339, 1.9153892 , 0.37639339, 1.09435167, 0.92629918],
               [2.54134935, 1.52545861, 2.04022114, 1.07959285, 0.6142512 ]])
    """
    # pylint: enable=line-too-long

    distribution: Literal["truncnorm"] = "truncnorm"
    mean: float
    sd: float
    a: float
    b: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the truncated normal distribution."""
        if rng is None:
            rng = np.random.default_rng()
        lower = (self.a - self.mean) / self.sd
        upper = (self.b - self.mean) / self.sd
        return truncnorm.rvs(
            a=lower,
            b=upper,
            loc=self.mean,
            scale=self.sd,
            size=size,
            random_state=rng,
        )


class PoissonDistribution(DistributionABC):
    """
    Represents a Poisson distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import PoissonDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = PoissonDistribution(lam=3.0)
        >>> dist
        PoissonDistribution(distribution='poisson', lam=3.0)
        >>> dist.sample(rng=rng)
        array([4])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[4, 5, 1, 7, 1],
               [4, 2, 2, 5, 4],
               [1, 6, 2, 5, 0]])
    """

    distribution: Literal["poisson"] = "poisson"
    lam: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.int64]:
        """Sample from the Poisson distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.poisson(lam=self.lam, size=size)


class BinomialDistribution(DistributionABC):
    """
    Represents a binomial distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import BinomialDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = BinomialDistribution(n=10, p=0.5)
        >>> dist
        BinomialDistribution(distribution='binomial', n=10, p=0.5)
        >>> dist.sample(rng=rng)
        array([6])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[5, 7, 6, 3, 8],
               [6, 6, 3, 5, 4],
               [7, 6, 6, 5, 4]])
    """

    distribution: Literal["binomial"] = "binomial"
    n: int
    p: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.int64]:
        """Sample from the binomial distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.binomial(n=self.n, p=self.p, size=size)


class GammaDistribution(DistributionABC):
    """
    Represents a gamma distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import GammaDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = GammaDistribution(shape=2.0, scale=1.5)
        >>> dist
        GammaDistribution(distribution='gamma', shape=2.0, scale=1.5)
        >>> dist.sample(rng=rng)
        array([2.78453141])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[1.0116844 , 5.1632733 , 6.51682397, 0.49053229, 0.82522731],
               [2.45524838, 1.4870423 , 1.95460596, 0.94191942, 5.92679803],
               [5.08819234, 2.21976694, 7.82570081, 3.42761858, 1.1070266 ]])
    """

    distribution: Literal["gamma"] = "gamma"
    shape: float
    scale: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the gamma distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.gamma(shape=self.shape, scale=self.scale, size=size)


class WeibullDistribution(DistributionABC):
    """
    Represents a weibull distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import WeibullDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = WeibullDistribution(shape=2.5, scale=5.0)
        >>> dist
        WeibullDistribution(distribution='weibull', shape=2.5, scale=5.0)
        >>> dist.sample(rng=rng)
        array([5.18664161])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[4.09267151, 6.25764353, 6.57560416, 2.72348545, 3.73147743],
           [5.08637841, 4.63044996, 4.96639599, 3.99658252, 6.42564251],
           [6.29525049, 5.03450379, 6.88371131, 5.65602432, 3.99307222]])
    """

    distribution: Literal["weibull"] = "weibull"
    shape: float
    scale: float

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Sample from the Weibull distribution."""
        if rng is None:
            rng = np.random.default_rng()
        # Multiply by scale b/c rng.weibull assumes standard weibull dist (scale of 1)
        return self.scale * rng.weibull(a=self.shape, size=size)


Distribution = Annotated[
    BinomialDistribution
    | FixedDistribution
    | GammaDistribution
    | LognormalDistribution
    | NormalDistribution
    | PoissonDistribution
    | TruncatedNormalDistribution
    | UniformDistribution
    | WeibullDistribution,
    Field(discriminator="distribution"),
]
