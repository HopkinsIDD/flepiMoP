"""
Representations of distributions to be used throughout gempyor.
"""

__all__: tuple[str, ...] = (
    "DistributionABC",
    "BetaDistribution",
    "BinomialDistribution",
    "FixedDistribution",
    "GammaDistribution",
    "LognormalDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
    "WeibullDistribution",
)


import confuse
from abc import ABC, abstractmethod
from math import isclose
from typing import Annotated, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt
from pydantic import BaseModel, PrivateAttr, Field, TypeAdapter, model_validator
import scipy.stats

from ._pydantic_ext import EvaledFloat, EvaledInt


class DistributionABC(ABC, BaseModel):
    """Base class for distributions used in random sampling."""

    distribution: str
    allow_edge_cases: bool = False

    _rng: Generator = PrivateAttr(default_factory=np.random.default_rng)

    def sample(
        self, size: int | tuple[int, ...] = 1, rng: Generator | None = None
    ) -> npt.NDArray[np.float64 | np.int64]:
        """
        Draw random sample(s) from the distribution.

        Args:
            size: The desired output size of samples to be drawn.
            rng: A NumPy random number generator instance used for sampling.

        Returns:
            A NumPy array of either floats/ints (depending on distribution)
            drawn from the distribution with shape `size`.
        """
        rng = rng if rng is not None else self._rng
        return self._sample_from_generator(size=size, rng=rng)

    def __call__(self) -> float | int:
        """A shortcut for `self.sample(size=1)`."""
        return self.sample(size=1).item()

    @abstractmethod
    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64 | np.int64]:
        """
        Distribution-specific sampling logic.

        Args:
            size: The desired output size of samples to be drawn.
            rng: A NumPy random number generator instance used for sampling.
        """
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
    value: EvaledFloat

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for fixed distributions."""
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
        NormalDistribution(distribution='norm', allow_edge_cases=False, mu=2.3, sigma=4.5)
        >>> dist.sample(rng=rng)
        array([3.67122686])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[-2.37992848,  5.67703038,  6.53254122, -6.47965835, -3.55980778],
               [ 2.87528181,  0.87690833,  2.22439479, -1.53869767,  6.25729089],
               [ 5.80006371,  2.59713814,  7.37258543,  4.40379204, -1.56681608]])
    """

    distribution: Literal["norm"] = "norm"
    mu: EvaledFloat
    sigma: EvaledFloat = Field(..., gt=0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for normal distributions."""
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
        UniformDistribution(distribution='uniform', allow_edge_cases=False, low=-0.5, high=1.5)
        >>> dist.sample(rng=rng)
        array([1.0479121])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[ 0.37775688,  1.21719584,  0.89473606, -0.3116453 ,  1.4512447 ],
               [ 1.0222794 ,  1.07212861, -0.24377273,  0.40077188,  0.24159605],
               [ 1.35352998,  0.78773024,  1.14552323,  0.3868284 , -0.04552256]])
        >>> # With `low == high` and `allow_edge_cases=True`, all samples == `low`.
        >>> dist_edge = UniformDistribution(low=5.0, high=5.0, allow_edge_cases=True)
        >>> dist_edge.sample(size=5)
        array([5., 5., 5., 5., 5.])
        >>> # Without `allow_edge_cases` set to True, it fails by default when `low == high`.
        >>> UniformDistribution(low=5.0, high=5.0)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for UniformDistribution
          Value error, Upper bound `high`, 5.0, must be > to lower bound `low`, 5.0. [type=value_error, ...
    """

    distribution: Literal["uniform"] = "uniform"
    low: EvaledFloat
    high: EvaledFloat

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for uniform distributions."""
        return rng.uniform(low=self.low, high=self.high, size=size)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "UniformDistribution":
        """Validate bounds based on whether or not edge cases are allowed."""
        if self.high < self.low or (
            not self.allow_edge_cases and isclose(self.high, self.low)
        ):
            op = ">=" if self.allow_edge_cases else ">"
            raise ValueError(
                f"Upper bound `high`, {self.high}, must be {op} to lower bound `low`, {self.low}."
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
        LognormalDistribution(distribution='lognorm', allow_edge_cases=False, meanlog=0.0, sdlog=1.0)
        >>> dist.sample(rng=rng)
        array([1.35624124])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[0.3534603 , 2.11795541, 2.56142749, 0.14212687, 0.27193845],
               [1.13637163, 0.72888261, 0.98333919, 0.42611589, 2.40944872],
               [2.17666075, 1.06825951, 3.08712799, 1.59601411, 0.42346159]])
    """

    distribution: Literal["lognorm"] = "lognorm"
    meanlog: EvaledFloat
    sdlog: EvaledFloat = Field(..., gt=0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for lognormal distributions."""
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
        TruncatedNormalDistribution(distribution='truncnorm', allow_edge_cases=False, mean=1.0, sd=1.0, a=0.0, b=10.0)
        >>> dist.sample(rng=rng)
        array([1.87722989])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[1.07000038, 2.18016199, 1.66002835, 0.28689654, 3.04332767],
               [1.83818339, 1.9153892 , 0.37639339, 1.09435167, 0.92629918],
               [2.54134935, 1.52545861, 2.04022114, 1.07959285, 0.6142512 ]])
        >>> # With `a == b` and `allow_edge_cases=True`, all samples == `a`.
        >>> dist_edge = TruncatedNormalDistribution(mean=5.0, sd=2.0, a=7.0, b=7.0, allow_edge_cases=True)
        >>> dist_edge.sample(size=5)
        array([7., 7., 7., 7., 7.])
        >>> # Withoug `allow_edge_cases` set to True, it fails by default when `a == b`.
        >>> TruncatedNormalDistribution(mean=5.0, sd=2.0, a=7.0, b=7.0)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for TruncatedNormalDistribution
          Value error, Upper bound `b`, 7.0, must be > to lower bound `a`, 7.0. [type=value_error, ...
    """
    # pylint: enable=line-too-long

    distribution: Literal["truncnorm"] = "truncnorm"
    mean: EvaledFloat
    sd: EvaledFloat = Field(..., gt=0)
    a: EvaledFloat
    b: EvaledFloat

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for truncated normal distributions."""
        if (
            isclose(self.a, self.b) and self.allow_edge_cases
        ):  # use this logic b/c scipy.truncnorm doesn't support equal bounds
            return np.full(size, self.a)

        lower = (self.a - self.mean) / self.sd
        upper = (self.b - self.mean) / self.sd
        return scipy.stats.truncnorm.rvs(
            a=lower,
            b=upper,
            loc=self.mean,
            scale=self.sd,
            size=size,
            random_state=rng,
        )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "TruncatedNormalDistribution":
        """Validate bounds based on whether or not edge cases are allowed.."""
        if self.b < self.a or (not self.allow_edge_cases and isclose(self.a, self.b)):
            op = ">=" if self.allow_edge_cases else ">"
            raise ValueError(
                f"Upper bound `b`, {self.b}, must be {op} to lower bound `a`, {self.a}."
            )
        return self


class PoissonDistribution(DistributionABC):
    """
    Represents a Poisson distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import PoissonDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = PoissonDistribution(lam=3.0)
        >>> dist
        PoissonDistribution(distribution='poisson', allow_edge_cases=False, lam=3.0)
        >>> dist.sample(rng=rng)
        array([4])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[4, 5, 1, 7, 1],
               [4, 2, 2, 5, 4],
               [1, 6, 2, 5, 0]])
        >>> # With `lam=0` and `allow_edge_cases=True`, all samples will be 0.
        >>> dist_edge = PoissonDistribution(lam=0.0, allow_edge_cases=True)
        >>> dist_edge.sample(size=5)
        array([0, 0, 0, 0, 0])
        >>> # Without `allow_edge_cases` explicitly set to True, it fails by default.
        >>> PoissonDistribution(lam=0.0)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for PoissonDistribution
          Value error, Input for `lam` cannot be zero when `allow_edge_cases` is `False`. [type=value_error, ...
    """

    distribution: Literal["poisson", "pois"] = "poisson"
    lam: EvaledFloat = Field(..., ge=0.0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.int64]:
        """Sampling logic for Poisson distributions."""
        return rng.poisson(lam=self.lam, size=size)

    @model_validator(mode="after")
    def _validate_lambda(self) -> "PoissonDistribution":
        if not self.allow_edge_cases and isclose(self.lam, 0.0):
            raise ValueError(
                "Input for `lam` cannot be zero when `allow_edge_cases` is `False`."
            )
        return self


class BinomialDistribution(DistributionABC):
    """
    Represents a binomial distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import BinomialDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = BinomialDistribution(n=10, p=0.5)
        >>> dist
        BinomialDistribution(distribution='binomial', allow_edge_cases=False, n=10, p=0.5)
        >>> dist.sample(rng=rng)
        array([6])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[5, 7, 6, 3, 8],
               [6, 6, 3, 5, 4],
               [7, 6, 6, 5, 4]])
        >>> # It succeeds with `p=0` or `p=1` when `allow_edge_cases=True`.
        >>> dist_edge = BinomialDistribution(n=10, p=1.0, allow_edge_cases=True)
        >>> dist_edge.sample(size=5)
        array([10, 10, 10, 10, 10])
        >>> # Without `allow_edge_cases` set to True, it fails by default when `p=0` or `p=1`.
        >>> BinomialDistribution(n=10, p=0.0)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for BinomialDistribution
          Value error, Input for `p` cannot be 0 or 1 when `allow_edge_cases` is `False`. [type=value_error, input_value={'n': 10, 'p': 0.0}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """

    distribution: Literal["binomial"] = "binomial"
    n: EvaledInt = Field(..., ge=0)
    p: EvaledFloat = Field(..., ge=0.0, le=1.0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.int64]:
        """Sampling logic for binomial distributions."""
        return rng.binomial(n=self.n, p=self.p, size=size)

    @model_validator(mode="after")
    def _validate_params(self) -> "BinomialDistribution":
        """Validate params based on whether or not edge cases are allowed."""
        if not self.allow_edge_cases:
            if self.n == 0:
                raise ValueError(
                    "Input for `n` cannot be zero when `allow_edge_cases` is `False`."
                )
            if isclose(self.p, 0.0) or isclose(self.p, 1.0):
                raise ValueError(
                    "Input for `p` cannot be 0 or 1 when `allow_edge_cases` is `False`."
                )
        return self


class GammaDistribution(DistributionABC):
    """
    Represents a gamma distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import GammaDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = GammaDistribution(shape=2.0, scale=1.5)
        >>> dist
        GammaDistribution(distribution='gamma', allow_edge_cases=False, shape=2.0, scale=1.5)
        >>> dist.sample(rng=rng)
        array([3.13772591])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[4.25301838, 2.75582337, 2.46760563, 4.61888299, 2.63006031],
               [3.51900762, 3.28422915, 4.61612039, 2.15883076, 5.69337578],
               [1.75889742, 3.67897959, 3.38745296, 1.6334585 , 3.89261212]])
    """

    distribution: Literal["gamma"] = "gamma"
    shape: EvaledFloat = Field(..., gt=0)
    scale: EvaledFloat = Field(..., gt=0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for Gamma distributions."""
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
        WeibullDistribution(distribution='weibull', allow_edge_cases=False, shape=2.5, scale=5.0)
        >>> dist.sample(rng=rng)
        array([7.10164344])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[7.02058405, 7.0786094 , 3.00403884, 1.87780582, 5.8054469 ],
               [5.73657673, 7.886256  , 1.81412232, 5.0918523 , 1.73016943],
               [5.17350562, 6.22761392, 3.4198509 , 5.43445321, 2.36440749]])
    """

    distribution: Literal["weibull"] = "weibull"
    shape: EvaledFloat = Field(..., gt=0)
    scale: EvaledFloat = Field(..., gt=0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for Weibull distributions."""
        # Multiply by scale b/c rng.weibull assumes standard weibull dist (scale of 1)
        return self.scale * rng.weibull(a=self.shape, size=size)


class BetaDistribution(DistributionABC):
    """
    Represents a beta distribution.

    Examples:
        >>> import numpy as np
        >>> from gempyor.distributions import BetaDistribution
        >>> rng = np.random.default_rng(42)
        >>> dist = BetaDistribution(alpha=2.0, beta=5.0)
        >>> dist
        BetaDistribution(distribution='beta', allow_edge_cases=False, alpha=2.0, beta=5.0)
        >>> dist.sample(rng=rng)
        array([0.24395464])
        >>> dist.sample(size=(3, 5), rng=rng)
        array([[0.28406092, 0.39027204, 0.29864681, 0.41835336, 0.49963165],
               [0.30396328, 0.15089427, 0.32937986, 0.52373987, 0.16127411],
               [0.32746504, 0.48761242, 0.2162056 , 0.29178583, 0.22819733]])
    """

    distribution: Literal["beta"] = "beta"
    alpha: EvaledFloat = Field(..., gt=0)
    beta: EvaledFloat = Field(..., gt=0)

    def _sample_from_generator(
        self, size: int | tuple[int, ...], rng: Generator
    ) -> npt.NDArray[np.float64]:
        """Sampling logic for beta distributions."""
        return rng.beta(a=self.alpha, b=self.beta, size=size)


Distribution = Annotated[
    BetaDistribution
    | BinomialDistribution
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

DISTRIBUTION_ADAPTER = TypeAdapter(Distribution)


def distribution_from_confuse_config(config: confuse.ConfigView) -> Distribution:
    """
    Creates a Distribution object from a `confuse.ConfigView`.

    Handles the case where the value is a simple number or string,
    interpreting it as a 'fixed' distribution.

    Args:
        config: A `confuse.ConfigView` for a single parameter.

    Returns:
        A Distribution object.
    """
    conf = config.get()
    if isinstance(conf, float | int | str):
        conf = {"distribution": "fixed", "value": conf}
    return DISTRIBUTION_ADAPTER.validate_python(conf)
