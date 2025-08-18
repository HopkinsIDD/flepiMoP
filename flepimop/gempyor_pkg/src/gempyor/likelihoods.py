"""
Representations of log-likelihood calculation methods to be used throughout gempyor.
"""

__all__: tuple[str, ...] = (
    "LoglikelihoodABC",
    "BetaLoglikelihood",
    "BinomialLoglikelihood",
    "FixedLoglikelihood",
    "GammaLoglikelihood",
    "LognormalLoglikelihood",
    "NormalLoglikelihood",
    "PoissonLoglikelihood",
    "TruncatedNormalLoglikelihood",
    "UniformLoglikelihood",
    "WeibullLoglikelihood",
    "AbsoluteErrorLoglikelihood",
    "RMSELoglikelihood",
)


import confuse
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, TypeAdapter, AliasChoices
import scipy.stats

from ._pydantic_ext import EvaledFloat, EvaledInt


class LoglikelihoodABC(ABC, BaseModel):
    """Base class for distributions used to calculate log-likelihoods."""

    distribution: str = Field(validation_alias=AliasChoices("distribution", "dist"))

    @abstractmethod
    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Establish distribution-specific log-likelihood logic."""
        raise NotImplemented

    def loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """
        Calculates the log-likelihood of observing data given the model's predictions.

        Args:
            gt_data: The observed ground truth data.
            model_data: The data produced by flepiMoP.

        Returns:
            An array of log-likelihood values.
        """
        return self._loglikelihood(gt_data, model_data)


class FixedLoglikelihood(LoglikelihoodABC):
    """
    Represents a fixed distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["fixed"] = "fixed"
    value: EvaledFloat

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for fixed distributions."""
        # ignores model_data and compares gt_data to its own value.
        return np.where(np.isclose(gt_data, self.value), 0.0, -np.inf)


class NormalLoglikelihood(LoglikelihoodABC):
    """
    Represents a normal distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["norm"] = "norm"
    sigma: EvaledFloat = Field(..., gt=0)

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for normal distributions."""
        return scipy.stats.norm.logpdf(x=gt_data, loc=model_data, scale=self.sigma)


class UniformLoglikelihood(LoglikelihoodABC):
    """
    Represents a uniform distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["uniform"] = "uniform"
    low: EvaledFloat
    high: EvaledFloat

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for uniform distributions."""
        loc = model_data - ((self.high - self.low) / 2.0)
        scale = self.high - self.low
        return scipy.stats.uniform.logpdf(x=gt_data, loc=loc, scale=scale)


class LognormalLoglikelihood(LoglikelihoodABC):
    """
    Represents a Lognormal distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["lognorm"] = "lognorm"
    sdlog: EvaledFloat = Field(..., gt=0)

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for lognormal distributions."""
        return scipy.stats.lognorm.logpdf(x=gt_data, s=self.sdlog, scale=model_data)


class TruncatedNormalLoglikelihood(LoglikelihoodABC):
    """
    Represents a truncated normal distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["truncnorm"] = "truncnorm"
    sd: EvaledFloat = Field(..., gt=0)
    a: EvaledFloat
    b: EvaledFloat

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for truncated normal distributions."""
        # clipping bounds
        a_prime = (self.a - model_data) / self.sd
        b_prime = (self.b - model_data) / self.sd
        return scipy.stats.truncnorm.logpdf(
            x=gt_data, a=a_prime, b=b_prime, loc=model_data, scale=self.sd
        )


class PoissonLoglikelihood(LoglikelihoodABC):
    """
    Represents a Poisson distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["poisson", "pois"] = "poisson"

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for Poisson distributions."""
        return scipy.stats.poisson.logpmf(k=gt_data, mu=model_data)


class BinomialLoglikelihood(LoglikelihoodABC):
    """
    Represents a binomial distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["binomial"] = "binomial"
    n: EvaledInt = Field(..., ge=0)

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for binomial distributions."""
        return scipy.stats.binom.logpmf(k=gt_data, n=self.n, p=np.clip(model_data, 0, 1))


class GammaLoglikelihood(LoglikelihoodABC):
    """
    Represents a gamma distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["gamma"] = "gamma"
    shape: EvaledFloat = Field(..., gt=0)

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for gamma distributions."""
        return scipy.stats.gamma.logpdf(x=gt_data, a=self.shape, scale=model_data)


class WeibullLoglikelihood(LoglikelihoodABC):
    """
    Represents a weibull distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["weibull"] = "weibull"
    shape: EvaledFloat = Field(..., gt=0)

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for weibull distributions."""
        return scipy.stats.weibull_min.logpdf(x=gt_data, c=self.shape, scale=model_data)


class BetaLoglikelihood(LoglikelihoodABC):  # probably we can remove this for now?
    """
    Represents a beta distribution for calculating log-likelihood.

    Examples:
        ...
    """

    distribution: Literal["beta"] = "beta"

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Log-likelihood calculations for bet distributions."""
        raise NotImplementedError(
            "Log-likelihood calculation is not yet implemented for the Beta distribution."
        )


class AbsoluteErrorLoglikelihood(LoglikelihoodABC):
    """
    Calculates a log-likelihood score using the sum of absolute errors..

    The final score is calculated as -log(sum_of_absolute_errors).
    """

    distribution: Literal["absolute_error"] = "absolute_error"

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Calculates the log-likelihood score from the sum of absolute errors."""
        absolute_error = np.abs(gt_data - model_data)
        total_absolute_error = np.nansum(absolute_error)
        return np.full(gt_data.shape, -np.log(total_absolute_error))


class RMSELoglikelihood(LoglikelihoodABC):
    """
    Calculates a log-likelihood score using RMSE.

    The final score is calculated as -log(RMSE).
    """

    distribution: Literal["rmse"] = "rmse"

    def _loglikelihood(self, gt_data: npt.NDArray, model_data: npt.NDArray) -> npt.NDArray:
        """Calculates the log-likelihood score from RMSE."""
        squared_error = (gt_data - model_data) ** 2
        mean_squared_error = np.nanmean(squared_error)
        rmse = np.sqrt(mean_squared_error)
        return np.full(gt_data.shape, -np.log(rmse))


LoglikelihoodShape = Annotated[
    BetaLoglikelihood
    | BinomialLoglikelihood
    | FixedLoglikelihood
    | GammaLoglikelihood
    | LognormalLoglikelihood
    | NormalLoglikelihood
    | PoissonLoglikelihood
    | TruncatedNormalLoglikelihood
    | UniformLoglikelihood
    | WeibullLoglikelihood
    | AbsoluteErrorLoglikelihood
    | RMSELoglikelihood,
    Field(discriminator="distribution"),
]

LOGLIKE_SHAPE_ADAPTER = TypeAdapter(LoglikelihoodShape)


def loglikelihood_from_confuse_config(config: confuse.ConfigView) -> LoglikelihoodShape:
    """
    Creates a log-likelihood calculation style from a `confuse.ConfigView`.

    Args:
        config: A `confuse.ConfigView` for single log-likelihood calculation.

    Returns:
        A LoglikelihoodShape object.
    """
    conf = config.get().copy()
    if "dist" in conf:
        conf["distribution"] = conf.pop("dist")
    return LOGLIKE_SHAPE_ADAPTER.validate_python(conf)
