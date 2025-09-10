"""
Representations of log-likelihood calculation methods to be used throughout gempyor.
"""

__all__: tuple[str, ...] = (
    "ObjectiveFunctionABC",
    "BetaLoglikelihood",
    "BinomialLoglikelihood",
    "FixedLoglikelihood",
    "GammaLoglikelihood",
    "LognormalLoglikelihood",
    "NormalLoglikelihood",
    "PoissonLoglikelihood",
    "WeibullLoglikelihood",
    "AbsoluteError",
    "RMSE",
)


import confuse
from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, TypeAdapter, AliasChoices
import scipy.stats

from ._pydantic_ext import EvaledFloat, EvaledInt


class ObjectiveFunctionABC(ABC, BaseModel):
    """Base class for distributions used to calculate log-likelihoods."""

    distribution: str = Field(validation_alias=AliasChoices("distribution", "dist"))

    @abstractmethod
    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Establish shape-specific error-metric calculation logic."""
        raise NotImplemented

    def error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """
        Calculates the error metric when observing data given the model's predictions.

        Args:
            gt_data: The observed ground truth data.
            model_data: The data produced by flepiMoP.

        gt_data and model_data must be the same size.

        Returns:
            An array of log-likelihood values.
        """
        return self._error_metric_calculation(gt_data, model_data)


class FixedLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a fixed distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import FixedLoglikelihood
        >>> dist = FixedLoglikelihood(value=10.0)
        >>> gt_data = np.array([5.0, 10.0, 10.0, 15.0])
        >>> model_data = np.array([1.0, 2.0, 3.0, 4.0])  # This data is ignored
        >>> dist.error_metric_calculation(gt_data=gt_data, model_data=model_data)
        array([-inf,   0.,   0., -inf])
    """

    distribution: Literal["fixed"] = "fixed"
    value: EvaledFloat

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, _model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for fixed distributions."""
        # ignores model_data and compares gt_data to its own value.
        return np.where(np.isclose(gt_data, self.value), 0.0, -np.inf)


class NormalLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a normal distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import NormalLoglikelihood
        >>> dist = NormalLoglikelihood(sigma=2.0)
        >>> gt_data = np.array([10.0, 12.0, 15.0])
        >>> model_data = np.array([11.0, 11.0, 16.0])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-1.7371, -1.7371, -1.7371])
    """

    distribution: Literal["norm"] = "norm"
    sigma: EvaledFloat = Field(..., gt=0, validation_alias=AliasChoices("sigma", "sd"))

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for normal distributions."""
        return scipy.stats.norm.logpdf(x=gt_data, loc=model_data, scale=self.sigma)


class LognormalLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a Lognormal distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import LognormalLoglikelihood
        >>> dist = LognormalLoglikelihood(sdlog=0.5)
        >>> gt_data = np.array([10.0, 20.0, 30.0])
        >>> model_data = np.array([12.0, 18.0, 35.0])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-2.5949, -3.2437, -3.6745])
    """

    distribution: Literal["lognorm"] = "lognorm"
    sigmalog: EvaledFloat = Field(
        ..., gt=0, validation_alias=AliasChoices("sigmalog", "sdlog")
    )

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for lognormal distributions."""
        return scipy.stats.lognorm.logpdf(x=gt_data, s=self.sigmalog, scale=model_data)


class PoissonLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a Poisson distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import PoissonLoglikelihood
        >>> dist = PoissonLoglikelihood()
        >>> gt_data = np.array([4, 10, 15])
        >>> model_data = np.array([5.5, 9.5, 16.0])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-1.8591, -2.0915, -2.3104])
    """

    distribution: Literal["poisson", "pois"] = "poisson"

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for Poisson distributions."""
        return scipy.stats.poisson.logpmf(k=gt_data, mu=model_data)


class BinomialLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a binomial distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import BinomialLoglikelihood
        >>> dist = BinomialLoglikelihood(n=20)
        >>> gt_data = np.array([5, 15, 10])
        >>> model_data = np.array([0.2, 0.8, 0.5])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-1.7455, -1.7455, -1.7362])
    """

    distribution: Literal["binomial"] = "binomial"
    n: EvaledInt = Field(..., ge=0)

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for binomial distributions."""
        if np.any((model_data < 0) | (model_data > 1)):
            raise ValueError(
                "With binomial llik calculations, probabilities in `model_data` must be in the range [0, 1]"
            )

        return scipy.stats.binom.logpmf(k=gt_data, n=self.n, p=model_data)


class GammaLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a gamma distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import GammaLoglikelihood
        >>> dist = GammaLoglikelihood(shape=2.0)
        >>> gt_data = np.array([5.0, 10.0, 15.0])
        >>> model_data = np.array([6.0, 9.0, 14.0])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-2.8074, -3.203 , -3.6415])
    """

    distribution: Literal["gamma"] = "gamma"
    shape: EvaledFloat = Field(..., gt=0)

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for gamma distributions."""
        return scipy.stats.gamma.logpdf(x=gt_data, a=self.shape, scale=model_data)


class WeibullLoglikelihood(ObjectiveFunctionABC):
    """
    Represents a weibull distribution for calculating log-likelihood.

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import WeibullLoglikelihood
        >>> dist = WeibullLoglikelihood(shape=1.5)
        >>> gt_data = np.array([5.0, 10.0, 15.0])
        >>> model_data = np.array([6.0, 12.0, 16.0])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-2.2382, -2.9313, -3.3071])
    """

    distribution: Literal["weibull"] = "weibull"
    shape: EvaledFloat = Field(..., gt=0)

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Log-likelihood calculations for weibull distributions."""
        return scipy.stats.weibull_min.logpdf(x=gt_data, c=self.shape, scale=model_data)


class AbsoluteError(ObjectiveFunctionABC):
    """
    Calculates an error metric using the sum of absolute errors..

    The final score is calculated as -log(sum_of_absolute_errors).

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import AbsoluteError
        >>> dist = AbsoluteError()
        >>> gt_data = np.array([1, 2, 6])
        >>> model_data = np.array([3, 2, 4])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-1.3863, -1.3863, -1.3863])
    """

    distribution: Literal["absolute_error"] = "absolute_error"

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Calculates the error metric from the sum of absolute errors."""
        absolute_error = np.abs(gt_data - model_data)
        total_absolute_error = np.nansum(absolute_error)
        return np.full(gt_data.shape, -np.log(total_absolute_error))


class RMSE(ObjectiveFunctionABC):
    """
    Calculates an error metric using random mean squared error.

    The final score is calculated as -log(RMSE).

    Examples:
        >>> import numpy as np
        >>> from gempyor.objective_functions import RMSE
        >>> dist = RMSE()
        >>> gt_data = np.array([1, 2, 6])
        >>> model_data = np.array([3, 4, 4])
        >>> np.round(dist.error_metric_calculation(gt_data=gt_data, model_data=model_data), 4)
        array([-0.6931, -0.6931, -0.6931])
    """

    distribution: Literal["rmse"] = "rmse"

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """Calculates the error metric from RMSE."""
        squared_error = (gt_data - model_data) ** 2
        mean_squared_error = np.nanmean(squared_error)
        rmse = np.sqrt(mean_squared_error)
        return np.full(gt_data.shape, -np.log(rmse))


ObjectiveFunction = Annotated[
    BinomialLoglikelihood
    | FixedLoglikelihood
    | GammaLoglikelihood
    | LognormalLoglikelihood
    | NormalLoglikelihood
    | PoissonLoglikelihood
    | WeibullLoglikelihood
    | AbsoluteError
    | RMSE,
    Field(discriminator="distribution"),
]

ERROR_METRIC_SHAPE_ADAPTER = TypeAdapter(ObjectiveFunction)


def objective_function_from_confuse_config(config: confuse.ConfigView) -> ObjectiveFunction:
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
    return ERROR_METRIC_SHAPE_ADAPTER.validate_python(conf)
