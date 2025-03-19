"""
Abstractions for interacting with output statistic configurations.

This module provides the `Statistic` class which represents a entry in the inference ->
statistics section.
"""

__all__ = (
    "Statistic",
    "StatisticConfig",
    "StatisticLikelihoodConfig",
    "StatisticRegularizeConfig",
    "StatisticResampleConfig",
)


import re
from typing import Final

import confuse
import numpy as np
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.special import gammaln
import scipy.stats
import xarray as xr
from xarray.core.resample import DataArrayResample


class StatisticLikelihoodConfig(BaseModel):
    """
    Configuration for the likelihood function of a statistic.

    Attributes:
        dist: The name of the distribution to use for calculating log-likelihood.
        params: Distribution parameters used in the log-likelihood calculation and
            dependent on `dist`.
    """

    dist: str
    params: dict[str, int | float] = {}


class StatisticResampleConfig(BaseModel):
    """
    Configuration for optional resampling of data before computing a statistic.

    Attributes:
        freq: The frequency to resample the data to.
        aggregator: The name of the aggregation function to use.
        skipna: If NAs should be skipped when aggregating.
    """

    freq: str
    aggregator: str
    skipna: bool = False

    @field_validator("freq", mode="after")
    @classmethod
    def validate_freq(cls, freq: str) -> str:
        """
        Statistic resample frequency validator.

        This pydantic field validator checks if the given frequency is a valid offset
        alias by parsing it with `pandas.tseries.frequencies.to_offset`. Pandas does not
        expose valid offset aliases as a constant.

        Args:
            freq: The frequency to resample the data to.

        Returns:
            The validated frequency.
        """
        to_offset(freq)
        return freq

    @field_validator("aggregator", mode="after")
    @classmethod
    def validate_aggregator(cls, aggregator: str) -> str:
        """
        Statistic resample aggregator validator.

        This pydantic field validator checks if the given aggregator name is a valid by
        checking if it is an attribute of `xarray.core.resample.DataArrayResample`.

        Args:
            aggregator: The name of the aggregation function to use.

        Returns:
            The validated aggregator name.

        Raises:
            ValueError: If the given aggregator name is not supported.
        """
        if not hasattr(DataArrayResample, aggregator):
            raise ValueError(
                f"Given an unsupported aggregator name, '{aggregator}', "
                "must be a valid xarray DataArrayResample method."
            )
        return aggregator


class StatisticRegularizeConfig(BaseModel):
    """
    Configuration for optional regularization of data before computing a statistic.

    Attributes:
        name: The name of the regularization function to use.
        extra: Additional configuration for the regularization function, dependent on
            `name`.
    """

    model_config = ConfigDict(extra="allow")

    name: str

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """
        Statistic regularization name validator.

        This pydantic field validator checks if the given regularization name is a valid
        by checking if it is a supported regularization name.

        Args:
            name: The name of the regularization function to use.

        Returns:
            The validated regularization name.

        Raises:
            ValueError: If the given regularization name is not supported.
        """
        if name not in _AVAILABLE_REGULARIZATIONS:
            raise ValueError(
                f"Given an unsupported regularization name, '{name}', "
                f"must be one of: {_AVAILABLE_REGULARIZATIONS}."
            )
        return name


class StatisticConfig(BaseModel):
    """
    Configuration for a statistic.

    Attributes:
        name: The human readable name for the statistic.
        sim_var: The variable in the model data.
        data_var: The variable in the ground truth data.
        likelihood: Configuration for the likelihood function of the statistic.
        resample: Optional configuration for resampling data before computing the
            statistic.
        zero_to_one: Should non-zero values be coerced to 1 when calculating
            log-likelihood.
        regularize: Optional configuration for regularizations of data before computing
            the statistic, applied in order given.
        scale: Optional configuration for scaling data before computing the statistic.
    """

    name: str
    sim_var: str
    data_var: str
    likelihood: StatisticLikelihoodConfig
    resample: StatisticResampleConfig | None = None
    zero_to_one: bool = False
    regularize: list[StatisticRegularizeConfig] = []
    scale: str | None = None

    @field_validator("scale", mode="after")
    @classmethod
    def validate_scale(cls, scale: str | None) -> str | None:
        """
        Statistic scale validator.

        This pydantic field validator checks if the given scale function is a valid
        numpy function.

        Args:
            scale: The function to use when rescaling the data.

        Returns:
            The validated scale function.

        Raises:
            ValueError: If the given scale function is not supported.
        """
        if isinstance(scale, str) and not hasattr(np, scale):
            raise ValueError(
                f"Given an unsupported scale function, '{scale}', "
                "must be a valid numpy function."
            )
        return scale


class Statistic:
    """
    Encapsulates logic for representing/implementing output statistic configurations.

    A statistic is a function that takes two time series and returns a scalar value. It
    applies resample, scale, and regularization to the data before computing the
    statistic's log-loss.

    Attributes:
        name: The human readable name for the statistic given during instantiation.
    """

    def __init__(self, name: str, statistic_config: confuse.ConfigView) -> None:
        """
        Create an `Statistic` instance from a confuse config view.

        Args:
            name: A human readable name for the statistic, mostly used for error
                messages.
            statistic_config: A confuse configuration view object describing an output
                statistic.

        Raises:
            ValueError: If an unsupported regularization name is provided via the
                `statistic_config` arg. Currently only 'forecast' and 'allsubpop' are
                supported.
        """
        self.name = name
        self._config = StatisticConfig.model_validate(dict(statistic_config.get()))

    @property
    def sim_var(self) -> str:
        """
        Accessor for the model output variable name.

        Returns:
            The name of the model output variable.
        """
        return self._config.sim_var

    @property
    def data_var(self) -> str:
        """
        Accessor for the ground truth variable name.

        Returns:
            The name of the ground truth variable.
        """
        return self._config.data_var

    def __str__(self) -> str:
        return (
            f"{self.name}: {self._config.likelihood.dist} between "
            f"{self._config.sim_var} (sim) and {self._config.data_var} (data)."
        )

    def __repr__(self) -> str:
        return f"A Statistic(): {self.__str__()}"

    def _forecast_regularize(
        self,
        model_data: xr.DataArray,
        gt_data: xr.DataArray,
        **kwargs: dict[str, int | float],
    ) -> float:
        """
        Regularization function to add weight to more recent forecasts.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.
            **kwargs: Optional keyword arguments that influence regularization.
                Currently uses `last_n` for the number of observations to up weight and
                `mult` for the coefficient of the regularization value.

        Returns:
            The log-likelihood of the `last_n` observation up weighted by a factor of
            `mult`.
        """
        last_n = kwargs.get("last_n", 4)
        mult = kwargs.get("mult", 2)
        last_n_llik = self.llik(
            model_data.isel(date=slice(-last_n, None)),
            gt_data.isel(date=slice(-last_n, None)),
        )
        return mult * last_n_llik.sum().sum().values

    def _allsubpop_regularize(
        self,
        model_data: xr.DataArray,
        gt_data: xr.DataArray,
        **kwargs: dict[str, int | float],
    ) -> float:
        """
        Regularization function to add the sum of all subpopulations.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.
            **kwargs: Optional keyword arguments that influence regularization.
                Currently uses `mult` for the coefficient of the regularization value.

        Returns:
            The sum of the subpopulations multiplied by `mult`.
        """
        mult = kwargs.get("mult", 1)
        llik_total = self.llik(model_data.sum("subpop"), gt_data.sum("subpop"))
        return mult * llik_total.sum().sum().values

    def apply_resample(self, data: xr.DataArray) -> xr.DataArray:
        """
        Resample a data set to the given frequency using the specified aggregation.

        Args:
            data: An xarray dataset with "date" and "subpop" dimensions.

        Returns:
            A resample dataset with similar dimensions to `data`.
        """
        if (r := self._config.resample) is not None:
            return getattr(data.resample(date=r.freq), r.aggregator)(skipna=r.skipna)
        return data

    def apply_scale(self, data: xr.DataArray) -> xr.DataArray:
        """
        Scale a data set using the specified scaling function.

        Args:
            data: An xarray dataset with "date" and "subpop" dimensions.

        Returns:
            An xarray dataset of the same shape and dimensions as `data` with the
            `scale_func` attribute applied.
        """
        if (s := self._config.scale) is not None:
            return getattr(np, s)(data)
        return data

    def apply_transforms(self, data: xr.DataArray):
        """
        Convenient wrapper for resampling and scaling a data set.

        The resampling is applied *before* scaling which can affect the log-likelihood.

        Args:
            data: An xarray dataset with "date" and "subpop" dimensions.

        Returns:
            An scaled and resampled dataset with similar dimensions to `data`.
        """
        return self.apply_scale(self.apply_resample(data))

    def llik(self, model_data: xr.DataArray, gt_data: xr.DataArray) -> xr.DataArray:
        """
        Compute the log-likelihood of observing the ground truth given model output.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.

        Returns:
            The log-likelihood of observing `gt_data` from the model `model_data` as an
            xarray DataArray with a "subpop" dimension.
        """

        dist_map = {
            "pois": lambda gt_data, model_data: -model_data
            + (gt_data * np.log(model_data))
            - gammaln(gt_data + 1),
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # OLD: # TODO: Swap out in favor of NEW
            "norm": lambda gt_data, model_data, scale: scipy.stats.norm.logpdf(
                gt_data,
                loc=model_data,
                scale=self._config.likelihood.params.get("scale", scale),
            ),
            "norm_cov": lambda gt_data, model_data, scale: scipy.stats.norm.logpdf(
                gt_data, loc=model_data, scale=scale * model_data.where(model_data > 5, 5)
            ),
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # NEW: names of distributions: `norm` --> `norm_homoskedastic`, `norm_cov`
            # --> `norm_heteroskedastic`; names of input `scale` --> `sd`
            "norm_homoskedastic": lambda gt_data, model_data, sd: scipy.stats.norm.logpdf(
                gt_data, loc=model_data, scale=self._config.likelihood.params.get("sd", sd)
            ),  # scale = standard deviation
            "norm_heteroskedastic": lambda gt_data, model_data, sd: scipy.stats.norm.logpdf(
                gt_data,
                loc=model_data,
                scale=self._config.likelihood.params.get("sd", sd) * model_data,
            ),  # scale = standard deviation
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            "nbinom": lambda gt_data, model_data, n, p: scipy.stats.nbinom.logpmf(
                k=gt_data,
                n=1.0 / self._config.likelihood.params.get("alpha"),
                p=1.0 / (1.0 + self._config.likelihood.params.get("alpha") * model_data),
            ),
            "rmse": lambda gt_data, model_data: -np.log(
                np.sqrt(np.nansum((gt_data - model_data) ** 2))
            ),
            "absolute_error": lambda gt_data, model_data: -np.log(
                np.nansum(np.abs(gt_data - model_data))
            ),
        }
        if self._config.likelihood.dist not in dist_map:
            raise ValueError(
                f"Invalid distribution specified: '{self._config.likelihood.dist}'. "
                f"Valid distributions: '{dist_map.keys()}'."
            )

        # pydata/xarray#4612
        if self._config.likelihood.dist in ["pois", "nbinom"]:
            model_data = model_data.where(model_data.isnull(), model_data.astype(int))
            gt_data = gt_data.where(gt_data.isnull(), gt_data.astype(int))

        # so confusing, wish I had not used xarray to do model_data[model_data==0]=1
        if self._config.zero_to_one:
            model_data = model_data.where(model_data != 0, 1)
            gt_data = gt_data.where(gt_data != 0, 1)

        # Use stored parameters in the distribution function call
        likelihood = dist_map[self._config.likelihood.dist](
            gt_data, model_data, **self._config.likelihood.params
        )

        # If the likelihood is a scalar, broadcast it to the shape of the data.
        # Xarray used to do this, but not anymore after numpy/numpy#26889?
        if len(likelihood.shape) == 0:
            likelihood = np.full(gt_data.shape, likelihood)
        likelihood = xr.DataArray(likelihood, coords=gt_data.coords, dims=gt_data.dims)

        return likelihood

    def compute_logloss(
        self, model_data: xr.Dataset, gt_data: xr.Dataset
    ) -> tuple[xr.DataArray, float]:
        """
        Compute the logistic loss of observing the ground truth given model output.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.

        Returns:
            The logistic loss of observing `gt_data` from the model `model_data`
            decomposed into the log-likelihood along the "subpop" dimension and
            regularizations.

        Raises:
            ValueError: If `model_data` and `gt_data` do not have the same shape.
        """
        model_data = self.apply_transforms(model_data[self.sim_var])
        gt_data = self.apply_transforms(gt_data[self.data_var])

        if not model_data.shape == gt_data.shape:
            raise ValueError(
                f"`model_data` and `gt_data` do not have "
                f"the same shape: `model_data.shape` = '{model_data.shape}' != "
                f"`gt_data.shape` = '{gt_data.shape}'."
            )

        regularization = 0.0
        if self._config.regularize:
            for reg_config in self._config.regularize:
                reg_func = getattr(self, f"_{reg_config.name}_regularize")
                regularization += reg_func(
                    model_data=model_data, gt_data=gt_data, **reg_config.extra
                )

        return self.llik(model_data, gt_data).sum("date"), regularization


_REGULARIZATION_NAME_REGEX: Final = re.compile(r"^_([a-z0-9_]+)_regularize$")

# pylint-dev/pylint#8486
_AVAILABLE_REGULARIZATIONS: Final = {
    m.group(1)  # pylint: disable=used-before-assignment
    for attr in dir(Statistic)
    if (m := _REGULARIZATION_NAME_REGEX.match(attr)) is not None
}
