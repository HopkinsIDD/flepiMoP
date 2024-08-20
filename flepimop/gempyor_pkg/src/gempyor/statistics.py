"""
Abstractions for interacting with output statistic configurations.

This module provides the `Statistic` class which represents a entry in the inference ->
statistics section.
"""

__all__ = ["Statistic"]


import confuse
import numpy as np
import scipy.stats
import xarray as xr


class Statistic:
    """
    Encapsulates logic for representing/implementing output statistic configurations.

    A statistic is a function that takes two time series and returns a scalar value. It
    applies resample, scale, and regularization to the data before computing the
    statistic's log-loss.

    Attributes:
        data_var: The variable in the ground truth data.
        dist: The name of the distribution to use for calculating log-likelihood.
        name: The human readable name for the statistic given during instantiation.
        params: Distribution parameters used in the log-likelihood calculation and
            dependent on `dist`.
        regularizations: Regularization functions that are added to the log loss of this
            statistic.
        resample: If the data should be resampled before computing the statistic.
        resample_aggregator_name: The name of the aggregation function to use.
        resample_freq: The frequency to resample the data to if the `resample` attribute
            is `True`.
        resample_skipna: If NAs should be skipped when aggregating. `False` by default.
        scale: If the data should be rescaled before computing the statistic.
        scale_func: The function to use when rescaling the data. Can be any function
            exported by `numpy`.
        sim_var: The variable in the simulation data.
        zero_to_one: Should non-zero values be coerced to 1 when calculating
            log-likelihood.
    """

    def __init__(self, name: str, statistic_config: confuse.ConfigView) -> None:
        """
        Create an `Statistic` instance from a confuse config view.

        Args:
            name: A human readable name for the statistic, mostly used for error
                messages.
            statistic_config: A confuse configuration view object describing an output
                statistic.
        """
        self.sim_var = statistic_config["sim_var"].as_str()
        self.data_var = statistic_config["data_var"].as_str()
        self.name = name

        self.regularizations = []  # A list to hold regularization functions and configs
        if statistic_config["regularize"].exists():
            for reg_config in statistic_config["regularize"]:  # Iterate over the list
                reg_name = reg_config["name"].get()
                reg_func = getattr(self, f"_{reg_name}_regularize", None)
                if reg_func is None:
                    raise ValueError(f"Unsupported regularization: {reg_name}")
                self.regularizations.append((reg_func, reg_config.get()))

        self.resample = False
        if statistic_config["resample"].exists():
            self.resample = True
            resample_config = statistic_config["resample"]
            self.resample_freq = ""
            if resample_config["freq"].exists():
                self.resample_freq = resample_config["freq"].as_str()
            self.resample_aggregator = ""
            if resample_config["aggregator"].exists():
                self.resample_aggregator_name = resample_config["aggregator"].get()
            self.resample_skipna = False  # TODO
            if (
                resample_config["aggregator"].exists()
                and resample_config["skipna"].exists()
            ):
                self.resample_skipna = resample_config["skipna"].get()

        self.scale = False
        if statistic_config["scale"].exists():
            self.scale_func = getattr(np, statistic_config["scale"].get())

        self.dist = statistic_config["likelihood"]["dist"].get()
        if statistic_config["likelihood"]["params"].exists():
            self.params = statistic_config["likelihood"]["params"].get()
        else:
            self.params = {}

        self.zero_to_one = False
        # TODO: this should be set_zeros_to and only do it for the probability
        if statistic_config["zero_to_one"].exists():
            self.zero_to_one = statistic_config["zero_to_one"].get()

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.dist} between {self.sim_var} "
            f"(sim) and {self.data_var} (data)."
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
        # scale the data so that the latest X items are more important
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
        if self.resample:
            aggregator_method = getattr(
                data.resample(date=self.resample_freq), self.resample_aggregator_name
            )
            return aggregator_method(skipna=self.resample_skipna)
        else:
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
        if self.scale:
            return self.scale_func(data)
        else:
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
        data_scaled_resampled = self.apply_scale(self.apply_resample(data))
        return data_scaled_resampled

    def llik(self, model_data: xr.DataArray, gt_data: xr.DataArray) -> float:
        """
        Compute the log-likelihood of observing the ground truth given model output.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.

        Returns:
            The log-likelihood of observing `gt_data` from the model `model_data`.
        """
        dist_map = {
            "pois": scipy.stats.poisson.logpmf,
            "norm": lambda x, loc, scale: scipy.stats.norm.logpdf(
                x, loc=loc, scale=self.params.get("scale", scale)
            ),  # wrong:
            "norm_cov": lambda x, loc, scale: scipy.stats.norm.logpdf(
                x, loc=loc, scale=scale * loc.where(loc > 5, 5)
            ),  # TODO: check, that it's really the loc
            "nbinom": lambda x, n, p: scipy.stats.nbinom.logpmf(
                x, n=self.params.get("n"), p=model_data
            ),
            "rmse": lambda x, y: -np.log(np.nansum(np.sqrt((x - y) ** 2))),
            "absolute_error": lambda x, y: -np.log(np.nansum(np.abs(x - y))),
        }
        if self.dist not in dist_map:
            raise ValueError(f"Invalid distribution specified: {self.dist}")
        if self.dist in ["pois", "nbinom"]:
            model_data = model_data.astype(int)
            gt_data = gt_data.astype(int)

        if self.zero_to_one:
            # so confusing, wish I had not used xarray to do model_data[model_data==0]=1
            model_data = model_data.where(model_data != 0, 1)
            gt_data = gt_data.where(gt_data != 0, 1)

        # Use stored parameters in the distribution function call
        likelihood = dist_map[self.dist](gt_data, model_data, **self.params)

        likelihood = xr.DataArray(likelihood, coords=gt_data.coords, dims=gt_data.dims)

        # TODO: check the order of the arguments
        return likelihood

    def compute_logloss(
        self, model_data: xr.DataArray, gt_data: xr.DataArray
    ) -> tuple[float, float]:
        """
        Compute the logistic loss of observing the ground truth given model output.

        Args:
            model_data: An xarray Dataset of the model data with date and subpop
                dimensions.
            gt_data: An xarray Dataset of the ground truth data with date and subpop
                dimensions.

        Returns:
            The logistic loss of observing `gt_data` from the model `model_data`
            decomposed into the log-likelihood and regularizations.
        """
        model_data = self.apply_transforms(model_data[self.sim_var])
        gt_data = self.apply_transforms(gt_data[self.data_var])

        if not model_data.shape == gt_data.shape:
            raise ValueError(
                (
                    f"{self.name} Statistic error: data and groundtruth do not have "
                    f"the same shape: model_data.shape={model_data.shape} != "
                    f"gt_data.shape={gt_data.shape}"
                )
            )

        regularization = 0
        for reg_func, reg_config in self.regularizations:
            regularization += reg_func(
                model_data=model_data, gt_data=gt_data, **reg_config
            )  # Pass config parameters

        return self.llik(model_data, gt_data).sum("date"), regularization
