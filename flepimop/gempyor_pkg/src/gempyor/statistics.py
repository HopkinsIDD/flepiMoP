import xarray as xr
import pandas as pd
import numpy as np
import confuse
import scipy.stats


class Statistic:
    """
    A statistic is a function that takes two time series and returns a scalar value.
    It applies resample, scale, and regularization to the data before computing the statistic's log-loss.
    Configuration:
    - sim_var: the variable in the simulation data
    - data_var: the variable in the ground truth data
    - resample: resample the data before computing the statistic
        - freq: the frequency to resample the data to
        - aggregator: the aggregation function to use
        - skipna: whether to skip NA values
    - regularize: apply a regularization term to the data before computing the statistic
    """

    def __init__(self, name, statistic_config: confuse.ConfigView):
        self.sim_var = statistic_config["sim_var"].as_str()
        self.data_var = statistic_config["data_var"].as_str()
        self.name = name

        self.regularizations = []  # A list to hold regularization functions and configs
        if statistic_config["regularize"].exists():
            for reg_config in statistic_config["regularize"]:  # Iterate over the list
                reg_name = reg_config["name"].get()
                reg_func = getattr(self, f"_{reg_name}_regularize")
                if reg_func is None:
                    raise ValueError(f"Unsupported regularization: {reg_name}")
                self.regularizations.append((reg_func, reg_config.get()))

        self.resample = False
        if statistic_config["resample"].exists():
            self.resample = True
            resample_config = statistic_config["resample"]
            self.resample_freq = ""
            if resample_config["freq"].exists():
                self.resample_freq = resample_config["freq"].get()
            self.resample_aggregator = ""
            if resample_config["aggregator"].exists():
                self.resample_aggregator = getattr(pd.Series, resample_config["aggregator"].get())
            self.resample_skipna = False  # TODO
            if resample_config["aggregator"].exists() and resample_config["skipna"].exists():
                self.resample_skipna = resample_config["skipna"].get()

        self.scale = False
        if statistic_config["scale"].exists():
            self.scale_func = getattr(np, statistic_config["scale"].get())

        self.dist = statistic_config["likelihood"]["dist"].get()
        if statistic_config["likelihood"]["params"].exists():
            self.params = statistic_config["likelihood"]["params"].get()
        else:
            self.params = {}

    def _forecast_regularize(self, model_data, gt_data, **kwargs):
        # scale the data so that the lastest X items are more important
        last_n = kwargs.get("last_n", 4)
        mult = kwargs.get("mult", 2)
        # multiply the last n items by mult
        model_data = model_data * np.concatenate([np.ones(model_data.shape[0] - last_n), np.ones(last_n) * mult])
        gt_data = gt_data * np.concatenate([np.ones(gt_data.shape[0] - last_n), np.ones(last_n) * mult])
        return self.llik(model_data, gt_data)

    def _allsubpop_regularize(self, model_data, gt_data, **kwargs):
        """add a regularization term that is the sum of all subpopulations"""
        mult = kwargs.get("mult", 1)
        # TODOODODODOOODO

    def __str__(self) -> str:
        return f"{self.name}: {self.dist} between {self.sim_var} (sim) and {self.data_var} (data)."

    def __repr__(self) -> str:
        return f"A Statistic(): {self.__str__()}"

    def apply_resample(self, data):
        if self.resample:
            return data.resample(self.resample_freq).agg(self.resample_aggregator, skipna=self.resample_skipna)
        else:
            return data

    def apply_scale(self, data):
        if self.scale:
            return self.scale_func(data)
        else:
            return data

    def apply_transforms(self, data):
        data_scaled_resampled = self.apply_scale(self.apply_resample(data))
        return data_scaled_resampled
    
    def llik(self, model_data, gt_data):
        dist_map = {
            "pois": scipy.stats.poisson.pmf,
            "norm": lambda x, loc, scale: scipy.stats.norm.pdf(
                x, loc=loc, scale=self.params.get("scale", scale)
            ),  # wrong:
            "nbinom": lambda x, n, p: scipy.stats.nbinom.pmf(x, n=self.params.get("n"), p=model_data),
            "rmse": lambda x, y: np.sqrt(np.mean((x - y) ** 2)),
            "absolute_error": lambda x, y: np.mean(np.abs(x - y)),
        }
        if self.dist not in dist_map:
            raise ValueError(f"Invalid distribution specified: {self.dist}")

        # Use stored parameters in the distribution function call
        likelihood = dist_map[self.dist](gt_data, model_data, **self.params)

        # TODO: check the order of the arguments

        return np.log(likelihood)


    def compute_logloss(self, model_data, gt_data):
        model_data = self.apply_transforms(model_data[self.sim_var])
        gt_data = self.apply_transforms(gt_data[self.data_var])
        
        if not model_data.shape == gt_data.shape:
            raise ValueError(f"{self.name} Statistic error: data and groundtruth do not have the same shape")

        regularization = 0
        for reg_func, reg_config in self.regularizations:
            regularization = reg_func(model_data=model_data, gt_data=gt_data, **reg_config)  # Pass config parameters

        return self.llik(model_data, gt_data), regularization
