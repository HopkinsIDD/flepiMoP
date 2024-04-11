import xarray as xr
import pandas as pd
import numpy as np
import confuse
import scipy.stats

class Statistic:
    """
        A statistic is a function that takes two time series and returns a scalar value.
        It applies resampling, scaling, and regularization to the data before computing the statistic's log-loss.
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
            
            self.resample_skipna = False # TODO
            if resample_config["aggregator"].exists() and resample_config["skipna"].exists():
                self.resample_skipna = resample_config["skipna"].get()
        
        self.regularize = False
        if statistic_config["regularize"].exists():
            raise ValueError("Regularization is not implemented")
            regularization_config = statistic_config["regularization"].get()

        self.regularization = None
        if statistic_config["regularize"].exists():
            # TODO: support several regularization
            self.regularization_config = statistic_config["regularize"]
            self.regularization_name = self.regularization_config["name"].get()
            self.regularization = getattr(self, f"_{self.regularization_name}_regularize")
            if self.regularization is None:
                raise ValueError(f"Unsupported regularization: {self.regularization_name}")

        self.scale = False
        if statistic_config["scale"].exists():
            self.scale_func = getattr(np, statistic_config["scale"].get())
    
        self.dist = statistic_config["likelihood"]["dist"].get()
        # TODO here get the parameter in a dictionnary

    def _forecast_regularize(self, data):
        # scale the data so that the lastest X items are more important
        last_n = self.regularization_config["last_n"].get()
        mult = self.regularization_config["mult"].get()
        # multiply the last n items by mult
        reg_data = data * np.concatenate([np.ones(data.shape[0]-last_n), np.ones(last_n)*mult])
        return reg_data
    
    def _allsubpop_regularize(self, data):
        """ add a regularization term that is the sum of all subpopulations
        """



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
        return self.apply_scale(self.apply_resampling(data))

    def compute_logloss(self, model_data, gt_data):
        model_data = self. apply_transforms(model_data[self.sim_var])
        gt_data = self.apply_transforms(gt_data[self.data_var])

                dist_map = {
            "pois": scipy.stats.poisson.pmf,
            "norm": scipy.stats.norm.pdf,
            "nbinom": scipy.stats.nbinom.pmf,
        }
        if self.dist not in dist_map:
            raise ValueError(f"Invalid distribution specified: {self.dist}")
        log_likelihood = dist_map[self.dist](round(gt_data), model_data)

        # Apply regularization if defined
        if self.regularization:
            log_likelihood -= self.regularization(model_data)


            dist_map = {
            "pois": scipy.stats.poisson.pmf,
            "norm": lambda x, loc, scale: scipy.stats.norm.pdf(x, loc=loc, scale=self.params.get("scale", scale)),
            "nbinom": lambda x, n, p: scipy.stats.nbinom.pmf(x, n=self.params.get("n"), p=model_data),
        }
        if self.dist not in dist_map:
            raise ValueError(f"Invalid distribution specified: {self.dist}")

        # Use stored parameters in the distribution function call
        log_likelihood = dist_map[self.dist](round(gt_data), model_data, **self.params)



        if not model_data.shape == gt_data.shape:
            raise ValueError(f"{self.name} Statistic error: data and groundtruth do not have the same shape")
        
        if self.dist == "pois":
            ll = np.log(scipy.stats.poisson.pmf(round(gt_data), model_data))
        elif self.dist == "norm":
            ll = np.log(scipy.stats.norm.pdf(gt_data, loc=model_data, scale=param[0]))
        elif self.dist == "nbinom":
            ll = np.log(scipy.stats.nbinom.pmf(gt_data, n=param[0], p=model_data))
        else:
            raise ValueError("Invalid distribution specified, got {self.dist}")
        return ll