import xarray as xr
import pandas as pd
import numpy as np
import confuse

## https://docs.xarray.dev/en/stable/user-guide/indexing.html#assigning-values-with-indexing
##


class Statistic:
    def __init__(self, name, statistic_config: confuse.ConfigView):
        self.sim_var = statistic_config["sim_var"].as_str()
        self.data_var = statistic_config["data_var"].as_str()
        self.name = name
        self.resample_config = statistic_config.get("resample_config", None)
        self.loss_function = statistic_config.get("loss_function", "rmse")

    def apply_resampling(self, data):
        if self.resample_config:
            freq = self.resample_config.get("freq", "W-SAT")
            agg_func = getattr(pd.Series, self.resample_config.get("agg_func", "sum"))
            skipna = self.resample_config.get("skipna", False)

            data_resampled = data.resample(freq).agg(agg_func, skipna=skipna)
            return data_resampled
        else:
            return data

    def compute_loss(self, model_data, gt_data):
        if self.loss_function == "rmse":
            return np.sqrt(np.mean((model_data - gt_data) ** 2))
        elif self.loss_function == "poisson_log_loss":
            epsilon = 1e-10  # to avoid log(0)
            return -np.mean(model_data - gt_data * np.log(model_data + epsilon))
        else:
            raise ValueError("Unsupported loss function")


class LogLoss:
    def __init__(self, inference_config: confuse.ConfigView, data_dir:str = "."):
        self.gt = pd.read_csv(f"{data_dir}/"inference_config["gt_data_path"].get())
        


def log_loss(model_df, gt, modinf, statistics):
    """
        model_df, gt: dataframe indexed by date
        TODO: support kwargs for emcee, and this looks very slow
    """
    
    log_loss = xr.DataArray(0, dims=["statistic", "subpop"],  
                coords={
                "statistic":statistics.key(),
                "subpop":modinf.subpop_struct.subpop_names})

    for subpop in modinf.subpop_struct.subpop_names:    
        # essential to sort by index (date here)
        gt_s = gt[gt["subpop"]==subpop].sort_index()
        model_df_s = model_df[model_df["subpop"]==subpop].sort_index()

        # places where data and model overlap
        first_date = max(gt_s.index.min(), model_df_s.index.min())
        last_date = min(gt_s.index.max(), model_df_s.index.max())

        gt_s = gt_s.loc[first_date:last_date].drop(["subpop"],axis=1).resample("W-SAT").agg(pd.Series.sum, skipna=False) # if one NA in the interval, skip the itnerval  (see https://stackoverflow.com/questions/54252106/while-resampling-put-nan-in-the-resulting-value-if-there-are-some-nan-values-in)
        model_df_s = model_df_s.drop(["subpop","time"],axis=1).loc[first_date:last_date].resample("W-SAT").agg(pd.Series.sum, skipna=False) # todo sub subpop here
        
        for key, value in statistics.items():
            assert model_df_s[key].shape == gt_s[value].shape

            log_loss.loc[dict(statistics=key, subpop=subpop)] += np.nansum((model_df_s[key]-gt_s[value])**2)

    return -log_loss,