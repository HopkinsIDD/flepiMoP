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

    def compute_logloss(self, model_data, gt_data, skip_resampling=False):
        if not skip_resampling:
            model_data = self.apply_resampling(model_data)
            gt_data = self.apply_resampling(gt_data)

        model_data = model_data[self.sim_var]
        gt_data = gt_data[self.data_var]

        assert model_data.shape == gt_data.shape

        if self.loss_function == "rmse":
            return -np.sqrt(np.mean((model_data - gt_data) ** 2))
        elif self.loss_function == "poisson_log_loss":
            epsilon = 1e-10  # to avoid log(0)
            return -np.mean(model_data - gt_data * np.log(model_data + epsilon))
        else:
            raise ValueError("Unsupported loss function")


class LogLoss:
    def __init__(self, inference_config: confuse.ConfigView, data_dir:str = "."):
        self.gt = pd.read_csv(f"{data_dir}/{inference_config['gt_data_path'].get()}")
        self.statistics = {}
        for key, value in inference_config["statistics"].items():
            self.statistics[key] = Statistic(key, value)

    def compute_logloss(self, model_df, modinf):
        """
        Compute logloss for all statistics
        model_df: DataFrame indexed by date
        modinf: model information
        TODO: support kwargs for emcee, and this looks very slow
        """
        logloss = xr.DataArray(0, dims=["statistic", "subpop"],  
                coords={
                "statistic":self.statistics.key(),
                "subpop":modinf.subpop_struct.subpop_names})

        for subpop in modinf.subpop_struct.subpop_names:
            # essential to sort by index (date here)
            gt_s = self.gt[self.gt["subpop"] == subpop].sort_index()
            model_df_s = model_df[model_df["subpop"] == subpop].sort_index()

            # Places where data and model overlap
            first_date = max(gt_s.index.min(), model_df_s.index.min())
            last_date = min(gt_s.index.max(), model_df_s.index.max())

            gt_s = gt_s.loc[first_date:last_date].drop(["subpop"], axis=1)
            model_df_s = model_df_s.drop(["subpop", "time"], axis=1).loc[first_date:last_date]

            for key, stat in self.statistics.items():
                logloss.loc[dict(statistics=key, subpop=subpop)] += stat.compute_logloss(model_df, gt_s)

        return logloss


