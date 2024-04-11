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

        self.resample_config = None
        if statistic_config["resample"].exists():
            self.resample_config = statistic_config["resample"].get()
        
        self.regularization_config = None
        if statistic_config["regularization"].exists():
            self.regularization_config = statistic_config["regularization"].get()
    
        self.loss_function = statistic_config["loss"].get()

    def __str__(self) -> str:
        return f"{self.name}: {self.loss_function} between {self.sim_var} (sim) and {self.data_var} (data)."
    
    def __repr__(self) -> str:
        return f"A Statistic(): {self.__str()}"

    def apply_resampling(self, data):
        if self.resample_config:
            freq = ""
            freq = self.resample_config["freq"].get("freq", "W-SAT")
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
        elif self.loss_function == "poisson":
            epsilon = 1e-10  # to avoid log(0)
            return -np.mean(model_data - gt_data * np.log(model_data + epsilon))
        else:
            raise ValueError("Unsupported loss function")

# TODO: add an autatic test that show that the loss is biggest when gt == modeldata

class LogLoss:
    def __init__(self, inference_config: confuse.ConfigView, data_dir:str = "."):
        self.gt = pd.read_csv(f"{data_dir}/{inference_config['gt_data_path'].get()}")
        self.gt["date"] = pd.to_datetime(self.gt['date'])
        self.gt = self.gt.set_index("date")
        self.statistics = {}
        for key, value in inference_config["statistics"].items():
            self.statistics[key] = Statistic(key, value)

    def plot_gt(self, ax):
        ax.plot(self.gt)

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

            # TODO: add whole US!! option

            for key, stat in self.statistics.items():
                logloss.loc[dict(statistics=key, subpop=subpop)] += stat.compute_logloss(model_df, gt_s)

        return logloss


