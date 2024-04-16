import xarray as xr
import pandas as pd
import numpy as np
import confuse
import scipy.stats
from . import statistics


## https://docs.xarray.dev/en/stable/user-guide/indexing.html#assigning-values-with-indexing

# TODO: add an autatic test that show that the loss is biggest when gt == modeldata
class LogLoss:
    def __init__(self, inference_config: confuse.ConfigView, data_dir:str = "."):
        self.gt = pd.read_csv(f"{data_dir}/{inference_config['gt_data_path'].get()}")
        self.gt["date"] = pd.to_datetime(self.gt['date'])
        self.gt = self.gt.set_index("date")
        self.statistics = {}
        for key, value in inference_config["statistics"].items():
            self.statistics[key] = statistics.Statistic(key, value)

    def plot_gt(self, ax, subpop, statistic, **kwargs):
        self.gt[self.gt["subpop"] == subpop].plot(y=statistic, ax=ax, **kwargs)


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
            model_df_s = model_df_s.drop(["subpop"], axis=1).loc[first_date:last_date]

            # TODO: add whole US!! option

            for key, stat in self.statistics.items():
                logloss.loc[dict(statistics=key, subpop=subpop)] += stat.compute_logloss(model_df, gt_s)

        return logloss


