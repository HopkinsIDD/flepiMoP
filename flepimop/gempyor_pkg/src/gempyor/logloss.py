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

    def plot_gt(self, ax=None, subpop=None, statistic=None, subplot=False, filename=None, **kwargs):
        """Plots ground truth data.

        Args:
            ax (matplotlib.axes.Axes, optional): An existing axis to plot on.
                If None, a new figure and axis will be created.
            subpop (str, optional): The subpopulation to plot. If None, plots all subpopulations.
            statistic (str, optional): The statistic to plot. If None, plots all statistics.
            subplot (bool, optional): If True, creates a subplot for each subpopulation/statistic combination.
                Defaults to False (single plot with all lines). 
            filename (str, optional): If provided, saves the plot to the specified filename.
            **kwargs: Additional keyword arguments passed to the matplotlib plot function.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            if subplot:
                fig, axes = plt.subplots(len(self.gt["subpop"].unique()), len(self.gt.columns.drop("subpop")), figsize=(4*len(self.gt.columns.drop("subpop")), 3*len(self.gt["subpop"].unique())), dpi=250, sharex=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=250)

        if subpop is None:
            subpops = self.gt["subpop"].unique()
        else:
            subpops = [subpop]

        if statistic is None:
            statistics = self.gt.columns.drop("subpop")  # Assuming other columns are statistics
        else:
            statistics = [statistic]

        if subplot:
            # One subplot for each subpop/statistic combination
            for i, subpop in enumerate(subpops):
                for j, stat in enumerate(statistics):
                    data_to_plot = self.gt[(self.gt["subpop"] == subpop)][stat].sort_index()
                    axes[i, j].plot(data_to_plot, **kwargs)
                    axes[i, j].set_title(f"{subpop} - {stat}")
        else:
            # All lines in a single plot
            for subpop in subpops:
                for stat in statistics:
                    data_to_plot = self.gt[(self.gt["subpop"] == subpop)][stat].sort_index()
                    data_to_plot.plot(ax=ax, **kwargs, label=f"{subpop} - {stat}")
            if len(statistics) > 1:
                ax.legend()
        
        if filename:
            if subplot:
                fig.tight_layout()  # Adjust layout for saving if using subplots
            plt.savefig(filename, **kwargs)  # Save the figure

        if subplot:
            return fig, axes  # Return figure and subplots for potential further customization
        else:
            return ax  # Optionally return the axis


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
    
    def __str__(self) -> str:
        return f"LogLoss: {len(self.statistics)} statistics and {len(self.gt)} data points," \
               f"number of NA for each statistic: \n{self.gt.drop('subpop', axis=1).isna().sum()}"


