import xarray as xr
import pandas as pd
import numpy as np
import confuse
import scipy.stats
from . import statistics
import os


## https://docs.xarray.dev/en/stable/user-guide/indexing.html#assigning-values-with-indexing
# TODO: add an autatic test that show that the loss is biggest when gt == modeldata


# A lot of things can go wrong here, in the previous approach where GT was cast to xarray as
#  self.gt_xr = xr.Dataset.from_dataframe(self.gt.reset_index().set_index(["date","subpop"]))
# then some NA were created if some dates where present in some gt but no other.


class LogLoss:
    def __init__(
        self,
        inference_config: confuse.ConfigView,
        subpop_struct,
        time_setup,
        path_prefix: str = ".",
    ):
        # TODO: bad format for gt because each date must have a value for each column, but if it doesn't and you add NA
        # then this NA has a meaning that depends on skip NA, which is annoying.
        # A lot of things can go wrong here, in the previous approach where GT was cast to xarray as
        #  self.gt_xr = xr.Dataset.from_dataframe(self.gt.reset_index().set_index(["date","subpop"]))
        # then some NA were created if some dates where present in some gt but no other.
        # FIXME THIS IS FUNDAMENTALLY WRONG, especially as groundtruth resample by statistic !!!!

        self.gt = pd.read_csv(
            os.path.join(path_prefix, inference_config["gt_data_path"].get()),
            converters={"subpop": lambda x: str(x)},
            skipinitialspace=True,
        )  # TODO: use read_df
        self.gt["date"] = pd.to_datetime(self.gt["date"])
        self.gt = self.gt.set_index("date")

        # made the controversial choice of storing the gt as an xarray dataset instead of a dictionary
        # of dataframes
        self.gt_xr = xr.Dataset.from_dataframe(
            self.gt.reset_index().set_index(["date", "subpop"])
        )
        # Very important: subsample the subpop in the population, in the right order, and sort by the date index.
        self.gt_xr = self.gt_xr.sortby("date").reindex(
            {"subpop": subpop_struct.subpop_names}
        )

        # This will force at 0, if skipna is False, data of some variable that don't exist if iother exist
        # and damn python datetime types are ugly...
        self.first_date = max(
            pd.to_datetime(self.gt_xr.date[0].values).date(), time_setup.ti
        )
        self.last_date = min(
            pd.to_datetime(self.gt_xr.date[-1].values).date(), time_setup.tf
        )

        self.statistics = {}
        for key, value in inference_config["statistics"].items():
            self.statistics[key] = statistics.Statistic(key, value)

    def plot_gt(
        self, ax=None, subpop=None, statistic=None, subplot=False, filename=None, **kwargs
    ):
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
                fig, axes = plt.subplots(
                    len(self.gt["subpop"].unique()),
                    len(self.gt.columns.drop("subpop")),
                    figsize=(
                        4 * len(self.gt.columns.drop("subpop")),
                        3 * len(self.gt["subpop"].unique()),
                    ),
                    dpi=250,
                    sharex=True,
                )
            else:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=250)

        if subpop is None:
            subpops = self.gt["subpop"].unique()
        else:
            subpops = [subpop]

        if statistic is None:
            statistics = self.gt.columns.drop(
                "subpop"
            )  # Assuming other columns are statistics
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
            return (
                fig,
                axes,
            )  # Return figure and subplots for potential further customization
        else:
            return ax  # Optionally return the axis

    def compute_logloss(self, model_df, subpop_names):
        """
        Compute logloss for all statistics
        model_df: DataFrame indexed by date
        subpop_names: list of subpop names
        TODO: support kwargs for emcee, and this looks very slow
        """
        coords = {"statistic": list(self.statistics.keys()), "subpop": subpop_names}

        logloss = xr.DataArray(
            np.zeros((len(coords["statistic"]), len(coords["subpop"]))),
            dims=["statistic", "subpop"],
            coords=coords,
        )

        regularizations = 0

        model_xr = (
            xr.Dataset.from_dataframe(model_df.reset_index().set_index(["date", "subpop"]))
            .sortby("date")
            .reindex({"subpop": subpop_names})
        )

        for key, stat in self.statistics.items():
            ll, reg = stat.compute_logloss(
                model_xr.sel(date=slice(self.first_date, self.last_date)),
                self.gt_xr.sel(date=slice(self.first_date, self.last_date)),
            )
            logloss.loc[dict(statistic=key)] = ll
            regularizations += reg

        ll_total = logloss.sum().sum().values + regularizations

        return ll_total, logloss, regularizations

    def __str__(self) -> str:
        return (
            f"LogLoss: {len(self.statistics)} statistics and {len(self.gt)} data points,"
            f"number of NA for each statistic: \n{self.gt.drop('subpop', axis=1).isna().sum()}"
        )
