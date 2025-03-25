"""
Construct and calculate custom logloss functions for a model.

This module contains the `LogLoss` class which is used to construct and calculate
custom logloss functions for a model. The class is initialized with ground truth data
and a configuration file containing the statistics to be used in the logloss
calculation. The `compute_logloss` method is used to calculate the logloss for a given
model and subpopulations.
"""

__all__ = ("LogLoss",)


import pathlib
import typing

import confuse
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import xarray as xr

from . import statistics
from .model_info import TimeSetup
from .subpopulation_structure import SubpopulationStructure
from .utils import read_df


class LogLoss:
    """
    Construct and calculate custom logloss functions for a model.

    The `LogLoss` class is used to construct and calculate custom logloss functions for
    a model. The class is initialized with ground truth data and a confuse configuration
    containing the statistics to be used in the logloss calculation.

    Attributes:
        gt: A DataFrame containing the ground truth data.
        gt_xr: An xarray dataset containing the ground truth data.
        first_date: The first date in the ground truth data.
        last_date: The last date in the ground truth data.
        statistics: A dictionary containing the statistics to be used in the logloss
            calculation.
    """

    def __init__(
        self,
        inference_config: confuse.ConfigView,
        subpop_struct: SubpopulationStructure,
        time_setup: TimeSetup,
        path_prefix: str = ".",
    ) -> None:
        """
        Initialize the `LogLoss` class.

        Args:
            inference_config: A configuration view containing the inference
                configuration.
            subpop_struct: A subpopulation structure object.
            time_setup: A time setup object.
            path_prefix: The path prefix for the ground truth data.
        """
        self.gt = read_df(
            pathlib.Path(path_prefix) / inference_config["gt_data_path"].get()
        )
        self.gt["date"] = pd.to_datetime(self.gt["date"])
        self.gt = self.gt.set_index("date")

        # made the controversial choice of storing the gt as an xarray dataset instead
        # of a dictionary of dataframes
        self.gt_xr = xr.Dataset.from_dataframe(
            self.gt.reset_index().set_index(["date", "subpop"])
        )

        # Very important: subsample the subpop in the population, in the right order,
        # and sort by the date index.
        self.gt_xr = self.gt_xr.sortby("date").reindex(
            {"subpop": subpop_struct.subpop_names}
        )

        # This will force at 0, if skipna is False, data of some variable that don't
        # exist if other exist, and damn python datetime types are ugly...
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
        self,
        ax: Axis | None = None,
        subpop: str | None = None,
        statistic: str | None = None,
        subplot: bool = False,
        filename: str | None = None,
        **kwargs: typing.Any,
    ) -> Axis | tuple[Figure, Axis]:
        """
        Plots ground truth data.

        Args:
            ax: An existing axis to plot on. If `None` a new figure and axis will be
                created.
            subpop: The subpopulation to plot. If `None` plots all subpopulations.
            statistic: The statistic to plot. If `None` plots all statistics.
            subplot: If `True` creates a subplot for each subpopulation/statistic
                combination otherwise plots all lines on a single plot.
            filename: If provided, saves the plot to the specified filename.
            **kwargs: Additional keyword arguments passed to the matplotlib plot
                function.

        Returns:
            If `subplot` is `True` returns a tuple of the figure and axes, otherwise
            returns the axis.
        """
        subpops = self.gt["subpop"].unique() if subpop is None else [subpop]
        stats = self.gt.columns.drop("subpop") if statistic is None else [statistic]

        if subplot:
            # One subplot for each subpop/statistic combination
            if ax is None:
                fig, ax = plt.subplots(
                    len(self.gt["subpop"].unique()),
                    len(self.gt.columns.drop("subpop")),
                    figsize=(
                        4 * len(self.gt.columns.drop("subpop")),
                        3 * len(self.gt["subpop"].unique()),
                    ),
                    dpi=250,
                    sharex=True,
                )
            for i, sp in enumerate(subpops):
                for j, stat in enumerate(stats):
                    data_to_plot = self.gt[(self.gt["subpop"] == sp)][stat].sort_index()
                    ax[i, j].plot(data_to_plot, **kwargs)
                    ax[i, j].set_title(f"{sp} - {stat}")
        else:
            # All lines in a single plot
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
            for sp in subpops:
                for stat in stats:
                    data_to_plot = self.gt[(self.gt["subpop"] == sp)][stat].sort_index()
                    data_to_plot.plot(ax=ax, **kwargs, label=f"{sp} - {stat}")
            if len(stats) > 1:
                ax.legend()

        if filename:
            if subplot:
                fig.tight_layout()
            plt.savefig(filename, **kwargs)

        if subplot:
            return (fig, ax)
        return ax

    def compute_logloss(
        self, model_df: pd.DataFrame, subpop_names: list[str]
    ) -> tuple[float, xr.DataArray, float]:
        """
        Compute logloss for all statistics.

        Args:
            model_df: DataFrame indexed by date
            subpop_names: list of subpop names

        Returns:
            A tuple of total logloss, logloss per statistic, and regularization term.

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
            logloss.loc[{"statistic": key}] = ll
            regularizations += reg

        ll_total = logloss.sum().sum().values + regularizations

        return ll_total, logloss, regularizations

    def __str__(self) -> str:
        return (
            f"Logloss of {len(self.statistics)} statistics "
            f"and {len(self.gt)} data points."
        )
