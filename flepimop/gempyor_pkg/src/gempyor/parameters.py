"""
Provides abstractions for interacting with the parameters configurations.

Classes:
    Parameters:
        Encapsulates logic for loading, parsing, and summarizing parameter configurations.
"""

__all__ = ["Parameters"]


from collections.abc import Callable
import copy
import datetime
from inspect import signature
import logging
import os
from typing import Any, Literal

import confuse
import numpy as np
from numpy import ndarray
import numpy.typing as npt
import pandas as pd
import pydantic

from . import NPI, utils
from .distributions import distribution_from_confuse_config


logger = logging.getLogger(__name__)


class Parameters:
    """
    Encapsulates logic for loading, parsing, and summarizing parameter configurations.

    Attributes:
        npar: The number of parameters contained within the given configuration.
        pconfig: A view subsetting to the parameters section of a given config file.
        pdata: A dictionary containing a processed and reformatted view of the `pconfig`
            attribute.
        pnames: The names of the parameters given.
        pnames2pindex: A mapping parameter names to their location in the `pnames`
            attribute.
        stacked_modifier_method: A mapping of modifier method to the parameters to which
            that modifier method is relevant for.
    """

    def __init__(
        self,
        parameter_config: confuse.ConfigView,
        *,
        ti: datetime.date,
        tf: datetime.date,
        subpop_names: list,
        path_prefix: str = ".",
    ):
        """
        Initialize a `Parameters` instance from a parameter config view.

        Encapsulates logic for loading, parsing, and summarizing parameter configurations.
        Parameters can be defined or drawn from distributions.

        Args:
            parameter_config: A view of the overall configuration object focused on the parameters
                section of a given config file.
            ti: An initial date for simulation.
            tf: A final date for simulation.
            subpop_names: A list of subpopulation names.
            path_prefix: A file path prefix to directory containing parameter values.

        Raises:
            ValueError: The parameter names for the SEIR model are not unique.
            ValueError: The dataframe file found for a given parameter contains an
                insufficient number of columns for the subpopulations being considered.
            ValueError: The dataframe file found for a given parameter does not have
                enough date entries to cover the time span being considered by the given
                `ti` and `tf`.
        """
        self.pconfig: confuse.ConfigView = parameter_config
        self.pnames: list[str] = []
        self.npar: int = len(self.pnames)

        self.pdata: dict[str, dict[str, Any]] = {}
        self.pnames2pindex: dict[str, int] = {}
        self.stacked_modifier_method: dict[
            Literal["sum", "product", "reduction_product"], list[str]
        ] = {"sum": [], "product": [], "reduction_product": []}

        self.pnames = self.pconfig.keys()
        self.npar = len(self.pnames)
        if self.npar != len(set([name.lower() for name in self.pnames])):
            raise ValueError(
                "Parameters of the SEIR model have the same "
                "name (remember that case is not sufficient!)"
            )

        # Attributes of dictionary
        for idx, pn in enumerate(self.pnames):
            self.pnames2pindex[pn] = idx
            self.pdata[pn] = {}
            self.pdata[pn]["idx"] = idx

            # Parameter characterized by its distribution
            if self.pconfig[pn]["value"].exists():
                self.pdata[pn]["dist"] = distribution_from_confuse_config(
                    self.pconfig[pn]["value"]
                )

            # Parameter given as a file
            elif self.pconfig[pn]["timeseries"].exists():
                fn_name = os.path.join(path_prefix, self.pconfig[pn]["timeseries"].get())
                df = utils.read_df(fn_name).set_index("date")
                df.index = pd.to_datetime(df.index)
                if len(df.columns) == 1:  # if only one ts, assume it applies to all subpops
                    df = pd.DataFrame(
                        pd.concat([df] * len(subpop_names), axis=1).values,
                        index=df.index,
                        columns=subpop_names,
                    )
                elif len(df.columns) >= len(subpop_names):  # one ts per subpop
                    df = df[
                        subpop_names
                    ]  # make sure the order of subpops is the same as the reference
                    # (subpop_names from spatial setup) and select the columns
                else:
                    print("loaded col :", sorted(list(df.columns)))
                    print("geodata col:", sorted(subpop_names))
                    raise ValueError(
                        f"Issue loading file '{fn_name}' for parameter '{pn}': "
                        f"the number of non-'date' columns is '{len(df.columns)}', "
                        f"expected '{len(subpop_names)}' (number of subpopulations) or one."
                    )

                df = df[str(ti) : str(tf)]
                if not (len(df.index) == len(pd.date_range(ti, tf))):
                    print("config dates:", pd.date_range(ti, tf))
                    print("loaded dates:", df.index)
                    raise ValueError(
                        f"Issue loading file '{fn_name}' for parameter '{pn}': "
                        f"Provided file dates span '{str(df.index[0])}' to "
                        f"'{str(df.index[-1])}', but the config dates "
                        f"span '{ti}' to '{tf}'."
                    )
                if not (pd.date_range(ti, tf) == df.index).all():
                    print("config dates:", pd.date_range(ti, tf))
                    print("loaded dates:", df.index)
                    raise ValueError(
                        f"Issue loading file '{fn_name}' for parameter '{pn}': "
                        f"Provided file dates span '{str(df.index[0])}' to "
                        f"'{str(df.index[-1])}', but the config dates "
                        f"span '{ti}' to '{tf}'."
                    )

                self.pdata[pn]["ts"] = df
            if self.pconfig[pn]["stacked_modifier_method"].exists():
                self.pdata[pn]["stacked_modifier_method"] = self.pconfig[pn][
                    "stacked_modifier_method"
                ].as_str()
            else:
                self.pdata[pn]["stacked_modifier_method"] = "product"
                logging.debug(
                    f"No 'stacked_modifier_method' for parameter {pn}, "
                    "assuming multiplicative NPIs."
                )

            if self.pconfig[pn]["rolling_mean_windows"].exists():
                self.pdata[pn]["rolling_mean_windows"] = self.pconfig[pn][
                    "rolling_mean_windows"
                ].get()

            self.stacked_modifier_method[self.pdata[pn]["stacked_modifier_method"]].append(
                pn.lower()
            )

        logging.debug(f"We have {self.npar} parameter: {self.pnames}")
        logging.debug(f"Data to sample is: {self.pdata}")
        logging.debug(f"Index in arrays are: {self.pnames2pindex}")
        logging.debug(f"NPI overlap operation is {self.stacked_modifier_method} ")

    def reinitialize_distributions(self) -> None:
        """
        Reinitialize all random distributions.

        This method reinitializes all random distributions for the parameters contained
        within this class. The random seed for each distribution is captured on
        initialization so this method will change those seeds.
        """
        for pn in self.pnames:
            if "dist" in self.pdata[pn]:
                self.pdata[pn]["dist"] = distribution_from_confuse_config(
                    self.pconfig[pn]["value"]
                )

    def get_pnames2pindex(self) -> dict:
        """
        Read the `pnames2pindex` attribute.

        This redundant method returns the `pnames2pindex` attribute of this class.

        Returns:
            A mapping parameter names to their location in the `pnames` attribute.
        """
        return self.pnames2pindex

    def parameters_quick_draw(self, n_days: int, nsubpops: int) -> ndarray:
        """
        Format all parameters as a numpy array including sampling.

        The entries in the output array are filled based on the input given in the
        parameters section of a yaml config file. If the given parameter is pulled from
        a distribution rather than fixed the values will be pulled from that
        distribution. If an appropriate value cannot be found for an entry then a
        `np.nan` is returned.

        Args:
            n_days: The number of days to generate an array for.
            nsubpops: The number of subpopulations to generate an array for.

        Returns:
            A numpy array of size (`npar`, `n_days`, `nsubpops`) where `npar`
            corresponds to the `npar` attribute of this class.

        Notes:
            If any of the parameters are 'timeseries' type parameters then `n_days` and
            `nsubpops` must be equal to the number of days between `ti` and `tf` given
            when initializing this class and the number of subpopulations given to this
            class via `subpop_names`.
        """
        param_arr = np.empty((self.npar, n_days, nsubpops), dtype="float64")
        param_arr[:] = np.nan  # fill with NaNs so we don't fail silently

        for idx, pn in enumerate(self.pnames):
            if "dist" in self.pdata[pn]:
                param_arr[idx] = np.full((n_days, nsubpops), self.pdata[pn]["dist"]())
            else:
                param_arr[idx] = self.pdata[pn]["ts"].values

        # we don't store it as a member because this object needs to be small to be pickable
        return param_arr

    def parameters_load(
        self, param_df: pd.DataFrame, n_days: int, nsubpops: int
    ) -> ndarray:
        """
        Format all parameters as a numpy array including sampling and overrides.

        This method serves largely the same purpose as the `parameters_quick_draw`, but
        has the ability to override the parameter specifications contained by this class
        with a given dataframe.

        Args:
            param_df: A DataFrame containing the columns 'parameter' and 'value'. If
                more than one entry for a given parameter is given then only the first
                value will be taken.
            n_days: The number of days to generate an array for.
            nsubpops: The number of subpopulations to generate an array for.

        Returns:
            A numpy array of size (`npar`, `n_days`, `nsubpops`) where `npar`
            corresponds to the `npar` attribute of this class.

        Notes:
            If any of the parameters are 'timeseries' type parameters and are not being
            overridden then `n_days` and `nsubpops` must be equal to the number of days
            between `ti` and `tf` given when initializing this class and the number of
            subpopulations given to this class via `subpop_names`.
        """
        param_arr = np.empty((self.npar, n_days, nsubpops), dtype="float64")
        param_arr[:] = np.nan  # fill with NaNs so we don't fail silently

        for idx, pn in enumerate(self.pnames):
            if pn in param_df["parameter"].values:
                pval = float(param_df[param_df["parameter"] == pn]["value"].iloc[0])
                param_arr[idx] = np.full((n_days, nsubpops), pval)
            elif "ts" in self.pdata[pn]:
                param_arr[idx] = self.pdata[pn]["ts"].values
            else:
                print(
                    f"PARAM: parameter {pn} NOT found in loadID file. "
                    "Drawing from config distribution"
                )
                param_arr[idx] = np.full((n_days, nsubpops), self.pdata[pn]["dist"]())

        return param_arr

    def getParameterDF(self, p_draw: ndarray) -> pd.DataFrame:
        """
        Serialize a parameter draw as a pandas `DataFrame`.

        This method only considers distribution parameters, which does include fixed
        parameters.

        Args:
            p_draw: A numpy array of shape (`npar`, `n_days`, `nsubpops`) like that
                returned by `parameters_quick_draw`.

        Returns:
            A pandas `DataFrame` with the columns 'parameter' and 'value' corresponding
            to the parameter name and value as well as an index containing the parameter
            name.
        """
        # we don't write to disk time series parameters.
        out_df = pd.DataFrame(
            [
                p_draw[idx, 0, 0]
                for idx, pn in enumerate(self.pnames)
                if "dist" in self.pdata[pn]
            ],
            columns=["value"],
            index=[pn for idx, pn in enumerate(self.pnames) if "dist" in self.pdata[pn]],
        )
        out_df["parameter"] = out_df.index
        return out_df

    def parameters_reduce(self, p_draw: ndarray, npi: object) -> ndarray:
        """
        Params reduced according to the NPI provided.

        Args:
            p_draw: A numpy array of shape (`npar`, `n_days`, `nsubpops`) like that
                returned by `parameters_quick_draw`.
            npi: An NPI object describing the parameter reduction to perform.

        Returns:
            An array the same shape as `p_draw` with the prescribed reductions
            performed.
        """
        p_reduced = copy.deepcopy(p_draw)
        if npi is not None:
            for idx, pn in enumerate(self.pnames):
                npi_val = NPI.reduce_parameter(
                    parameter=p_draw[idx],
                    modification=npi.getReduction(pn.lower()),
                    method=self.pdata[pn]["stacked_modifier_method"],
                )
                p_reduced[idx] = npi_val
                if "rolling_mean_windows" in self.pdata[pn]:
                    p_reduced[idx] = utils.rolling_mean_pad(
                        data=npi_val, window=self.pdata[pn]["rolling_mean_windows"]
                    )

        return p_reduced


def _inspect_requested_parameters(
    f: Callable[..., Any],
    ignore_args: int,
    pdata: dict[str, dict[str, Any]],
    p_draw: npt.NDArray[np.float64],
) -> list[float | int | npt.NDArray[np.float64 | np.int64]]:
    """
    Inspect a function for requested parameters.

    Args:
        f: The function to inspect the arguments of.
        ignore_args: The number of initial arguments to ignore.
        pdata: A dictionary of parameter data, like the `pdata` attribute of
            :obj:`gempyor.parameters.Parameters`.
        p_draw: A numpy array of parameter draws to extract values from.

    Returns:
        A list of parameter values extracted from `p_draw` that can be passed to the
        given function `f` after ignoring the first `ignore_args` arguments. The
        values are extracted based on the parameter data in `pdata`.
        * 'dist' parameters are expected to either be single values or 1D arrays if
            they vary along the subpopulation dimension.
        * 'ts' parameters are expected to be time series data and will be returned as
            a numpy array.

    Raises:
        ValueError: If the function does not have enough arguments to ignore.
        ValueError: If a requested parameter is not found in `pdata`.
        NotImplementedError: If a requested parameter is not supported, only 'dist' or
            'ts' parameters are currently supported.

    Examples:
        >>> from pprint import pprint
        >>> import numpy as np
        >>> from gempyor.parameters import _inspect_requested_parameters
        >>> def example_function(a, b, c, d):
        ...     return (a + b) * c * d
        >>> pdata = {
        ...     'a': {'idx': 0, 'dist': True},
        ...     'b': {'idx': 1, 'dist': True},
        ...     'c': {'idx': 2, 'ts': True},
        ...     'd': {'idx': 3, 'dist': True},
        ... }
        >>> rng = np.random.default_rng(123)
        >>> p_draw = np.stack([
        ...     0.1 * np.ones((3, 2)),
        ...     1.5 * np.ones((3, 2)),
        ...     rng.uniform(size=(3, 2)),
        ...     rng.normal(size=2) * np.ones((3, 2)),
        ... ])
        >>> p_draw.shape
        (4, 3, 2)
        >>> args = _inspect_requested_parameters(example_function, 1, pdata, p_draw)
        >>> pprint(args)
        [1.5,
         array([[0.68235186, 0.05382102],
               [0.22035987, 0.18437181],
               [0.1759059 , 0.81209451]]),
         array([-0.63646365,  0.54195222])]
        >>> example_function(1.0, *args)
        array([[-1.08573039,  0.07292105],
               [-0.35062762,  0.24980178],
               [-0.27989428,  1.10029105]])

    """
    sig = signature(f)
    arg_names = list(sig.parameters.keys())
    if len(arg_names) < ignore_args:
        msg = (
            f"Function '{f.__name__}' does not have enough arguments to ignore "
            f"the first {ignore_args} arguments. It has {len(arg_names)} arguments "
            f"instead. The arguments are: {arg_names}."
        )
        raise ValueError(msg)
    if not (parameter_args := arg_names[ignore_args:]):
        return []
    args = []
    for param_name in parameter_args:
        if (param_data := pdata.get(param_name)) is None:
            msg = (
                f"The requested parameter, '{param_name}', not "
                f"found in the arguments of {f.__name__}. The "
                f"available parameters are: {pdata.keys()}."
            )
            raise ValueError(msg)
        if "dist" in param_data:
            p_draw_row = p_draw[param_data["idx"], 0, :]
            p_draw_item = p_draw_row[0].item()
            if np.allclose(p_draw_row, p_draw_item):
                args.append(p_draw_item)
            else:
                args.append(p_draw_row)
            continue
        if "ts" in param_data:
            args.append(p_draw[param_data["idx"], :, :])
            continue
        msg = (
            f"Parameter '{param_name}' in function '{f.__name__}' is not supported. "
            "Only parameters with 'dist' or 'ts' in their data are currently "
            f"supported. Instead has the following data: {param_data.keys()}."
        )
        raise NotImplementedError(msg)
    return args
