"""
Abstractions for interacting with the parameters configurations.

This module contains abstractions for interacting with the parameters section of given
config files. Namely it contains the `Parameters` class.
"""

__all__ = ["Parameters"]


import copy
import datetime
import logging
import os

import confuse
import numpy as np
from numpy import ndarray
import pandas as pd

from . import NPI, utils


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
        pnames2index: A mapping parameter names to their location in the `pnames` 
            attribute.
        stacked_modifier_method: A mapping of modifier method to the parameters to which
            that modifier method is relevant for.
    """
    
    # Minimal object to be easily picklable for // runs
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
        
        Args:
            parameter_config: A view subsetting to the parameters section of a given
                config file.
            ti: An initial date.
            tf: A final date.
            subpop_names: A list of subpopulation names.
            path_prefix: A file path prefix to use when reading in parameter values from
                a dataframe like file.
        
        Raises:
            ValueError: The parameter names for the SEIR model are not unique.
            ValueError: The dataframe file found for a given parameter contains an
                insufficient number of columns for the subpopulations being considered.
            ValueError: The dataframe file found for a given parameter does not have
                enough date entries to cover the time span being considered by the given
                `ti` and `tf`.
        """
        self.pconfig = parameter_config
        self.pnames = []
        self.npar = len(self.pnames)

        self.pdata = {}
        self.pnames2pindex = {}
        self.stacked_modifier_method = {"sum": [], "product": [], "reduction_product": []}

        self.pnames = self.pconfig.keys()
        self.npar = len(self.pnames)
        if self.npar != len(set([name.lower() for name in self.pnames])):
            raise ValueError("Parameters of the SEIR model have the same name (remember that case is not sufficient!)")

        # Attributes of dictionary
        for idx, pn in enumerate(self.pnames):
            self.pnames2pindex[pn] = idx
            self.pdata[pn] = {}
            self.pdata[pn]["idx"] = idx

            # Parameter characterized by it's distribution
            if self.pconfig[pn]["value"].exists():
                self.pdata[pn]["dist"] = self.pconfig[pn]["value"].as_random_distribution()

            # Parameter given as a file
            elif self.pconfig[pn]["timeseries"].exists():
                fn_name = os.path.join(path_prefix, self.pconfig[pn]["timeseries"].get())
                df = utils.read_df(fn_name).set_index("date")
                df.index = pd.to_datetime(df.index)
                if len(df.columns) == 1:  # if only one ts, assume it applies to all subpops
                    df = pd.DataFrame(
                        pd.concat([df] * len(subpop_names), axis=1).values, index=df.index, columns=subpop_names
                    )
                elif len(df.columns) >= len(subpop_names):  # one ts per subpop
                    df = df[subpop_names]  # make sure the order of subpops is the same as the reference
                    # (subpop_names from spatial setup) and select the columns
                else:
                    print("loaded col :", sorted(list(df.columns)))
                    print("geodata col:", sorted(subpop_names))
                    raise ValueError(
                        f"""ERROR loading file {fn_name} for parameter {pn}: the number of non 'date'
                    columns are {len(df.columns)}, expected {len(subpop_names)} (the number of subpops) or one."""
                    )

                df = df[str(ti) : str(tf)]
                if not (len(df.index) == len(pd.date_range(ti, tf))):
                    print("config dates:", pd.date_range(ti, tf))
                    print("loaded dates:", df.index)
                    raise ValueError(
                        f"""ERROR loading file {fn_name} for parameter {pn}: 
                    the 'date' entries of the provided file do not include all the days specified to be modeled by 
                    the config. the provided file includes {len(df.index)} days between {str(df.index[0])} to {str(df.index[-1])}, 
                    while there are {len(pd.date_range(ti, tf))} days in the config time span of {ti}->{tf}. The file must contain entries for the
                    the exact start and end dates from the config. """
                    )
                if not (pd.date_range(ti, tf) == df.index).all():
                    print("config dates:", pd.date_range(ti, tf))
                    print("loaded dates:", df.index)
                    raise ValueError(
                        f"""ERROR loading file {fn_name} for parameter {pn}: 
                    the 'date' entries of the provided file do not include all the days specified to be modeled by 
                    the config. the provided file includes {len(df.index)} days between {str(df.index[0])} to {str(df.index[-1])}, 
                    while there are {len(pd.date_range(ti, tf))} days in the config time span of {ti}->{tf}. The file must contain entries for the
                    the exact start and end dates from the config. """
                    )

                self.pdata[pn]["ts"] = df
            if self.pconfig[pn]["stacked_modifier_method"].exists():
                self.pdata[pn]["stacked_modifier_method"] = self.pconfig[pn]["stacked_modifier_method"].as_str()
            else:
                self.pdata[pn]["stacked_modifier_method"] = "product"
                logging.debug(f"No 'stacked_modifier_method' for parameter {pn}, assuming multiplicative NPIs")

            if self.pconfig[pn]["rolling_mean_windows"].exists():
                self.pdata[pn]["rolling_mean_windows"] = self.pconfig[pn]["rolling_mean_windows"].get()

            self.stacked_modifier_method[self.pdata[pn]["stacked_modifier_method"]].append(pn.lower())

        logging.debug(f"We have {self.npar} parameter: {self.pnames}")
        logging.debug(f"Data to sample is: {self.pdata}")
        logging.debug(f"Index in arrays are: {self.pnames2pindex}")
        logging.debug(f"NPI overlap operation is {self.stacked_modifier_method} ")

    def picklable_lamda_alpha(self):
        """
        Read the `alpha_val` attribute.
        
        This defunct method returns the `alpha_val` attribute of this class which is
        never set by this class. If this method is called and the `alpha_val` attribute
        is not set an AttributeError will be raised.
        
        Returns:
            The `alpha_val` attribute.
        """
        return self.alpha_val

    def picklable_lamda_sigma(self):
        """
        Read the `sigma_val` attribute.
        
        This defunct method returns the `sigma_val` attribute of this class which is
        never set by this class. If this method is called and the `sigma_val` attribute
        is not set an AttributeError will be raised.
        
        Returns:
            The `sigma_val` attribute.
        """
        return self.sigma_val

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
        """
        param_arr = np.empty((self.npar, n_days, nsubpops), dtype="float64")
        param_arr[:] = np.nan  # fill with NaNs so we don't fail silently

        for idx, pn in enumerate(self.pnames):
            if "dist" in self.pdata[pn]:
                param_arr[idx] = np.full((n_days, nsubpops), self.pdata[pn]["dist"]())
            else:
                param_arr[idx] = self.pdata[pn]["ts"].values

        return param_arr  # we don't store it as a member because this object needs to be small to be pickable

    def parameters_load(self, param_df: pd.DataFrame, n_days: int, nsubpops: int) -> ndarray:
        """
        Format all parameters as a numpy array including sampling and overrides.
        
        This method serves largely the same purpose as the `parameters_quick_draw`, but
        has the ability to override the parameter specifications contained by this class
        with a given dataframe.
        
        Args:
            param_df: A dataframe containing the columns 'parameter' and 'value'. If 
                more than one entry for a given parameter is given then only the first 
                value will be taken.
            n_days: The number of days to generate an array for.
            nsubpops: The number of subpopulations to generate an array for.
        
        Returns:
            A numpy array of size (`npar`, `n_days`, `nsubpops`) where `npar` 
            corresponds to the `npar` attribute of this class.
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
                print(f"PARAM: parameter {pn} NOT found in loadID file. Drawing from config distribution")
                pval = self.pdata[pn]["dist"]()
                param_arr[idx] = np.full((n_days, nsubpops), pval)

        return param_arr

    def getParameterDF(self, p_draw: ndarray) -> pd.DataFrame:
        """
        Serialize a parameter draw as a pandas `DataFrame`.
        
        This method only considers distribution parameters and will pull the first 
        sample from the `p_draw` given.
        
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
            [p_draw[idx, 0, 0] for idx, pn in enumerate(self.pnames) if "dist" in self.pdata[pn]],
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
                    p_reduced[idx] = utils.rolling_mean_pad(data=npi_val, window=self.pdata[pn]["rolling_mean_windows"])

        return p_reduced
