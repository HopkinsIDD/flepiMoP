import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import copy
import confuse
from numpy import ndarray
import logging
from . import model_info, NPI, utils
import datetime
import xarray as xr
import os

logger = logging.getLogger(__name__)

# TODO: it should work like
# import xarray as xr
# xr.DataArray(p_draw, dims=["parameter", "date", "subpop"],
#                 coords={"parameter":modinf.parameters.pnames,
#                 "date":pd.date_range(modinf.ti, modinf.tf, freq="D"),
#                 "subpop":modinf.subpop_struct.subpop_names}).sel(parameter="gamma_0").plot()


## TODO: ideally here path_prefix should not be used and all files loaded from modinf


class Parameters:
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
        """These two functions were lambda in __init__ before, it was more elegant. but as the object needs to be pickable,
        we cannot use second order function, hence these ugly definitions"""
        return self.alpha_val

    def picklable_lamda_sigma(self):
        return self.sigma_val

    def get_pnames2pindex(self) -> dict:
        return self.pnames2pindex

    def parameters_quick_draw(self, n_days: int, nsubpops: int) -> ndarray:
        """
        Returns all parameter in an array. These are drawn based on the seir::parameters section of the config, passed in as p_config.
        :param n_days: number of time interval
        :param nsubpops: number of spatial nodes
        :return:  array of shape (nparam, n_days, nsubpops) with all parameters for all nodes and all time (same value)
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
        drop-in equivalent to param_quick_draw() that take a file as written parameter_write()
        :param fname:
        :param n_days:
        :param nsubpops:
        :param extension:
        :return: array of shape (nparam, n_days, nsubpops) with all parameters for all nodes and all time.
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
        return parameters generated by parameters_quick_draw() as dataframe, just the first value as they are all similar.
        :param p_draw:
        :return: The dataframe (to be written to disk, or not)
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
        :param p_draw: array of shape (nparam, n_days, nsubpops) from p_draw
        :param npi: NPI object with the reduction
        :return: array of shape (nparam, n_days, nsubpops) with all parameters for all nodes and all time, reduced
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
