import pandas as pd
import numpy as np
from . import helpers
from .base import NPIBase


class MultiPeriodModifier(NPIBase):
    def __init__(
        self,
        *,
        npi_config,
        global_config,
        subpops,
        loaded_df=None,
        pnames_overlap_operation_sum=[],
        sanitize=False,
    ):
        super().__init__(
            name=getattr(
                npi_config,
                "key",
                (npi_config["scenario"].exists() and npi_config["scenario"].get()) or "unknown",
            )
        )

        self.sanitize = sanitize
        self.start_date = global_config["start_date"].as_date()
        self.end_date = global_config["end_date"].as_date()

        self.subpops = subpops

        self.npi = pd.DataFrame(
            0.0,
            index=self.subpops,
            columns=pd.date_range(self.start_date, self.end_date),
        )

        self.parameters = pd.DataFrame(
            data={
                "npi_name": [""] * len(self.subpops),
                "parameter": [""] * len(self.subpops),
                "start_date": [[self.start_date]] * len(self.subpops),
                "end_date": [[self.end_date]] * len(self.subpops),
                "reduction": [0.0] * len(self.subpops),
            },
            index=self.subpops,
        )

        self.param_name = npi_config["parameter"].as_str().lower()

        if (loaded_df is not None) and self.name in loaded_df["npi_name"].values:
            self.__createFromDf(loaded_df, npi_config)
        else:
            self.__createFromConfig(npi_config)

        # if parameters are exceeding global start/end dates, index of parameter df will be out of range so check first
        if self.sanitize:
            too_early = min([min(i) for i in self.parameters["start_date"]]) < self.start_date
            too_late = max([max(i) for i in self.parameters["end_date"]]) > self.end_date
            if too_early or too_late:
                raise ValueError("at least one period start or end date is not between global dates")

        for grp_config in npi_config["groups"]:
            affected_subpops_grp = self.__get_affected_subpops_grp(grp_config)
            for sub_index in range(len(self.parameters["start_date"][affected_subpops_grp[0]])):
                period_range = pd.date_range(
                    self.parameters["start_date"][affected_subpops_grp[0]][sub_index],
                    self.parameters["end_date"][affected_subpops_grp[0]][sub_index],
                )
                self.npi.loc[affected_subpops_grp, period_range] = np.tile(
                    self.parameters["reduction"][affected_subpops_grp],
                    (len(period_range), 1),
                ).T

        # for index in self.parameters.index:
        #    for sub_index in range(len(self.parameters["start_date"][index])):
        #        period_range = pd.date_range(self.parameters["start_date"][index][sub_index], self.parameters["end_date"][index][sub_index])
        #        ## This the line that does the work
        #        self.npi_old.loc[index, period_range] = np.tile(self.parameters["reduction"][index], (len(period_range), 1)).T
        # print(f'{self.name}, : {(self.npi_old == self.npi).all().all()}')

        # self.__checkErrors()

    def __checkErrors(self):
        if not self.sanitize:
            return
        min_start_date = min([min(i) for i in self.parameters["start_date"]])
        max_start_date = max([max(i) for i in self.parameters["start_date"]])
        min_end_date = min([min(i) for i in self.parameters["end_date"]])
        max_end_date = max([max(i) for i in self.parameters["end_date"]])
        if not ((self.start_date <= min_start_date) & (max_start_date <= self.end_date)):
            raise ValueError(
                f"at least one period_start_date [{min_start_date}, {max_start_date}] is not between global dates [{self.start_date}, {self.end_date}]"
            )
        if not ((self.start_date <= min_end_date) & (max_end_date <= self.end_date)):
            raise ValueError(
                f"at least one period_end_date ([{min_end_date}, {max_end_date}] is not between global dates [{self.start_date}, {self.end_date}]"
            )

        if not (self.parameters["start_date"] <= self.parameters["end_date"]).all():
            raise ValueError(f"at least one period_start_date is greater than the corresponding period end date")

        for n in self.affected_subpops:
            if n not in self.subpops:
                raise ValueError(f"Invalid config value {n} not in subpops")

        ### if self.param_name not in REDUCE_PARAMS:
        ###     raise ValueError(f"Invalid parameter name: {self.param_name}. Must be one of {REDUCE_PARAMS}")

        # Validate
        if (self.npi == 0).all(axis=None):
            print(f"Warning: The intervention in config: {self.name} does nothing.")

        if (self.npi > 1).any(axis=None):
            raise ValueError(
                f"The intervention in config: {self.name} has reduction of {self.param_name} is greater than 1"
            )

    def __createFromConfig(self, npi_config):
        # Get name of the parameter to reduce
        self.param_name = npi_config["parameter"].as_str().lower().replace(" ", "")

        self.affected_subpops = self.__get_affected_subpops(npi_config)

        self.parameters = self.parameters[self.parameters.index.isin(self.affected_subpops)]
        dist = npi_config["value"].as_random_distribution()
        self.parameters["npi_name"] = self.name
        self.parameters["parameter"] = self.param_name

        self.spatial_groups = []
        for grp_config in npi_config["groups"]:
            affected_subpops_grp = self.__get_affected_subpops_grp(grp_config)
            # Create reduction
            start_dates = []
            end_dates = []
            if grp_config["periods"].exists():
                for period in grp_config["periods"]:
                    start_dates = start_dates + [period["start_date"].as_date()]
                    end_dates = end_dates + [period["end_date"].as_date()]
            else:
                start_dates = [self.start_date]
                end_dates = [self.end_date]
            this_spatial_group = helpers.get_spatial_groups(grp_config, affected_subpops_grp)
            self.spatial_groups.append(this_spatial_group)
            # print(self.name, this_spatial_groups)

            # unfortunately, we cannot use .loc here, because it is not possible to assign a list of list
            # to a subset of a dataframe... so we iterate.
            for subpop in this_spatial_group["ungrouped"]:
                self.parameters.at[subpop, "start_date"] = start_dates
                self.parameters.at[subpop, "end_date"] = end_dates
                self.parameters.at[subpop, "reduction"] = dist(size=1)
            for group in this_spatial_group["grouped"]:
                drawn_value = dist(size=1)
                for subpop in group:
                    self.parameters.at[subpop, "start_date"] = start_dates
                    self.parameters.at[subpop, "end_date"] = end_dates
                    self.parameters.at[subpop, "reduction"] = drawn_value

    def __get_affected_subpops_grp(self, grp_config):
        if grp_config["subpop"].get() == "all":
            affected_subpops_grp = self.subpops
        else:
            affected_subpops_grp = [str(n.get()) for n in grp_config["subpop"]]
        return affected_subpops_grp

    def __createFromDf(self, loaded_df, npi_config):
        loaded_df.index = loaded_df.subpop
        loaded_df = loaded_df[loaded_df["npi_name"] == self.name]
        self.affected_subpops = self.__get_affected_subpops(npi_config)

        self.parameters = self.parameters[self.parameters.index.isin(self.affected_subpops)]
        self.parameters["npi_name"] = self.name
        self.parameters["parameter"] = self.param_name

        # self.parameters = loaded_df[["npi_name", "start_date", "end_date", "parameter", "reduction"]].copy()
        # self.parameters["start_date"] = [[datetime.date.fromisoformat(date) for date in strdate.split(",")] for strdate in self.parameters["start_date"]]
        # self.parameters["end_date"] =   [[datetime.date.fromisoformat(date) for date in strdate.split(",")] for strdate in self.parameters["end_date"]]
        # self.affected_subpops = set(self.parameters.index)

        if self.sanitize:
            if len(self.affected_subpops) != len(self.parameters):
                print(f"loading {self.name} and we got {len(self.parameters)} subpops")
                print(f"getting from config that it affects {len(self.affected_subpops)}")

        self.spatial_groups = []
        for grp_config in npi_config["groups"]:
            affected_subpops_grp = self.__get_affected_subpops_grp(grp_config)
            # Create reduction
            start_dates = []
            end_dates = []
            if grp_config["periods"].exists():
                for period in grp_config["periods"]:
                    start_dates = start_dates + [period["start_date"].as_date()]
                    end_dates = end_dates + [period["end_date"].as_date()]
            else:
                start_dates = [self.start_date]
                end_dates = [self.end_date]
            this_spatial_group = helpers.get_spatial_groups(grp_config, affected_subpops_grp)
            self.spatial_groups.append(this_spatial_group)

            for subpop in this_spatial_group["ungrouped"]:
                if not subpop in loaded_df.index:
                    self.parameters.at[subpop, "start_date"] = start_dates
                    self.parameters.at[subpop, "end_date"] = end_dates
                    dist = npi_config["value"].as_random_distribution()
                    self.parameters.at[subpop, "reduction"] = dist(size=1)
                else:
                    self.parameters.at[subpop, "start_date"] = start_dates
                    self.parameters.at[subpop, "end_date"] = end_dates
                    self.parameters.at[subpop, "reduction"] = loaded_df.at[subpop, "reduction"]
            for group in this_spatial_group["grouped"]:
                if ",".join(group) in loaded_df.index:  # ordered, so it's ok
                    for subpop in group:
                        self.parameters.at[subpop, "start_date"] = start_dates
                        self.parameters.at[subpop, "end_date"] = end_dates
                        self.parameters.at[subpop, "reduction"] = loaded_df.at[",".join(group), "reduction"]
                else:
                    dist = npi_config["value"].as_random_distribution()
                    drawn_value = dist(size=1)
                    for subpop in group:
                        self.parameters.at[subpop, "start_date"] = start_dates
                        self.parameters.at[subpop, "end_date"] = end_dates
                        self.parameters.at[subpop, "reduction"] = drawn_value

        self.parameters = self.parameters.loc[list(self.affected_subpops)]
        # self.parameters = self.parameters[self.parameters.index.isin(self.affected_subpops) ]
        # self.parameters = self.parameters[self.affected_subpops]

        # parameter name is picked from config too: (before: )
        # self.param_name = self.parameters["parameter"].unique()[0]  # [0] to convert ndarray to str
        # now:
        self.param_name = npi_config["parameter"].as_str().lower().replace(" ", "")
        self.parameters["parameter"] = self.param_name

    def __get_affected_subpops(self, npi_config):
        # Optional config field "subpop"
        # If values of "subpop" is "all" or unspecified, run on all subpops.
        # Otherwise, run only on subpops specified.
        affected_subpops_grp = []
        for grp_config in npi_config["groups"]:
            if grp_config["subpop"].get() == "all":
                affected_subpops_grp = self.subpops
            else:
                affected_subpops_grp += [str(n.get()) for n in grp_config["subpop"]]
        affected_subpops = set(affected_subpops_grp)
        if len(affected_subpops) != len(affected_subpops_grp):
            raise ValueError(f"In NPI {self.name}, some subpops belong to several groups. This is unsupported.")
        return affected_subpops

    def getReduction(self, param, default=0.0):
        "Return the reduction for this param, `default` if no reduction defined"

        if param == self.param_name:
            return self.npi
        return default

    def getReductionToWrite(self):
        df_list = []
        # self.parameters.index is a list of subpops
        for this_spatial_groups in self.spatial_groups:
            # spatially ungrouped dataframe
            df_ungroup = self.parameters[self.parameters.index.isin(this_spatial_groups["ungrouped"])].copy()
            df_ungroup.index.name = "subpop"
            df_ungroup["start_date"] = df_ungroup["start_date"].apply(
                lambda l: ",".join([d.strftime("%Y-%m-%d") for d in l])
            )
            df_ungroup["end_date"] = df_ungroup["end_date"].apply(
                lambda l: ",".join([d.strftime("%Y-%m-%d") for d in l])
            )
            df_list.append(df_ungroup)
            # spatially grouped dataframe. They are nested within multitime reduce groups,
            # so we can set the same dates for allof them
            for group in this_spatial_groups["grouped"]:
                # we use the first subpop to represent the group
                df_group = self.parameters[self.parameters.index == group[0]].copy()

                row_group = pd.DataFrame.from_dict(
                    {
                        "subpop": ",".join(group),
                        "npi_name": df_group["npi_name"],
                        "parameter": df_group["parameter"],
                        "start_date": df_group["start_date"].apply(
                            lambda l: ",".join([d.strftime("%Y-%m-%d") for d in l])
                        ),
                        "end_date": df_group["end_date"].apply(lambda l: ",".join([d.strftime("%Y-%m-%d") for d in l])),
                        "reduction": df_group["reduction"],
                    }
                ).set_index("subpop")
                df_list.append(row_group)

        df = pd.concat(df_list)

        df = df.reset_index()
        return df
