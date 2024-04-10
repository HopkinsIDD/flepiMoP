import xarray as xr
import pandas as pd
import numpy as np
import confuse


class InferenceParameter:
    def __init__(self, ptype, pname, subpop, pdist, lb, ub):
        self.ptype = ptype
        self.pname = pname
        self.subpop = subpop
        self.pdist = pdist
        self.lb = lb
        self.ub = ub
    # TODO: does not support the subpop groups here !!!!!!!
    def print_summary(self):
        print(f"{self.ptype}::{self.pname} in [{self.lb}, {self.ub}]"
              f"   >> affected subpop: {self.subpop}"
        )

class InferenceParameters:
    def __init__(self):
        self.params = []

    def add_param(self, ptype, pname, subpop, pdist, lb, ub):
        param = InferenceParameter(ptype, pname, subpop, pdist, lb, ub)
        self.params.append(param)


    def 



    def build_from_config(self, global_config, modinf):
        for npi in global_config["seir_modifiers"]["modifiers"].get():
            if global_config["seir_modifiers"]["modifiers"][npi]["perturbation"].exists():
                c = global_config["seir_modifiers"]["modifiers"][npi]
                for sp in modinf.subpop_struct.subpop_names:
                    self.add_param(
                        ptype="snpi",
                        pname=npi,
                        subpop=sp,
                        pdist=c["value"].as_random_distribution(),
                        lb=c["value"]["a"].get(),
                        ub=c["value"]["b"].get()
                    )

        for npi in global_config["outcome_modifiers"]["modifiers"].get():
            if global_config["outcome_modifiers"]["modifiers"][npi]["perturbation"].exists():
                c = global_config["outcome_modifiers"]["modifiers"][npi]
                for sp in modinf.subpop_struct.subpop_names:
                    self.add_param(
                        ptype="hnpi",
                        pname=npi,
                        subpop=sp,
                        pdist=c["value"].as_random_distribution(),
                        lb=c["value"]["a"].get(),
                        ub=c["value"]["b"].get()
                    )

    def print_summary(self):
        print(f"There are {len(self.params)} parameters in the configuration.")
        for param in self.params:
            param.print_summary()

# Create an instance of FittedParams and build it from the config
fitted_params = InferenceParameters()
fitted_params.build_from_config(gempyor.config, modinf)

# Print summary
fitted_params.print_summary()


class FittedParameters:
    def __init__(self, global_config: confuse.ConfigView):
