import xarray as xr
import pandas as pd
import numpy as np
import confuse

class InferenceParameters:
    def __init__(self):
        self.params = []

    def add_modifier(self, pname, ptype, parameter_config):
        for sp in modinf.subpop_struct.subpop_names:
            param = {
                "ptype": ptype,
                "pname": pname,
                "subpop": sp,
                "pdist": parameter_config["value"].as_random_distribution(),
                "lb": ["value"]["a"].get(),
                "ub": parameter_config["value"]["b"].get()
            }
            self.params.append(param)
            # TODO: does not support the subpop groups here !!!!!!!

    def build_from_config(self, global_config, modinf):
        for config_part in ["seir_modifiers", "outcome_modifiers"]:
            if  global_config[config_part].exists():
                for npi in global_config[config_part]["modifiers"].get():
                    if global_config[config_part]["modifiers"][npi]["perturbation"].exists():
                        self.add_modifier(pname=npi, ptype=config_part, parameter_config=global_config[config_part]["modifiers"][npi])

    def print_summary(self):
        print(f"There are {len(self.params)} parameters in the configuration.")
        for param in self.params:
            print(f"{param['ptype']}::{param['pname']} in [{param['lb']}, {param['ub']}]"
                f"   >> affected subpop: {param['subpop']}"
            )

# Create an instance of FittedParams and build it from the config
fitted_params = InferenceParameters()
fitted_params.build_from_config(gempyor.config, modinf)

# Print summary
fitted_params.print_summary()


class FittedParameters:
    def __init__(self, global_config: confuse.ConfigView):