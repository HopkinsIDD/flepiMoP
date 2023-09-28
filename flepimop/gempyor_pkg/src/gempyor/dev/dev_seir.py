import numpy as np
import os
import pytest
import warnings
import shutil

import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import filecmp
import pandas as pd
import matplotlib.pyplot as plt
from . import compartments, seir, NPI, file_paths, model_info

from .utils import config

DATA_DIR = "data"

config.clear()
config.read(user=False)
config.set_file(f"{DATA_DIR}/config.yml")

first_sim_index = 1
run_id = "test_SeedOneNode"
prefix = ""
s = model_info.ModelInfo(
    setup_name="test_seir",
    nslots=1,
    seir_modifiers_scenario="None",
    write_csv=False,
    first_sim_index=first_sim_index,
    in_run_id=run_id,
    in_prefix=prefix,
    out_run_id=run_id,
    out_prefix=prefix,
)

seeding_data = s.seedingAndIC.draw_seeding(sim_id=100, setup=s)
initial_conditions = s.seedingAndIC.draw_ic(sim_id=100, setup=s)

mobility_subpop_indices = s.mobility.indices
mobility_data_indices = s.mobility.indptr
mobility_data = s.mobility.data

npi = NPI.NPIBase.execute(npi_config=s.npi_config_seir, global_config=config, subpops=s.subpop_struct.subpop_names)

params = s.parameters.parameters_quick_draw(s.n_days, s.nsubpops)
params = s.parameters.parameters_reduce(params, npi)

(
    parsed_parameters,
    unique_strings,
    transition_array,
    proportion_array,
    proportion_info,
) = s.compartments.get_transition_array(params, s.parameters.pnames)


states = seir.steps_SEIR_nb(
    s.compartments.compartments.shape[0],
    s.nsubpops,
    s.n_days,
    parsed_parameters,
    s.dt,
    transition_array,
    proportion_info,
    proportion_array,
    initial_conditions,
    seeding_data,
    mobility_data,
    mobility_subpop_indices,
    mobility_data_indices,
    s.subpop_pop,
    True,
)
df = seir.states2Df(s, states)
assert df[(df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")].loc[str(s.tf), "20002"] > 1
print(df)
ts = df
cp = "R"
ts = ts[(ts["mc_infection_stage"] == cp) & (ts["mc_value_type"] == "prevalence")]
ts = ts.drop(["mc_value_type", "mc_infection_stage", "mc_name"], axis=1)
ts = ts.pivot(columns="mc_vaccination_stage").sum(axis=1, level=1)
ts["unvaccinated"].plot()

out_df = df
out_df["date"] = out_df.first_sim_index
pa_df = pa.Table.from_pandas(out_df, preserve_index=False)
pa.parquet.write_table(pa_df, "testlol.parquet")

df2 = SEIR.seir.onerun_SEIR(100, s)
