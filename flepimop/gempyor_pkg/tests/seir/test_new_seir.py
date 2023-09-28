import numpy as np
import os
import pytest
import warnings
import shutil
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from functools import reduce

from gempyor import model_info, seir, NPI, file_paths, compartments, subpopulation_structure

from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


def test_constant_population():
    config.set_file(f"{DATA_DIR}/config.yml")

    s = model_info.ModelInfo(
        config=config,
        nslots=1,
        write_csv=False,
        stoch_traj_flag=False,
    )

    initial_conditions = s.seedingAndIC.draw_ic(sim_id=0, setup=s)
    seeding_data, seeding_amounts = s.seedingAndIC.load_seeding(sim_id=100, setup=s)

    npi = NPI.NPIBase.execute(npi_config=s.npi_config_seir, global_config=config, subpops=s.subpop_struct.subpop_names)

    parameters = s.parameters.parameters_quick_draw(n_days=s.n_days, nsubpops=s.nsubpops)
    parameter_names = [x for x in s.parameters.pnames]

    print("RUN_FUN_START")
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = s.compartments.get_transition_array()
    parsed_parameters = s.compartments.parse_parameters(parameters, s.parameters.pnames, unique_strings)
    print("RUN_FUN_END")
    print(proportion_array)

    states = seir.steps_SEIR(
        s,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data,
        seeding_amounts,
    )
