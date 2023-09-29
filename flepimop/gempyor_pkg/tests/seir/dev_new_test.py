import numpy as np
import pandas as pd
import os
import pytest
import warnings
import shutil

import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import filecmp

from gempyor import model_info, seir, NPI, file_paths, parameters

from gempyor.utils import config, write_df, read_df
import gempyor

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


# def test_parameters_from_timeserie_file():
#if True:
#    config.clear()
#    config.read(user=False)
#    config.set_file(f"{DATA_DIR}/config_compartmental_model_format_with_covariates.yml")
#    inference_simulator = gempyor.GempyorSimulator(
#        config_path=f"{DATA_DIR}/config_compartmental_model_format_with_covariates.yml",
#        run_id=1,
#        prefix="",
#        first_sim_index=1,
#        stoch_traj_flag=False,
#    )
#
#    # p = parameters.Parameters(
#    #    parameter_config=config["seir"]["parameters"])
#
#    p = inference_simulator.modinf.parameters
#    p_draw = p.parameters_quick_draw(n_days=inference_simulator.modinf.n_days, nsubpops=inference_simulator.modinf.nsubpops)
#
#    p_df = p.getParameterDF(p_draw)["parameter"]
#
#    for pn in p.pnames:
#        if pn == "R0s":
#            assert pn not in p_df
#        else:
#            assert pn in p_df
#
#    initial_df = read_df("data/r0s_ts.csv").set_index("date")
#
#    assert (p_draw[p.pnames2pindex["R0s"]] == initial_df.values).all()
#
#    ### test what happen when the order of subpops is not respected (expected: reput them in order)
#
#    ### test what happens with incomplete data (expected: fail)
#
#    ### test what happens when loading from file
#    # write_df(fname="test_pwrite.parquet", df=p.getParameterDF(p_draw=p_draw))
#