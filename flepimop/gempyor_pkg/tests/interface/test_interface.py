import pytest
import datetime
import os
import pandas as pd

# import dask.dataframe as dd
import pyarrow as pa
import time
import confuse

from gempyor import utils, interface, seir, parameters
from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))

tmp_path = "/tmp"


class TestGempyorSimulator:
    def test_GempyorSimulator_success(self):
        os.chdir(os.path.dirname(__file__))
        # the minimum model test, choices are: npi_scenario="None"
        #     config.set_file(f"{DATA_DIR}/config_min_test.yml")
        #     i = interface.GempyorSimulator(config_path=f"{DATA_DIR}/config.yml", npi_scenario="None")
        i = interface.GempyorSimulator(config_path=f"{DATA_DIR}/config_test.yml", seir_modifiers_scenario="None")
        """ run_id="test_run_id" = in_run_id,
            prefix="test_prefix" = in_prefix = out_prefix,
            out_run_id = in_run_id,
        """

        i.update_prefix("test_new_in_prefix")
        assert i.modinf.in_prefix == "test_new_in_prefix"
        assert i.modinf.out_prefix == "test_new_in_prefix"

        i.update_prefix("test_newer_in_prefix", "test_newer_out_prefix")
        assert i.modinf.in_prefix == "test_newer_in_prefix"
        assert i.modinf.out_prefix == "test_newer_out_prefix"
        i.update_prefix("", "")

        i.update_run_id("test_new_run_id")
        assert i.modinf.in_run_id == "test_new_run_id"
        assert i.modinf.out_run_id == "test_new_run_id"

        i.update_run_id("test_newer_in_run_id", "test_newer_out_run_id")
        assert i.modinf.in_run_id == "test_newer_in_run_id"
        assert i.modinf.out_run_id == "test_newer_out_run_id"

        i.update_run_id("test", "test")

        i.one_simulation_legacy(sim_id2write=0)
        i.build_structure()
        assert i.already_built

        i.one_simulation_legacy(sim_id2write=0, load_ID=True, sim_id2load=0)

        i.already_built = False
        i.one_simulation(sim_id2write=0)

        i.already_built = False
        i.one_simulation(sim_id2write=0, load_ID=True, sim_id2load=0)

        i.already_built = False
        i.one_simulation(sim_id2write=0, parallel=True)
