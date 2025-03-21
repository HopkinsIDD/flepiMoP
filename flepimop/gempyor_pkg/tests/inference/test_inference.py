import pytest
import datetime
import os
import pandas as pd

# import dask.dataframe as dd
import pyarrow as pa
import time
import confuse

from gempyor import utils, inference, seir, parameters
from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))

tmp_path = "/tmp"


class TestGempyorInference:
    def test_GempyorInference_success(self):
        os.chdir(os.path.dirname(__file__))

        i = inference.GempyorInference(config_filepath=f"{DATA_DIR}/config_test.yml")
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

    # HopkinsIDD/flepiMoP#269
    def test_inference_without_seir_modifiers(self):
        # Setup
        os.chdir(os.path.dirname(__file__))
        gempyor_inference = inference.GempyorInference(
            f"{DATA_DIR}/config_inference_without_seir_modifiers.yml",
        )

        # Test
        simulation_int = gempyor_inference.one_simulation(0)
        assert simulation_int == 0
