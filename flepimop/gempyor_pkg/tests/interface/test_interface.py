import pytest
import datetime
import os
import pandas as pd
#import dask.dataframe as dd
import pyarrow as pa
import time
import confuse

from gempyor import utils, interface, seir, setup, parameters
from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))

tmp_path = "/tmp"

class TestInferenceSimulator:
    def test_InferenceSimulator_success(self):
   # the minimum model test, choices are: npi_scenario="None"
   #     config.set_file(f"{DATA_DIR}/config_min_test.yml")
        i = interface.InferenceSimulator(config_path=f"{DATA_DIR}/config_min_test.yml", npi_scenario="None")
        ''' run_id="test_run_id" = in_run_id,
            prefix="test_prefix" = in_prefix = out_prefix,
            out_run_id = in_run_id,
        ''' 
   
        i.update_prefix("test_new_in_prefix")
        assert i.s.in_prefix == "test_new_in_prefix"  
        assert i.s.out_prefix == "test_new_in_prefix"  

        i.update_prefix("test_newer_in_prefix", "test_newer_out_prefix")
        assert i.s.in_prefix == "test_newer_in_prefix"  
        assert i.s.out_prefix == "test_newer_out_prefix"  
        i.update_prefix("", "")

        i.update_run_id("test_new_run_id")
        assert i.s.in_run_id == "test_new_run_id"  
        assert i.s.out_run_id == "test_new_run_id"  

        i.update_run_id("test_newer_in_run_id", "test_newer_out_run_id")
        assert i.s.in_run_id == "test_newer_in_run_id"  
        assert i.s.out_run_id == "test_newer_out_run_id" 

        i.update_run_id("test", "test")

      #  i.one_simulation_legacy(sim_id2write=0)
        i.build_structure()
        assert i.already_built 

        i.one_simulation(sim_id2write=0)
