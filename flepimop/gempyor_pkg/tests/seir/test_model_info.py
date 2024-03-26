import datetime
import numpy as np
import os
import pandas as pd
import pytest
import confuse

from gempyor.model_info import ModelInfo, subpopulation_structure

from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestModelInfo:
    def test_ModelInfo_init_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        s = ModelInfo(
            config=config,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            spatial_path_prefix="",
            write_csv=False,
            write_parquet=False,
            first_sim_index=1,
            in_run_id=None,
            in_prefix=None,
            out_run_id=None,
            out_prefix=None,
            stoch_traj_flag=False,
            inference_filename_prefix="",
            inference_filepath_suffix="",
            setup_name=None,  # override config setup_name
        )
        assert isinstance(s, ModelInfo)

    def test_ModelInfo_init_tf_is_ahead_of_ti_fail(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        config["start_date"] = "2022-01-02"
        with pytest.raises(ValueError, match=r"tf.*is less than or equal to ti.*"):
            s = ModelInfo(
                config=config,
                seir_modifiers_scenario=None,
                outcome_modifiers_scenario=None,
                spatial_path_prefix="",
                write_csv=False,
                write_parquet=False,
                first_sim_index=1,
                in_run_id=None,
                in_prefix=None,
                out_run_id=None,
                out_prefix=None,
                stoch_traj_flag=False,
                inference_filename_prefix="",
                inference_filepath_suffix="",
                setup_name=None,  # override config setup_name
            )

    def test_ModelInfo_init_seir_modifiers_scenario_set(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")

        s = ModelInfo(
            config=config,
            seir_modifiers_scenario="Scenario1",
            outcome_modifiers_scenario=None,
            spatial_path_prefix="",
            write_csv=False,
            write_parquet=False,
            first_sim_index=1,
            in_run_id=None,
            in_prefix=None,
            out_run_id=None,
            out_prefix=None,
            stoch_traj_flag=False,
            inference_filename_prefix="",
            inference_filepath_suffix="",
            setup_name=None,  # override config setup_name
        )

    def test_ModelInfo_init_setup_name_set(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")

        s = ModelInfo(
            config=config,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            spatial_path_prefix="",
            write_csv=False,
            write_parquet=False,
            first_sim_index=1,
            in_run_id=None,
            in_prefix=None,
            out_run_id=None,
            out_prefix=None,
            stoch_traj_flag=False,
            inference_filename_prefix="",
            inference_filepath_suffix="",
            setup_name=TEST_SETUP_NAME,
        )
