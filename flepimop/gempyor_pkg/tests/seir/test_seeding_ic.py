import numpy as np
import os
import pytest
import warnings
import shutil

import pathlib
import pyarrow as pa
import pyarrow.parquet as pq

from gempyor import seir, NPI, file_paths, seeding_ic, model_info

from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestSeedingAndIC:
    def test_SeedingAndIC_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_seeding and ic",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )
        sic = seeding_ic.SeedingAndIC(
            seeding_config=s.seeding_config, initial_conditions_config=s.initial_conditions_config
        )
        assert sic.seeding_config == s.seeding_config
        assert sic.initial_conditions_config == s.initial_conditions_config

    def test_SeedingAndIC_allow_missing_node_compartments_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_seeding and ic",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )

        s.initial_conditions_config["allow_missing_nodes"] = True
        s.initial_conditions_config["allow_missing_compartments"] = True
        sic = seeding_ic.SeedingAndIC(
            seeding_config=s.seeding_config, initial_conditions_config=s.initial_conditions_config
        )

        initial_conditions = sic.draw_ic(sim_id=100, setup=s)

    # print(initial_conditions)
    # integration_method = "legacy"

    def test_SeedingAndIC_IC_notImplemented_fail(self):
        with pytest.raises(NotImplementedError, match=r".*unknown.*initial.*conditions.*"):
            config.clear()
            config.read(user=False)
            config.set_file(f"{DATA_DIR}/config.yml")

            s = model_info.ModelInfo(
                config=config,
                setup_name="test_seeding and ic",
                nslots=1,
                seir_modifiers_scenario=None,
                outcome_modifiers_scenario=None,
                write_csv=False,
            )
            s.initial_conditions_config["method"] = "unknown"
            sic = seeding_ic.SeedingAndIC(
                seeding_config=s.seeding_config, initial_conditions_config=s.initial_conditions_config
            )

            sic.draw_ic(sim_id=100, setup=s)

    def test_SeedingAndIC_draw_seeding_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_seeding and ic",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )
        sic = seeding_ic.SeedingAndIC(
            seeding_config=s.seeding_config, initial_conditions_config=s.initial_conditions_config
        )
        s.seeding_config["method"] = "NoSeeding"

        seeding = sic.draw_seeding(sim_id=100, setup=s)
        print(seeding)

    # print(initial_conditions)
