import os
import pytest
from gempyor import seeding, model_info, initial_conditions
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestIC:
    def test_IC_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_ic",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )
        sic = initial_conditions.InitialConditionsFactory(config=s.initial_conditions_config)
        assert sic.initial_conditions_config == s.initial_conditions_config

    def test_IC_allow_missing_node_compartments_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config.yml")

        s = model_info.ModelInfo(
            config=config,
            setup_name="test_ic",
            nslots=1,
            seir_modifiers_scenario=None,
            outcome_modifiers_scenario=None,
            write_csv=False,
        )

        s.initial_conditions_config["allow_missing_nodes"] = True
        s.initial_conditions_config["allow_missing_compartments"] = True
        sic = initial_conditions.InitialConditionsFactory(config=s.initial_conditions_config)
        sic.get_from_config(sim_id=100, modinf=s)

    def test_IC_IC_notImplemented_fail(self):
        with pytest.raises(NotImplementedError, match=r"^Unknown initial conditions method \[received: .*?\]\.$"):
            config.clear()
            config.read(user=False)
            config.set_file(f"{DATA_DIR}/config.yml")

            s = model_info.ModelInfo(
                config=config,
                setup_name="test_ic",
                nslots=1,
                seir_modifiers_scenario=None,
                outcome_modifiers_scenario=None,
                write_csv=False,
            )
            s.initial_conditions_config["method"] = "unknown"
            sic = initial_conditions.InitialConditionsFactory(config=s.initial_conditions_config)

            sic.get_from_config(sim_id=100, modinf=s)
