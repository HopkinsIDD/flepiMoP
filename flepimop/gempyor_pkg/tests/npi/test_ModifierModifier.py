import pandas as pd
import numpy as np
import os
import pathlib
import confuse
import pytest
import datetime

from gempyor import NPI, model_info
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class Test_ModifierModifier:
    def test_ModifierModifier_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")

        s = model_info.ModelInfo(
            setup_name="test_seir",
            config=config,
            nslots=1,
            seir_modifiers_scenario="Fatigue",
            outcome_modifiers_scenario=None,
            write_csv=False,
        )

        test = NPI.ModifierModifier(
            npi_config=s.npi_config_seir,
            modinf=s,
            modifiers_library="",
            subpops=s.subpop_struct.subpop_names,
            loaded_df=None,
        )
        """
        test2 = NPI.SinglePeriodModifier(
            npi_config=s.npi_config_seir,
            modinf=s,
            modifiers_library="",
            subpops=s.subpop_struct.subpop_names,
            loaded_df=test.parameters,
        )
        """

    def test_ModifierModifier_start_date_fail(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        with pytest.raises(ValueError, match=r".*at least one period start or end date is not between.*"):
            s = model_info.ModelInfo(
                setup_name="test_seir",
                config=config,
                nslots=1,
                seir_modifiers_scenario="None",
                outcome_modifiers_scenario=None,
                write_csv=False,
            )
            s.ti = datetime.datetime.strptime("2020-04-02", "%Y-%m-%d").date()

            test = NPI.ModifierModifier(
                npi_config=s.npi_config_seir,
                modinf=s,
                modifiers_library="",
                subpops=s.subpop_struct.subpop_names,
                loaded_df=None,
            )

    def test_ModifierModifier_end_date_fail(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        with pytest.raises(ValueError, match=r".*at least one period start or end date is not between.*"):
            s = model_info.ModelInfo(
                setup_name="test_seir",
                config=config,
                nslots=1,
                seir_modifiers_scenario="None",
                outcome_modifiers_scenario=None,
                write_csv=False,
            )
            s.tf = datetime.datetime.strptime("2020-05-14", "%Y-%m-%d").date()

            test = NPI.ModifierModifier(
                npi_config=s.npi_config_seir,
                modinf=s,
                modifiers_library="",
                subpops=s.subpop_struct.subpop_names,
                loaded_df=None,
            )

    def test_ModifierModifier_checkerrors(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        s = model_info.ModelInfo(
            setup_name="test_seir",
            config=config,
            nslots=1,
            seir_modifiers_scenario="None",
            outcome_modifiers_scenario=None,
            write_csv=False,
        )

        test = NPI.ModifierModifier(
            npi_config=s.npi_config_seir,
            modinf=s,
            modifiers_library="",
            subpops=s.subpop_struct.subpop_names,
            loaded_df=None,
        )

        # Test
        test._SinglePeriodModifier__checkErrors()
