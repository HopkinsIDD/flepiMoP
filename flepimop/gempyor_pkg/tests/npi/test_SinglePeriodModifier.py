import pandas as pd
import numpy as np
import os
import pathlib
import confuse

from gempyor import NPI, model_info
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class Test_SinglePeriodModifier:
    def test_SinglePeriodModifier_success(self):
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

        test = NPI.SinglePeriodModifier(
            npi_config=s.npi_config_seir,
            modinf=s,
            modifiers_library="",
            subpops=s.subpop_struct.subpop_names,
            loaded_df=None,
        )
