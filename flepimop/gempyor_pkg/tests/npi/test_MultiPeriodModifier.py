import pandas as pd
import os

from gempyor import seir, NPI, model_info
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"


class Test_MultiPeriodModifier:
    def test_MultiPeriodModifier_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        s = model_info.ModelInfo(
            config=config, nslots=1, seir_modifiers_scenario="KansasCity"
        )
        modifier = seir.build_npi_SEIR(
            modinf=s, config=config, load_ID=False, sim_id2load=None
        )
        assert modifier is not None
        assert isinstance(modifier, NPI.MultiPeriodModifier)
        assert modifier.name == "KansasCity"

    def test_MultiPeriodModifier_createFromDf(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")

        s_from_config = model_info.ModelInfo(
            config=config, nslots=1, seir_modifiers_scenario="KansasCity"
        )
        modifier_from_config = seir.build_npi_SEIR(
            modinf=s_from_config, config=config, load_ID=False, sim_id2load=None
        )
        params_df = modifier_from_config.getReductionToWrite()
        modifier_from_df = seir.build_npi_SEIR(
            modinf=s_from_config,
            config=config,
            bypass_DF=params_df,
            load_ID=False,
            sim_id2load=None,
        )
        pd.testing.assert_frame_equal(modifier_from_config.npi, modifier_from_df.npi)
