import pytest
import os
import importlib

from gempyor import seir, model_info, NPI
from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"


class Test_StackedModifier:
    def test_stacked_modifier_success(self):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")

        s = model_info.ModelInfo(
            config=config, nslots=1, seir_modifiers_scenario="Scenario1"
        )
        npi_stack = seir.build_npi_SEIR(
            modinf=s, config=config, load_ID=False, sim_id2load=None
        )
        assert isinstance(npi_stack, NPI.StackedModifier)
        assert npi_stack.name.endswith("Scenario1")
        assert "r0" in npi_stack.param_name

    def test_stacked_modifier_cap_fail(self, monkeypatch):
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_test.yml")
        module_path = NPI.StackedModifier.__module__
        stacked_module = importlib.import_module(module_path)
        monkeypatch.setattr(stacked_module, "REDUCTION_METADATA_CAP", 2)
        s = model_info.ModelInfo(
            config=config, nslots=1, seir_modifiers_scenario="Scenario1"
        )
        npi_stack = seir.build_npi_SEIR(
            modinf=s, config=config, load_ID=False, sim_id2load=None
        )
        assert npi_stack.reduction_cap_exceeded is True
        with pytest.raises(RuntimeError, match=r".*memory buffer cap exceeded.*"):
            with pytest.warns(UserWarning, match=r".*memory buffer cap exceeded.*"):
                npi_stack.getReductionToWrite()
