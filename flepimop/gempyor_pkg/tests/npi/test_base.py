import pytest
import datetime
import confuse

from gempyor.NPI.base import NPIBase
from gempyor.NPI.SinglePeriodModifier import SinglePeriodModifier
from gempyor.NPI.MultiPeriodModifier import MultiPeriodModifier
from gempyor.NPI.StackedModifier import StackedModifier
from gempyor.utils import config
import os

DATA_DIR = os.path.dirname(__file__) + "/data"


@pytest.mark.parametrize(
    "modifier_name, expected_class",
    [
        ("None", SinglePeriodModifier),
        ("KansasCity", MultiPeriodModifier),
        ("Scenario1", StackedModifier),
    ],
)
def test_execute_success(modifier_name, expected_class):
    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config_test.yml")
    npi_config = config["seir_modifiers"]["modifiers"][modifier_name]
    modifiers_library = config["seir_modifiers"]["modifiers"].get()
    result_modifier = NPIBase.execute(
        npi_config=npi_config,
        modifiers_library=modifiers_library,
        modinf_ti=datetime.date(2020, 1, 1),
        modinf_tf=datetime.date(2020, 12, 31),
        subpops=["A", "B"],
    )
    assert isinstance(result_modifier, expected_class)


def test_execute_failure_unknown_method():
    config_fail = confuse.Configuration("TestApp", __name__)
    config_fail.set({"method": "ThisMethodDoesNotExist"})
    with pytest.raises(KeyError):
        NPIBase.execute(
            npi_config=config_fail,
            modifiers_library={},
            modinf_ti=datetime.date(2020, 1, 1),
            modinf_tf=datetime.date(2020, 12, 31),
            subpops=["A", "B"],
        )
