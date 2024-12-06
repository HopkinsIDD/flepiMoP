import os

import pytest

from gempyor.inference import get_static_arguments
from gempyor.model_info import ModelInfo
from gempyor.testing import create_confuse_configview_from_dict


def test_modinf_missing_compartments_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(os.path.dirname(__file__))
    config = create_confuse_configview_from_dict(
        {
            "name": "fooobar",
            "setup_name": "test1",
            "start_date": "2020-04-01",
            "end_date": "2020-05-15",
            "nslots": 1,
            "subpop_setup": {
                "geodata": "data/geodata.csv",
            },
            "outcomes": {
                "method": "delayframe",
                "param_from_file": False,
                "outcomes": {
                    "incidI": {
                        "source": {
                            "incidence": {
                                "infection_stage": ["I1"],
                            },
                        },
                        "probability": {
                            "value": {
                                "distribution": "fixed",
                                "value": 1,
                            },
                        },
                        "delay": {
                            "value": {
                                "distribution": "fixed",
                                "value": 0,
                            },
                        },
                    }
                },
            },
        }
    )
    modinf = ModelInfo(
        config=config,
    )
    with pytest.raises(
        RuntimeError,
        match="^The `modinf` is required to have a parsed `compartments` attribute.$",
    ):
        get_static_arguments(modinf)
