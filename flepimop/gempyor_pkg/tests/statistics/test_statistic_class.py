import pathlib
from typing import Any, Callable

import confuse
import pytest
import xarray as xr

from gempyor.statistics import Statistic
from gempyor.testing import create_confuse_configview_from_dict


class MockStatisticInput:
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        model_data: xr.DataArray | None = None,
        gt_data: xr.DataArray | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.model_data = model_data
        self.gt_data = gt_data
        self._confuse_subview = None

    def create_confuse_subview(self) -> confuse.Subview:
        if self._confuse_subview is None:
            self._confuse_subview = create_confuse_configview_from_dict(
                self.config, name=self.name
            )
        return self._confuse_subview

    def create_statistic_instance(self) -> Statistic:
        return Statistic(self.name, self.create_confuse_subview())


def invalid_regularization_factory(tmp_path: pathlib.Path) -> MockStatisticInput:
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "aggregator": "sum",
            "period": "1 months",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "pois"},
            "regularize": [{"name": "forecast"}, {"name": "invalid"}],
        },
    )


class TestStatistic:
    @pytest.mark.parametrize("factory", [(invalid_regularization_factory)])
    def test_unsupported_regularizations_value_error(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockStatisticInput],
    ) -> None:
        mock_inputs = factory(tmp_path)
        unsupported_name = next(
            reg_name
            for reg_name in [
                reg["name"] for reg in mock_inputs.config.get("regularize", [])
            ]
            if reg_name not in ["forecast", "allsubpop"]
        )
        with pytest.raises(
            ValueError, match=rf"^Unsupported regularization\: {unsupported_name}$"
        ):
            mock_inputs.create_statistic_instance()
