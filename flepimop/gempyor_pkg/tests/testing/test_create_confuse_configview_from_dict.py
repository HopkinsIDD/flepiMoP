from datetime import date
from typing import Any

import confuse
import pytest

from gempyor.testing import create_confuse_configview_from_dict


class TestCreateConfuseConfigviewFromDict:
    @pytest.mark.parametrize(
        "name,data",
        [
            (None, {}),
            ("nil", {}),
            (None, {"foo": "bar"}),
            ("basic", {"foo": "bar"}),
            (None, {"a": "b", "c": 1}),
            ("small", {"a": "b", "c": 1}),
            (
                None,
                {
                    "alphabet": ["a", "b", "c", "d", "e"],
                    "integers": [1, 2, 3, 4, 5],
                    "floats": [1.2, 2.3, 3.4, 4.5, 5.6],
                },
            ),
            (
                "big",
                {
                    "alphabet": ["a", "b", "c", "d", "e"],
                    "integers": [1, 2, 3, 4, 5],
                    "floats": [1.2, 2.3, 3.4, 4.5, 5.6],
                },
            ),
            (None, {"as_of_date": date(2024, 1, 1)}),
            ("date_data_type", {"as_of_date": date(2024, 1, 1)}),
            (
                None,
                {
                    "foo": "bar",
                    "fizz": 123,
                    "alphabet": ["a", "b", "c"],
                    "mapping": {"x": 1, "y": 2},
                },
            ),
            (
                "root",
                {
                    "foo": "bar",
                    "fizz": 123,
                    "alphabet": ["a", "b", "c"],
                    "mapping": {"x": 1, "y": 2},
                },
            ),
        ],
    )
    def test_output_validation(self, name: str, data: dict[str, Any]) -> None:
        view = create_confuse_configview_from_dict(data, name=name)
        assert isinstance(view, confuse.ConfigView)
        assert (
            isinstance(view, confuse.RootView)
            if name is None
            else isinstance(view, confuse.Subview)
        )
        assert view == view.root() if name is None else view != view.root()
        assert view.name == "root" if name is None else view.name == name
        assert view.get() == data
