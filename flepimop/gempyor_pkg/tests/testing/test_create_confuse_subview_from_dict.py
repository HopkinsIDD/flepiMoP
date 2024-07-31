from datetime import date
from typing import Any

import confuse
import pytest

from gempyor.testing import create_confuse_subview_from_dict


class TestCreateConfuseSubviewFromDict:
    @pytest.mark.parametrize(
        "name,data",
        [
            ("nil", {}),
            ("basic", {"foo": "bar"}),
            ("small", {"a": "b", "c": 1}),
            (
                "big",
                {
                    "alphabet": ["a", "b", "c", "d", "e"],
                    "integers": [1, 2, 3, 4, 5],
                    "floats": [1.2, 2.3, 3.4, 4.5, 5.6],
                },
            ),
            ("date_data_type", {"as_of_date": date(2024, 1, 1)}),
        ],
    )
    def test_output_validation(self, name: str, data: dict[str, Any]) -> None:
        root_view = create_confuse_subview_from_dict(name, data)
        assert isinstance(root_view, confuse.Subview)
        assert root_view != root_view.root()
        assert root_view.name == name
        assert root_view.get() == data
