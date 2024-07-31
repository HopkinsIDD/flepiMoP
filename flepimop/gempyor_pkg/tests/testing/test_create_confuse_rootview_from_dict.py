from datetime import date
from typing import Any

import confuse
import pytest

from gempyor.testing import create_confuse_rootview_from_dict


class TestCreateConfuseRootviewFromDict:
    @pytest.mark.parametrize(
        "data",
        [
            ({}),
            ({"foo": "bar"}),
            ({"a": "b", "c": 1}),
            (
                {
                    "alphabet": ["a", "b", "c", "d", "e"],
                    "integers": [1, 2, 3, 4, 5],
                    "floats": [1.2, 2.3, 3.4, 4.5, 5.6],
                }
            ),
            ({"as_of_date": date(2024, 1, 1)}),
        ],
    )
    def test_output_validation(self, data: dict[str, Any]) -> None:
        root_view = create_confuse_rootview_from_dict(data)
        assert isinstance(root_view, confuse.RootView)
        assert root_view == root_view.root()
        assert root_view.name == "root"
        assert root_view.get() == data
