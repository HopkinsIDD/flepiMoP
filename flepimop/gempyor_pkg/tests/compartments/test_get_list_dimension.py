from typing import Any

import pytest

from gempyor.compartments import get_list_dimension


@pytest.mark.parametrize(
    "thing",
    (
        1,
        2,
        3,
        "abc",
        "def",
        "xyz",
        (1, 2, 3),
        (),
        {"a": 1, "b": 2},
        {},
        3.14,
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        object,
        None,
        {1, 2, 3},
        set(),
    ),
)
def test_get_list_dimension_output_validation(thing: Any) -> None:
    assert get_list_dimension(thing) == len(thing) if isinstance(thing, list) else 1
