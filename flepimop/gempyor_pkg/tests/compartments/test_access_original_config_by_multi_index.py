from typing import Any

import pytest

from gempyor.compartments import _access_original_config_by_multi_index


@pytest.mark.parametrize(
    ("config_piece", "index", "dimension", "encapsulate_as_list", "expected_output"),
    (
        (["S"], (0,), None, False, ["S"]),
        (["S"], (0,), None, True, [["S"]]),
        (["S", "vaccinated"], (0, 1), None, False, ["S", "vaccinated"]),
        (["S", "vaccinated"], (0, 1), None, True, [["S"], ["vaccinated"]]),
        (["I"], (0,), [1], False, ["I"]),
        (["I"], (0,), [1], True, [["I"]]),
        (["I"], (0,), [2], False, ["I"]),
        (["I"], (0,), [2], True, [["I"]]),
    ),
)
def test_access_original_config_by_multi_index_output_validation(
    config_piece: list[Any],
    index: int,
    dimension: list[int | None] | None,
    encapsulate_as_list: bool,
    expected_output: Any,
) -> None:
    assert (
        _access_original_config_by_multi_index(
            config_piece,
            index,
            dimension=dimension,
            encapsulate_as_list=encapsulate_as_list,
        )
        == expected_output
    )
