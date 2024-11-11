from collections.abc import Sequence

import pytest

from gempyor.compartments import NestedIterableOfStr, _format_deep_nested_iterables


@pytest.mark.parametrize(
    ("x", "sep", "expected"),
    (
        (["a", "b", "c"], "_", ["a", "b", "c"]),
        ([["a", "b"], "c", ["d", "e", "f"]], "-", ["a-b", "c", "d-e-f"]),
        ([["a", ["b", "c"]], ["d", ["e"], "f"]], "#", ["a#['b', 'c']", "d#['e']#f"]),
        ("abc", "*", ["abc"]),
        ([["a", "b", ["c", "d"]]], ["_", "*"], ["a_b_c*d"]),
        (
            [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]],
            ["%", "#"],
            ["a#b%c#d", "e#f%g#h"],
        ),
        (
            [["a", ["b", ["c"]]], ["d", "e", ["f", "g"]]],
            ("!", "~"),
            ["a!b~['c']", "d!e!f~g"],
        ),
        ((("a", ("b", ("c", ("d")))),), ("*", "_", "+"), ["a*b_c+d"]),
        (
            [["a", ["b", ["c", ["d", ["e", "f"]]]]]],
            ["?", "&", "@"],
            ["a?b&c@['d', ['e', 'f']]"],
        ),
    ),
)
def test_output_validation(
    x: NestedIterableOfStr, sep: str | Sequence[str], expected: list[str]
) -> None:
    assert _format_deep_nested_iterables(x, sep) == expected
