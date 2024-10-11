import pytest

from gempyor.compartments import (
    list_recursive_convert_to_string,
    NestedListOfAny,
    NestedListOfStr,
)


@pytest.mark.parametrize(
    ("thing", "expected_output"),
    (
        (1, "1"),
        (-1, "-1"),
        ("abc", "abc"),
        ("xyz", "xyz"),
        (object, "<class 'object'>"),
        (None, "None"),
        ([], []),
        ([1], ["1"]),
        ([-1], ["-1"]),
        (["abc"], ["abc"]),
        (["xyz"], ["xyz"]),
        ([object], ["<class 'object'>"]),
        ([None], ["None"]),
        ([1, 2, 3], ["1", "2", "3"]),
        ([[1], [2, 3], [4, [5, [6]]]], [["1"], ["2", "3"], ["4", ["5", ["6"]]]]),
    ),
)
def test_list_recursive_convert_to_string_output_validation(
    thing: NestedListOfAny, expected_output: NestedListOfStr
) -> None:
    assert list_recursive_convert_to_string(thing) == expected_output
