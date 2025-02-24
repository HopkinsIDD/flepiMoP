from typing import Literal

import pytest
from pydantic import ValidationError, TypeAdapter

from gempyor.sync._sync_filter import WithFilters, FilterParts

class MockFilters(WithFilters):
    def _formatter(self, f: FilterParts) -> list[str]:
        return [f[0], f[1]]

# valid vs invalid filters successfully parsed
@pytest.mark.parametrize(
    "filter", [
        ["somefilter"],
        ["somefilter", "another"],
        [],
    ],
)
def test_valid_filters(filter: list[str] | str):
    """
    A string that doesn't start with a space or a plus/minus without a
    subsequent space is a valid filter
    """
    _ = MockFilters(filters = filter)

@pytest.mark.parametrize(
    "filter", [
        " somefilter",
        "-something",
        "+something",
        1,
    ],
)
def test_invalid_filters(filter: list[str] | str):
    """
    a string that starts with a space or a plus/minus without a subsequent space
    is an invalid filter
    """
    with pytest.raises(ValidationError):
        _ = MockFilters(filters = filter)

# ensure automatic list promotion
@pytest.mark.parametrize(
    "filter", [
        "somefilter",
    ],
)
def test_valid_filters(filter: list[str] | str):
    """
    A string that doesn't start with a space or a plus/minus without a
    subsequent space is a valid filter
    """
    mf = MockFilters(filters = filter)
    assert mf.filters.__len__() == 1

# include vs exclude filters successfully parsed, including default include, including mixtures
@pytest.mark.parametrize(
    "filter,mode", [
        (["somefilter"], ["+"]),
        (["somefilter", "- another"], ["+", "-"]),
        (["- somefilter", "+ another"], ["-", "+"]),
    ],
)
def test_filter_modes(filter: list[str] | str, mode: list[Literal["+", "-"]]):
    """
    a string that starts with a space or a plus/minus without a subsequent space
    is an invalid filter
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters()
    for i, m in enumerate(mode):
        assert lfs[i][0] == m

# convert convert single filters to list

# handle single include filter expanding to add an exclude all
