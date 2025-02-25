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


@pytest.mark.parametrize(
    "filter", [
        "somefilter",
    ],
)
def test_valid_filters(filter: list[str] | str):
    """
    an a single string filter is promoted to a list
    """
    mf = MockFilters(filters = filter)
    assert isinstance(mf.filters, list)
    assert mf.filters.__len__() == 1

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

@pytest.mark.parametrize(
    "filter,addneg", [
        (["somefilter"], True),
        (["somefilter", "another"], True),
        (["- somefilter", "another"], False),
    ],
)
def test_filter_all_include(filter: list[str], addneg: bool):
    """
    if all filters are include filters, add an exclude all filter at the end
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters()
    if addneg:
        assert lfs[-1][0] == "-"
        assert lfs[-1][1] == "**"
    else:
        assert lfs.__len__() == filter.__len__()

@pytest.mark.parametrize(
    "filter,prefix", [
        ([], ["+ foo"]),
        (["somefilter"], ["foo", "- bar"]),
        (["somefilter", "another"], ["foo", "- bar"]),
    ],
)
def test_filter_all_include(filter: list[str], prefix: list[str]):
    """
    prefix filters should appear, in order, before core filters
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters(prefix=prefix)
    for i, p in enumerate(prefix):
        assert lfs[i][-1] == p

# suffix filters should appear, in order, after core filters

# asking for reverse view should present filters in reverse order