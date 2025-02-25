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
def test_filter_prefix(filter: list[str], prefix: list[str]):
    """
    prefix filters should appear, in order, before core filters
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters(prefix=prefix)
    for i, p in enumerate(prefix):
        assert lfs[i][-1] == p

@pytest.mark.parametrize(
    "filter,suffix", [
        ([], ["+ foo"]),
        (["somefilter"], ["foo", "- bar"]),
        (["somefilter", "another"], ["foo", "- bar"]),
    ],
)
def test_filter_suffix(filter: list[str], suffix: list[str]):
    """
    suffix filters should appear, in order, after core filters
    """
    obj = MockFilters(filters = filter)
    offset = obj.filters.__len__()
    lfs = obj.list_filters(suffix=suffix)
    for i, p in enumerate(suffix):
        assert lfs[offset+i][-1] == p

@pytest.mark.parametrize(
    "filter,prefix,suffix,addneg", [
        (["somefilter"],["more"],["more"],True),
        (["- somefilter", "another"],["more"],["more"],False),
    ],
)
def test_filter_presuff_autoexclude_same(filter:list[str], prefix: list[str], suffix: list[str], addneg: bool):
    """
    prefix and suffix include additions should preserve the autoexclude rule
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters(prefix=prefix, suffix=suffix)
    if addneg:
        assert lfs[-1][0] == "-"
        assert lfs[-1][1] == "**"
    else:
        assert lfs.__len__() == filter.__len__() + prefix.__len__() + suffix.__len__()
    
@pytest.mark.parametrize(
    "filter,prefix,suffix,addneg", [
        (["somefilter"],["- more"],["more"],False),
        (["somefilter"],["more"],["- more"],False),
        (["somefilter", "another"],["more"],["- more"],False),
    ],
)
def test_filter_presuff_autoexclude(filter:list[str], prefix: list[str], suffix: list[str], addneg: bool):
    """
    prefix or suffix exclude additions should no longer autoexclude
    """
    obj = MockFilters(filters = filter)
    assert obj.list_filters().__len__() == filter.__len__() + 1
    lfs = obj.list_filters(prefix=prefix, suffix=suffix)
    assert lfs.__len__() == filter.__len__() + prefix.__len__() + suffix.__len__()

# override should replace core filter, if present
@pytest.mark.parametrize(
    "filter,override", [
        (["somefilter"], ["another"]),
        (["somefilter"], []),
        (["somefilter"], None),
    ],
)
def test_override_filter(filter : list[str], override: list[str] | None):
    """
    override filters should replace core filters
    """
    obj = MockFilters(filters = filter)
    lfs = obj.list_filters(overrides=override)
    if override is not None:
        if override.__len__() == 0:
            assert lfs.__len__() == 0
        else:
            assert all(f[-1] == o for f, o in zip(lfs, override))
    else:
        assert all(f[-1] == o for f, o in zip(lfs, filter))

# asking for reverse view should present filters in reverse order