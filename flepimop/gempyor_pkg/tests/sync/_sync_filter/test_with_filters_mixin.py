"""Unit tests for the `gempyor.sync._sync_filter.WithFilters` mixin."""

import pytest

from gempyor.sync._sync_filter import FilterParts, ListSyncFilter, WithFilters


class ExampleWithFilters(WithFilters):
    """Example class for the `WithFilters` mixin."""

    @staticmethod
    def _formatter(f: FilterParts) -> list[str]:
        _, _, raw_filter = f
        return [raw_filter]


@pytest.mark.parametrize(
    "filters",
    [[], ["- /foo/bar.txt"], ["+ /fizz/buzz.txt"], ["/a/b/c.txt"], ["s /x/y/z.txt"]],
)
@pytest.mark.parametrize(
    "overrides",
    [
        None,
        [],
        ["- /x/y/z.txt"],
        ["+ /x/y/z.txt"],
        ["/x/y/z.txt"],
        ["+ /qrs.txt", "- /tuv.txt"],
        ["s /abc.txt"],
    ],
)
@pytest.mark.parametrize(
    "prefix",
    [
        None,
        [],
        ["- /abc/def.txt"],
        ["+ /ghi.txt"],
        ["/jkl.txt"],
        ["+ /mno.txt", "- /pqr.txt"],
        ["s /stu.txt"],
    ],
)
@pytest.mark.parametrize(
    "suffix",
    [
        None,
        [],
        ["- /fed/cba.txt"],
        ["+ /ihg.txt"],
        ["/lkj.txt"],
        ["+ /omn.txt", "- /rqp.txt"],
        ["s /uts.txt"],
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_list_filters(
    filters: ListSyncFilter,
    overrides: ListSyncFilter | None,
    prefix: ListSyncFilter | None,
    suffix: ListSyncFilter | None,
    reverse: bool,
) -> None:
    """Test the `list_filters` method."""
    example_with_filters = ExampleWithFilters(filters=filters)
    parsed_filters = example_with_filters.list_filters(
        overrides=overrides, prefix=prefix, suffix=suffix, reverse=reverse
    )
    assert (
        isinstance(parsed_filters, list)
        and all(isinstance(f, tuple) for f in parsed_filters)
        and all(isinstance(p, str) for f in parsed_filters for p in f)
    )
    any_minus_or_substring = (
        any(
            f.startswith("- ") or f.startswith("s ")
            for f in (filters if overrides is None else overrides)
        )
        or any(f.startswith("- ") or f.startswith("s ") for f in (prefix or []))
        or any(f.startswith("- ") or f.startswith("s ") for f in (suffix or []))
    )
    parsed_count = sum(
        (
            len(filters if overrides is None else overrides),
            len(prefix or []),
            len(suffix or []),
        )
    )
    assert len(parsed_filters) == parsed_count + int(not any_minus_or_substring) * min(
        parsed_count, 1
    )
    if reverse:
        # It's easier to undo reverse and assert than conditionally assert
        parsed_filters.reverse()
    if prefix:
        assert [rf for _, _, rf in parsed_filters[: len(prefix)]] == prefix
    if suffix:
        if any_minus_or_substring:
            assert [rf for _, _, rf in parsed_filters[-len(suffix) :]] == suffix
        else:
            assert [rf for _, _, rf in parsed_filters[-(len(suffix) + 1) : -1]] == suffix
