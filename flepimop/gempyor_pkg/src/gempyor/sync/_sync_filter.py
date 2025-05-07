"""Sync filtering utilities for sync protocols provided by `gempyor`."""

__all__ = ("FilterParts", "ListSyncFilter", "WithFilters")


import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from .._pydantic_ext import _ensure_list, _override_or_val

# filters can begin with a `+` or `-` followed by a space,
# but not just those symbols or a space
_FRCOMPILED: Final = re.compile(r"^([\+\-] )?")

SyncFilter = Annotated[str, Field(pattern=r"^([\+\-] |[^\+\- ])")]

ListSyncFilter = Annotated[list[SyncFilter], BeforeValidator(_ensure_list)]
_LIST_SYNC_FILTER_ADAPTER: Final = TypeAdapter(ListSyncFilter)

# mode, pattern, whole specification (for debugging)
FilterParts = tuple[Literal["+", "-"], str, str]


def _filter_mode(filter_: SyncFilter) -> Literal["+", "-"]:
    """
    Returns the mode of a filter.

    Args:
        filter: The sync filter string to parse.

    Returns:
        A string representing the mode, either `+` or `-`.

    Examples:
        >>> from gempyor.sync._sync_filter import _filter_mode
        >>> _filter_mode("+ /foo/bar.txt")
        '+'
        >>> _filter_mode("- /fizz/buzz.txt")
        '-'
        >>> _filter_mode("/a/b/c.txt")
        '+'

    """
    return "-" if filter_.startswith("- ") else "+"


def _filter_pattern(filter_: SyncFilter) -> str:
    """
    Removes the mode from a filter string.

    Args:
        filter: The sync filter string to parse.

    Returns:
        The filter string without the mode, '+ ' or '- ', prefix.

    Examples:
        >>> from gempyor.sync._sync_filter import _filter_pattern
        >>> _filter_pattern("+ /foo/bar.txt")
        '/foo/bar.txt'
        >>> _filter_pattern("- /fizz/buzz.txt")
        '/fizz/buzz.txt'
        >>> _filter_pattern("/a/b/c.txt")
        '/a/b/c.txt'

    """
    return _FRCOMPILED.sub("", filter_)


def _filter_parse(filter_: SyncFilter) -> FilterParts:
    """
    Parses a filter string into its mode, pattern, and whole specification.

    Args:
        filter: The sync filter string to parse.

    Returns:
        A tuple containing the mode, pattern, and the original filter string.

    Examples:
        >>> from gempyor.sync._sync_filter import _filter_parse
        >>> _filter_parse("+ /foo/bar.txt")
        ('+', '/foo/bar.txt', '+ /foo/bar.txt')
        >>> _filter_parse("- /fizz/buzz.txt")
        ('-', '/fizz/buzz.txt', '- /fizz/buzz.txt')
        >>> _filter_parse("/a/b/c.txt")
        ('+', '/a/b/c.txt', '/a/b/c.txt')

    """
    return (_filter_mode(filter_), _filter_pattern(filter_), filter_)


class WithFilters(BaseModel, ABC):
    """
    A filters mixin for `SyncABC` models that provide a `filters` attribute.

    Attributes:
        filters: A list of filters to be applied to the sync operation.
    """

    filters: ListSyncFilter = []

    def list_filters(
        self,
        overrides: ListSyncFilter | None = None,
        prefix: ListSyncFilter | None = None,
        suffix: ListSyncFilter | None = None,
        reverse: bool = False,
    ) -> list[FilterParts]:
        """
        Create a list of filters in the order they should be applied.

        This method combines the default filters with any overrides, prefix, and suffix
        filters. The order of the filters is determined by the following rules:
        1. The filters provided in the `overrides` argument take precedence over the
            default filters.
        2. The filters in the `prefix` argument are added to the beginning of the filter
            list.
        3. The filters in the `suffix` argument are added to the end of the filter list.
        4. If `reverse` is True, the order of the filters is reversed.

        Args:
            overrides: A list of filters to override the default filters.
            prefix: A list of filters to be added to the beginning of the filter list.
            suffix: A list of filters to be added to the end of the filter list.
            reverse: If True, reverse the order of the filters.

        Returns:
            A list of tuples containing the mode, pattern, and whole specification of
            each filter to be applied in order.
        """
        if not overrides is None:
            overrides = _LIST_SYNC_FILTER_ADAPTER.validate_python(overrides)
        prefix = _LIST_SYNC_FILTER_ADAPTER.validate_python(prefix or [])
        suffix = _LIST_SYNC_FILTER_ADAPTER.validate_python(suffix or [])
        chn = chain(prefix, _override_or_val(overrides, self.filters), suffix)
        res = [_filter_parse(f) for f in chn]
        if len(res) and all(m == "+" for m, _, _ in res):
            res.append(_filter_parse("- **"))
        if reverse:
            res.reverse()
        return res

    def format_filters(
        self,
        overrides: ListSyncFilter | None = None,
        prefix: ListSyncFilter | None = None,
        suffix: ListSyncFilter | None = None,
        reverse: bool = False,
    ) -> list[str]:
        """
        Create a list of formatted filters in the order they should be applied.

        A thin wrapper around :method list_filters: that applies formatting (supplied by
        the child class via `_formatter`) to the filters.

        Args:
            overrides: A list of filters to override the default filters.
            prefix: A list of filters to be added to the beginning of the filter list.
            suffix: A list of filters to be added to the end of the filter list.
            reverse: If True, reverse the order of the filters.

        Returns:
            A list of formatted strings representing the filters to be applied in order.
        """
        return list(
            chain.from_iterable(
                self._formatter(f)
                for f in self.list_filters(overrides, prefix, suffix, reverse)
            )
        )

    @staticmethod
    @abstractmethod
    def _formatter(f: FilterParts) -> list[str]:
        """
        Internal method to format a filter.

        Args:
            f: A tuple containing the mode, pattern, and whole specification of a
                filter.

        Returns:
            A list of formatted strings representing the filter.
        """
