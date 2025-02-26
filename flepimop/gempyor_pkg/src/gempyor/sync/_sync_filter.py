import re
from typing import Annotated, Literal
from itertools import chain
from abc import abstractmethod

from pydantic import BaseModel, Field, BeforeValidator, TypeAdapter

from .._pydantic_ext import _ensure_list, _override_or_val

__all__ = ["ListSyncFilter", "FilterParts", "WithFilters"]

# filters can begin with a `+` or `-` followed by a space, but not just those symbols or a space
FilterRegex = r"^([\+\-] |[^\+\- ])"
frcompiled = re.compile(r"^([\+\-] )?")

SyncFilter = Annotated[str, Field(pattern=FilterRegex)]

ListSyncFilter = Annotated[list[SyncFilter], BeforeValidator(_ensure_list)]

# mode, pattern, whole specification (for debugging)
FilterParts = tuple[Literal["+", "-"], str, str]


def _filter_mode(filter: SyncFilter) -> Literal["+", "-"]:
    return "-" if filter.startswith("- ") else "+"


def _filter_pattern(filter: SyncFilter) -> str:
    return frcompiled.sub("", filter)


def _filter_parse(filter: SyncFilter) -> FilterParts:
    return (_filter_mode(filter), _filter_pattern(filter), filter)


class WithFilters(BaseModel):
    """
    A mixin to be applied to SyncModels that have a `sync::filters` key.
    :method list_filters: creates a list filters in the order they should be applied,
      including overrides, prefix, and suffix filters. By default, in the `rsync` order convention,
      but reversible.
    :method format_filters: creates the formatted version of :method list_filters:.
    """

    filters: ListSyncFilter = []

    def list_filters(
        self,
        overrides: ListSyncFilter | None = None,
        prefix: ListSyncFilter = [],
        suffix: ListSyncFilter = [],
        reverse: bool = False,
    ) -> list[FilterParts]:

        if not overrides is None:
            overrides = TypeAdapter(ListSyncFilter).validate_python(overrides)
        prefix = TypeAdapter(ListSyncFilter).validate_python(prefix)
        suffix = TypeAdapter(ListSyncFilter).validate_python(suffix)

        chn = chain(prefix, _override_or_val(overrides, self.filters), suffix)
        res = [_filter_parse(f) for f in chn]
        if (res.__len__() > 0) and all(f[0] == "+" for f in res):
            res.append(_filter_parse("- **"))
        if reverse:
            res.reverse()
        return res

    def format_filters(
        self,
        overrides: ListSyncFilter | None = None,
        prefix: ListSyncFilter = [],
        suffix: ListSyncFilter = [],
        reverse: bool = False,
    ) -> list[str]:
        return list(
            chain.from_iterable(
                self._formatter(f)
                for f in self.list_filters(overrides, prefix, suffix, reverse)
            )
        )

    @abstractmethod
    def _formatter(self, f: FilterParts) -> list[str]: ...
