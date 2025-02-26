from typing import Literal

import pytest
from pydantic import ValidationError

from gempyor.sync._sync import SyncOptions, SyncProtocols
from gempyor._pydantic_ext import _ensure_list

# test filter validation / construction for overrides, prefixes, and suffixes (see test__sync_filter.py)

# ensure that None vs [] is handled correctly for override

# ensure that all the other options properly bind / fail when given garbage


@pytest.mark.parametrize(
    "opts",
    [
        {"filter_override": "somefilter"},
        {"filter_override": ["somefilter", "another"]},
        {"filter_override": []},
    ],
)
def test_sync_opts_filters(opts: dict):
    """
    Ensures SyncOptions can instantiate valid objects w.r.t to filters
    """
    sp = SyncOptions(**opts)
    assert sp.filter_override == _ensure_list(opts["filter_override"])
