from typing import Literal

import pytest
from pydantic import ValidationError, TypeAdapter

from gempyor.sync._sync_filter import *

# include vs exclude filters successfully parsed, including default include

# convert convert single filters to list

# handle single include filter expanding to add an exclude all

# valid vs invalid filters successfully parsed
@pytest.mark.parametrize(
    "filter", [
        "somefilter",
        ["somefilter", "another"],
        [],
    ],
)
def test_valid_filters(filter: list[str] | str):
    """
    A string that doesn't start with a space or a plus/minus without a
    subsequent space is a valid filter
    """
    _ = TypeAdapter(ListSyncFilter).validate_python(filter)

@pytest.mark.parametrize(
    "filter", (
        " somefilter",
        "-something",
        "+something",
        1,
    ),
)
def test_invalid_filters(filter: list[str] | str):
    """
    a string that starts with a space or a plus/minus without a subsequent space
    is an invalid filter
    """
    with pytest.raises(ValidationError):
        _ = TypeAdapter(ListSyncFilter).validate_python(filter)