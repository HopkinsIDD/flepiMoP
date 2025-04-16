"""Unit tests for `gempyor.utils._nslots_random_seeds`."""

import random

import pytest

from gempyor.utils import _nslots_random_seeds


@pytest.mark.parametrize("nslots", [0, 1, 10, 25, 100])
def test_nslots_random_seeds(nslots: int) -> None:
    """Test that the output of _nslots_random_seeds is a list of integers."""
    random.seed(123)

    result1 = _nslots_random_seeds(nslots)
    assert isinstance(result1, list) and all(isinstance(i, int) for i in result1)
    assert len(result1) == len(set(result1)) == nslots
    assert all(nslots < i < 2**32 for i in result1)

    result2 = _nslots_random_seeds(nslots)
    assert isinstance(result2, list) and all(isinstance(i, int) for i in result2)
    assert len(result2) == len(set(result2)) == nslots
    assert all(nslots < i < 2**32 for i in result2)

    assert (result1 != result2) if nslots > 0 else (result1 == result2 == [])
