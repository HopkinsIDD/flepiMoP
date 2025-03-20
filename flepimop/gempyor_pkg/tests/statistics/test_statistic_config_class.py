"""Unit tests for the `gempyor.statistics.StatisticConfig` class."""

import numpy as np
import pytest

from gempyor.statistics import StatisticConfig


@pytest.mark.parametrize("scale", ("invalid", "nonexistent", "false"))
def test_validate_scale(scale: str) -> None:
    """Test the `validate_scale` class method."""
    assert scale not in dir(np)
    with pytest.raises(
        ValueError,
        match=(
            f"Given an unsupported scale function, '{scale}', "
            "must be a valid numpy function."
        ),
    ):
        StatisticConfig.validate_scale(scale)
