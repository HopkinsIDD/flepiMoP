"""Unit tests for the `gempyor.statistics.StatisticRegularizeConfig` class."""

import pytest

from gempyor.statistics import _AVAILABLE_REGULARIZATIONS, StatisticRegularizeConfig


@pytest.mark.parametrize("name", ("invalid", "false", "nonexistent"))
def test_validate_name(name: str) -> None:
    """Test the `validate_name` class method."""
    assert name not in _AVAILABLE_REGULARIZATIONS
    with pytest.raises(
        ValueError,
        match=(
            f"Given an unsupported regularization name, '{name}', " f"must be one of: .*."
        ),
    ):
        StatisticRegularizeConfig.validate_name(name)
