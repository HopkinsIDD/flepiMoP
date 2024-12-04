from datetime import timedelta
from typing import Any

from click.exceptions import BadParameter
import pytest

from gempyor.shared_cli import DurationParamType


@pytest.mark.parametrize("nonnegative", (True, False))
@pytest.mark.parametrize("value", ("abc", "$12.34", "12..3", "12years", "12.a2"))
def test_invalid_duration_bad_parameter(nonnegative: bool, value: Any) -> None:
    duration = DurationParamType(nonnegative=nonnegative)
    with pytest.raises(BadParameter, match="^'.*' is not a valid duration$"):
        duration.convert(value, None, None)


@pytest.mark.parametrize("value", ("-1", "-123", "-99.45", "-.9"))
def test_negative_duration_bad_parameter(value: Any) -> None:
    duration = DurationParamType(nonnegative=True)
    with pytest.raises(BadParameter, match="^'.*' is a negative duration$"):
        duration.convert(value, None, None)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("1", timedelta(minutes=1)),
        ("2s", timedelta(seconds=2)),
        ("3hrs", timedelta(hours=3)),
        ("-4min", timedelta(minutes=-4)),
        ("-5d", timedelta(days=-5)),
        ("12.3", timedelta(minutes=12.3)),
        ("-45.6h", timedelta(hours=-45.6)),
        ("-.1w", timedelta(weeks=-0.1)),
    ),
)
def test_exact_results_for_select_inputs(value: Any, expected: timedelta) -> None:
    duration = DurationParamType(nonnegative=False)
    assert duration.convert(value, None, None) == expected
