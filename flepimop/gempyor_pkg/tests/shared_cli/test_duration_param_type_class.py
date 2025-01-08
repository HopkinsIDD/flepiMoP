from datetime import timedelta
from typing import Any, Literal

from click.exceptions import BadParameter
import pytest

from gempyor.shared_cli import DurationParamType


@pytest.mark.parametrize("nonnegative", (True, False))
@pytest.mark.parametrize("value", ("abc", "$12.34", "12..3", "12years", "12.a2"))
def test_invalid_duration_bad_parameter(nonnegative: bool, value: str) -> None:
    duration = DurationParamType(nonnegative=nonnegative, default_unit="seconds")
    with pytest.raises(BadParameter, match="^'.*' is not a valid duration$"):
        duration.convert(value, None, None)


@pytest.mark.parametrize("value", ("-1", "-123", "-99.45", "-.9"))
def test_negative_duration_bad_parameter(value: str) -> None:
    duration = DurationParamType(nonnegative=True, default_unit="seconds")
    with pytest.raises(BadParameter, match="^'.*' is a negative duration$"):
        duration.convert(value, None, None)


@pytest.mark.parametrize("value", ("1", "-123", "99.45", "-.9"))
def test_unitless_duration_bad_paramter(value: str) -> None:
    duration = DurationParamType(nonnegative=False, default_unit=None)
    with pytest.raises(BadParameter, match="^'.*' is a unitless duration$"):
        duration.convert(value, None, None)


@pytest.mark.parametrize(
    ("value", "default_unit", "expected"),
    (
        ("1", "minutes", timedelta(minutes=1)),
        ("1", "days", timedelta(days=1)),
        ("2s", None, timedelta(seconds=2)),
        ("3hrs", None, timedelta(hours=3)),
        ("-4min", None, timedelta(minutes=-4)),
        ("-5d", None, timedelta(days=-5)),
        ("12.3", "seconds", timedelta(seconds=12.3)),
        ("12.3", "hours", timedelta(hours=12.3)),
        ("12.3", "weeks", timedelta(weeks=12.3)),
        ("-45.6h", None, timedelta(hours=-45.6)),
        ("-.1w", None, timedelta(weeks=-0.1)),
        ("0.0Weeks", "days", timedelta(weeks=0)),
    ),
)
def test_exact_results_for_select_inputs(
    value: str,
    default_unit: Literal["seconds", "minutes", "hours", "days", "weeks"] | None,
    expected: timedelta,
) -> None:
    duration = DurationParamType(nonnegative=False, default_unit=default_unit)
    assert duration.convert(value, None, None) == expected
    assert duration.convert(value.upper(), None, None) == expected
    assert duration.convert(value.lower(), None, None) == expected
