import random
from typing import Any

from click.exceptions import BadParameter
import pytest

from gempyor.shared_cli import MemoryParamType


@pytest.mark.parametrize("unit", ("Nope", "NO CHANCE", "wrong", "bb"))
def test_invalid_unit_value_error(unit: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^The `unit` given is not valid, given "
            f"'{unit.lower()}' and must be one of:.*.$"
        ),
    ):
        MemoryParamType(unit)


@pytest.mark.parametrize("value", ("1..2MB", "3.4cb", "56.abc", "-1GB"))
def test_invalid_value_bad_parameter(value: Any) -> None:
    memory = MemoryParamType("mb")
    with pytest.raises(BadParameter, match="^.* is not a valid memory size.$"):
        memory.convert(value, None, None)


@pytest.mark.parametrize("unit", MemoryParamType._units.keys())
@pytest.mark.parametrize(
    "number",
    [random.randint(1, 1000) for _ in range(3)]  # int
    + [random.random() for _ in range(3)]  # float without numbers left of decimal
    + [
        random.randint(1, 25) + random.random() for _ in range(3)
    ],  # float with numbers left of the decimal
)
def test_convert_acts_as_identity(unit: str, number: int) -> None:
    memory = MemoryParamType(unit)
    assert memory.convert(f"{number}{unit}".lstrip("0"), None, None) == number
    assert memory.convert(f"{number}{unit.upper()}".lstrip("0"), None, None) == number


@pytest.mark.parametrize(
    ("unit", "value", "expected"),
    (
        ("gb", "1.2gb", 1.2),
        ("kb", "1mb", 1024.0),
        ("gb", "30mb", 30.0 / 1024.0),
        ("kb", "2tb", 2.0 * (1024.0**3.0)),
        ("mb", "0.1gb", 0.1 * 1024.0),
    ),
)
def test_exact_results_for_select_inputs(unit: str, value: Any, expected: float) -> None:
    memory = MemoryParamType(unit)
    assert memory.convert(value, None, None) == expected
