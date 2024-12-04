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
@pytest.mark.parametrize("as_int", (True, False))
@pytest.mark.parametrize(
    "number",
    [random.randint(1, 1000) for _ in range(3)]  # int
    + [random.random() for _ in range(3)]  # float without numbers left of decimal
    + [
        random.randint(1, 25) + random.random() for _ in range(3)
    ],  # float with numbers left of the decimal
)
def test_convert_acts_as_identity(unit: str, as_int: bool, number: int | float) -> None:
    memory = MemoryParamType(unit, as_int=as_int)
    for u in (unit, unit.upper()):
        result = memory.convert(f"{number}{u}".lstrip("0"), None, None)
        assert isinstance(result, int if as_int else float)
        assert abs(result - number) <= 1 if as_int else result == number


@pytest.mark.parametrize(
    ("unit", "as_int", "value", "expected"),
    (
        ("gb", False, "1.2gb", 1.2),
        ("gb", True, "1.2gb", 2),
        ("kb", False, "1mb", 1024.0),
        ("kb", True, "1mb", 1024),
        ("gb", False, "30mb", 30.0 / 1024.0),
        ("gb", True, "30mb", 1),
        ("kb", False, "2tb", 2.0 * (1024.0**3.0)),
        ("kb", True, "2tb", 2147483648),
        ("mb", False, "0.1gb", 0.1 * 1024.0),
        ("mb", True, "0.1gb", 103),
        ("gb", False, "4", 4.0),
        ("gb", True, "4", 4),
        ("mb", False, "1234.56", 1234.56),
        ("mb", True, "1234.56", 1235),
    ),
)
def test_exact_results_for_select_inputs(
    unit: str, as_int: bool, value: Any, expected: float | int
) -> None:
    memory = MemoryParamType(unit, as_int=as_int)
    assert memory.convert(value, None, None) == expected
