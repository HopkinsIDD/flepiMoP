import random

from click.exceptions import BadParameter
import pytest

from gempyor._click import MemoryParamType


@pytest.mark.parametrize("unit", ("Nope", "NO CHANCE", "wrong", "bb"))
def test_invalid_unit_value_error(unit: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^The `unit` given is not valid, given "
            f"'{unit.lower()}' and must be one of:.*.$"
        ),
    ):
        MemoryParamType(False, unit, True)


@pytest.mark.parametrize("value", ("1..2MB", "3.4cb", "56.abc", "-1GB"))
def test_invalid_value_bad_parameter(value: str) -> None:
    memory = MemoryParamType(False, "mb", True)
    with pytest.raises(BadParameter, match="^.* is not a valid memory size.$"):
        memory.convert(value, None, None)


@pytest.mark.parametrize("value", ("1", "123", "99.45", ".9"))
def test_unitless_value_bad_parameter(value: str) -> None:
    memory = MemoryParamType(False, "mb", False)
    with pytest.raises(BadParameter, match="^'.*' is a unitless memory size.$"):
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
    memory = MemoryParamType(as_int, unit, True)
    for u in (unit, unit.upper()):
        result = memory.convert(f"{number}{u}".lstrip("0"), None, None)
        assert isinstance(result, int if as_int else float)
        assert abs(result - number) <= 1 if as_int else result == number


@pytest.mark.parametrize(
    ("as_int", "unit", "allow_unitless", "value", "expected"),
    (
        (False, "gb", False, "1.2gb", 1.2),
        (True, "gb", False, "1.2gb", 2),
        (False, "kb", False, "1mb", 1024.0),
        (True, "kb", False, "1mb", 1024),
        (False, "gb", False, "30mb", 30.0 / 1024.0),
        (True, "gb", False, "30mb", 1),
        (False, "kb", False, "2tb", 2.0 * (1024.0**3.0)),
        (True, "kb", False, "2tb", 2147483648),
        (False, "mb", False, "0.1gb", 0.1 * 1024.0),
        (True, "mb", False, "0.1gb", 103),
        (False, "gb", True, "4", 4.0),
        (True, "gb", True, "4", 4),
        (False, "mb", True, "1234.56", 1234.56),
        (True, "mb", True, "1234.56", 1235),
    ),
)
def test_exact_results_for_select_inputs(
    unit: str, as_int: bool, allow_unitless: bool, value: str, expected: float | int
) -> None:
    memory = MemoryParamType(as_int, unit, allow_unitless)
    assert memory.convert(value, None, None) == expected
