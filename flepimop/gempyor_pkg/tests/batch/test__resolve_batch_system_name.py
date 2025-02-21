import pytest

from gempyor.batch import _resolve_batch_system_name


def test_multiple_flags_value_error() -> None:
    with pytest.raises(
        ValueError,
        match=f"^There were 2 boolean flags given, expected either 0 or 1.$",
    ):
        _resolve_batch_system_name(None, True, True)


@pytest.mark.parametrize(
    ("name", "local", "slurm"),
    (
        ("local", False, True),
        ("slurm", True, False),
    ),
)
def test_name_flag_mismatch_value_error(name: str, local: bool, slurm: bool) -> None:
    flag_name = "local" if local else "slurm"
    with pytest.raises(
        ValueError,
        match=(
            "^Conflicting batch systems given. The batch system name "
            f"is '{name}' and the flags indicate '{flag_name}'.$"
        ),
    ):
        _resolve_batch_system_name(name, local, slurm)


@pytest.mark.parametrize(
    ("name", "local", "slurm", "expected"), (("LoCaL", False, False, "local"),)
)
def test_exact_output_for_select_values(
    name: str | None, local: bool, slurm: bool, expected: str
) -> None:
    assert _resolve_batch_system_name(name, local, slurm) == expected
