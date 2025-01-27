from typing import Literal

import pytest

from gempyor.batch import _resolve_batch_system


@pytest.mark.parametrize(
    ("aws", "local", "slurm"),
    ((True, True, False), (False, True, True), (True, False, True), (True, True, True)),
)
def test_multiple_flags_value_error(aws: bool, local: bool, slurm: bool) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"^There were {sum((aws, local, slurm))} boolean "
            "flags given, expected either 0 or 1.$"
        ),
    ):
        _resolve_batch_system(None, aws, local, slurm)


@pytest.mark.parametrize(
    ("batch_system", "aws", "local", "slurm"),
    (
        ("aws", False, True, False),
        ("aws", False, False, True),
        ("slurm", True, False, False),
        ("slurm", False, True, False),
        ("local", True, False, False),
        ("local", False, False, True),
    ),
)
def test_batch_system_flag_mismatch_value_error(
    batch_system: Literal["aws", "local", "slurm"], aws: bool, local: bool, slurm: bool
) -> None:
    name = next(n for n, f in zip(("aws", "local", "slurm"), (aws, local, slurm)) if f)
    with pytest.raises(
        ValueError,
        match=(
            "^Conflicting batch systems given. The batch system name "
            f"is '{batch_system}' and the flags indicate '{name}'.$"
        ),
    ):
        _resolve_batch_system(batch_system, aws, local, slurm)


@pytest.mark.parametrize(
    ("batch_system", "aws", "local", "slurm", "expected"),
    (
        (None, True, False, False, "aws"),
        ("aws", False, False, False, "aws"),
        ("aws", True, False, False, "aws"),
        (None, False, True, False, "local"),
        ("local", False, False, False, "local"),
        ("local", False, True, False, "local"),
        (None, False, False, True, "slurm"),
        ("slurm", False, False, False, "slurm"),
        ("slurm", False, False, True, "slurm"),
    ),
)
def test_output_validation(
    batch_system: Literal["aws", "local", "slurm"] | None,
    aws: bool,
    local: bool,
    slurm: bool,
    expected: Literal["aws", "local", "slurm"],
) -> None:
    assert _resolve_batch_system(batch_system, aws, local, slurm) == expected
