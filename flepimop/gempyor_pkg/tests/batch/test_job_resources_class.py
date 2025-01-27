from typing import Literal

import pytest

from gempyor.batch import JobResources


@pytest.mark.parametrize(
    "kwargs",
    (
        {"nodes": 0, "cpus": 1, "memory": 1},
        {"nodes": 1, "cpus": 0, "memory": 1},
        {"nodes": 1, "cpus": 1, "memory": 0},
        {"nodes": 0, "cpus": 0, "memory": 1},
        {"nodes": 1, "cpus": 0, "memory": 0},
        {"nodes": 0, "cpus": 1, "memory": 0},
        {"nodes": 0, "cpus": 0, "memory": 0},
    ),
)
def test_less_than_one_value_error(
    kwargs: dict[Literal["nodes", "cpus", "memory"], int]
) -> None:
    param = next(k for k, v in kwargs.items() if v < 1)
    with pytest.raises(
        ValueError,
        match=(
            f"^The '{param}' attribute must be greater than 0, "
            f"but instead was given '{kwargs.get(param)}'.$"
        ),
    ):
        JobResources(**kwargs)
