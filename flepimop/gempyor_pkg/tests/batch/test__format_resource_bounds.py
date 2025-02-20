import random
from typing import Literal

import pytest

from gempyor.batch import _format_resource_bounds


@pytest.mark.parametrize(
    "bounds",
    (
        {},
        {"cpu": 1.0},
        {"memory": 1.0},
        {"time": 1.0},
        {"cpu": 1.0, "memory": 1.0},
        {"cpu": 1.0, "time": 1.0},
        {"memory": 1.0, "time": 1.0},
        {"cpu": 1.0, "memory": 1.0, "time": 1.0},
        {"cpu": 3.45678, "memory": 3.45678, "time": 3.45678},
        {"cpu": random.random(), "memory": random.random(), "time": random.random()},
    ),
)
def test_output_validation(bounds: dict[Literal["cpu", "memory", "time"], float]) -> None:
    formatted_bounds = _format_resource_bounds(bounds)
    assert isinstance(formatted_bounds, str)
    for v in bounds.values():
        assert f"{v:.2f}" in formatted_bounds
