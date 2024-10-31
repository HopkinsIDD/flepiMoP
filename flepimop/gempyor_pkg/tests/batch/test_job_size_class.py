from typing import Literal

import pytest

from gempyor.batch import JobSize


@pytest.mark.parametrize(
    "kwargs",
    (
        {"jobs": 0, "simulations": 1, "blocks": 1},
        {"jobs": 1, "simulations": 0, "blocks": 1},
        {"jobs": 1, "simulations": 1, "blocks": 0},
        {"jobs": 0, "simulations": 0, "blocks": 1},
        {"jobs": 1, "simulations": 0, "blocks": 0},
        {"jobs": 0, "simulations": 1, "blocks": 0},
        {"jobs": 0, "simulations": 0, "blocks": 0},
    ),
)
def test_less_than_one_value_error(
    kwargs: dict[Literal["jobs", "simulations", "blocks"], int]
) -> None:
    param = next(k for k, v in kwargs.items() if v < 1)
    with pytest.raises(
        ValueError,
        match=(
            f"^The '{param}' attribute must be greater than 0, "
            f"but instead was given '{kwargs.get(param)}'.$"
        ),
    ):
        JobSize(**kwargs)
