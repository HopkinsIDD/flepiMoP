from itertools import product
from typing import Generator, Literal

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


def generate_size_from_jobs_sims_blocks(
    *args: int | None,
) -> Generator[tuple[int | None, ...], None, None]:
    for combo in product(args, repeat=3):
        yield combo


@pytest.mark.parametrize("combo", generate_size_from_jobs_sims_blocks(1, 5, 10))
def test_size_from_jobs_sims_blocks_output(combo: tuple[int, ...]) -> None:
    jobs, simulations, blocks = combo
    for inference_method in (None, ""):
        job_size = JobSize.size_from_jobs_sims_blocks(
            jobs, simulations, blocks, inference_method
        )
        assert job_size.jobs == jobs
        assert job_size.simulations >= simulations
        assert job_size.blocks >= 1
