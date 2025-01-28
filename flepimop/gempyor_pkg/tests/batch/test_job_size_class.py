from itertools import product
from typing import Generator, Literal

import pytest

from gempyor.batch import BatchSystem, JobSize


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


# def generate_from_jobs_simulations_blocks(
#     *args: int | None,
# ) -> Generator[tuple[int | None, ...], None, None]:
#     for combo in product(args, repeat=3):
#         yield combo


# @pytest.mark.parametrize("combo", generate_from_jobs_simulations_blocks(1, 5, 10))
# def test_from_jobs_simulations_blocks_output(combo: tuple[int, ...]) -> None:
#     jobs, simulations, blocks = combo
#     for inference_method in (None, ""):
#         for batch_system in (None, BatchSystem.AWS, BatchSystem.LOCAL, BatchSystem.SLURM):
#             job_size = JobSize.from_jobs_simulations_blocks(
#                 jobs, simulations, blocks, inference_method, batch_system
#             )
#             assert job_size.jobs >= 1
#             assert job_size.simulations >= simulations
#             assert job_size.blocks >= 1


# @pytest.mark.parametrize(
#     ("jobs", "simulations", "blocks", "inference_method", "batch_system", "expected"),
#     (
#         (4, 32, 8, None, None, JobSize(jobs=4, simulations=32, blocks=8)),
#         (4, 32, 8, None, BatchSystem.LOCAL, JobSize(jobs=1, simulations=10, blocks=1)),
#         (4, 4, 1, "emcee", BatchSystem.LOCAL, JobSize(jobs=1, simulations=4, blocks=1)),
#         (4, 16, 8, "emcee", None, JobSize(jobs=4, simulations=8 * 16, blocks=1)),
#     ),
# )
# def test_from_jobs_simulations_blocks_exact_results_for_select_inputs(
#     jobs: int | None,
#     simulations: int | None,
#     blocks: int | None,
#     inference_method: Literal["emcee"] | None,
#     batch_system: BatchSystem,
#     expected: JobSize,
# ) -> None:
#     job_size = JobSize.from_jobs_simulations_blocks(
#         jobs, simulations, blocks, inference_method, batch_system
#     )
#     assert job_size == expected
