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


@pytest.mark.parametrize(("simulations", "blocks"), [(None, None), (1, None), (None, 1)])
def test_size_from_jobs_sims_blocks_iteration_value_error(
    simulations: int | None, blocks: int | None
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^If simulations and blocks are not all explicitly "
            "provided then an iterations per slot must be given.$"
        ),
    ):
        JobSize.size_from_jobs_sims_blocks(1, simulations, blocks, None, 1, 1, "aws")


def test_size_from_jobs_sims_blocks_slots_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="^If jobs is not explicitly provided, it must be given via slots.$",
    ):
        JobSize.size_from_jobs_sims_blocks(None, 1, 1, 1, None, 1, "aws")


def test_size_from_jobs_sims_blocks_subpops_value_error() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^If simulations and blocks are not explicitly "
            "provided, then a subpops must be given.$"
        ),
    ):
        JobSize.size_from_jobs_sims_blocks(1, None, None, 1, 1, None, "aws")


def generate_size_from_jobs_sims_blocks(
    *args: int | None,
) -> Generator[tuple[int | None, ...], None, None]:
    for combo in product(args, repeat=6):
        jobs, simulations, blocks, iterations_per_slot, slots, subpops = combo
        if iterations_per_slot is None and (simulations is None or blocks is None):
            continue
        elif jobs is None and slots is None:
            continue
        elif simulations is None and blocks is None and subpops is None:
            continue
        yield combo


@pytest.mark.parametrize("combo", generate_size_from_jobs_sims_blocks(None, 1, 10))
def test_size_from_jobs_sims_blocks_output(combo: tuple[int | None, ...]) -> None:
    jobs, simulations, blocks, iterations_per_slot, slots, subpops = combo
    job_sizes_by_batch_system = {}
    for batch_system in ("aws", "local", "slurm"):
        job_size = JobSize.size_from_jobs_sims_blocks(
            jobs, simulations, blocks, iterations_per_slot, slots, subpops, batch_system
        )
        assert (
            job_size.jobs == jobs
            if jobs is not None
            else isinstance(job_size.jobs, int) and job_size.jobs > 0
        )
        assert (
            job_size.simulations == simulations
            if simulations is not None
            else isinstance(job_size.simulations, int) and job_size.simulations > 0
        )
        assert (
            job_size.blocks == blocks
            if blocks is not None
            else isinstance(job_size.blocks, int) and job_size.blocks > 0
        )
        job_sizes_by_batch_system[batch_system] = job_size
    assert job_sizes_by_batch_system["local"] == job_sizes_by_batch_system["slurm"]
    assert job_sizes_by_batch_system["local"].jobs == job_sizes_by_batch_system["aws"].jobs
