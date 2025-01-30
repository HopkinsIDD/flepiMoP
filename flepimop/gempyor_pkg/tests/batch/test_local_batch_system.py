from datetime import timedelta

import pytest

from gempyor.batch import JobSize, LocalBatchSystem, get_batch_system


def test_local_batch_system_registered_by_default() -> None:
    batch_system = get_batch_system("local")
    assert isinstance(batch_system, LocalBatchSystem)
    assert batch_system.name == "local"


@pytest.mark.parametrize(
    ("jobs", "simulations", "blocks", "expected"),
    (
        (1, 1, 1, JobSize(jobs=1, simulations=1, blocks=1)),
        (2, 1, 1, JobSize(jobs=1, simulations=1, blocks=1)),
        (1, 20, 1, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 1, 20, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 5, 5, JobSize(jobs=1, simulations=10, blocks=1)),
        (5, 5, 5, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 10, 1, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 1, 10, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 2, 5, JobSize(jobs=1, simulations=10, blocks=1)),
        (1, 3, 3, JobSize(jobs=1, simulations=9, blocks=1)),
    ),
)
def test_size_from_jobs_simulations_blocks_for_select_values(
    jobs: int, simulations: int, blocks: int, expected: JobSize
) -> None:
    batch_system = get_batch_system("local")
    jobs_warning = jobs != 1
    simulations_warning = blocks * simulations > 10
    if jobs_warning and simulations_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 1 job "
                f"but was given {jobs}, overriding.$"
            ),
        ):
            with pytest.warns(
                UserWarning,
                match=(
                    "^Local batch system only supports 10 blocks x simulations "
                    f"but was given {blocks * simulations}, overriding.$"
                ),
            ):
                assert (
                    batch_system.size_from_jobs_simulations_blocks(
                        jobs, simulations, blocks
                    )
                    == expected
                )
    elif jobs_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 1 job "
                f"but was given {jobs}, overriding.$"
            ),
        ):
            assert (
                batch_system.size_from_jobs_simulations_blocks(jobs, simulations, blocks)
                == expected
            )
    elif simulations_warning:
        with pytest.warns(
            UserWarning,
            match=(
                "^Local batch system only supports 10 blocks x simulations "
                f"but was given {blocks * simulations}, overriding.$"
            ),
        ):
            assert (
                batch_system.size_from_jobs_simulations_blocks(jobs, simulations, blocks)
                == expected
            )
    else:
        assert (
            batch_system.size_from_jobs_simulations_blocks(jobs, simulations, blocks)
            == expected
        )
