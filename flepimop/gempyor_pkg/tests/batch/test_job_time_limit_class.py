from datetime import timedelta
import re
from typing import Literal

import pytest

from gempyor.batch import BatchSystem, JobSize, JobTimeLimit


NONPOSITIVE_TIMEDELTAS = (timedelta(), timedelta(hours=-1.0), timedelta(days=-3.0))


@pytest.mark.parametrize("time_limit", NONPOSITIVE_TIMEDELTAS)
def test_time_limit_non_positive_value_error(time_limit: timedelta) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^The \`time\_limit\` attribute has [0-9\,\-]+ seconds\, "
            r"which is less than or equal to 0\.$"
        ),
    ):
        JobTimeLimit(time_limit=time_limit)


@pytest.mark.parametrize(
    "time_limit",
    (
        timedelta(hours=1),
        timedelta(hours=2, minutes=34, seconds=56),
        timedelta(days=1, seconds=3),
        timedelta(minutes=12345),
    ),
)
@pytest.mark.parametrize(
    "batch_system", (None, BatchSystem.AWS, BatchSystem.LOCAL, BatchSystem.SLURM)
)
def test_format_output_validation(
    time_limit: timedelta, batch_system: BatchSystem | None
) -> None:
    job_time_limit = JobTimeLimit(time_limit=time_limit)
    formatted_time_limit = job_time_limit.format(batch_system=batch_system)
    assert isinstance(formatted_time_limit, str)
    if batch_system == BatchSystem.SLURM:
        assert re.match(r"^[0-9]+\:[0-9]{2}\:[0-9]{2}$", formatted_time_limit)
    else:
        assert formatted_time_limit.isdigit()


@pytest.mark.parametrize(
    ("time_limit", "batch_system", "expected"),
    (
        (timedelta(hours=1), None, "60"),
        (timedelta(seconds=20), None, "1"),
        (timedelta(days=2, hours=3, minutes=45), None, "3105"),
        (timedelta(hours=1), BatchSystem.SLURM, "1:00:00"),
        (timedelta(seconds=20), BatchSystem.SLURM, "0:00:20"),
        (timedelta(days=1, hours=2, minutes=34, seconds=5), BatchSystem.SLURM, "26:34:05"),
    ),
)
def test_format_exact_results(
    time_limit: timedelta, batch_system: BatchSystem | None, expected: str
) -> None:
    job_time_limit = JobTimeLimit(time_limit=time_limit)
    assert job_time_limit.format(batch_system=batch_system) == expected


@pytest.mark.parametrize("time_per_simulation", NONPOSITIVE_TIMEDELTAS)
def test_from_per_simulation_time_per_simulation_nonpositive_value_error(
    time_per_simulation: timedelta,
) -> None:
    job_size = JobSize(jobs=1, simulations=1, blocks=1)
    with pytest.raises(
        ValueError,
        match=(
            r"^The \`time\_per\_simulation\` is \'[0-9\,\-]+\' seconds\, "
            r"which is less than or equal to 0\.$"
        ),
    ):
        JobTimeLimit.from_per_simulation_time(
            job_size, time_per_simulation, timedelta(minutes=10)
        )


@pytest.mark.parametrize("initial_time", NONPOSITIVE_TIMEDELTAS)
def test_from_per_simulation_initial_time_nonpositive_value_error(
    initial_time: timedelta,
) -> None:
    job_size = JobSize(jobs=1, simulations=1, blocks=1)
    with pytest.raises(
        ValueError,
        match=(
            r"^The \`initial\_time\` is \'[0-9\,\-]+\' seconds\, "
            r"which is less than or equal to 0\.$"
        ),
    ):
        JobTimeLimit.from_per_simulation_time(job_size, timedelta(minutes=10), initial_time)


@pytest.mark.parametrize(
    "job_size",
    (
        JobSize(jobs=1, simulations=10, blocks=1),
        JobSize(jobs=10, simulations=25, blocks=15),
    ),
)
@pytest.mark.parametrize(
    "time_per_simulation",
    (timedelta(minutes=5), timedelta(seconds=120), timedelta(hours=1.5)),
)
@pytest.mark.parametrize(
    "initial_time", (timedelta(minutes=10), timedelta(seconds=30), timedelta(hours=2))
)
def test_from_per_simulation_time(
    job_size: JobSize, time_per_simulation: timedelta, initial_time: timedelta
) -> None:
    job_time_limit = JobTimeLimit.from_per_simulation_time(
        job_size, time_per_simulation, initial_time
    )
    assert job_time_limit.time_limit >= initial_time

    double_job_time_limit = JobTimeLimit.from_per_simulation_time(
        job_size, 2 * time_per_simulation, initial_time
    )
    assert double_job_time_limit > job_time_limit
