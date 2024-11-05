from datetime import timedelta
import re
from typing import Literal

import pytest

from gempyor.batch import JobTimeLimit


@pytest.mark.parametrize(
    "time_limit", (timedelta(), timedelta(hours=-1.0), timedelta(days=-3.0))
)
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
@pytest.mark.parametrize("batch_system", (None, "aws", "local", "slurm"))
def test_format_output_validation(
    time_limit: timedelta, batch_system: Literal["aws", "local", "slurm"] | None
) -> None:
    job_time_limit = JobTimeLimit(time_limit=time_limit)
    formatted_time_limit = job_time_limit.format(batch_system=batch_system)
    assert isinstance(formatted_time_limit, str)
    if batch_system == "slurm":
        assert re.match(r"^[0-9]+\:[0-9]{2}\:[0-9]{2}$", formatted_time_limit)
    else:
        assert formatted_time_limit.isdigit()


@pytest.mark.parametrize(
    ("time_limit", "batch_system", "expected"),
    (
        (timedelta(hours=1), None, "60"),
        (timedelta(seconds=20), None, "1"),
        (timedelta(days=2, hours=3, minutes=45), None, "3105"),
        (timedelta(hours=1), "slurm", "1:00:00"),
        (timedelta(seconds=20), "slurm", "0:00:20"),
        (timedelta(days=1, hours=2, minutes=34, seconds=5), "slurm", "26:34:05"),
    ),
)
def test_format_exact_results(
    time_limit: timedelta,
    batch_system: Literal["aws", "local", "slurm"] | None,
    expected: str,
) -> None:
    job_time_limit = JobTimeLimit(time_limit=time_limit)
    assert job_time_limit.format(batch_system=batch_system) == expected
