from datetime import datetime, timedelta, timezone
import re

import pytest

from gempyor.batch import _job_name


VALID_JOB_NAME = re.compile(
    r"^([a-z]{1}([a-z0-9\_\-]+)?\-)?[0-9]{8}T[0-9]{6}$", flags=re.IGNORECASE
)


@pytest.mark.parametrize("name", ("123", "_abc", "", "abc!@#"))
def test_invalid_name_value_error(name: str) -> None:
    with pytest.raises(
        ValueError, match=f"^The given `name`, '{name}', is not a valid safe name.$"
    ):
        _job_name(name, None)


@pytest.mark.parametrize("name", (None, "abc", "flu_usa", "covid-19"))
@pytest.mark.parametrize(
    "timestamp",
    (
        None,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=3))),
    ),
)
def test_output_validation(name: str | None, timestamp: datetime | None) -> None:
    job_name = _job_name(name, timestamp)
    if name is not None:
        assert job_name.startswith(name)
    if timestamp is not None:
        assert job_name.endswith(timestamp.strftime("%Y%m%dT%H%M%S"))
    assert VALID_JOB_NAME.match(job_name)
