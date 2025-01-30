from datetime import timedelta

import pytest

from gempyor.batch import JobResources, SlurmBatchSystem, get_batch_system


def test_slurm_batch_system_registered_by_default() -> None:
    batch_system = get_batch_system("slurm")
    assert isinstance(batch_system, SlurmBatchSystem)
    assert batch_system.name == "slurm"


@pytest.mark.parametrize(
    ("job_resources", "expected"),
    (
        (JobResources(nodes=1, cpus=1, memory=1), "1MB"),
        (JobResources(nodes=1, cpus=1, memory=1024), "1024MB"),
    ),
)
def test_memory_formatting_for_select_values(
    job_resources: JobResources, expected: str
) -> None:
    batch_system = get_batch_system("slurm")
    assert batch_system.format_memory(job_resources) == expected


@pytest.mark.parametrize(
    ("job_time_limit", "expected"),
    (
        (timedelta(hours=1), "1:00:00"),
        (timedelta(hours=1, minutes=30), "1:30:00"),
        (timedelta(days=1), "24:00:00"),
        (timedelta(days=1, hours=2, minutes=34, seconds=56), "26:34:56"),
    ),
)
def test_time_limit_formatting_for_select_values(
    job_time_limit: timedelta, expected: str
) -> None:
    batch_system = get_batch_system("slurm")
    assert batch_system.format_time_limit(job_time_limit) == expected
