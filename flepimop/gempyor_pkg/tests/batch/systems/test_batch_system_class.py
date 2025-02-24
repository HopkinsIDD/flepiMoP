from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

from confuse import Configuration
import pytest

from gempyor.batch import BatchSystem, JobResources, JobSize, JobSubmission


class SimpleBatchSystem(BatchSystem):
    name = "simple"

    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
    ) -> JobSubmission | None:
        return None


@pytest.mark.parametrize(
    "resources",
    (
        JobResources(nodes=1, cpus=1, memory=1),
        JobResources(nodes=1, cpus=1, memory=1024),
        JobResources(nodes=4, cpus=8, memory=16 * 1024),
    ),
)
def test_job_resource_formatting(resources: JobResources) -> None:
    batch_system = SimpleBatchSystem()
    assert isinstance(batch_system.format_nodes(resources), str)
    assert isinstance(batch_system.format_cpus(resources), str)
    assert isinstance(batch_system.format_memory(resources), str)


@pytest.mark.parametrize(
    "time_limit",
    (
        timedelta(hours=1),
        timedelta(days=1, hours=2, minutes=34, seconds=56),
        timedelta(days=1),
        timedelta(hours=1, minutes=30),
    ),
)
def test_job_time_limit_formatting(time_limit: timedelta) -> None:
    batch_system = SimpleBatchSystem()
    assert isinstance(batch_system.format_time_limit(time_limit), str)


@pytest.mark.parametrize(
    ("blocks", "chains", "samples", "simulations"),
    ((1, 1, 1, 4), (2, 3, 4, 8), (10, 25, 100, 250)),
)
def test_size_from_jobs_simulations_blocks(
    blocks: int, chains: int, samples: int, simulations: int
) -> None:
    batch_system = SimpleBatchSystem()
    size = batch_system.size_from_jobs_simulations_blocks(
        blocks, chains, samples, simulations
    )
    assert isinstance(size, JobSize)
    assert size == JobSize(
        blocks=blocks, chains=chains, samples=samples, simulations=simulations
    )


def test_options_from_config_and_cli() -> None:
    batch_system = SimpleBatchSystem()
    assert (
        batch_system.options_from_config_and_cli(
            Configuration("foobar", read=False), {}, None
        )
        is None
    )
