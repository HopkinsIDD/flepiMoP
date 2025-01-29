from typing import Literal

import pytest

from gempyor.batch import BatchSystem, JobResources, JobSize


@pytest.mark.parametrize(
    "kwargs",
    (
        {"nodes": 0, "cpus": 1, "memory": 1},
        {"nodes": 1, "cpus": 0, "memory": 1},
        {"nodes": 1, "cpus": 1, "memory": 0},
        {"nodes": 0, "cpus": 0, "memory": 1},
        {"nodes": 1, "cpus": 0, "memory": 0},
        {"nodes": 0, "cpus": 1, "memory": 0},
        {"nodes": 0, "cpus": 0, "memory": 0},
    ),
)
def test_less_than_one_value_error(
    kwargs: dict[Literal["nodes", "cpus", "memory"], int]
) -> None:
    param = next(k for k, v in kwargs.items() if v < 1)
    with pytest.raises(
        ValueError,
        match=(
            f"^The '{param}' attribute must be greater than 0, "
            f"but instead was given '{kwargs.get(param)}'.$"
        ),
    ):
        JobResources(**kwargs)


@pytest.mark.parametrize("nodes", (1, 2, 4, 8))
@pytest.mark.parametrize("cpus", (1, 2, 4, 8))
@pytest.mark.parametrize("memory", (1024, 2 * 1024, 4 * 1024, 8 * 1024))
def test_instance_attributes(nodes: int, cpus: int, memory: int) -> None:
    job_resources = JobResources(nodes=nodes, cpus=cpus, memory=memory)
    assert job_resources.total_cpus >= cpus
    assert job_resources.total_memory >= memory
    assert job_resources.total_resources() == (nodes, nodes * cpus, nodes * memory)
