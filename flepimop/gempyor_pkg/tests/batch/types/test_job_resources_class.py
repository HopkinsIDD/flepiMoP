from typing import Literal

import pytest

from gempyor.batch import JobResources


@pytest.mark.parametrize("nodes", (1, 2, 4, 8))
@pytest.mark.parametrize("cpus", (1, 2, 4, 8))
@pytest.mark.parametrize("memory", (1024, 2 * 1024, 4 * 1024, 8 * 1024))
def test_instance_attributes(nodes: int, cpus: int, memory: int) -> None:
    job_resources = JobResources(nodes=nodes, cpus=cpus, memory=memory)
    assert job_resources.total_cpus >= cpus
    assert job_resources.total_memory >= memory
    assert job_resources.total_resources() == (nodes, nodes * cpus, nodes * memory)
