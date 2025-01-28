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


# @pytest.mark.parametrize("nodes", (1, 2, 4, 8))
# @pytest.mark.parametrize("cpus", (1, 2, 4, 8))
# @pytest.mark.parametrize("memory", (1024, 2 * 1024, 4 * 1024, 8 * 1024))
# @pytest.mark.parametrize(
#     "batch_system", (BatchSystem.AWS, BatchSystem.LOCAL, BatchSystem.SLURM, None)
# )
# def test_formatting(
#     nodes: int, cpus: int, memory: int, batch_system: BatchSystem | None
# ) -> None:
#     job_resources = JobResources(nodes=nodes, cpus=cpus, memory=memory)

#     formatted_nodes = job_resources.format_nodes(batch_system)
#     assert isinstance(formatted_nodes, str)
#     assert str(nodes) in formatted_nodes

#     formatted_cpus = job_resources.format_cpus(batch_system)
#     assert isinstance(formatted_cpus, str)
#     assert str(cpus) in formatted_cpus

#     formatted_memory = job_resources.format_memory(batch_system)
#     assert isinstance(formatted_memory, str)
#     assert str(memory) in formatted_memory


# @pytest.mark.parametrize("jobs", (1, 4, 16, 32))
# @pytest.mark.parametrize("simulations", (250, 4 * 250, 16 * 250, 32 * 250))
# @pytest.mark.parametrize("blocks", (1, 4, 16, 32))
# @pytest.mark.parametrize("inference_method", ("emcee", None))
# def test_from_presets_for_select_inputs(
#     jobs: int, simulations: int, blocks: int, inference_method: Literal["emcee"] | None
# ) -> None:
#     job_size = JobSize(jobs=jobs, simulations=simulations, blocks=blocks)
#     job_resources = JobResources.from_presets(job_size, inference_method)
#     if inference_method == "emcee":
#         assert job_resources.nodes == 1
#         assert job_resources.cpus % 2 == 0
#         assert job_resources.memory % (2 * 1024) == 0
#     else:
#         assert job_resources.cpus == 2
#         assert job_resources.memory == 2 * 1024


# @pytest.mark.parametrize("inference_method", ("emcee", None))
# @pytest.mark.parametrize("nodes", (1, 2, 4, 8))
# @pytest.mark.parametrize("cpus", (1, 2, 4, 8))
# @pytest.mark.parametrize("memory", (1024, 2 * 1024, 4 * 1024, 8 * 1024))
# def test_from_presets_overrides(
#     inference_method: Literal["emcee"] | None, nodes: int, cpus: int, memory: int
# ) -> None:
#     job_size = JobSize(jobs=1, simulations=1, blocks=1)
#     job_resources = JobResources.from_presets(
#         job_size, inference_method, nodes=nodes, cpus=cpus, memory=memory
#     )
#     assert job_resources.nodes == nodes
#     assert job_resources.cpus == cpus
#     assert job_resources.memory == memory
