import pytest

from gempyor.batch import JobSize, _job_resources_from_size_and_inference


@pytest.mark.parametrize("nodes", (2, 4, 8))
def test_warning_for_more_than_one_node_with_emcee(nodes: int) -> None:
    with pytest.warns(
        UserWarning,
        match=f"^EMCEE inference only supports 1 node given {nodes}, overriding.$",
    ):
        _job_resources_from_size_and_inference(
            JobSize(jobs=10, simulations=200, blocks=1), "emcee", nodes=nodes
        )


@pytest.mark.parametrize("jobs", (5, 10, 20))
@pytest.mark.parametrize("simulations", (100, 200, 400))
@pytest.mark.parametrize("blocks", (1, 2, 4))
def test_resources_scales_with_size_for_emcee(
    jobs: int, simulations: int, blocks: int
) -> None:
    size_1x = JobSize(jobs=jobs, simulations=simulations, blocks=blocks)
    resources_1x = _job_resources_from_size_and_inference(size_1x, "emcee")
    size_2x = JobSize(jobs=2 * jobs, simulations=2 * simulations, blocks=2 * blocks)
    resources_2x = _job_resources_from_size_and_inference(size_2x, "emcee")
    assert resources_1x.nodes == resources_2x.nodes == 1
    assert resources_2x.cpus >= 2 * resources_1x.cpus
    assert resources_2x.memory >= 2 * resources_1x.memory


@pytest.mark.parametrize("jobs", (5, 10, 20))
@pytest.mark.parametrize("simulations", (100, 200, 400))
@pytest.mark.parametrize("blocks", (1, 2, 4))
def test_resources_is_constant_with_size_for_legacy(
    jobs: int, simulations: int, blocks: int
) -> None:
    size_1x = JobSize(jobs=jobs, simulations=simulations, blocks=blocks)
    resources_1x = _job_resources_from_size_and_inference(size_1x, "r")
    size_2x = JobSize(jobs=2 * jobs, simulations=2 * simulations, blocks=2 * blocks)
    resources_2x = _job_resources_from_size_and_inference(size_2x, "r")
    assert resources_2x.nodes >= 2 * resources_1x.nodes
    assert resources_2x.cpus == resources_1x.cpus
    assert resources_2x.memory == resources_1x.memory
