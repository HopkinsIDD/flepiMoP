import pytest

from gempyor.batch import JobSize, _create_estimate_job_size_from_reference


example_job_sizes = (
    JobSize(blocks=None, chains=None, samples=None, simulations=None),
    JobSize(blocks=1, chains=1, samples=None, simulations=10),
    JobSize(blocks=10, chains=25, samples=100, simulations=1000),
    JobSize(blocks=100, chains=100, samples=1000, simulations=10000),
    JobSize(blocks=1, chains=2, samples=10, simulations=None),
    JobSize(blocks=1, chains=None, samples=10, simulations=100),
    JobSize(blocks=None, chains=None, samples=50, simulations=100),
)


@pytest.mark.parametrize(
    "reference_job_size",
    example_job_sizes,
)
def test_is_identity_when_no_overrides_are_given(reference_job_size: JobSize) -> None:
    assert (
        _create_estimate_job_size_from_reference(reference_job_size, {})
        == reference_job_size
    )


@pytest.mark.filterwarnings(
    "ignore:The samples to simulations ratio is .*, "
    "which is higher than the recommended limit of .*."
)
@pytest.mark.parametrize("reference_job_size", example_job_sizes)
@pytest.mark.parametrize(
    "overrides",
    (
        {"blocks": 1},
        {"chains": 1},
        {"samples": 1},
        {"simulations": 1},
        {"blocks": 100},
        {"chains": 25},
        {"samples": 1000},
        {"simulations": 250},
        {"blocks": 1, "chains": 1},
        {"blocks": 1, "samples": 10},
        {"blocks": 1, "simulations": 10},
        {"chains": 1, "samples": 10},
        {"chains": 1, "simulations": 10},
        {"samples": 10, "simulations": 100},
        {"blocks": 10, "chains": 25, "samples": 100, "simulations": 1000},
        {"blocks": 100, "chains": 100, "samples": 1000, "simulations": 100},
        {"samples": None, "simulations": 100},
    ),
)
def test_is_overridden_when_overrides_are_given(
    reference_job_size: JobSize, overrides: dict
) -> None:
    new_job_size = _create_estimate_job_size_from_reference(reference_job_size, overrides)
    assert isinstance(new_job_size, JobSize)
    assert new_job_size.blocks == overrides.get("blocks", reference_job_size.blocks)
    assert new_job_size.chains == overrides.get("chains", reference_job_size.chains)
    if (samples := overrides.get("samples", reference_job_size.samples)) is not None:
        assert new_job_size.samples <= samples
    assert new_job_size.simulations == overrides.get(
        "simulations", reference_job_size.simulations
    )
