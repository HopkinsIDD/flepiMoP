from pydantic import PositiveInt
import pytest

from gempyor.batch import _SAMPLES_SIMULATIONS_RATIO, JobSize


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    (
        (None, None, None),
        (1, None, 1),
        (None, 1, None),
        (1, 1, 1),
        (2, 4, 8),
        (4, 2, 8),
        (2, 2, 4),
    ),
)
def test__scale_method_for_select_values(
    x: PositiveInt | None, y: PositiveInt | None, expected: PositiveInt | None
) -> None:
    assert JobSize._scale(x, y) == expected


@pytest.mark.parametrize("blocks", (None, 2, 4))
@pytest.mark.parametrize("chains", (None, 8, 16))
@pytest.mark.parametrize(
    ("samples", "simulations"), ((100, 10), (100, 99), (2, 1), (25, 15))
)
def test_samples_and_simulations_consistent_value_error(
    blocks: PositiveInt | None,
    chains: PositiveInt | None,
    samples: PositiveInt,
    simulations: PositiveInt,
) -> None:
    assert samples > simulations
    with pytest.raises(
        ValueError,
        # Not an exact match since pydantic captures the
        # ValueError and converts it to a ValidationError.
        match=(
            f"The number of samples, {samples}, must be less than or equal "
            f"to the number of simulations, {simulations}, per a block."
        ),
    ):
        JobSize(blocks=blocks, chains=chains, samples=samples, simulations=simulations)


@pytest.mark.parametrize("blocks", (None, 2, 4))
@pytest.mark.parametrize("chains", (None, 8, 16))
@pytest.mark.parametrize(("samples", "simulations"), ((61, 100), (75, 100)))
def test_samples_to_simulations_ratio_warning(
    blocks: PositiveInt | None,
    chains: PositiveInt | None,
    samples: PositiveInt,
    simulations: PositiveInt,
) -> None:
    assert samples < simulations
    assert (samples / simulations) >= _SAMPLES_SIMULATIONS_RATIO
    with pytest.warns(UserWarning):
        JobSize(blocks=blocks, chains=chains, samples=samples, simulations=simulations)


@pytest.mark.parametrize("blocks", (None, 2, 4))
@pytest.mark.parametrize("chains", (None, 8, 16))
@pytest.mark.parametrize("samples", (None, 100, 200))
@pytest.mark.parametrize("simulations", (None, 400, 600))
def test_output_validation(
    blocks: PositiveInt | None,
    chains: PositiveInt | None,
    samples: PositiveInt | None,
    simulations: PositiveInt | None,
) -> None:
    size = JobSize(blocks=blocks, chains=chains, samples=samples, simulations=simulations)
    if samples is not None:
        assert size.samples_per_chain >= size.samples
        assert size.total_samples >= size.samples_per_chain
    if simulations is not None:
        assert size.simulations_per_chain >= size.simulations
        assert size.total_simulations >= size.simulations_per_chain
