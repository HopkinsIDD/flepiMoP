import logging

from pydantic import PositiveInt
import pytest

from gempyor.batch import JobSize, _generate_job_sizes_grid


@pytest.mark.parametrize(
    ("reference_job_size", "expected_none_fields"),
    (
        (
            JobSize(blocks=None, chains=None, samples=None, simulations=None),
            "'blocks', 'chains', 'simulations'",
        ),
        (
            JobSize(blocks=None, chains=None, samples=None, simulations=10),
            "'blocks', 'chains'",
        ),
        (
            JobSize(blocks=10, chains=None, samples=None, simulations=None),
            "'chains', 'simulations'",
        ),
        (
            JobSize(blocks=None, chains=1, samples=None, simulations=None),
            "'blocks', 'simulations'",
        ),
        (
            JobSize(blocks=None, chains=4, samples=None, simulations=10),
            "'blocks'",
        ),
        (
            JobSize(blocks=1, chains=None, samples=None, simulations=10),
            "'chains'",
        ),
        (
            JobSize(blocks=1, chains=1, samples=None, simulations=None),
            "'simulations'",
        ),
    ),
)
def test_invalid_reference_job_size(
    reference_job_size: JobSize, expected_none_fields: str
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "^The reference job size has `None` for the following fields "
            f"which is not allowed for estimation: {expected_none_fields}.$"
        ),
    ):
        _generate_job_sizes_grid(reference_job_size, 1, logging.INFO)


@pytest.mark.parametrize(
    "reference_job_size",
    (
        JobSize(blocks=1, chains=1, samples=None, simulations=10),
        JobSize(blocks=10, chains=25, samples=100, simulations=1000),
        JobSize(blocks=100, chains=100, samples=1000, simulations=10000),
    ),
)
@pytest.mark.parametrize("estimate_runs", (1, 2, 4, 8, 16))
@pytest.mark.parametrize(
    "verbosity",
    (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL),
)
def test_output_validation(
    caplog: pytest.LogCaptureFixture,
    reference_job_size: JobSize,
    estimate_runs: PositiveInt,
    verbosity: int,
) -> None:
    estimate_job_sizes = _generate_job_sizes_grid(
        reference_job_size, estimate_runs, verbosity
    )
    assert len(estimate_job_sizes) == estimate_runs
    assert isinstance(estimate_job_sizes, list)
    assert all(isinstance(job_size, JobSize) for job_size in estimate_job_sizes)
    assert all(job_size.samples is None for job_size in estimate_job_sizes)
    for field in ("blocks", "chains", "simulations"):
        assert max(getattr(job_size, field) for job_size in estimate_job_sizes) <= max(
            getattr(reference_job_size, field) // 3, 1
        )
        assert min(getattr(job_size, field) for job_size in estimate_job_sizes) >= max(
            getattr(reference_job_size, field) // 10, 1
        )
    assert len(caplog.records) == (2 * (verbosity <= logging.INFO)) + (
        estimate_runs * (verbosity <= logging.DEBUG)
    )
