from collections.abc import Sequence
import logging
import math

from pydantic import PositiveInt
import pytest

from gempyor.batch import JobSize, _generate_job_sizes_grid


@pytest.mark.parametrize("vary_fields", ([], ()))
def test_empty_vary_fields_value_error(vary_fields: Sequence[str]) -> None:
    with pytest.raises(
        ValueError,
        match=f"^The vary fields must not be empty.$",
    ):
        _generate_job_sizes_grid(
            JobSize(blocks=1, chains=1, samples=1, simulations=2),
            vary_fields,
            10,
            3,
            1,
            logging.INFO,
        )


@pytest.mark.parametrize(
    ("lower_scale", "upper_scale"), ((3, 3), (2.3, 2.3), (10, 20), (2.3, 2.4))
)
def test_lower_and_upper_scale_value_error(
    lower_scale: float | int, upper_scale: float | int
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"^The lower scale, {lower_scale}, must be "
            f"greater than the upper scale, {upper_scale}.$"
        ),
    ):
        _generate_job_sizes_grid(
            JobSize(blocks=1, chains=1, samples=1, simulations=2),
            ("blocks", "chains", "simulations"),
            lower_scale,
            upper_scale,
            1,
            logging.INFO,
        )


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
        _generate_job_sizes_grid(
            reference_job_size, ("blocks", "chains", "simulations"), 10, 3, 1, logging.INFO
        )


@pytest.mark.filterwarnings(
    "ignore:The samples to simulations ratio is .*, "
    "which is higher than the recommended limit of .*."
)
@pytest.mark.parametrize(
    "reference_job_size",
    (
        JobSize(blocks=1, chains=1, samples=None, simulations=10),
        JobSize(blocks=10, chains=25, samples=100, simulations=1000),
        JobSize(blocks=100, chains=100, samples=1000, simulations=10000),
    ),
)
@pytest.mark.parametrize(
    "vary_fields",
    (
        ("blocks", "chains", "simulations"),
        ("blocks", "chains"),
        ("blocks", "simulations"),
        ("chains", "simulations"),
        ("blocks",),
        ("chains",),
        ("simulations",),
    ),
)
@pytest.mark.parametrize(
    ("lower_scale", "upper_scale"), ((10, 3), (5, 2), (5.5, 2.5), (6, 2.4))
)
@pytest.mark.parametrize("estimate_runs", (1, 2, 4, 8, 16))
@pytest.mark.parametrize(
    "verbosity",
    (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL),
)
def test_output_validation(
    caplog: pytest.LogCaptureFixture,
    reference_job_size: JobSize,
    vary_fields: tuple[str, ...],
    lower_scale: float | int,
    upper_scale: float | int,
    estimate_runs: PositiveInt,
    verbosity: int,
) -> None:
    estimate_job_sizes = _generate_job_sizes_grid(
        reference_job_size,
        vary_fields,
        lower_scale,
        upper_scale,
        estimate_runs,
        verbosity,
    )
    assert len(estimate_job_sizes) == estimate_runs
    assert isinstance(estimate_job_sizes, list)
    assert all(isinstance(job_size, JobSize) for job_size in estimate_job_sizes)
    for field in vary_fields:
        assert max(getattr(job_size, field) for job_size in estimate_job_sizes) <= max(
            math.ceil(getattr(reference_job_size, field) / upper_scale), 1
        )
        assert min(getattr(job_size, field) for job_size in estimate_job_sizes) >= max(
            math.floor(getattr(reference_job_size, field) / lower_scale), 1
        )
    for field in {"blocks", "chains", "simulations"} - set(vary_fields):
        assert all(
            getattr(job_size, field) == getattr(reference_job_size, field)
            for job_size in estimate_job_sizes
        )
    assert len(caplog.records) == (2 * (verbosity <= logging.INFO)) + (
        estimate_runs * (verbosity <= logging.DEBUG)
    )
