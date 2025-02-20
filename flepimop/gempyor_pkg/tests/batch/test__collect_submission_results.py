from collections.abc import Iterable

import pytest

from gempyor.batch import JobResources, JobSize, _collect_submission_results


@pytest.mark.parametrize("estimate_factors", ((), []))
def test_empty_estimate_factors_value_error(estimate_factors: Iterable[str]) -> None:
    with pytest.raises(ValueError, match="^The estimate factors must not be empty.$"):
        _collect_submission_results(
            estimate_factors,
            ("memory", "time"),
            0.9,
            JobSize(blocks=1, chains=4, samples=None, simulations=10),
            JobResources(nodes=1, cpus=4, memory=16),
            [],
            ["None"],
            ["None"],
            {},
            None,
            0,
        )


@pytest.mark.parametrize(
    "estimate_factors",
    (("not_a_valid_factor",), ("not_a_valid_factor", "time"), ("memory", "invalid")),
)
def test_invalid_estimate_factors_value_error(estimate_factors: Iterable[str]) -> None:
    with pytest.raises(
        ValueError, match="^The estimate factors .* are not valid job size fields.$"
    ):
        _collect_submission_results(
            estimate_factors,
            ("memory", "time"),
            0.9,
            JobSize(blocks=1, chains=4, samples=None, simulations=10),
            JobResources(nodes=1, cpus=4, memory=16),
            [],
            ["None"],
            ["None"],
            {},
            None,
            0,
        )


@pytest.mark.parametrize("estimate_measurements", ((), []))
def test_empty_estimate_measurements_value_error(
    estimate_measurements: Iterable[str],
) -> None:
    with pytest.raises(ValueError, match="^The estimate measurements must not be empty.$"):
        _collect_submission_results(
            ("total_simulations",),
            estimate_measurements,
            0.9,
            JobSize(blocks=1, chains=4, samples=None, simulations=10),
            JobResources(nodes=1, cpus=4, memory=16),
            [],
            ["None"],
            ["None"],
            {},
            None,
            0,
        )
