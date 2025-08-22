"""Unit tests for the `gempyor.NPI.helpers.get_spatial_groups` function."""

import pytest
from gempyor.NPI.helpers import get_spatial_groups


@pytest.mark.parametrize(
    ("subpopulations", "subpopulation_groups", "expected_groups", "expected_ungrouped"),
    [
        (
            ["A", "B", "C"],
            None,
            [],
            ["A", "B", "C"],
        ),
        (
            ["A", "B", "C"],
            [],
            [],
            ["A", "B", "C"],
        ),
        (
            ["A", "B", "C"],
            [["A", "B"], ["C"]],
            [["A", "B"], ["C"]],
            [],
        ),
        (
            ["A", "B", "C"],
            [["A", "B"]],
            [["A", "B"]],
            ["C"],
        ),
        (
            ["A", "B", "C"],
            "all",
            [["A", "B", "C"]],
            [],
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B", "C"], ["D", "E", "F"]],
            [["A", "B", "C"], ["D", "E", "F"]],
            [],
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B"], [], ["E", "F"]],
            [["A", "B"], ["E", "F"]],
            ["C", "D"],
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B"], [], ["E", "F"]],
            [["A", "B"], ["E", "F"]],
            ["C", "D"],
        ),
    ],
)
def test_exact_results_for_select_inputs(
    subpopulations: list[str],
    subpopulation_groups: list[list[str]] | list[str] | str | None,
    expected_groups: list[list[str]],
    expected_ungrouped: list[str],
):
    """Test the exact results of `get_spatial_groups` for select inputs."""
    result = get_spatial_groups(subpopulations, subpopulation_groups)
    assert isinstance(result, dict) and set(result.keys()) == {"grouped", "ungrouped"}
    assert result["grouped"] == expected_groups
    assert result["ungrouped"] == expected_ungrouped
    assert result["ungrouped"] == sorted(result["ungrouped"])
    assert all(sorted(group) == group for group in result["grouped"])
