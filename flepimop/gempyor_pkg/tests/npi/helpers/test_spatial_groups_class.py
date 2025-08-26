"""Unit tests for the `gempyor.NPI.helpers.SpatialGroups` class."""

import pytest
from gempyor.NPI.helpers import SpatialGroups


@pytest.mark.parametrize(
    ("subpopulations", "subpopulation_groups", "expected_groups", "expected_ungrouped"),
    [
        (
            ["A", "B", "C"],
            None,
            (),
            ("A", "B", "C"),
        ),
        (
            ["A", "B", "C"],
            [],
            (),
            ("A", "B", "C"),
        ),
        (
            ["A", "B", "C"],
            [["A", "B"], ["C"]],
            (("A", "B"), ("C",)),
            (),
        ),
        (
            ["A", "B", "C"],
            [["A", "B"]],
            (("A", "B"),),
            ("C",),
        ),
        (
            ["A", "B", "C"],
            "all",
            (("A", "B", "C"),),
            (),
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B", "C"], ["D", "E", "F"]],
            (("A", "B", "C"), ("D", "E", "F")),
            (),
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B"], [], ["E", "F"]],
            (("A", "B"), ("E", "F")),
            ("C", "D"),
        ),
        (
            ["A", "B", "C", "D", "E", "F"],
            [["A", "B"], [], ["E", "F"]],
            (("A", "B"), ("E", "F")),
            ("C", "D"),
        ),
        (
            ["VA", "NC", "SC", "GA", "FL"],
            [["VA", "FL"], ["SC", "NC"]],
            (("FL", "VA"), ("NC", "SC")),
            ("GA",),
        ),
    ],
)
def test_from_subpopulations_for_exact_results_for_select_inputs(
    subpopulations: list[str],
    subpopulation_groups: list[list[str]] | list[str] | str | None,
    expected_groups: tuple[tuple[str]],
    expected_ungrouped: tuple[str],
):
    """Test the exact results of `from_subpopulations` cls method for select inputs."""
    result = SpatialGroups.from_subpopulations(subpopulations, subpopulation_groups)
    assert result.grouped == expected_groups
    assert result.ungrouped == expected_ungrouped
    assert result.ungrouped == tuple(sorted(result.ungrouped))
    assert all(tuple(sorted(group)) == group for group in result.grouped)
