"""Unit test the `gempyor.model_meta._construct_setup_name` internal helper."""

import pytest

from gempyor.model_meta import _construct_setup_name


@pytest.mark.parametrize(
    (
        "setup_name",
        "name",
        "seir_modifiers_scenario",
        "outcome_modifiers_scenario",
        "expected",
    ),
    [
        ("setup-name", "name", None, None, "setup-name"),
        (None, "name", None, None, "name"),
        (None, "name", "seir-scenario", None, "name_seir-scenario"),
        (None, "name", None, "outcome-scenario", "name_outcome-scenario"),
        (
            None,
            "name",
            "seir-scenario",
            "outcome-scenario",
            "name_seir-scenario_outcome-scenario",
        ),
        ("setup-name", "name", "seir-scenario", None, "setup-name"),
        ("setup-name", "name", None, "outcome-scenario", "setup-name"),
        ("setup-name", "name", "seir-scenario", "outcome-scenario", "setup-name"),
    ],
)
def test_exact_results_for_select_inputs(
    setup_name: str | None,
    name: str,
    seir_modifiers_scenario: str | None,
    outcome_modifiers_scenario: str | None,
    expected: str,
) -> None:
    """Test exact results for a select set of inputs."""
    assert (
        _construct_setup_name(
            setup_name, name, seir_modifiers_scenario, outcome_modifiers_scenario
        )
        == expected
    )
