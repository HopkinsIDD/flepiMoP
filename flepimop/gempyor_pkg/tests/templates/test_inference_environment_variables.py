from pathlib import Path
from typing import Any

import pytest

from gempyor._jinja import _render_template


@pytest.mark.parametrize(
    ("template_data", "expected"),
    (
        (
            {
                "config_path": "/home/foobar/project/config.yml",
                "flepi_path": "/home/foobar/flepiMoP",
                "job_name": "sample_2pop-20240101T0000_None_None",
                "outcome_modifiers_scenario": None,
                "project_path": "/home/foobar/project",
                "reset_chimerics": False,
                "seir_modifiers_scenario": None,
                "stoch_traj_flag": True,
            },
            [
                "# Set environment variables",
                'if [ -z "$SLURM_ARRAY_TASK_ID" ]; then',
                '    FLEPI_SLOT_INDEX="1"',
                "else",
                "    FLEPI_SLOT_INDEX=$SLURM_ARRAY_TASK_ID",
                "fi",
                'export FLEPI_PATH="/home/foobar/flepiMoP"',
                'export PROJECT_PATH="/home/foobar/project"',
                'export CONFIG_PATH="/home/foobar/project/config.yml"',
                'export FLEPI_STOCHASTIC_RUN="TRUE"',
                'export FLEPI_OUTCOME_SCENARIOS="None"',
                'export FLEPI_SEIR_SCENARIOS="None"',
                'export FLEPI_RESET_CHIMERICS="FALSE"',
                "export LOG_FILE="
                '"out_sample_2pop-20240101T0000_None_None_$FLEPI_SLOT_INDEX.out"',
            ],
        ),
        (
            {
                "config_path": Path(
                    "/users/f/o/foobar/flepimop_sample/config_sample_2pop_inference.yml"
                ),
                "flepi_path": Path("/users/f/o/foobar/flepiMoP"),
                "job_name": "sample_2pop-20240101T0000_Ro_all_test_limits",
                "outcome_modifiers_scenario": "test_limits",
                "project_path": Path("/users/f/o/foobar/flepimop_sample"),
                "reset_chimerics": True,
                "seir_modifiers_scenario": "Ro_all",
                "stoch_traj_flag": False,
            },
            [
                "# Set environment variables",
                'if [ -z "$SLURM_ARRAY_TASK_ID" ]; then',
                '    FLEPI_SLOT_INDEX="1"',
                "else",
                "    FLEPI_SLOT_INDEX=$SLURM_ARRAY_TASK_ID",
                "fi",
                'export FLEPI_PATH="/users/f/o/foobar/flepiMoP"',
                'export PROJECT_PATH="/users/f/o/foobar/flepimop_sample"',
                "export CONFIG_PATH="
                '"/users/f/o/foobar/flepimop_sample/config_sample_2pop_inference.yml"',
                'export FLEPI_STOCHASTIC_RUN="FALSE"',
                'export FLEPI_OUTCOME_SCENARIOS="test_limits"',
                'export FLEPI_SEIR_SCENARIOS="Ro_all"',
                'export FLEPI_RESET_CHIMERICS="TRUE"',
                "export LOG_FILE="
                '"out_sample_2pop-20240101T0000_Ro_all_test_limits_$FLEPI_SLOT_INDEX.out"',
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(
    template_data: dict[str, Any], expected: list[str]
) -> None:
    lines = _render_template(
        "inference_environment_variables.bash.j2", template_data
    ).split("\n")
    assert lines == expected
