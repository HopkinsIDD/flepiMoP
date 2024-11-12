from pathlib import Path
from typing import Any

import pytest

from gempyor._jinja import _render_template


@pytest.mark.parametrize(
    ("template_data", "expected"),
    (
        (
            {
                "flepi_path": "/home/foobar/flepiMoP",
                "project_path": "/home/foobar/project",
                "config_path": "/home/foobar/project/config.yml",
            },
            [
                "# Set environment variables",
                'if [ -z "$SLURM_ARRAY_TASK_ID" ]; then',
                '    FLEPI_SLOT_INDEX="1"',
                "else",
                "    FLEPI_SLOT_INDEX=$SLURM_ARRAY_TASK_ID",
                "fi",
                "export FLEPI_PATH=/home/foobar/flepiMoP",
                "export PROJECT_PATH=/home/foobar/project",
                "export CONFIG_PATH=/home/foobar/project/config.yml",
            ],
        ),
        (
            {
                "flepi_path": Path("/users/f/o/foobar/flepiMoP"),
                "project_path": Path("/users/f/o/foobar/flepimop_sample"),
                "config_path": Path(
                    "/users/f/o/foobar/flepimop_sample/config_sample_2pop_inference.yml"
                ),
            },
            [
                "# Set environment variables",
                'if [ -z "$SLURM_ARRAY_TASK_ID" ]; then',
                '    FLEPI_SLOT_INDEX="1"',
                "else",
                "    FLEPI_SLOT_INDEX=$SLURM_ARRAY_TASK_ID",
                "fi",
                "export FLEPI_PATH=/users/f/o/foobar/flepiMoP",
                "export PROJECT_PATH=/users/f/o/foobar/flepimop_sample",
                "export CONFIG_PATH="
                "/users/f/o/foobar/flepimop_sample/config_sample_2pop_inference.yml",
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
