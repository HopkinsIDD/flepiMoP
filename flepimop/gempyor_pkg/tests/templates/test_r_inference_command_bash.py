from pathlib import Path
from typing import Any

import pytest

from gempyor._jinja import _jinja_environment


@pytest.mark.parametrize(
    ("template_data", "expected"),
    (
        (
            {
                "config": Path("/path/to/config.yaml"),
                "run_id": "20250206_160655",
                "seir_modifiers_scenario": "None",
                "outcome_modifiers_scenario": "None",
                "simulations_per_chain": 400,
                "stoch_traj_flag": True,
                "flepi_path": Path("/path/to/flepiMoP"),
                "log_output": "/dev/null",
            },
            [
                "flepimop-inference-slot --config /path/to/config.yaml \\",
                "    --run_id 20250206_160655 \\",
                "    --seir_modifiers_scenarios None \\",
                "    --outcome_modifiers_scenarios None \\",
                "    --jobs 1 \\",
                "    --iterations_per_slot 400 \\",
                "    --this_slot $SLURM_ARRAY_TASK_ID \\",
                "    --this_block 1 \\",
                "    --stoch_traj_flag TRUE \\",
                "    --flepi_path /path/to/flepiMoP \\",
                "    --python $WHICH_PYTHON \\",
                "    --rpath $WHICH_RSCRIPT \\",
                "    --is-interactive FALSE \\",
                "    > /dev/null 2>&1",
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(
    template_data: dict[str, Any], expected: list[str]
) -> None:
    lines = (
        _jinja_environment.get_template("r_inference_command.bash.j2")
        .render(template_data)
        .split("\n")
    )
    assert lines == expected
