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
                "project_path": Path("/my/project/dir"),
                "chains": 8,
                "simulations_per_chain": 400,
                "samples_per_chain": 200,
                "run_id": "20250206_160655",
                "prefix": "foobar_None_None",
                "log_output": "/dev/null",
            },
            [
                "flepimop-calibrate \\",
                "    --config /path/to/config.yaml \\",
                "    --project_path /my/project/dir \\",
                "    --nwalkers 8 \\",
                "    --niterations 400 \\",
                "    --nsamples 200 \\",
                "    --id 20250206_160655 \\",
                "    --prefix foobar_None_None \\",
                "    > /dev/null 2>&1",
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(
    template_data: dict[str, Any], expected: list[str]
) -> None:
    lines = (
        _jinja_environment.get_template("emcee_inference_command.bash.j2")
        .render(template_data)
        .split("\n")
    )
    assert lines == expected
