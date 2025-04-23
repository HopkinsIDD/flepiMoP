from pathlib import Path
from typing import Any

import pytest

from gempyor._jinja import _jinja_environment


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        (
            {
                "job_name": "Foobar",
                "job_comment": "My custom comment",
                "project_path": Path("/foo/bar"),
                "job_time_limit": "1:00:00",
                "job_resources_nodes": "1",
                "job_resources_cpus": "1",
                "job_resources_memory": "1024G",
                "debug": False,
                "command": "echo 'Hello, world!'",
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --job-name="Foobar"',
                '#SBATCH --comment="My custom comment"',
                '#SBATCH --chdir="/foo/bar"',
                '#SBATCH --time="1:00:00"',
                '#SBATCH --nodes="1"',
                '#SBATCH --ntasks="1"',
                '#SBATCH --cpus-per-task="1"',
                '#SBATCH --mem="1024G"',
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "echo 'Hello, world!'",
            ],
        ),
        (
            {
                "command": "echo 'Foobar'",
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --ntasks="1"',
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "echo 'Foobar'",
            ],
        ),
        (
            {
                "job_name": "flu_20250101",
                "project_path": Path("/path/to/project"),
                "job_resources_nodes": "10",
                "job_resources_cpus": "4",
                "job_resources_memory": "2048MB",
                "debug": True,
                "command": "do-analysis-cmd",
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --job-name="flu_20250101"',
                '#SBATCH --chdir="/path/to/project"',
                '#SBATCH --nodes="10"',
                '#SBATCH --ntasks="1"',
                '#SBATCH --cpus-per-task="4"',
                '#SBATCH --mem="2048MB"',
                "",
                "",
                "# Debugging",
                "set -x",
                "",
                "",
                "",
                "",
                "",
                "do-analysis-cmd",
            ],
        ),
        (
            {
                "command": """my-complicated-command \\
    --flag-one 'abc' \\
    --flag-two 'def' \\
    --flag-three 'ghi' \\
    > /dev/null""",
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --ntasks="1"',
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "my-complicated-command \\",
                "    --flag-one 'abc' \\",
                "    --flag-two 'def' \\",
                "    --flag-three 'ghi' \\",
                "    > /dev/null",
            ],
        ),
        (
            {
                "job_name": "rsv_2024",
                "project_path": Path("/path/to/rsv/project"),
                "job_resources_nodes": "10",
                "job_resources_cpus": "4",
                "job_resources_memory": "2048MB",
                "debug": True,
                "command": "do-rsv-analysis",
                "array_capable": True,
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --job-name="rsv_2024"',
                '#SBATCH --chdir="/path/to/rsv/project"',
                '#SBATCH --nodes="1"',
                '#SBATCH --array="1-10"',
                '#SBATCH --ntasks="1"',
                '#SBATCH --cpus-per-task="4"',
                '#SBATCH --mem="2048MB"',
                "",
                "",
                "# Debugging",
                "set -x",
                "",
                "",
                "",
                "",
                "",
                "do-rsv-analysis",
            ],
        ),
        (
            {
                "job_name": "measles_2024",
                "job_dependency": 12345,
                "command": "do-measles-analysis",
            },
            [
                "#!/usr/bin/env bash",
                '#SBATCH --job-name="measles_2024"',
                '#SBATCH --ntasks="1"',
                '#SBATCH --dependency="afterok:12345"',
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "do-measles-analysis",
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(data: dict[str, Any], expected: list[str]) -> None:
    lines = (
        _jinja_environment.get_template("sbatch_submit_command.bash.j2")
        .render(data)
        .split("\n")
    )
    print(lines)
    print(expected)
    assert lines == expected
