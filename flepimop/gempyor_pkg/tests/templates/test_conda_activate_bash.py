import pytest

from gempyor._jinja import _render_template


@pytest.mark.parametrize(
    ("conda_env", "expected"),
    (
        (
            "flepimop-env",
            [
                "# Load conda env",
                'eval "$( conda shell.bash hook )"',
                "conda activate flepimop-env",
                "WHICH_PYTHON=$( which python )",
                "WHICH_RSCRIPT=$( which Rscript )",
            ],
        ),
        (
            "/path/to/conda/env",
            [
                "# Load conda env",
                'eval "$( conda shell.bash hook )"',
                "conda activate /path/to/conda/env",
                "WHICH_PYTHON=$( which python )",
                "WHICH_RSCRIPT=$( which Rscript )",
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(conda_env: str, expected: list[str]) -> None:
    lines = _render_template("conda_activate.bash.j2", {"conda_env": conda_env}).split("\n")
    assert lines == expected
