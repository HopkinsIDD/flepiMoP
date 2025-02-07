import pytest

from gempyor._jinja import _jinja_environment


@pytest.mark.parametrize(
    ("conda_env", "expected"),
    (
        (None, [""]),
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
def test_exact_results_for_select_inputs(
    conda_env: str | None, expected: list[str]
) -> None:
    lines = (
        _jinja_environment.get_template("conda_activate.bash.j2")
        .render({} if conda_env is None else {"conda_env": conda_env})
        .split("\n")
    )
    assert lines == expected
