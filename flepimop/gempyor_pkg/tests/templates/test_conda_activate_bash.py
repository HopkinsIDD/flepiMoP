import pytest

from gempyor._jinja import _render_template


@pytest.mark.parametrize("conda_env", ("abc", "flepimop-env"))
def test_output_validation(conda_env: str) -> None:
    rendered_template = _render_template("conda_activate.bash.j2", {"conda_env": conda_env})
    lines = rendered_template.split("\n")
    assert len(lines) == 4
    assert conda_env in lines[1]
