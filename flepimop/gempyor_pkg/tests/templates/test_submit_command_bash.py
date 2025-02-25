import pytest

from gempyor._jinja import _jinja_environment


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("echo 'Foobar!'", ["#!/usr/bin/env bash", "", "", "", "echo 'Foobar!'"]),
        (
            "echo 'Hello, world!'",
            ["#!/usr/bin/env bash", "", "", "", "echo 'Hello, world!'"],
        ),
    ),
)
def test_exact_results_for_select_inputs(command: str, expected: list[str]) -> None:
    lines = (
        _jinja_environment.get_template("submit_command.bash.j2")
        .render({"command": command})
        .splitlines()
    )
    assert lines == expected
