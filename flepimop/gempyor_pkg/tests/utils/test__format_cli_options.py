from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gempyor.utils import _format_cli_options


@pytest.mark.parametrize(
    ("options", "formatted_options"),
    (
        ({"name": "foo bar fizz buzz"}, ["--name='foo bar fizz buzz'"]),
        ({"o": Path("/path/to/output.log")}, ["-o=/path/to/output.log"]),
        (
            {"opt1": "```", "opt2": "$( echo 'Hello!')"},
            ["--opt1='```'", "--opt2='$( echo '\"'\"'Hello!'\"'\"')'"],
        ),
        (
            {"start": datetime(2024, 1, 1, 12, 34, 56), "J": "rm -rf ~"},
            ["--start='2024-01-01 12:34:56'", "-J='rm -rf ~'"],
        ),
    ),
)
def test_output_validation(options: dict[str, Any], formatted_options: list[str]) -> None:
    assert _format_cli_options(options) == formatted_options
