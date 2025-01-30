from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from gempyor.utils import _format_cli_options


@pytest.mark.parametrize("options", ({"name": "foo"},))
@pytest.mark.parametrize("always_single", (True, False))
def test_affect_of_always_single(options: dict[str, Any], always_single: bool) -> None:
    formatted_options = _format_cli_options(options, always_single=always_single)
    assert all(fo.startswith("-") for fo in formatted_options)
    assert (
        all(not fo.startswith("--") for fo in formatted_options)
        if always_single
        else any(fo.startswith("--") for fo in formatted_options)
    )


@pytest.mark.parametrize(
    ("options", "always_single", "formatted_options"),
    (
        ({}, False, []),
        ({}, True, []),
        (None, False, []),
        (None, True, []),
        ({"name": "foo bar fizz buzz"}, False, ["--name='foo bar fizz buzz'"]),
        ({"name": "foo bar fizz buzz"}, True, ["-name='foo bar fizz buzz'"]),
        ({"o": Path("/path/to/output.log")}, False, ["-o=/path/to/output.log"]),
        ({"o": Path("/path/to/output.log")}, True, ["-o=/path/to/output.log"]),
        (
            {"opt1": "```", "opt2": "$( echo 'Hello!')"},
            False,
            ["--opt1='```'", "--opt2='$( echo '\"'\"'Hello!'\"'\"')'"],
        ),
        (
            {"opt1": "```", "opt2": "$( echo 'Hello!')"},
            True,
            ["-opt1='```'", "-opt2='$( echo '\"'\"'Hello!'\"'\"')'"],
        ),
        (
            {"start": datetime(2024, 1, 1, 12, 34, 56), "J": "rm -rf ~"},
            False,
            ["--start='2024-01-01 12:34:56'", "-J='rm -rf ~'"],
        ),
        (
            {"start": datetime(2024, 1, 1, 12, 34, 56), "J": "rm -rf ~"},
            True,
            ["-start='2024-01-01 12:34:56'", "-J='rm -rf ~'"],
        ),
    ),
)
def test_output_validation_for_select_values(
    options: dict[str, Any] | None, always_single: bool, formatted_options: list[str]
) -> None:
    assert _format_cli_options(options, always_single=always_single) == formatted_options
