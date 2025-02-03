from typing import Any, Literal

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


@pytest.mark.parametrize("options", ({}, None))
@pytest.mark.parametrize("always_single", (True, False))
@pytest.mark.parametrize("iterable_formatting", ("split", "comma"))
def test_output_validation_for_empty_values(
    options: dict[str, Any] | None,
    always_single: bool,
    iterable_formatting: Literal["split", "comma"],
) -> None:
    assert (
        _format_cli_options(
            options, always_single=always_single, iterable_formatting=iterable_formatting
        )
        == []
    )


@pytest.mark.parametrize(
    ("options", "always_single", "iterable_formatting", "formatted_options"),
    (
        ({"name": "foo bar fizz buzz"}, False, "split", ["--name='foo bar fizz buzz'"]),
        ({"name": "foo bar fizz buzz"}, True, "split", ["-name='foo bar fizz buzz'"]),
        ({"o": "/path/to/output.log"}, False, "split", ["-o=/path/to/output.log"]),
        ({"o": "/path/to/output.log"}, True, "split", ["-o=/path/to/output.log"]),
        (
            {"opt1": "```", "opt2": "$( echo 'Hello!')"},
            False,
            "split",
            ["--opt1='```'", "--opt2='$( echo '\"'\"'Hello!'\"'\"')'"],
        ),
        (
            {"opt1": "```", "opt2": "$( echo 'Hello!')"},
            True,
            "split",
            ["-opt1='```'", "-opt2='$( echo '\"'\"'Hello!'\"'\"')'"],
        ),
        (
            {"start": "2024-01-01 12:34:56", "J": "rm -rf ~"},
            False,
            "split",
            ["--start='2024-01-01 12:34:56'", "-J='rm -rf ~'"],
        ),
        (
            {"start": "2024-01-01 12:34:56", "J": "rm -rf ~"},
            True,
            "split",
            ["-start='2024-01-01 12:34:56'", "-J='rm -rf ~'"],
        ),
        (
            {"opt1": ["foo", "bar", "fizz", "buzz"]},
            False,
            "split",
            ["--opt1=foo", "--opt1=bar", "--opt1=fizz", "--opt1=buzz"],
        ),
        (
            {"opt1": ["foo", "bar", "fizz", "buzz"]},
            False,
            "comma",
            ["--opt1=foo,bar,fizz,buzz"],
        ),
        (
            {"opt1": ["foo", "bar", "fizz", "buzz"]},
            True,
            "split",
            ["-opt1=foo", "-opt1=bar", "-opt1=fizz", "-opt1=buzz"],
        ),
        (
            {"opt1": ["foo", "bar", "fizz", "buzz"]},
            True,
            "comma",
            ["-opt1=foo,bar,fizz,buzz"],
        ),
        (
            {
                "letters": ["abc", "def", "ghi"],
                "numbers": ["1", "2", "3"],
                "symbols": "!@#",
                "cmd": "echo $( cat foobar.txt | grep -v 'baz' )",
            },
            False,
            "split",
            [
                "--letters=abc",
                "--letters=def",
                "--letters=ghi",
                "--numbers=1",
                "--numbers=2",
                "--numbers=3",
                "--symbols='!@#'",
                """--cmd=\'echo $( cat foobar.txt | grep -v \'"\'"\'baz\'"\'"\' )\'""",
            ],
        ),
        (
            {
                "letters": ["abc", "def", "ghi"],
                "numbers": ["1", "2", "3"],
                "symbols": "!@#",
                "cmd": "echo $( cat foobar.txt | grep -v 'baz' )",
            },
            True,
            "split",
            [
                "-letters=abc",
                "-letters=def",
                "-letters=ghi",
                "-numbers=1",
                "-numbers=2",
                "-numbers=3",
                "-symbols='!@#'",
                """-cmd=\'echo $( cat foobar.txt | grep -v \'"\'"\'baz\'"\'"\' )\'""",
            ],
        ),
        (
            {
                "letters": ["abc", "def", "ghi"],
                "numbers": ["1", "2", "3"],
                "symbols": "!@#",
                "cmd": "echo $( cat foobar.txt | grep -v 'baz' )",
            },
            False,
            "comma",
            [
                "--letters=abc,def,ghi",
                "--numbers=1,2,3",
                "--symbols='!@#'",
                """--cmd=\'echo $( cat foobar.txt | grep -v \'"\'"\'baz\'"\'"\' )\'""",
            ],
        ),
        (
            {
                "letters": ["abc", "def", "ghi"],
                "numbers": ["1", "2", "3"],
                "symbols": "!@#",
                "cmd": "echo $( cat foobar.txt | grep -v 'baz' )",
            },
            True,
            "comma",
            [
                "-letters=abc,def,ghi",
                "-numbers=1,2,3",
                "-symbols='!@#'",
                """-cmd=\'echo $( cat foobar.txt | grep -v \'"\'"\'baz\'"\'"\' )\'""",
            ],
        ),
    ),
)
def test_output_validation_for_select_values(
    options: dict[str, Any] | None,
    always_single: bool,
    iterable_formatting: Literal["split", "comma"],
    formatted_options: list[str],
) -> None:
    assert (
        _format_cli_options(
            options, always_single=always_single, iterable_formatting=iterable_formatting
        )
        == formatted_options
    )
