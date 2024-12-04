import logging
from pathlib import Path
from typing import Any

import pytest

from gempyor.logging import _get_logging_level
from gempyor.shared_cli import log_cli_inputs


@pytest.mark.parametrize(
    "kwargs",
    (
        {"abc": 123},
        {"abc": 123, "verbosity": 3},
        {"str": "foobar", "int": 123, "float": 3.14, "bool": True},
        {"str": "foobar", "int": 123, "float": 3.14, "bool": True, "verbosity": 3},
        {"file_path": Path("fizzbuzz.log")},
        {"file_path": Path("fizzbuzz.log"), "verbosity": 3},
    ),
)
@pytest.mark.parametrize("verbosity", (None, 0, 1, 2, 3, logging.DEBUG, logging.INFO))
def test_number_of_messages(
    caplog: pytest.LogCaptureFixture, kwargs: dict[str, Any], verbosity: int | None
) -> None:
    log_cli_inputs(kwargs, verbosity=verbosity)
    assert len(caplog.records) == (
        len(kwargs) + 1
        if _get_logging_level(
            kwargs.get("verbosity", 0) if verbosity is None else verbosity
        )
        <= logging.DEBUG
        else 0
    )
