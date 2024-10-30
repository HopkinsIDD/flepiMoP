import logging
import random
import string

import pytest

from gempyor.logging import _get_logging_level, get_script_logger, DEFAULT_LOG_FORMAT


@pytest.mark.parametrize("name", ("foobar", "fizzbuzz"))
@pytest.mark.parametrize(
    "verbosity",
    (
        0,
        1,
        2,
        3,
        4,
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ),
)
@pytest.mark.parametrize("handler", (None, logging.StreamHandler()))
@pytest.mark.parametrize("log_format", (None, "%(message)s"))
def test_get_script_logger(
    caplog: pytest.LogCaptureFixture,
    name: str,
    verbosity: int,
    handler: logging.Handler | None,
    log_format: str | None,
) -> None:
    caplog.set_level(logging.DEBUG, logger=name)

    if log_format is None:
        logger = get_script_logger(name, verbosity, handler=handler)
    else:
        logger = get_script_logger(
            name, verbosity, handler=handler, log_format=log_format
        )

    assert isinstance(logger, logging.Logger)
    assert logger.name == name
    assert logger.level == _get_logging_level(verbosity)
    if handler is not None:
        assert len(logger.handlers) == 1
        assert logger.handlers[0] == handler
    assert logger.handlers[0].formatter._fmt == (
        log_format if log_format else DEFAULT_LOG_FORMAT
    )

    i = 0
    for val, lvl in [
        (10, "debug"),
        (20, "info"),
        (30, "warning"),
        (40, "error"),
        (50, "critical"),
    ]:
        msg = "".join(random.choice(string.ascii_letters) for _ in range(20))
        getattr(logger, lvl)(msg)
        if val < _get_logging_level(verbosity):
            assert len(caplog.records) == i
            continue
        assert caplog.records[i].levelno == val
        assert caplog.records[i].message == msg
        i += 1
