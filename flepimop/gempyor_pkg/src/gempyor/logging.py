"""
Logging utilities for consistent script output.

This module provides functionality for creating consistent outputs from CLI tools 
provided by this package. Currently exported are:
- `ClickHandler`: Custom logging handler specifically designed for CLI output using 
    click.
- `get_script_logger`: Factory for creating a logger instance with a consistent style
    across CLI tools.
"""

__all__ = ["ClickHandler", "get_script_logger"]


import logging
from typing import Any, IO

import click


DEFAULT_LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s> %(message)s"


class ClickHandler(logging.Handler):
    """
    Custom logging handler specifically for click based CLI tools.
    """

    _punctuation = (".", ",", "?", "!", ":")

    def __init__(
        self,
        level: int | str = 0,
        file: IO[Any] | None = None,
        nl: bool = True,
        err: bool = False,
        color: bool | None = None,
        punctuate: bool = True,
    ) -> None:
        """
        Initialize an instance of the click handler.

        Args:
            level: The logging level to use for this handler.
            file: The file to write to. Defaults to stdout.
            nl: Print a newline after the message. Enabled by default.
            err: Write to stderr instead of stdout.
            color: Force showing or hiding colors and other styles. By default click
                will remove color if the output does not look like an interactive
                terminal.
            punctuate: A boolean indicating if punctuation should be added to the end
                of a log message provided if missing.

        Notes:
            For more details on the `file`, `nl`, `err`, and `color` args please refer
            to [`click.echo`](https://click.palletsprojects.com/en/8.1.x/api/#click.echo).
        """
        super().__init__(level)
        self._file = file
        self._nl = nl
        self._err = err
        self._color = color
        self._punctuate = punctuate

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a given log record via `click.echo`

        Args:
            record: The log record to output.

        See Also:
            [`logging.Handler.emit`](https://docs.python.org/3/library/logging.html#logging.Handler.emit)
        """
        msg = self.format(record)
        msg = (
            f"{msg}."
            if self._punctuate and not msg.endswith(self._punctuation)
            else msg
        )
        click.echo(
            message=msg, file=self._file, nl=self._nl, err=self._err, color=self._color
        )


def get_script_logger(
    name: str,
    verbosity: int,
    handler: logging.Handler | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> logging.Logger:
    """
    Create a logger for use in scripts.

    Args:
        name: The name to display in the log message, useful for locating the source
            of logging messages. Almost always `__name__`.
        verbosity: A non-negative integer for the verbosity level.
        handler: An optional logging handler to use in creating the logger returned, or
            `None` to just use the `ClickHandler`.
        log_format: The format to use for logged messages. Passed directly to the `fmt`
            argument of [logging.Formatter](https://docs.python.org/3/library/logging.html#logging.Formatter).

    Returns:
        An instance of `logging.Logger` that has the appropriate level set based on
        `verbosity` and a custom handler for outputting for CLI tools.

    Examples:
        >>> from gempyor.logging import get_script_logger
        >>> logger = get_script_logger(__name__, 3)
        >>> logger.info("This is a log info message")
        2024-10-29 16:07:20,272:INFO:__main__> This is a log info message.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_get_logging_level(verbosity))
    handler = ClickHandler() if handler is None else handler
    log_formatter = logging.Formatter(log_format)
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _get_logging_level(verbosity: int) -> int:
    """
    An internal method to convert verbosity to a logging level.

    Args:
        verbosity: A non-negative integer for the verbosity level or level from
            `logging` that will be returned as is.

    Examples:
        >>> _get_logging_level(0)
        40
        >>> _get_logging_level(1)
        30
        >>> _get_logging_level(2)
        20
        >>> _get_logging_level(3)
        10
        >>> _get_logging_level(4)
        10
        >>> import logging
        >>> _get_logging_level(logging.ERROR) == logging.ERROR
        True

    Raises:
        ValueError: If `verbosity` is less than zero.

    Returns:
        The log level from the `logging` module corresponding to the given `verbosity`.
    """
    if verbosity < 0:
        raise ValueError(f"`verbosity` must be non-negative, was given '{verbosity}'.")
    if verbosity in (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ):
        return verbosity
    verbosity_to_logging_level = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
    }
    return verbosity_to_logging_level.get(verbosity, logging.DEBUG)
