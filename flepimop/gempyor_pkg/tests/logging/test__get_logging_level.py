import logging

import pytest

from gempyor.logging import _get_logging_level


@pytest.mark.parametrize("verbosity", (-1, -100))
def test__get_logging_level_negative_verbosity_value_error(verbosity: int) -> None:
    with pytest.raises(
        ValueError, match=f"`verbosity` must be non-negative, was given '{verbosity}'."
    ):
        _get_logging_level(verbosity)


@pytest.mark.parametrize(
    ("verbosity", "expected_level"),
    (
        (logging.ERROR, logging.ERROR),
        (logging.WARNING, logging.WARNING),
        (logging.INFO, logging.INFO),
        (logging.DEBUG, logging.DEBUG),
        (0, logging.ERROR),
        (1, logging.WARNING),
        (2, logging.INFO),
        (3, logging.DEBUG),
        (4, logging.DEBUG),
        (5, logging.DEBUG),
    ),
)
def test__get_logging_level_output_validation(
    verbosity: int, expected_level: int
) -> None:
    assert _get_logging_level(verbosity) == expected_level
