import io
import logging
from typing import Any, IO

import pytest

from gempyor.logging import ClickHandler


@pytest.mark.parametrize(
    "level",
    (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL),
)
@pytest.mark.parametrize("file", (None, io.StringIO))
@pytest.mark.parametrize("nl", (True, False))
@pytest.mark.parametrize("err", (False, True))
@pytest.mark.parametrize("color", (None, False))
@pytest.mark.parametrize("punctuate", (True, False))
def test_click_handler_init(
    level: int | str,
    file: IO[Any] | None,
    nl: bool,
    err: bool,
    color: bool | None,
    punctuate: bool,
) -> None:
    handler = ClickHandler(
        level=level, file=file, nl=nl, err=err, color=color, punctuate=punctuate
    )
    assert handler.level == level
    assert handler._file == file
    assert handler._nl == nl
    assert handler._err == err
    assert handler._color == color
    assert handler._punctuate == punctuate


@pytest.mark.parametrize("punctuate", (True, False))
@pytest.mark.parametrize(
    "msg",
    (
        "This is a message",
        "Another message.",
        "Start to a list:",
        "Middle of a sentence,",
        "Oh-no!",
        "Question?",
    ),
)
def test_click_handler_punctuation_formatting(punctuate: bool, msg: str) -> None:
    buffer = io.StringIO()
    handler = ClickHandler(level=logging.DEBUG, file=buffer, punctuate=punctuate)
    log_record = logging.LogRecord(
        name="",
        level=logging.DEBUG,
        pathname="",
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    handler.emit(log_record)
    buffer.seek(0)
    handler_msg = buffer.getvalue()
    if not punctuate:
        assert handler_msg == msg + "\n"
    else:
        assert (
            handler_msg
            == (msg if msg.endswith(ClickHandler._punctuation) else f"{msg}.") + "\n"
        )
