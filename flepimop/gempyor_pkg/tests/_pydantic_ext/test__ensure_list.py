"""Unit tests for `gempyor._pydantic_ext._ensure_list`."""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator
import pytest

from gempyor._pydantic_ext import _ensure_list


@pytest.mark.parametrize(
    "value", (None, "abc", 123, [True, False, True], (True, False, True), (1, 2, 3, 4)),
)
def test_value_output_validation(value) -> None:
    """Ensure that the output of `_ensure_list` is as expected."""
    list_value = _ensure_list(value)
    if value is None:
        assert list_value is None
    elif isinstance(value, (list, tuple)):
        assert list_value == list(value)
    else:
        assert list_value == [value]


@pytest.mark.parametrize(
    "letters", (None, "a", "b", "abc", ["a", "b", "c"], ("a", "b", "c")),
)
def test_pydantic_compatibility(letters: list[str] | tuple[str] | str | None) -> None:
    """Ensure that `_ensure_list` can be used as a Pydantic validator."""
    class Alphabet(BaseModel):
        letters: Annotated[list[str] | None, BeforeValidator(_ensure_list)]

    ab = Alphabet(letters=letters)
    assert ab.letters == _ensure_list(letters)
