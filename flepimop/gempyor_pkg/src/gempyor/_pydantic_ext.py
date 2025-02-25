"""
Gempyor specific Pydantic extensions.

This module contains functions that are useful for working with and creating Pydantic
models.
"""

__all__ = ()


from typing import TypeVar, overload


T = TypeVar("T")
U = TypeVar("U")


def _ensure_list(value: list[T] | tuple[T] | T | None) -> list[T] | None:
    """
    Ensure that a list, tuple, or single value is returned as a list.
    
    Args:
        value: A value to ensure is a list.
    
    Returns:
        A list of the value(s), if the `value` is not None.

    Examples:
        >>> from gempyor._pydantic_ext import _ensure_list
        >>> _ensure_list(None) is None
        True
        >>> _ensure_list("abc")
        ['abc']
        >>> _ensure_list(123)
        [123]
        >>> _ensure_list([True, False, True])
        [True, False, True]
        >>> _ensure_list((True, False, True))
        [True, False, True]
        >>> _ensure_list((1, 2, 3, 4))
        [1, 2, 3, 4]
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@overload
def _override_or_val(override: T, value: U) -> T: ...


@overload
def _override_or_val(override: None, value: U) -> U: ...


def _override_or_val(override: T | None, value: U) -> T | U:
    """
    Return the override value if it is not None, otherwise return the value.
    
    Args:
        override: Optional override value.
        value: The value to return if the override is None.
    
    Returns:
        The `override` value if it is not None, otherwise `value`.
    
    Examples:
        >>> from gempyor._pydantic_ext import _override_or_val
        >>> _override_or_val(None, 1)
        1
        >>> _override_or_val(None, "abc")
        'abc'
        >>> _override_or_val(1, "abc")
        1
        >>> _override_or_val("", "foo")
        ''
    """
    return value if override is None else override
