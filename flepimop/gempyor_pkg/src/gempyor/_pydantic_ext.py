from typing import Any

__all__ = ["_ensure_list", "_override_or_val"]


# once 3.12, use type parametrization
# def _ensure_list[T](value: T | list[T] | tuple) -> list[T]:
def _ensure_list(value: Any) -> list | None:
    if value is None:
        return None
    elif isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        return list(value)
    else:
        return [value]


# once 3.12, use type parametrization
# def _override_or_val[T](override : T | None, value : T) -> T:
def _override_or_val(override: Any, value: Any) -> Any:
    return value if override is None else override
