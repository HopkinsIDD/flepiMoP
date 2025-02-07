__all__ = []


from datetime import timedelta
from math import ceil
import re
import click
from typing import Any, Literal


class DurationParamType(click.ParamType):
    """
    A custom Click parameter type for parsing duration strings into `timedelta` objects.

    Attributes:
        name: The name of the parameter type.

    Examples:
        >>> from gempyor._click import DurationParamType
        >>> DurationParamType(False, "minutes").convert("23min", None, None)
        datetime.timedelta(seconds=1380)
        >>> DurationParamType(False, None).convert("2.5hr", None, None)
        datetime.timedelta(seconds=9000)
        >>> DurationParamType(False, "minutes").convert("-2", None, None)
        datetime.timedelta(days=-1, seconds=86280)
    """

    name = "duration"
    _abbreviations = {
        "s": "seconds",
        "sec": "seconds",
        "secs": "seconds",
        "second": "seconds",
        "seconds": "seconds",
        "m": "minutes",
        "min": "minutes",
        "mins": "minutes",
        "minute": "minutes",
        "minutes": "minutes",
        "h": "hours",
        "hr": "hours",
        "hrs": "hours",
        "hour": "hours",
        "hours": "hours",
        "d": "days",
        "day": "days",
        "days": "days",
        "w": "weeks",
        "week": "weeks",
        "weeks": "weeks",
    }

    def __init__(
        self,
        nonnegative: bool,
        default_unit: Literal["seconds", "minutes", "hours", "days", "weeks"] | None,
    ) -> None:
        """
        Initialize the instance based on parameter settings.

        Args:
            nonnegative: If `True` negative durations are not allowed.
            default_unit: The default unit to use if no unit is specified in the input
                string. If `None` a unitless duration is not allowed.

        Notes:
            It's on the user of this param type to document in their CLI help text what
            the default unit is if they set it to a non-`None` value.
        """
        super().__init__()
        self._nonnegative = nonnegative
        self._duration_regex = re.compile(
            rf"^((-)?([0-9]+)?(\.[0-9]+)?)({'|'.join(self._abbreviations.keys())})?$",
            flags=re.IGNORECASE,
        )
        self._default_unit = default_unit

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> timedelta:
        """
        Converts a string representation of a duration into a `timedelta` object.

        Args:
            value: The value to convert, expected to be a string like representation of
                a duration. Allowed durations are limited to seconds, minutes, hours,
                days, and weeks.
            param: The Click parameter object for context in errors.
            ctx: The Click context object for context in errors.

        Returns:
            The converted duration as a `timedelta` object.

        Raises:
            click.BadParameter: If the value is not a valid duration based on the
                format.
            click.BadParameter: If the duration is negative and the class was
                initialized with `nonnegative` set to `True`.
            click.BadParameter: If the duration is unitless and the class was
                initialized with `default_unit` set to `None`.
        """
        value = str(value).strip()
        if (m := self._duration_regex.match(value)) is None:
            self.fail(f"{value!r} is not a valid duration", param, ctx)
        number, posneg, _, _, unit = m.groups()
        if self._nonnegative and posneg == "-":
            self.fail(f"{value!r} is a negative duration", param, ctx)
        if unit is None:
            if self._default_unit is None:
                self.fail(f"{value!r} is a unitless duration", param, ctx)
            unit = self._default_unit
        kwargs = {}
        kwargs[self._abbreviations.get(unit.lower())] = float(number)
        return timedelta(**kwargs)


class MemoryParamType(click.ParamType):
    """
    A custom Click parameter type for parsing memory strings.

    Attributes:
        name: The name of the parameter type.

    Examples:
        >>> from gempyor._click import MemoryParamType
        >>> MemoryParamType(False, "mb", False).convert("12.34MB", None, None)
        12.34
        >>> MemoryParamType(True, "mb", True).convert("78.9", None, None)
        79
        >>> MemoryParamType(False, "gb", False).convert("123kb", None, None)
        0.00011730194091796875
    """

    name = "memory"
    _units = {
        "kb": 1024.0**1.0,
        "k": 1024.0**1.0,
        "mb": 1024.0**2.0,
        "m": 1024.0**2.0,
        "gb": 1024.0**3.0,
        "g": 1024.0**3.0,
        "t": 1024.0**4.0,
        "tb": 1024.0**4.0,
    }

    def __init__(self, as_int: bool, unit: str, allow_unitless: bool) -> None:
        """
        Initialize the instance based on parameter settings.

        Args:
            as_int: if `True` the `convert` method returns an integer instead of a
                float.
            unit: The output unit to use in the `convert` method.

        Raises:
            ValueError: If `unit` is not a valid memory unit size.
        """
        super().__init__()
        if (unit := unit.lower()) not in self._units.keys():
            raise ValueError(
                f"The `unit` given is not valid, given '{unit}' and "
                f"must be one of: {', '.join(self._units.keys())}."
            )
        self._unit = unit
        self._regex = re.compile(
            rf"^(([0-9]+)?(\.[0-9]+)?)({'|'.join(self._units.keys())})?$",
            flags=re.IGNORECASE,
        )
        self._as_int = as_int
        self._allow_unitless = allow_unitless

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> float | int:
        """
        Converts a string representation of a memory size into a numeric.

        Args:
            value: The value to convert, expected to be a string like representation of
                memory size.
            param: The Click parameter object for context in errors.
            ctx: The Click context object for context in errors.

        Returns:
            The converted memory size as a numeric. Specifically an integer if the
            `as_int` attribute is `True` and float otherwise.

        Raises:
            click.BadParameter: If the value is not a valid memory size based on the
                format.
            click.BadParameter: If the memory size is unitless and the class was
                initialized with `allow_unitless` set to `False`.
        """
        value = str(value).strip()
        if (m := self._regex.match(value)) is None:
            self.fail(f"{value!r} is not a valid memory size.", param, ctx)
        number, _, _, unit = m.groups()
        if unit is None:
            if not self._allow_unitless:
                self.fail(f"{value!r} is a unitless memory size.", param, ctx)
            unit = self._unit
        else:
            unit = unit.lower()
        if unit == self._unit:
            result = float(number)
        else:
            result = (self._units.get(unit, self._unit) * float(number)) / (
                self._units.get(self._unit)
            )
        return ceil(result) if self._as_int else result
