"""Represents the time setup for a simulation, including start and end dates."""

__all__ = ("TimeSetup",)


from datetime import date

import pandas as pd
from pydantic import BaseModel, computed_field, model_validator


class TimeSetup(BaseModel):
    # pylint: disable=line-too-long
    """
    Time setup for the simulation.

    Attributes:
        start_date: The start date of the simulation.
        end_date: The end date of the simulation.

    Examples:
        >>> from gempyor.time_setup import TimeSetup
        >>> from datetime import date
        >>> ts = TimeSetup(start_date=date(2024, 1, 1), end_date=date(2024, 1, 10))
        >>> ts.start_date
        datetime.date(2024, 1, 1)
        >>> ts.end_date
        datetime.date(2024, 1, 10)
        >>> ts.n_days
        10
        >>> ts.dates
        [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), ..., datetime.date(2024, 1, 10)]
        >>> TimeSetup(start_date='2024-01-01', end_date='2024-01-01')
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for TimeSetup
          Value error, End date, 2024-01-01, is on or before the start date, 2024-01-01. [type=value_error, input_value={'start_date': '2024-01-0...end_date': '2024-01-01'}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """
    # pylint: enable=line-too-long

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _validate_dates(self) -> "TimeSetup":
        """
        Validate that the start date is before the end date.

        Raises:
            ValueError: If the end date is on or before the start date.
        """
        if self.end_date <= self.start_date:
            msg = (
                f"End date, {self.end_date}, is on or "
                f"before the start date, {self.start_date}."
            )
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def n_days(self) -> int:
        """
        Number of days in the simulation.

        Returns:
            The number of days in the simulation.
        """
        return (self.end_date - self.start_date).days + 1

    @computed_field
    @property
    def dates(self) -> list[date]:
        """
        Dates in the simulation.

        Returns:
            A list containing the dates in the simulation ranging from `start_date`
            to `end_date` inclusive.
        """
        return [self.start_date + pd.Timedelta(days=i) for i in range(self.n_days)]

    @computed_field
    @property
    def ti(self) -> date:
        """
        Alias for `start_date`.

        Alias for the start date attribute of this object for backward compatibility.

        Returns:
            The start date of the simulation.
        """
        return self.start_date

    @computed_field
    @property
    def tf(self) -> date:
        """
        Alias for `end_date`.

        Alias for the end date attribute of this object for backward compatibility.

        Returns:
            The end date of the simulation.
        """
        return self.end_date
