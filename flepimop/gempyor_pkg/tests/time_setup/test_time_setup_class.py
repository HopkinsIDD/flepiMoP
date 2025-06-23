"""Unit tests for the `gempyor.time_setup.TimeSetup` class."""

from datetime import date

import pytest

from gempyor.time_setup import TimeSetup


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2024, 1, 1), date(2024, 1, 1)),
        (date(2025, 2, 1), date(2025, 1, 31)),
        (date(2023, 10, 1), date(2023, 9, 1)),
    ],
)
def test_end_date_on_or_before_start_date(start_date: date, end_date: date) -> None:
    """Test that the end date is on or before the start date."""
    assert end_date <= start_date
    with pytest.raises(
        ValueError,
        match=rf"End date, {end_date}, is on or before the start date, {start_date}.",
    ):
        TimeSetup(start_date=start_date, end_date=end_date)


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2024, 1, 1), date(2024, 1, 10)),
        (date(2025, 1, 1), date(2025, 12, 31)),
        (date(2023, 10, 1), date(2024, 10, 1)),
    ],
)
def test_instance_attributes(start_date: date, end_date: date) -> None:
    """Test that the instance attributes are set correctly."""
    assert end_date > start_date
    time_setup = TimeSetup(start_date=start_date, end_date=end_date)
    assert time_setup.start_date == time_setup.ti == start_date
    assert time_setup.end_date == time_setup.tf == end_date
    assert isinstance(time_setup.dates, list) and all(
        isinstance(d, date) for d in time_setup.dates
    )
    assert time_setup.dates[0] == start_date
    assert time_setup.dates[-1] == end_date
    assert len(time_setup.dates) == time_setup.n_days
