"""Definition for a single period modifier."""

__all__: tuple[str, ...] = ()

from datetime import date
from typing import Literal

from pydantic import field_validator, model_validator

from ..distributions import Distribution
from ._base import ModifierABC


class SinglePeriodModifier(ModifierABC):
    """Modifier that applies to a single period."""

    method: Literal["SinglePeriodModifier"] = "SinglePeriodModifier"
    parameter: str
    subpop: Literal["all"] | list[str] = "all"
    period_start_date: date
    period_end_date: date
    value: Distribution
    perturbation: Distribution

    @field_validator("subpop", mode="before")
    @classmethod
    def _format_subpop_as_list(cls, subpop: str | list[str]) -> Literal["all"] | list[str]:
        """
        Coerce the subpop to a list of strings.

        If subpop is a string and not "all", it will be converted to a list of strings
        for consistency and ease of user specification.

        Args:
            subpop: The subpopulation identifier, which can be a string or a list of
                strings.

        Returns:
            A list of strings representing the subpopulation identifiers or "all" if
            that literal string was specified.

        """
        return [subpop] if isinstance(subpop, str) and subpop != "all" else subpop

    @model_validator(mode="after")
    def _validate_date_order(self) -> "SinglePeriodModifier":
        """
        Validate that the period start date is before the period end date.

        Raises:
            ValueError: If the period start date is not before the period end date.

        Returns:
            The validated SinglePeriodModifier instance.

        """
        if self.period_start_date > self.period_end_date:
            raise ValueError(
                f"The period start date, {self.period_start_date}, must "
                f"be before the period end date, {self.period_end_date}."
            )
        return self
