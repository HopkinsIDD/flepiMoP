"""Definition for a single period modifier."""

__all__: tuple[str, ...] = ()

import numpy as np
import pandas as pd
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

    def apply(
        self,
        parameter: np.ndarray,
        modification: pd.DataFrame | float,
        method: str = "product",
    ) -> np.ndarray:
        """
        Applies a modification to parameters.

        Args:
            parameter: Array of parameters
            modification: Modification to be applied
            method: Method of modification (e.g., "product", "sum", "reduction_product")

        Returns:
            The array of parameters with the modification applied
        """
        if isinstance(modification, pd.DataFrame):
            modification = modification.T
            modification.index = pd.to_datetime(modification.index.astype(str))
            modification = modification.resample("1D").ffill().to_numpy()

        if method == "reduction_product":
            return parameter * (1 - modification)
        elif method == "sum":
            return parameter + modification
        elif method == "product":
            return parameter * modification
        else:
            raise ValueError(f"Unknown modifier method, received: {method}")

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
