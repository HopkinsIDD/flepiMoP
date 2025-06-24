"""General purpose discriminated union for modifiers."""

__all__: tuple[str, ...] = ()

from typing import Annotated

from pydantic import Field

from ._single_period import SinglePeriodModifier


Modifier = Annotated[SinglePeriodModifier, Field(discriminator="method")]
