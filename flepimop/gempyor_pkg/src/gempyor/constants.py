__all__ = ()


import re
from typing import Annotated

from pydantic import Field


_JOB_NAME_REGEX = re.compile(r"^[a-z]{1}([a-z0-9\_\-]+)?$", flags=re.IGNORECASE)
_SAMPLES_SIMULATIONS_RATIO: Annotated[float, Field(gt=0.0)] = 0.6
