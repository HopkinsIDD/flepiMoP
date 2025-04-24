"""
Gempyor specific Pydantic extensions.

This module contains functions that are useful for working with and creating Pydantic
models.
"""

__all__ = ()


from pathlib import Path
from typing import Any, TypeVar, overload

import pyarrow.parquet as pq
import pandas as pd
from pydantic import BaseModel, RootModel


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


def _read_and_validate_dataframe(
    file: Path, model: type[BaseModel] | None = None, **kwargs: Any
) -> pd.DataFrame:
    """
    Read a tabular file and validate its contents against a Pydantic model.

    Args:
        file: Path to the file to read.
        model: Pydantic model to validate the data against or `None` to skip validation.
            If `model` is a `RootModel`, the data is expected to be a list of the row
            types otherwise `model` is expected to be a type representing the row type
            and will be wrapped in a `RootModel`.
        **kwargs: Additional arguments passed to the reader function.

    Returns:
        A DataFrame containing the data read from the file.

    Notes:
        The function supports reading CSV and Parquet files. The file type is
        determined by the file extension. The supported file types are:
        - CSV: The file must have a `.csv` extension and uses `pandas.read_csv` to read
            the file.
        - Parquet: The file must have a `.parquet` extension and uses
            `pyarrow.parquet.read_table` to read the file.

    Raises:
        ValueError: If the file type is not supported.

    Examples:
        >>> import math
        >>> from pathlib import Path
        >>> from typing import Annotated
        >>> import pandas as pd
        >>> from pydantic import BaseModel, Field, RootModel, model_validator
        >>> from gempyor._pydantic_ext import _read_and_validate_dataframe
        >>> file = Path("foobar.csv")
        >>> pd.DataFrame(
        ...     data={"name": ["Jack", "Jill"], "age": [23, 25]},
        ... ).to_csv(file, index=False)
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> _read_and_validate_dataframe(file, model=Person)
        name  age
        0  Jack   23
        1  Jill   25
        >>> pd.DataFrame(data={"name": [32], "age": ["Jane"]}).to_csv(file, index=False)
        >>> _read_and_validate_dataframe(file, model=Person)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for RootModel[list[Person]]
        0.age
        Input should be a valid integer, unable to parse string as an integer ...
        >>> class PartitionSlice(BaseModel):
        ...     name: str
        ...     amount: Annotated[float, Field(gt=0.0, lt=1.0)]
        >>> class Partition(RootModel):
        ...     root: list[PartitionSlice]
        ...
        ...     @model_validator(mode='after')
        ...     def check_sum(self) -> 'Partition':
        ...         if not math.isclose(sum([s.amount for s in self.root]), 1.0):
        ...             raise ValueError("The sum of the amounts must be equal to 1.0")
        ...         return self
        >>> pd.DataFrame(
        ...     data={"name": ["A", "B"], "amount": [0.5, 0.5]},
        ... ).to_csv(file, index=False)
        >>> _read_and_validate_dataframe(file, model=Partition)
        name  amount
        0    A     0.5
        1    B     0.5
        >>> pd.DataFrame(
        ...     data={"name": ["A", "B"], "amount": [0.5, 0.1]},
        ... ).to_csv(file, index=False)
        >>> _read_and_validate_dataframe(file, model=Partition)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for Partition
        Value error, The sum of the amounts must be equal to 1.0 ...

    """
    if file.suffix == ".csv":
        with file.open("r") as f:
            data = pd.read_csv(f, **kwargs).to_dict(orient="records")
    elif file.suffix == ".parquet":
        data = pq.read_table(file, **kwargs).to_pylist()
    else:
        raise ValueError(f"Unsupported file type '{file.suffix}'.")
    if model is not None:
        if not issubclass(model, RootModel):
            model = RootModel[list[model]]
        data = [r.model_dump() for r in model.model_validate(data).root]
    return pd.DataFrame.from_records(data)
