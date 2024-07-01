import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Callable, Any, Literal

import pytest
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype

from gempyor.utils import read_df


class TestReadDf:
    """
    Unit tests for the `gempyor.utils.read_df` function.
    """

    sample_df: pd.DataFrame = pd.DataFrame(
        {
            "abc": [1, 2, 3, 4, 5],
            "def": ["v", "w", "x", "y", "z"],
            "ghi": [True, False, False, None, True],
            "jkl": [1.2, 3.4, 5.6, 7.8, 9.0],
        }
    )

    subpop_df: pd.DataFrame = pd.DataFrame(
        {
            "subpop": [1, 2, 3, 4],
            "value": [5, 6, 7, 8],
        }
    )

    def test_raises_not_implemented_error(self) -> None:
        """
        Tests that read_df raises a NotImplementedError for unsupported file
        extensions.
        """
        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Invalid extension txt. Must be 'csv' or 'parquet'.",
        ) as _:
            with NamedTemporaryFile(suffix=".txt") as temp_file:
                read_df(fname=temp_file.name)
        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Invalid extension txt. Must be 'csv' or 'parquet'.",
        ) as _:
            with NamedTemporaryFile(suffix=".txt") as temp_file:
                fname = temp_file.name[:-4]
                read_df(fname=fname, extension="txt")

    @pytest.mark.parametrize(
        "fname_transformer,extension",
        [
            (lambda x: str(x), ""),
            (lambda x: x, ""),
            (lambda x: str(x), None),
            (lambda x: x, None),
            (lambda x: f"{x.parent}/{x.stem}", "csv"),
            (lambda x: Path(f"{x.parent}/{x.stem}"), "csv"),
        ],
    )
    def test_read_csv_dataframe(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        extension: Literal[None, "", "csv", "parquet"],
    ) -> None:
        """
        Tests reading a DataFrame from a CSV file.

        Args:
            fname_transformer: A function that transforms the file name to create the
                `fname` arg.
            extension: The file extension to use, provided directly to
                `gempyor.utils.read_df`.
        """
        self._test_read_df(
            fname_transformer=fname_transformer,
            extension=extension,
            suffix=".csv",
            path_writer=lambda p, df: df.to_csv(p, index=False),
        )

    @pytest.mark.parametrize(
        "fname_transformer,extension",
        [
            (lambda x: str(x), ""),
            (lambda x: x, ""),
            (lambda x: str(x), None),
            (lambda x: x, None),
            (lambda x: f"{x.parent}/{x.stem}", "parquet"),
            (lambda x: Path(f"{x.parent}/{x.stem}"), "parquet"),
        ],
    )
    def test_read_parquet_dataframe(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        extension: Literal[None, "", "csv", "parquet"],
    ) -> None:
        """
        Tests reading a DataFrame from a Parquet file.

        Args:
            fname_transformer: A function that transforms the file name to create the
                `fname` arg.
            extension: The file extension to use, provided directly to
                `gempyor.utils.read_df`.
        """
        self._test_read_df(
            fname_transformer=fname_transformer,
            extension=extension,
            suffix=".parquet",
            path_writer=lambda p, df: df.to_parquet(p, engine="pyarrow", index=False),
        )

    def test_subpop_is_cast_as_str(self) -> None:
        """
        Tests that read_df returns an object dtype for the column 'subpop' when reading
        a csv file, but not when reading a parquet file.
        """
        # First confirm the dtypes of our test DataFrame
        assert is_numeric_dtype(self.subpop_df["subpop"])
        assert is_numeric_dtype(self.subpop_df["value"])
        # Test that the subpop column is converted to a string for a csv file
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            temp_path = Path(temp_file.name)
            assert temp_path.stat().st_size == 0
            assert self.subpop_df.to_csv(temp_path, index=False) is None
            assert temp_path.stat().st_size > 0
            test_df = read_df(fname=temp_path)
            assert isinstance(test_df, pd.DataFrame)
            assert is_object_dtype(test_df["subpop"])
            assert is_numeric_dtype(test_df["value"])
        # Test that the subpop column remains unaltered for a parquet file
        with NamedTemporaryFile(suffix=".parquet") as temp_file:
            temp_path = Path(temp_file.name)
            assert temp_path.stat().st_size == 0
            assert (
                self.subpop_df.to_parquet(temp_path, engine="pyarrow", index=False)
                is None
            )
            assert temp_path.stat().st_size > 0
            test_df = read_df(fname=temp_path)
            assert isinstance(test_df, pd.DataFrame)
            assert is_numeric_dtype(test_df["subpop"])
            assert is_numeric_dtype(test_df["value"])

    def _test_read_df(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        extension: Literal[None, "", "csv", "parquet"],
        suffix: str | None,
        path_writer: Callable[[os.PathLike, pd.DataFrame], None],
    ) -> None:
        """
        Helper method to test writing a DataFrame to a file.

        Args:
            fname_transformer: A function that transforms the file name.
            extension: The file extension to use.
            suffix: The suffix to use for the temporary file.
            path_writer: A function to write the DataFrame to the file.
        """
        with NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            assert temp_path.stat().st_size == 0
            path_writer(temp_path, self.sample_df)
            test_df = read_df(fname=fname_transformer(temp_path), extension=extension)
            assert isinstance(test_df, pd.DataFrame)
            assert temp_path.stat().st_size > 0
            assert test_df.equals(self.sample_df)
