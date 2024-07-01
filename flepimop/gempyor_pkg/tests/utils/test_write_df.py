import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Callable, Any

import pytest
import pandas as pd

from gempyor.utils import write_df


class TestWriteDf:
    """
    Unit tests for the `gempyor.utils.write_df` function.
    """

    sample_df: pd.DataFrame = pd.DataFrame(
        {
            "abc": [1, 2, 3, 4, 5],
            "def": ["v", "w", "x", "y", "z"],
            "ghi": [True, False, False, None, True],
            "jkl": [1.2, 3.4, 5.6, 7.8, 9.0],
        }
    )

    def test_raises_not_implemented_error(self) -> None:
        """
        Tests that write_df raises a NotImplementedError for unsupported file
        extensions.
        """
        with pytest.raises(
            expected_exception=NotImplementedError,
            match="Invalid extension txt. Must be 'csv' or 'parquet'",
        ) as _:
            with NamedTemporaryFile(suffix=".txt") as temp_file:
                write_df(fname=temp_file.name, df=self.sample_df)

    @pytest.mark.parametrize(
        "fname_transformer,extension",
        [
            (lambda x: str(x), ""),
            (lambda x: x, ""),
            (lambda x: f"{x.parent}/{x.stem}", "csv"),
            (lambda x: Path(f"{x.parent}/{x.stem}"), "csv"),
        ],
    )
    def test_write_csv_dataframe(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        extension: str,
    ) -> None:
        """
        Tests writing a DataFrame to a CSV file.

        Args:
            fname_transformer: A function that transforms the file name to create the
                `fname` arg.
            extension: The file extension to use, provided directly to
                `gempyor.utils.write_df`.
        """
        self._test_write_df(
            fname_transformer=fname_transformer,
            df=self.sample_df,
            extension=extension,
            suffix=".csv",
            path_reader=lambda x: pd.read_csv(x, index_col=False),
        )

    @pytest.mark.parametrize(
        "fname_transformer,extension",
        [
            (lambda x: str(x), ""),
            (lambda x: x, ""),
            (lambda x: f"{x.parent}/{x.stem}", "parquet"),
            (lambda x: Path(f"{x.parent}/{x.stem}"), "parquet"),
        ],
    )
    def test_write_parquet_dataframe(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        extension: str,
    ) -> None:
        """
        Tests writing a DataFrame to a Parquet file.

        Args:
            fname_transformer: A function that transforms the file name to create the
                `fname` arg.
            extension: The file extension to use, provided directly to
                `gempyor.utils.write_df`.
        """
        self._test_write_df(
            fname_transformer=fname_transformer,
            df=self.sample_df,
            extension=extension,
            suffix=".parquet",
            path_reader=lambda x: pd.read_parquet(x, engine="pyarrow"),
        )

    def _test_write_df(
        self,
        fname_transformer: Callable[[os.PathLike], Any],
        df: pd.DataFrame,
        extension: str,
        suffix: str | None,
        path_reader: Callable[[os.PathLike], pd.DataFrame],
    ) -> None:
        """
        Helper method to test writing a DataFrame to a file.

        Args:
            fname_transformer: A function that transforms the file name.
            df: The DataFrame to write.
            extension: The file extension to use.
            suffix: The suffix to use for the temporary file.
            path_reader: A function to read the DataFrame from the file.
        """
        with NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            assert temp_path.stat().st_size == 0
            assert (
                write_df(fname=fname_transformer(temp_path), df=df, extension=extension)
                is None
            )
            assert temp_path.stat().st_size > 0
            test_df = path_reader(temp_path)
            assert test_df.equals(df)
