import os.path
from pathlib import Path

import pytest

from gempyor.file_paths import create_file_name
from gempyor.testing import *


class TestCreateFileName:
    """
    Unit tests for the `gempyor.file_paths.create_file_name` function.
    """

    @pytest.mark.usefixtures("change_directory_to_temp_directory")
    @pytest.mark.parametrize(
        (
            "run_id",
            "prefix",
            "index",
            "ftype",
            "extension",
            "inference_filepath_suffix",
            "inference_filename_prefix",
            "create_directory",
        ),
        [
            ("abc", "def", "ghi", "jkl", "csv", "mno", "pqr", False),
            ("abc", "def", "ghi", "jkl", "csv", "mno", "pqr", True),
            ("abc", "def", 123, "jkl", "parquet", "mno", "pqr", False),
            ("abc", "def", 123, "jkl", "parquet", "mno", "pqr", True),
            ("20240101_000000", "test0001", "0", "seed", "csv", "", "", True),
            ("20240101_000000", "test0002", "0", "seed", "parquet", "", "", True),
            ("20240101_000000", "test0003", "0", "seed", "csv", "", "", False),
            ("20240101_000000", "test0004", "0", "seed", "parquet", "", "", False),
            ("20240101_000000", "test0001", "1", "seed", "csv", "", "", True),
            ("20240101_000000", "test0002", "1", "seed", "parquet", "", "", True),
            ("20240101_000000", "test0003", "1", "seed", "csv", "", "", False),
            ("20240101_000000", "test0004", "1", "seed", "parquet", "", "", False),
        ],
    )
    def test_create_file_name(
        self,
        run_id: str,
        prefix: str,
        index: str | int,
        ftype: str,
        extension: str,
        inference_filepath_suffix: str,
        inference_filename_prefix: str,
        create_directory: bool,
    ) -> None:
        # Setup
        path = create_file_name(
            run_id=run_id,
            prefix=prefix,
            index=index,
            ftype=ftype,
            extension=extension,
            inference_filepath_suffix=inference_filepath_suffix,
            inference_filename_prefix=inference_filename_prefix,
            create_directory=create_directory,
        )
        expected_path = str(
            Path(
                "model_output",
                prefix,
                ftype,
                inference_filepath_suffix,
                f"{inference_filename_prefix}{index:>09}.{run_id}.{ftype}.{extension}",
            )
        )
        parent = os.path.dirname(path)

        # Assertions
        assert isinstance(path, str)
        assert not os.path.exists(path)
        assert path == expected_path
        if create_directory:
            assert os.path.exists(parent)
        else:
            assert not os.path.exists(parent)
