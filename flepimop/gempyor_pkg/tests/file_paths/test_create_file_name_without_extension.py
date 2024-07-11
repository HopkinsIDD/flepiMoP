# from collections.abc import Generator
# import os
from pathlib import Path
# from tempfile import TemporaryDirectory

import pytest

from gempyor.file_paths import create_file_name_without_extension
from gempyor.testing import *


class TestCreateFileNameWithoutExtension:
    """
    Unit tests for the `gempyor.file_paths.create_file_name_without_extension` function.
    """

    @pytest.mark.usefixtures("change_directory_to_temp_directory")
    @pytest.mark.parametrize(
        (
            "run_id",
            "prefix",
            "index",
            "ftype",
            "inference_filepath_suffix",
            "inference_filename_prefix",
            "create_directory",
        ),
        [
            ("abc", "def", "ghi", "jkl", "mno", "pqr", False),
            ("abc", "def", "ghi", "jkl", "mno", "pqr", True),
            ("abc", "def", 123, "jkl", "mno", "pqr", False),
            ("abc", "def", 123, "jkl", "mno", "pqr", True),
        ],
    )
    def test_create_file_name(
        self,
        run_id: str,
        prefix: str,
        index: str | int,
        ftype: str,
        inference_filepath_suffix: str,
        inference_filename_prefix: str,
        create_directory: bool,
    ) -> None:
        # Setup
        path = create_file_name_without_extension(
            run_id=run_id,
            prefix=prefix,
            index=index,
            ftype=ftype,
            inference_filepath_suffix=inference_filepath_suffix,
            inference_filename_prefix=inference_filename_prefix,
            create_directory=create_directory,
        )
        expected_path = Path(
            "model_output",
            prefix,
            ftype,
            inference_filepath_suffix,
            f"{inference_filename_prefix}{index:>09}.{run_id}.{ftype}",
        )
        
        # Assertions
        assert isinstance(path, Path)
        assert path.exists() == False
        assert path == expected_path
        if create_directory:
            assert path.parent.exists()
        else:
            assert not path.parent.exists()
