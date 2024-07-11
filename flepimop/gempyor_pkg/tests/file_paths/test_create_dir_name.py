import os.path
from pathlib import Path

import pytest

from gempyor.file_paths import create_dir_name
from gempyor.testing import *


class TestCreateDirName:
    """
    Unit tests for the `gempyor.file_paths.create_dir_name` function.
    """

    @pytest.mark.parametrize(
        (
            "run_id",
            "prefix",
            "ftype",
            "inference_filepath_suffix",
            "inference_filename_prefix",
        ),
        [
            ("abc", "def", "jkl", "mno", "pqr"),
        ],
    )
    def test_create_file_name(
        self,
        run_id: str, 
        prefix: str, 
        ftype: str, 
        inference_filepath_suffix: str, 
        inference_filename_prefix: str,
    ) -> None:
        # Setup
        path = create_dir_name(
            run_id=run_id,
            prefix=prefix,
            ftype=ftype,
            inference_filepath_suffix=inference_filepath_suffix,
            inference_filename_prefix=inference_filename_prefix,
        )
        expected_path = str(
            Path(
                "model_output",
                prefix,
                ftype,
                inference_filepath_suffix,
            )
        )
        
        # Assertions
        assert isinstance(path, str)
        assert not os.path.exists(path)
        assert path == expected_path
