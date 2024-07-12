import os.path
from pathlib import Path

import pytest

from gempyor.file_paths import create_dir_name


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
            ("20240101_000000", "test0001", "seed", "", ""),
            ("20240101_000000", "test0002", "seed", "", ""),
            ("20240101_000000", "test0003", "seed", "", ""),
            ("20240101_000000", "test0004", "seed", "", ""),
            ("20240101_000000", "test0005", "hosp", "", ""),
            ("20240101_000000", "test0006", "hosp", "", ""),
            ("20240101_000000", "test0007", "hosp", "", ""),
            ("20240101_000000", "test0008", "hosp", "", ""),
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
