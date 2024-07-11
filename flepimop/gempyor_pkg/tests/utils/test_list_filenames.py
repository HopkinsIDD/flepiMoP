"""Unit tests for the `gempyor.utils.list_filenames` function.

These tests cover scenarios for finding files in both flat and nested directories.
"""

from collections.abc import Generator
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gempyor.utils import list_filenames


@pytest.fixture(scope="class")
def create_directories_with_files(
    request: pytest.FixtureRequest,
) -> Generator[tuple[TemporaryDirectory, TemporaryDirectory], None, None]:
    """Fixture to create temporary directories with files for testing.

    This fixture creates two temporary directories:
    - A flat directory with files.
    - A nested directory with files organized in subdirectories.

    The directories and files are cleaned up after the tests are run.

    Args:
        request: The pytest fixture request object.

    Yields:
        tuple: A tuple containing the flat and nested TemporaryDirectory objects.
    """
    # Create a flat and nested directories
    flat_temp_dir = TemporaryDirectory()
    nested_temp_dir = TemporaryDirectory()
    # Setup flat directory
    for file in ["hosp.csv", "hosp.parquet", "spar.csv", "spar.parquet"]:
        Path(f"{flat_temp_dir.name}/{file}").touch()
    # Setup nested directory structure
    for file in [
        "hpar/chimeric/001.parquet",
        "hpar/chimeric/002.parquet",
        "hpar/global/001.parquet",
        "hpar/global/002.parquet",
        "seed/001.csv",
        "seed/002.csv",
    ]:
        path = Path(f"{nested_temp_dir.name}/{file}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # Yield
    request.cls.flat_temp_dir = flat_temp_dir
    request.cls.nested_temp_dir = nested_temp_dir
    yield (flat_temp_dir, nested_temp_dir)
    # Clean up directories on test end
    flat_temp_dir.cleanup()
    nested_temp_dir.cleanup()


@pytest.mark.usefixtures("create_directories_with_files")
class TestListFilenames:
    """Unit tests for the `gempyor.utils.list_filenames` function."""

    @pytest.mark.parametrize(
        "filters,expected_basenames",
        [
            ("", ["hosp.csv", "hosp.parquet", "spar.csv", "spar.parquet"]),
            ([], ["hosp.csv", "hosp.parquet", "spar.csv", "spar.parquet"]),
            ("hosp", ["hosp.csv", "hosp.parquet"]),
            (["hosp"], ["hosp.csv", "hosp.parquet"]),
            ("spar", ["spar.csv", "spar.parquet"]),
            (["spar"], ["spar.csv", "spar.parquet"]),
            (".parquet", ["hosp.parquet", "spar.parquet"]),
            ([".parquet"], ["hosp.parquet", "spar.parquet"]),
            (".csv", ["hosp.csv", "spar.csv"]),
            ([".csv"], ["hosp.csv", "spar.csv"]),
            (".tsv", []),
            ([".tsv"], []),
            (["hosp", ".csv"], ["hosp.csv"]),
            (["spar", ".parquet"], ["spar.parquet"]),
            (["hosp", "spar"], []),
            ([".csv", ".parquet"], []),
        ],
    )
    def test_finds_files_in_flat_directory(
        self,
        filters: str | list[str],
        expected_basenames: list[str],
    ) -> None:
        """Test `list_filenames` in a flat directory.

        Args:
            filters: List of filters to apply to filenames.
            expected_basenames: List of expected filenames that match the filters.
        """
        self._test_list_filenames(
            folder=self.flat_temp_dir.name,
            filters=filters,
            expected_basenames=expected_basenames,
        )
        self._test_list_filenames(
            folder=self.flat_temp_dir.name.encode(),
            filters=filters,
            expected_basenames=expected_basenames,
        )
        self._test_list_filenames(
            folder=Path(self.flat_temp_dir.name),
            filters=filters,
            expected_basenames=expected_basenames,
        )

    @pytest.mark.parametrize(
        "filters,expected_basenames",
        [
            (
                "",
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/chimeric/002.parquet",
                    "hpar/global/001.parquet",
                    "hpar/global/002.parquet",
                    "seed/001.csv",
                    "seed/002.csv",
                ],
            ),
            (
                [],
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/chimeric/002.parquet",
                    "hpar/global/001.parquet",
                    "hpar/global/002.parquet",
                    "seed/001.csv",
                    "seed/002.csv",
                ],
            ),
            (
                "hpar",
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/chimeric/002.parquet",
                    "hpar/global/001.parquet",
                    "hpar/global/002.parquet",
                ],
            ),
            (
                ["hpar"],
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/chimeric/002.parquet",
                    "hpar/global/001.parquet",
                    "hpar/global/002.parquet",
                ],
            ),
            ("seed", ["seed/001.csv", "seed/002.csv"]),
            (["seed"], ["seed/001.csv", "seed/002.csv"]),
            ("global", ["hpar/global/001.parquet", "hpar/global/002.parquet"]),
            (["global"], ["hpar/global/001.parquet", "hpar/global/002.parquet"]),
            (
                "001",
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/global/001.parquet",
                    "seed/001.csv",
                ],
            ),
            (
                ["001"],
                [
                    "hpar/chimeric/001.parquet",
                    "hpar/global/001.parquet",
                    "seed/001.csv",
                ],
            ),
            (["hpar", "001"], ["hpar/chimeric/001.parquet", "hpar/global/001.parquet"]),
            (["seed", "002"], ["seed/002.csv"]),
            (["hpar", "001", "global"], ["hpar/global/001.parquet"]),
            (".tsv", []),
            ([".tsv"], []),
        ],
    )
    def test_find_files_in_nested_directory(
        self,
        filters: str | list[str],
        expected_basenames: list[str],
    ) -> None:
        """Test `list_filenames` in a nested directory.

        Args:
            filters: List of filters to apply to filenames.
            expected_basenames: List of expected filenames that match the filters.
        """
        self._test_list_filenames(
            folder=self.nested_temp_dir.name,
            filters=filters,
            expected_basenames=expected_basenames,
        )
        self._test_list_filenames(
            folder=self.nested_temp_dir.name.encode(),
            filters=filters,
            expected_basenames=expected_basenames,
        )
        self._test_list_filenames(
            folder=Path(self.nested_temp_dir.name),
            filters=filters,
            expected_basenames=expected_basenames,
        )

    def _test_list_filenames(
        self,
        folder: str | bytes | os.PathLike,
        filters: str | list[str],
        expected_basenames: list[str],
    ) -> None:
        """Helper method to test `list_filenames`.

        Args:
            folder: The directory to search for files.
            filters: List of filters to apply to filenames.
            expected_basenames: List of expected filenames that match the filters.
        """
        files = list_filenames(folder=folder, filters=filters)
        assert len(files) == len(expected_basenames)
        folder = folder.decode() if isinstance(folder, bytes) else str(folder)
        basenames = [f.removeprefix(f"{folder}/") for f in files]
        assert sorted(basenames) == sorted(expected_basenames)
