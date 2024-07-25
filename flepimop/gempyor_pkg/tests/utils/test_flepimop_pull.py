import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch
from gempyor.resume_pull import fetching_resume_files


@pytest.fixture
def runner():
    return CliRunner()


class TestFetchingResumeFiles:
    @pytest.fixture(autouse=True)
    def set_env(self):
        with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}):
            yield

    def test_s3_resume_location(self, runner):
        with patch(
            "gempyor.resume_pull.create_resume_file_names_map", return_value="dummy_map"
        ) as mock_create_map, patch("gempyor.resume_pull.download_file_from_s3") as mock_download:
            result = runner.invoke(
                fetching_resume_files,
                [
                    "--resume_location",
                    "s3://some/location",
                    "--discard_seeding",
                    "true",
                    "--block_index",
                    1,
                    "--resume_run_index",
                    "1",
                    "--flepi_run_index",
                    "1",
                    "--flepi_prefix",
                    "prefix123",
                ],
            )
            assert result.exit_code == 0
            mock_create_map.assert_called_once()
            mock_download.assert_called_once()

    def test_local_resume_location(self, runner):
        with patch("gempyor.resume_pull.move_file_at_local") as mock_move:
            result = runner.invoke(
                fetching_resume_files,
                [
                    "--resume_location",
                    "local/path",
                    "--discard_seeding",
                    "true",
                    "--block_index",
                    1,
                    "--resume_run_index",
                    "run123",
                    "--flepi_run_index",
                    "run123",
                    "--flepi_prefix",
                    "prefix123",
                ],
            )
            assert result.exit_code == 0
            mock_move.assert_called_once()


if __name__ == "__main__":
    pytest.main()
