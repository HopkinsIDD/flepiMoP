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
        with patch("gempyor.resume_pull.download_file_from_s3") as mock_download, patch(
            "gempyor.resume_pull.pull_check_for_s3"
        ) as mock_pull_check_for_s3:
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
            mock_download.assert_called_once()
            mock_pull_check_for_s3.assert_called_once()

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
