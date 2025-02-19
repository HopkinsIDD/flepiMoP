from typing import Literal

import pytest
from pydantic import ValidationError

from gempyor.sync._sync import SyncOptions, _ensure_list, SyncProtocols, _filter_mode


@pytest.mark.parametrize(
    "opts",
    [
        {"filter_override": "somefilter"},
        {"filter_override": ["somefilter", "another"]},
        {"filter_override": []},
    ],
)
def test_sync_opts_filters(opts: dict):
    """
    Ensures SyncOptions can instantiate valid objects w.r.t to filters
    """
    sp = SyncOptions(**opts)
    assert sp.filter_override == _ensure_list(opts["filter_override"])


@pytest.mark.parametrize(
    "opts",
    [
        {"filter_override": 1},
        {"filter_override": "-something"},
        {"filter_override": "+something"},
        {"filter_override": " something"},
    ],
)
def test_sync_opts_filters(opts: dict):
    """
    Ensures SyncOptions can identify invalid objects w.r.t to filters
    """
    with pytest.raises(ValidationError):
        SyncOptions(**opts)


@pytest.mark.parametrize(
    "opts,mode",
    [
        ({"filter_override": "+ something"}, "+"),
        ({"filter_override": "- something"}, "-"),
        ({"filter_override": ["- something", "other"]}, ["-", "+"]),
    ],
)
def test_sync_opts_filters_mode(opts: dict, mode: Literal["+", "-"]):
    """
    Ensures SyncOptions can identify invalid objects w.r.t to filters
    """
    assert [_filter_mode(f) for f in SyncOptions(**opts).filter_override] == _ensure_list(
        mode
    )


@pytest.mark.parametrize(
    "protocols",
    [
        {
            "demorsync": {"type": "rsync", "source": ".", "target": "host:~/some/path"},
            "demos3sync": {"type": "s3sync", "source": ".", "target": "some/path"},
            "demogit": {"type": "git", "mode": "push"},
        },
        {"justone": {"type": "git", "mode": "pull"}},
        {},
    ],
)
def test_successfully_construct_from_valid_protocols(protocols: dict):
    """
    Ensures SyncProtocols can instantiate valid objects
    """

    SyncProtocols(sync=protocols)


@pytest.mark.parametrize(
    "protocols",
    [
        {
            "demorsync": {
                "type": "unsupported",
                "source": ".",
                "target": "host:~/some/path",
            }
        },
        {"missingtar": {"type": "rsync", "source": "."}},
        {"badgit": {"type": "git", "source": "."}},
    ],
)
def test_fail_construct_from_invalid_protocols(protocols: dict):
    """
    Ensures SyncProtocols doesn't instantiate invalid objects
    """
    with pytest.raises(ValidationError):
        SyncProtocols(protocols=protocols)


# construct from yaml file(s)

# @pytest.mark.parametrize("data", (

# ))
# @pytest.mark.parametrize("which", (
#     1, (1, 2), (1, 3)
# ))
# def test_sync_yaml_load(
#     tmp_path: Path,
#     monkeypatch: pytest.MonkeyPatch,
#     dat: dict[str, Any],
#     which: str | ,
# ) -> None:
#     # Setup the test
#     monkeypatch.chdir(tmp_path)
#     config_one = tmp_path / "config_one.yml"
#     config_one.write_text(yaml.dump(data_one))
#     config_two = tmp_path / "config_two.yml"
#     config_two.write_text(yaml.dump(data_two))

#     # Invoke the command
#     runner = CliRunner()
#     result = runner.invoke(patch, [config_one.name, config_two.name])
#     assert result.exit_code == 1
#     assert isinstance(result.exception, ValueError)
#     assert str(result.exception) == (
#         "Configuration files contain overlapping keys, seir, introduced by config_two.yml."
#     )

# construct options

# import os
# import pytest
# from click.testing import CliRunner
# from unittest.mock import patch
# from gempyor.resume_pull import fetching_resume_files


# @pytest.fixture
# def runner():
#     return CliRunner()


# class TestFetchingResumeFiles:
#     @pytest.fixture(autouse=True)
#     def set_env(self):
#         with patch.dict(os.environ, {"SLURM_ARRAY_TASK_ID": "1"}):
#             yield

#     def test_s3_resume_location(self, runner):
#         with patch("gempyor.resume_pull.download_file_from_s3") as mock_download, patch(
#             "gempyor.resume_pull.pull_check_for_s3"
#         ) as mock_pull_check_for_s3:
#             result = runner.invoke(
#                 fetching_resume_files,
#                 [
#                     "--resume_location",
#                     "s3://some/location",
#                     "--discard_seeding",
#                     "true",
#                     "--block_index",
#                     1,
#                     "--resume_run_index",
#                     "1",
#                     "--flepi_run_index",
#                     "1",
#                     "--flepi_prefix",
#                     "prefix123",
#                 ],
#             )
#             assert result.exit_code == 0
#             mock_download.assert_called_once()
#             mock_pull_check_for_s3.assert_called_once()

#     def test_local_resume_location(self, runner):
#         with patch("gempyor.resume_pull.move_file_at_local") as mock_move:
#             result = runner.invoke(
#                 fetching_resume_files,
#                 [
#                     "--resume_location",
#                     "local/path",
#                     "--discard_seeding",
#                     "true",
#                     "--block_index",
#                     1,
#                     "--resume_run_index",
#                     "run123",
#                     "--flepi_run_index",
#                     "run123",
#                     "--flepi_prefix",
#                     "prefix123",
#                 ],
#             )
#             assert result.exit_code == 0
#             mock_move.assert_called_once()


# if __name__ == "__main__":
#     pytest.main()
