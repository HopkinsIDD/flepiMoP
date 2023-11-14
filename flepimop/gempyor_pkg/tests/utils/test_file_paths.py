import pytest
import datetime
import os
from mock import MagicMock

from typing import Callable, Any
from gempyor import file_paths

FAKE_TIME = datetime.datetime(2023, 8, 9, 16, 00, 0)

"""
@pytest.fixture(scope="module")
def mock_datetime_now(monkeypatch):
	datetime_mock = MagicMock(wraps=datetime.datetime)
	datetime_mock.now.return_value = FAKE_TIME
	monkeypatch.setattr(datetime, "datetime", datetime_mock)
@pytest.fixture(scope="module")
def test_datetime(mock_datetime_now):
	assert datetime.datetime.now() == FAKE_TIME
"""


def test_run_id(monkeypatch: pytest.MonkeyPatch):
    datetime_mock = MagicMock(wraps=datetime.datetime)
    datetime_mock.now.return_value = FAKE_TIME
    monkeypatch.setattr(datetime, "datetime", datetime_mock)

    run_id = file_paths.run_id()
    assert run_id == datetime.datetime.strftime(FAKE_TIME, "%Y%m%d_%H%M%S%Z")


@pytest.fixture(scope="module")
def set_run_id():
    return lambda: file_paths.run_id()


tmp_path = "/tmp"


@pytest.mark.parametrize(
    ("prefix", "ftype"),
    [
        ("test0001", "seed"),
        ("test0002", "seed"),
        ("test0003", "seed"),
        ("test0004", "seed"),
        ("test0005", "hosp"),
        ("test0006", "hosp"),
        ("test0007", "hosp"),
        ("test0008", "hosp"),
    ],
)
def test_create_dir_name(set_run_id, prefix, ftype):
    os.chdir(tmp_path)
    os.path.exists(file_paths.create_dir_name(set_run_id, prefix, ftype))


@pytest.mark.parametrize(
    ("prefix", "ftype", "inference_filepath_suffix", "inference_filename_prefix"),
    [
        ("test0001", "seed", "", ""),
        ("test0002", "seed", "", ""),
        ("test0003", "seed", "", ""),
        ("test0004", "seed", "", ""),
        ("test0005", "hosp", "", ""),
        ("test0006", "hosp", "", ""),
        ("test0007", "hosp", "", ""),
        ("test0008", "hosp", "", ""),
    ],
)
def test_create_dir_name(
    set_run_id: Callable[[], Any],
    prefix,
    ftype,
    inference_filepath_suffix,
    inference_filename_prefix,
):
    os.chdir(tmp_path)
    os.path.exists(
        file_paths.create_dir_name(set_run_id, prefix, ftype, inference_filepath_suffix, inference_filename_prefix)
    )


@pytest.mark.parametrize(
    (
        "prefix",
        "index",
        "ftype",
        "extension",
        "inference_filepath_suffix",
        "inference_filename_prefix",
        "create_directory",
    ),
    [
        ("test0001", "0", "seed", "csv", "", "", True),
        ("test0002", "0", "seed", "parquet", "", "", True),
        ("test0003", "0", "seed", "csv", "", "", False),
        ("test0004", "0", "seed", "parquet", "", "", False),
        ("test0001", "1", "seed", "csv", "", "", True),
        ("test0002", "1", "seed", "parquet", "", "", True),
        ("test0003", "1", "seed", "csv", "", "", False),
        ("test0004", "1", "seed", "parquet", "", "", False),
    ],
)
def test_create_file_name(
    set_run_id: Callable[[], Any],
    prefix,
    index,
    ftype,
    extension,
    inference_filepath_suffix,
    inference_filename_prefix,
    create_directory,
):
    os.chdir(tmp_path)
    os.path.isfile(
        file_paths.create_file_name(
            set_run_id,
            prefix,
            int(index),
            ftype,
            extension,
            inference_filepath_suffix,
            inference_filename_prefix,
            create_directory,
        )
    )


@pytest.mark.parametrize(
    ("prefix", "index", "ftype", "inference_filepath_suffix", "inference_filename_prefix", "create_directory"),
    [
        ("test0001", "0", "seed", "", "", True),
        ("test0002", "0", "seed", "", "", True),
        ("test0003", "0", "seed", "", "", False),
        ("test0004", "0", "seed", "", "", False),
        ("test0001", "1", "seed", "", "", True),
        ("test0002", "1", "seed", "", "", True),
        ("test0003", "1", "seed", "", "", False),
        ("test0004", "1", "seed", "", "", False),
    ],
)
def test_create_file_name_without_extension(
    set_run_id: Callable[[], Any],
    prefix,
    index,
    ftype,
    inference_filepath_suffix,
    inference_filename_prefix,
    create_directory,
):
    os.chdir(tmp_path)
    os.path.isfile(
        file_paths.create_file_name_without_extension(
            set_run_id,
            prefix,
            int(index),
            ftype,
            inference_filepath_suffix,
            inference_filename_prefix,
            create_directory,
        )
    )
