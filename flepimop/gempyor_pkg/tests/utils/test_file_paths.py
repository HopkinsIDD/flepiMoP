import pytest
import datetime
import os
from mock import MagicMock

from gempyor import file_paths 

FAKE_TIME = datetime.datetime(2023,8,9,16,00,0)

@pytest.fixture(scope="module")
def mock_datetime_now(monkeypatch):
	datetime_mock = MagicMock(wraps=datetime.datetime)
	datetime_mock.now.return_value = FAKE_TIME
	monkeypatch.setattr(datetime, "datetime", datetime_mock)

@pytest.fixture(scope="module")
def test_datetime(mock_datetime_now):
	assert datetime.datetime.now() == FAKE_TIME

def test_run_id():
	run_id = file_paths.run_id()
	assert run_id == datetime.datetime.strftime(datetime.datetime.now(), "%Y.%m.%d.%H:%M:%S.%Z")

@pytest.fixture(scope="module")
def set_run_id():
	return lambda: file_path.run_id() 


tmp_path = "/tmp"

@pytest.mark.parametrize(('prefix','ftype'),[
        ('test0001','seed'),
        ('test0002','seed'),
        ('test0003','seed'),
        ('test0004','seed'),
        ('test0001','seed'),
        ('test0002','seed'),
        ('test0003','seed'),
        ('test0004','seed'),
])
def test_create_dir_name(set_run_id, prefix, ftype):
	#run_id = set_run_id()
	os.chdir(tmp_path)
	os.path.exists(file_paths.create_dir_name(set_run_id, prefix, ftype))	


@pytest.mark.parametrize(('prefix','index','ftype','extension','create_directory'),[
        ('test0001','0','seed','csv', True),
        ('test0002','0','seed','parquet', True),
        ('test0003','0','seed','csv', False),
        ('test0004','0','seed','parquet', False),
        ('test0001','1','seed','csv', True),
        ('test0002','1','seed','parquet', True),
        ('test0003','1','seed','csv', False),
        ('test0004','1','seed','parquet', False),
])
def test_create_file_name(set_run_id, prefix, index, ftype, extension, create_directory):
	os.chdir(tmp_path)
	os.path.isfile(file_paths.create_file_name(set_run_id, prefix, int(index), ftype, extension, create_directory))	


@pytest.mark.parametrize(('prefix','index','ftype','create_directory'),[
        ('test0001','0','seed', True),
        ('test0002','0','seed', True),
        ('test0003','0','seed', False),
        ('test0004','0','seed', False),
        ('test0001','1','seed', True),
        ('test0002','1','seed', True),
        ('test0003','1','seed', False),
        ('test0004','1','seed', False),
])
def test_create_file_name_without_extension(set_run_id, prefix, index, ftype, create_directory):
	os.chdir(tmp_path)
	os.path.isfile(file_paths.create_file_name_without_extension(set_run_id, prefix, int(index), ftype, create_directory))	

