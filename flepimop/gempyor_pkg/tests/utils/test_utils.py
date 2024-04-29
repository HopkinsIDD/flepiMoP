import pytest
import os
import pandas as pd

# import dask.dataframe as dd
import pyarrow as pa
import time
from typing import List
from unittest.mock import patch
from gempyor import utils

DATA_DIR = os.path.dirname(__file__) + "/data"
tmp_path = "/tmp"


@pytest.mark.parametrize(
    ("fname", "extension"),
    [
        ("mobility", "csv"),
        ("usa-geoid-params-output", "parquet"),
    ],
)
def test_read_df_and_write_success(fname, extension):
    os.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.chdir("data")
    df1 = utils.read_df(fname=f"{DATA_DIR}/" + fname, extension=extension)
    if extension == "csv":
        df2 = pd.read_csv(f"{DATA_DIR}/" + fname + "." + extension)
        assert df2.equals(df1)
        utils.write_df(tmp_path + "/data/" + fname, df2, extension=extension)
        assert os.path.isfile(tmp_path + "/data/" + fname + "." + extension)
    elif extension == "parquet":
        df2 = pa.parquet.read_table(f"{DATA_DIR}/" + fname + "." + extension).to_pandas()
        assert df2.equals(df1)
        utils.write_df(tmp_path + "/data/" + fname, df2, extension=extension)
        assert os.path.isfile(tmp_path + "/data/" + fname + "." + extension)


@pytest.mark.parametrize(("fname", "extension"), [("mobility", "csv"), ("usa-geoid-params-output", "parquet")])
def test_read_df_and_write_fail(fname, extension):
    with pytest.raises(NotImplementedError, match=r".*Invalid.*extension.*Must.*"):
        os.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)
        os.chdir("data")
        df1 = utils.read_df(fname=f"{DATA_DIR}/" + fname, extension=extension)
        if extension == "csv":
            df2 = pd.read_csv(f"{DATA_DIR}/" + fname + "." + extension)
            assert df2.equals(df1)
            utils.write_df(tmp_path + "/data/" + fname, df2, extension="")
        elif extension == "parquet":
            df2 = pa.parquet.read_table(f"{DATA_DIR}/" + fname + "." + extension).to_pandas()
            assert df2.equals(df1)
            utils.write_df(tmp_path + "/data/" + fname, df2, extension="")


@pytest.mark.parametrize(("fname", "extension"), [("mobility", "")])
def test_read_df_fail(fname, extension):
    with pytest.raises(NotImplementedError, match=r".*Invalid.*extension.*"):
        os.chdir(tmp_path)
        utils.read_df(fname=f"{DATA_DIR}/" + fname, extension=extension)


def test_Timer_with_statement_success():
    with utils.Timer(name="test") as t:
        time.sleep(1)


def test_print_disk_diagnosis_success():
    utils.print_disk_diagnosis()


def test_profile_success():
    utils.profile()
    utils.profile(output_file="test")


def test_ISO8601Date_success():
    t = utils.ISO8601Date("2020-02-01")
    # dt = datetime.datetime.strptime("2020-02-01", "%Y-%m-%d")

    # assert t == datetime.datetime("2020-02-01").strftime("%Y-%m-%d")


def test_get_truncated_normal_success():
    utils.get_truncated_normal(mean=0, sd=1, a=-2, b=2)


def test_get_log_normal_success():
    utils.get_log_normal(meanlog=0, sdlog=1)


@pytest.fixture
def env_vars(monkeypatch):
    # Setting environment variables for the test
    monkeypatch.setenv("RESUME_RUN_INDEX", "321")
    monkeypatch.setenv("FLEPI_PREFIX", "output")
    monkeypatch.setenv("FLEPI_SLOT_INDEX", "2")
    monkeypatch.setenv("FLEPI_BLOCK_INDEX", "2")
    monkeypatch.setenv("FLEPI_RUN_INDEX", "123")


def test_create_resume_out_filename(env_vars):
    result = utils.create_resume_out_filename("spar", "global")
    expected_filename = "model_output/output/123/spar/global/intermidate/000000002.000000001.000000001.123.spar.parquet"
    assert result == expected_filename
    
    result2 = utils.create_resume_out_filename("seed", "chimeric")
    expected_filename2 = "model_output/output/123/seed/chimeric/intermidate/000000002.000000001.000000001.123.seed.csv"
    assert result2 == expected_filename2


def test_create_resume_input_filename(env_vars):

    result = utils.create_resume_input_filename("spar", "global")
    expect_filename = 'model_output/output/321/spar/global/final/000000002.321.spar.parquet' 

    assert result == expect_filename
    
    result2 = utils.create_resume_input_filename("seed", "chimeric")
    expect_filename2 = 'model_output/output/321/seed/chimeric/final/000000002.321.seed.csv'
    assert result2 == expect_filename2


@patch.dict(os.environ, {"RESUME_DISCARD_SEEDING": "true", "FLEPI_BLOCK_INDEX": "1"})
def test_get_parquet_types_resume_discard_seeding_true_flepi_block_index_1():
    expected_types = ["spar", "snpi", "hpar", "hnpi", "init"]
    assert utils.get_parquet_types() == expected_types


@patch.dict(os.environ, {"RESUME_DISCARD_SEEDING": "false", "FLEPI_BLOCK_INDEX": "1"})
def test_get_parquet_types_resume_discard_seeding_false_flepi_block_index_1():
    expected_types = ["seed", "spar", "snpi", "hpar", "hnpi", "init"]
    assert utils.get_parquet_types() == expected_types


@patch.dict(os.environ, {"FLEPI_BLOCK_INDEX": "2"})
def test_get_parquet_types_flepi_block_index_2():
    expected_types = ["seed", "spar", "snpi", "hpar", "hnpi", "host", "llik", "init"]
    assert utils.get_parquet_types() == expected_types