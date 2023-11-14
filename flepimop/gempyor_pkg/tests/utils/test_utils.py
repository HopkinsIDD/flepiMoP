import pytest
import datetime
import os
import pandas as pd

# import dask.dataframe as dd
import pyarrow as pa
import time

from gempyor import utils

DATA_DIR = os.path.dirname(__file__) + "/data"
# os.chdir(os.path.dirname(__file__))

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


def test_aws_disk_diagnosis_success():
    utils.aws_disk_diagnosis()


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
