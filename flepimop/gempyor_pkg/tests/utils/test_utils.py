import pytest
import os
import pandas as pd
import pyarrow as pa
import time
from gempyor import utils

DATA_DIR = os.path.dirname(__file__) + "/data"
tmp_path = "/tmp"


@pytest.mark.parametrize(
    ("fname", "extension"), [("mobility", "csv"), ("usa-geoid-params-output", "parquet"),],
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


def test_create_resume_out_filename():
    result = utils.create_resume_out_filename(
        flepi_run_index="123",
        flepi_prefix="output",
        flepi_slot_index="2",
        flepi_block_index="2",
        filetype="spar",
        liketype="global",
    )
    expected_filename = (
        "model_output/output/123/spar/global/intermediate/000000002.000000001.000000001.123.spar.parquet"
    )
    assert result == expected_filename

    result2 = utils.create_resume_out_filename(
        flepi_run_index="123",
        flepi_prefix="output",
        flepi_slot_index="2",
        flepi_block_index="2",
        filetype="seed",
        liketype="chimeric",
    )
    expected_filename2 = "model_output/output/123/seed/chimeric/intermediate/000000002.000000001.000000001.123.seed.csv"
    assert result2 == expected_filename2


def test_create_resume_input_filename():

    result = utils.create_resume_input_filename(
        flepi_slot_index="2", resume_run_index="321", flepi_prefix="output", filetype="spar", liketype="global"
    )
    expect_filename = "model_output/output/321/spar/global/final/000000002.321.spar.parquet"

    assert result == expect_filename

    result2 = utils.create_resume_input_filename(
        flepi_slot_index="2", resume_run_index="321", flepi_prefix="output", filetype="seed", liketype="chimeric"
    )
    expect_filename2 = "model_output/output/321/seed/chimeric/final/000000002.321.seed.csv"
    assert result2 == expect_filename2


def test_get_filetype_resume_discard_seeding_true_flepi_block_index_1():
    expected_types = ["spar", "snpi", "hpar", "hnpi", "init"]
    assert utils.get_filetype_for_resume(resume_discard_seeding="true", flepi_block_index="1") == expected_types


def test_get_filetype_resume_discard_seeding_false_flepi_block_index_1():
    expected_types = ["seed", "spar", "snpi", "hpar", "hnpi", "init"]
    assert utils.get_filetype_for_resume(resume_discard_seeding="false", flepi_block_index="1") == expected_types


def test_get_filetype_flepi_block_index_2():
    expected_types = ["seed", "spar", "snpi", "hpar", "hnpi", "host", "llik", "init"]
    assert utils.get_filetype_for_resume(resume_discard_seeding="false", flepi_block_index="2") == expected_types


def test_create_resume_file_names_map():
    name_map = utils.create_resume_file_names_map(
        resume_discard_seeding="false",
        flepi_block_index="2",
        resume_run_index="321",
        flepi_prefix="output",
        flepi_slot_index="2",
        flepi_run_index="123",
        last_job_output="s3://bucket",
    )
    for k in name_map:
        assert k.find("s3://bucket") >= 0
