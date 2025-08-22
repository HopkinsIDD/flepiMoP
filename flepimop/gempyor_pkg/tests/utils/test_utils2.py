import pytest
import datetime
import os
import pandas as pd

# import dask.dataframe as dd
import numpy as np
from scipy.stats import rv_continuous
import pyarrow as pa
import cProfile
import pstats
import datetime
import confuse
from unittest.mock import MagicMock, patch

from gempyor import utils
from gempyor.utils import ISO8601Date

DATA_DIR = os.path.dirname(__file__) + "/data"
# os.chdir(os.path.dirname(__file__))

tmp_path = "/tmp"


class SampleClass:
    def __init__(self):
        self.value = 11

    @utils.profile(
        output_file="get_value.prof", sort_by="time", lines_to_print=10, strip_dirs=True
    )
    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


class Test_utils2:
    @utils.add_method(SampleClass)
    def get_a(self):
        return "a"

    def test_add_method(self):
        assert SampleClass.get_a(self) == "a"

    def test_get_value_w_profile(self):
        s = SampleClass()
        s.get_value()

        # display profile information
        stats = pstats.Stats("get_value.prof")
        stats.sort_stats("time")
        stats.print_stats(10)

    def test_ISO8601Date_success(self):
        iso_date = utils.ISO8601Date("2020-01-01")
        input_date = datetime.date(2020, 1, 1)
        result = iso_date.convert(input_date, None)  # dummy for view
        assert result == input_date

        iso_date2 = utils.ISO8601Date()
        result = iso_date2.convert(str(input_date), None)  # dummy for view
        assert result == input_date

    """
	def test_ISO8601Date_invalid_value(self):
		iso_date2 = utils.ISO8601Date()
		invalid_value = "2020-01-01"
		with pytest.raises(ValueError, match=r".*must.*be.*ISO8601.*"):
			iso_date2.convert(invalid_value, None) # dummy for view
	"""


"""
def test_profile_success():
	utils.profile()
	utils.profile(output_file="test")

def test_ISO8601Date_success():
	t = utils.ISO8601Date("2020-02-01")
	#dt = datetime.datetime.strptime("2020-02-01", "%Y-%m-%d")

	#assert t == datetime.datetime("2020-02-01").strftime("%Y-%m-%d")



"""


def test_as_date_with_valid_date_string():
    # created MockConfigView object
    mock_config_view = MagicMock(spec=confuse.ConfigView)

    # ConfigViewのgetメソッドをモックし、適切な日付文字列を返すように設定
    mock_config_view.get.return_value = "2022-01-15"

    # ISO8601Dateのconvertメソッドをモックし、適切な日付オブジェクトを返すように設定
    with patch.object(ISO8601Date, "convert", return_value=datetime.date(2022, 1, 15)):
        result = ISO8601Date().convert(mock_config_view.get(), None)

    # 正しい日付オブジェクトが返されることを確認
    assert result == datetime.date(2022, 1, 15)


@pytest.fixture
def config():
    config = confuse.Configuration("myapp", __name__)
    return config


def test_as_date(config):
    config.add({"myvalue": "2022-01-15"})
    assert config["myvalue"].as_date() == datetime.date(2022, 1, 15)
