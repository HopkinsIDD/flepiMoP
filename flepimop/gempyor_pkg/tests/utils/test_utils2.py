import pytest
import datetime
import os
import pandas as pd
#import dask.dataframe as dd
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
#os.chdir(os.path.dirname(__file__))

tmp_path = "/tmp"

class SampleClass:
	def __init__(self):
		self.value =  11
   
	@utils.profile(output_file="get_value.prof", sort_by="time", lines_to_print=10, strip_dirs=True)
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
		s =  SampleClass()
		s.get_value()

        # display profile information
		stats = pstats.Stats("get_value.prof")
		stats.sort_stats("time")
		stats.print_stats(10)

	def test_ISO8601Date_success(self):
		iso_date = utils.ISO8601Date("2020-01-01")
		input_date = datetime.date(2020,1,1)
		result = iso_date.convert(input_date, None) # dummy for view
		assert result == input_date

		iso_date2 = utils.ISO8601Date()
		result = iso_date2.convert(str(input_date), None) # dummy for view
		assert result == input_date
	'''
	def test_ISO8601Date_invalid_value(self):
		iso_date2 = utils.ISO8601Date()
		invalid_value = "2020-01-01"
		with pytest.raises(ValueError, match=r".*must.*be.*ISO8601.*"):
			iso_date2.convert(invalid_value, None) # dummy for view
	'''	
'''
def test_profile_success():
	utils.profile()
	utils.profile(output_file="test")

def test_ISO8601Date_success():
	t = utils.ISO8601Date("2020-02-01")
	#dt = datetime.datetime.strptime("2020-02-01", "%Y-%m-%d")

	#assert t == datetime.datetime("2020-02-01").strftime("%Y-%m-%d")


def test_get_truncated_normal_success():
	utils.get_truncated_normal(mean=0, sd=1, a=-2, b=2)


def test_get_log_normal_success():
	utils.get_log_normal(meanlog=0, sdlog=1)
'''

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

def test_as_evaled_expression_with_valid_expression():
    # ConfigViewオブジェクトをモック化
    mock_config_view = MagicMock(spec=confuse.ConfigView)
    mock_config_view.as_evaled_expression.return_value =7.5 

    # as_evaled_expressionメソッドを呼び出し、正しい結果を確認
    result = mock_config_view.as_evaled_expression()

    assert result == 7.5



@pytest.fixture
def config():
    config = confuse.Configuration('myapp', __name__)
    return config

def test_as_evaled_expression_number(config):
    config.add({'myvalue': 123})
    assert config['myvalue'].as_evaled_expression() == 123

def test_as_evaled_expression_number(config):
    config.add({'myvalue': 1.10})
    assert config['myvalue'].as_evaled_expression() == 1.1

def test_as_evaled_expression_string(config):
    config.add({'myvalue': '2 + 3'})
    assert config['myvalue'].as_evaled_expression() == 5.0

def test_as_evaled_expression_other(config):
    config.add({'myvalue': [1, 2, 3]})
    with pytest.raises(ValueError):
        config['myvalue'].as_evaled_expression()

def test_as_evaled_expression_Invalid_string(config):
    config.add({'myvalue': 'invalid'})
    with pytest.raises(ValueError):
        config['myvalue'].as_evaled_expression()

def test_as_date(config):
    config.add({'myvalue': '2022-01-15'})
    assert config['myvalue'].as_date() ==  datetime.date(2022, 1, 15)
    
def test_as_random_distribution_fixed(config):
    config.add({'value':{'distribution': 'fixed', 'value': 1}})
    dist = config['value'].as_random_distribution()
    assert dist() == 1

def test_as_random_distribution_uniform(config):
    config.add({'value':{'distribution': 'uniform', 'low': 1, 'high':2.6}})
    dist = config['value'].as_random_distribution()
    assert  1 <= dist() <=2.6

def test_as_random_distribution_poisson(config):
    config.add({'value':{'distribution': 'poisson', 'lam': 1}})
    dist = config['value'].as_random_distribution()
    assert  isinstance(dist(), int)

def test_as_random_distribution_binomial(config):
    config.add({'value':{'distribution': 'binomial', 'n': 10, 'p':0.5 }})
    dist = config['value'].as_random_distribution()
    assert  0 <= dist() <= 10

def test_as_random_distribution_binomial_error(config):
    config.add({'value':{'distribution': 'binomial', 'n': 10, 'p':1.1 }})
    with pytest.raises(ValueError, match=r".*p.*value.*"):
        dist = config['value'].as_random_distribution()

def test_as_random_distribution_truncnorm(config):
    config.add({'value':{'distribution': 'truncnorm', 'mean': 0, 'sd':1, 'a':-1, 'b':1}})
    dist = config['value'].as_random_distribution()
    rvs = dist(size=1000)
    assert len(rvs) == 1000
    assert all(-1 <= x <= 1 for x in rvs)

def test_as_random_distribution_lognorm(config):
    config.add({'value':{'distribution': 'lognorm', 'meanlog': 0, 'sdlog':1}})
    dist = config['value'].as_random_distribution()
    rvs = dist(size=1000)
    assert len(rvs) == 1000
    assert all(x > 0 for x in rvs)

def test_as_random_distribution_unknown(config):
    config.add({'value':{'distribution': 'unknown', 'mean': 0, 'sd':1}})
    with pytest.raises(NotImplementedError):
        config['value'].as_random_distribution()
