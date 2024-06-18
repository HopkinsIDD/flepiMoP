
from click.testing import CliRunner
from gempyor.simulate import simulate
import os

# See here to test click application https://click.palletsprojects.com/en/8.1.x/testing/
# would be useful to also call the command directly

def test_config_sample_2pop():
  os.chdir(os.path.dirname(__file__) + "/tutorial_two_subpops")
  runner = CliRunner()
  result = runner.invoke(simulate, ['-c', 'config_sample_2pop.yml'])
  print(result.output) # useful for debug
  print(result.exit_code) # useful for debug
  print(result.exception) # useful for debug
  assert result.exit_code == 0
  assert 'completed in' in result.output


def test_sample_2pop_interventions_test():
  os.chdir(os.path.dirname(__file__) + "/tutorial_two_subpops")
  runner = CliRunner()
  result = runner.invoke(simulate, ['-c', 'config_sample_2pop_interventions_test.yml'])
  print(result.output) # useful for debug
  print(result.exit_code) # useful for debug
  print(result.exception) # useful for debug
  assert result.exit_code == 0
  assert 'completed in' in result.output


def test_simple_usa_statelevel():
  os.chdir(os.path.dirname(__file__) + "/simple_usa_statelevel")
  runner = CliRunner()
  result = runner.invoke(simulate, ['-c', 'simple_usa_statelevel.yml', '-n', '1'])
  print(result.output) # useful for debug
  print(result.exit_code) # useful for debug
  print(result.exception) # useful for debug
  assert result.exit_code == 0
  assert 'completed in' in result.output

def test_simple_usa_statelevel():
  os.chdir(os.path.dirname(__file__) + "/simple_usa_statelevel")
  runner = CliRunner()
  result = runner.invoke(simulate, ['-c', 'simple_usa_statelevel.yml', '-n', '1'])
  print(result.output) # useful for debug
  print(result.exit_code) # useful for debug
  print(result.exception) # useful for debug
  assert result.exit_code == 0
  assert 'completed in' in result.output