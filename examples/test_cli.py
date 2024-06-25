
from click.testing import CliRunner
from gempyor.simulate import simulate
from gempyor.utils import command_safe_run
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

def test_Rscript_inference_main():
  # test the same as above using an R script, instead of CliRunner use a command line
  os.chdir(os.path.dirname(__file__) + "/tutorial_two_subpops")

  returncode, stdout, stderr = command_safe_run("Rscript ../../flepimop/main_scripts/inference_main.R -c config_sample_2pop_inference.yml -n 1",
                                                command_name="testing Rscript",
                                                fail_on_fail=False)
  assert returncode == 0
  assert os.path.isfile('model_output/sample_2pop_inference_all/20240621_164508EDT/hosp/global/intermediate/000000001.000000001.000000000.20240621_164508EDT.hosp.parquet')
  assert 'Successfully' in stdout
