
import yaml
import pathlib
from typing import Callable, Any

import pytest

from gempyor.shared_cli import parse_config_files

from gempyor.testing import *

def config_file(
    tmp_path: pathlib.Path,
    config_dict: dict[str, Any],
    filename: str = "config.yaml",
) -> pathlib.Path:
    config_file = tmp_path / filename
    with open(config_file, "w") as f:
        f.write(create_confuse_config_from_dict(config_dict).dump())
    return config_file

# collection of bad option values
def bad_opt_args(opt : str) -> Any:
    return {
        "write_csv": "foo", "write_parquet": "bar", "first_sim_index": -1,
        "config_filepath": -1, "seir_modifiers_scenario": 1, "outcome_modifiers_scenario": 1,
        "jobs": -1, "nslots": -1, "stoch_traj_flag": "foo", "in_run_id": 42, "out_run_id": 42,
        "in_prefix": 42,
    }[opt]

# collection of good option values
def good_opt_args(opt : str) -> Any:
    return {
        "write_csv": False, "write_parquet": False, "first_sim_index": 42,
        "config_filepath": "foo", "seir_modifiers_scenario": "example", "outcome_modifiers_scenario": "example",
        "jobs": 10, "nslots": 10, "stoch_traj_flag": False, "in_run_id": "foo", "out_run_id": "foo",
        "in_prefix": "foo",
    }[opt]

# collection of good configuration entries, to be overwritten by good_opt_args
def ref_cfg_kvs(opt : str) -> Any:
    return { opt: {
        "write_csv": True, "write_parquet": True, "first_sim_index": 1,
        "config_filepath": "notfoo", "seir_modifiers_scenario": ["example", "ibid"], "outcome_modifiers_scenario": ["example", "ibid"],
        "jobs": 1, "nslots": 1, "stoch_traj_flag": True, "in_run_id": "bar", "out_run_id": "bar",
        "in_prefix": "bar",
    }[opt]}

class TestParseConfigFiles:
    
    def test_deprecated_config(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that a -c config file work."""
        testdict = {"foo": "bar", "test": 123 }
        tmpconfigfile = config_file(tmp_path, testdict)
        mockconfig = mock_empty_config()
        parse_config_files(mockconfig, config_filepath=tmpconfigfile)
        for k, v in testdict.items():
           assert mockconfig[k].get(v) == v
        assert mockconfig["config_src"].as_str_seq() == [str(tmpconfigfile)]

    def test_preferred_config(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that a -c config file work."""
        testdict = {"foo": "bar", "test": 123 }
        tmpconfigfile = config_file(tmp_path, testdict)
        mockconfig = mock_empty_config()
        parse_config_files(mockconfig, config_files=tmpconfigfile)
        for k, v in testdict.items():
           assert mockconfig[k].get(v) == v
        assert mockconfig["config_src"].as_str_seq() == [str(tmpconfigfile)]

    def test_conflict_config_opts_error(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that both -c and argument style config file raise an error."""
        testdict = {"foo": "bar", "test": 123 }
        tmpconfigfile = config_file(tmp_path, testdict)
        mockconfig = mock_empty_config()
        with pytest.raises(ValueError):
            parse_config_files(mockconfig, config_filepath=tmpconfigfile, config_files=tmpconfigfile)

    def test_multifile_config(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that multiple config files are merged."""
        testdict1 = {"foo": "bar", "test": 123 }
        testdict2 = {"bar": "baz" }
        tmpconfigfile1 = config_file(tmp_path, testdict1, "config1.yaml")
        tmpconfigfile2 = config_file(tmp_path, testdict2, "config2.yaml")
        mockconfig = mock_empty_config()
        parse_config_files(mockconfig, config_files=[tmpconfigfile1, tmpconfigfile2])
        for k, v in (testdict1 | testdict2).items():
           assert mockconfig[k].get(v) == v
        assert mockconfig["config_src"].as_str_seq() == [str(tmpconfigfile1), str(tmpconfigfile2)]

    # for all the options:
    # - test the default
    # - test the envvar
    # - test invalid values => error
    # - test valid values => present in config
    # - test override: if not present in config => assigned
    # - test override: if present in config => not default (i.e. actually provided) overridden
    # - test override: if present in config => default => not overridden
