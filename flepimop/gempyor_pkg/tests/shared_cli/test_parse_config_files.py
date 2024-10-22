
import pathlib
from typing import Any

import pytest
import click

from gempyor.shared_cli import parse_config_files, config_file_options
from gempyor.testing import *


def config_file(
    tmp_path: pathlib.Path,
    config_dict: dict[str, Any] = {},
    filename: str = "config.yaml",
) -> pathlib.Path:
    config_file = tmp_path / filename
    with open(config_file, "w") as f:
        f.write(create_confuse_config_from_dict(config_dict).dump())
    return config_file


other_single_opt_args = [
    "write_csv",
    "write_parquet",
    "first_sim_index",
    "jobs",
    "nslots",
    "stoch_traj_flag",
    "in_run_id",
    "out_run_id",
    "in_prefix",
]


# collection of bad option values
def bad_opt_args(opt: str) -> dict[str, Any]:
    return {
        opt: {
            "write_csv": "foo",
            "write_parquet": "bar",
            "first_sim_index": -1,
            "seir_modifiers_scenario": 1,
            "outcome_modifiers_scenario": 1,
            "jobs": -1,
            "nslots": -1,
            "stoch_traj_flag": "foo",
            "in_run_id": -1,
            "out_run_id": -1,
            "in_prefix": [42],
        }[opt]
    }


# collection of good option values
def good_opt_args(opt: str) -> dict[str, Any]:
    return {
        opt: {
            "write_csv": False,
            "write_parquet": False,
            "first_sim_index": 42,
            "seir_modifiers_scenario": "example",
            "outcome_modifiers_scenario": "example",
            "jobs": 10,
            "nslots": 10,
            "stoch_traj_flag": False,
            "in_run_id": "foo",
            "out_run_id": "foo",
            "in_prefix": "foo",
        }[opt]
    }


# collection of good configuration entries, to be overwritten by good_opt_args
def ref_cfg_kvs(opt: str) -> dict[str, Any]:
    return {
        opt: {
            "write_csv": True,
            "write_parquet": True,
            "first_sim_index": 1,
            "seir_modifiers_scenario": ["example", "ibid"],
            "outcome_modifiers_scenario": ["example", "ibid"],
            "jobs": 1,
            "nslots": 1,
            "stoch_traj_flag": True,
            "in_run_id": "bar",
            "out_run_id": "bar",
            "in_prefix": "bar",
        }[opt]
    }


class TestParseConfigFiles:

    def test_deprecated_config(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that a -c config file work."""
        testdict = {"foo": "bar", "test": 123}
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
        testdict = {"foo": "bar", "test": 123}
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
        testdict = {"foo": "bar", "test": 123}
        tmpconfigfile = config_file(tmp_path, testdict)
        mockconfig = mock_empty_config()
        with pytest.raises(ValueError):
            parse_config_files(
                mockconfig, config_filepath=tmpconfigfile, config_files=tmpconfigfile
            )

    def test_multifile_config(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that multiple config files are merged."""
        testdict1 = {"foo": "bar", "test": 123}
        testdict2 = {"bar": "baz"}
        tmpconfigfile1 = config_file(tmp_path, testdict1, "config1.yaml")
        tmpconfigfile2 = config_file(tmp_path, testdict2, "config2.yaml")
        mockconfig = mock_empty_config()
        parse_config_files(mockconfig, config_files=[tmpconfigfile1, tmpconfigfile2])
        for k, v in (testdict1 | testdict2).items():
            assert mockconfig[k].get(v) == v
        assert mockconfig["config_src"].as_str_seq() == [
            str(tmpconfigfile1),
            str(tmpconfigfile2),
        ]

    @pytest.mark.parametrize("opt", [(k) for k in other_single_opt_args])
    def test_other_opts(self, tmp_path: pathlib.Path, opt: str) -> None:
        """for the non-scenario modifier parameters, test default, envvar, invalid values, valid values, override"""

        goodopt = good_opt_args(opt)
        badopt = bad_opt_args(opt)
        refopt = ref_cfg_kvs(opt)

        # the config file, with option set
        tmpconfigfile_wi_ref = config_file(tmp_path, refopt, "withref.yaml")
        # the config, without option set
        tmpconfigfile_wo_ref = config_file(tmp_path, filename="noref.yaml")
        mockconfig = mock_empty_config()

        for cfg in [tmpconfigfile_wi_ref, tmpconfigfile_wo_ref]:
            # both versions error on bad values
            with pytest.raises(click.exceptions.BadParameter):
                parse_config_files(mockconfig, config_files=cfg, **badopt)
            # when supplied an override, both should have the override
            parse_config_files(mockconfig, config_files=cfg, **goodopt)
            for k, v in goodopt.items():
                assert mockconfig[k].get(v) == v
            mockconfig.clear()

        # the config file with the option set should override the default
        parse_config_files(mockconfig, config_files=tmpconfigfile_wi_ref)
        for k, v in refopt.items():
            assert mockconfig[k].get(v) == v
        mockconfig.clear()

        # the config file without the option set should adopt the default
        parse_config_files(mockconfig, config_files=tmpconfigfile_wo_ref)
        defopt = config_file_options[opt].default
        if defopt is not None:
            assert mockconfig[opt].get(defopt) == defopt
