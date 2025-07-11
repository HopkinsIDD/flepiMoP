import pathlib
from typing import Any

import pytest
import click
import itertools

from gempyor.shared_cli import parse_config_files, config_file_options
from gempyor.testing import *


def config_file(
    tmp_path: pathlib.Path,
    config_dict: dict[str, Any] = {},
    filename: str = "config.yaml",
) -> pathlib.Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    "method",
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
            "method": "foo",
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
            "method": "euler",
            "in_run_id": "foo",
            "out_run_id": "foo",
            "in_prefix": "foo",
            "populations": "test",
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
            "method": "rk4",
            "in_run_id": "bar",
            "out_run_id": "bar",
            "in_prefix": "bar",
            "populations": ["reference"],
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

    @pytest.mark.parametrize(
        ("option_configs", "argument_configs"),
        itertools.combinations([["A"], ["B", "C"], ["D", "E", "F"], ["G", "H", "I"]], 2),
    )
    def test_conflict_config_via_arg_and_opts_error(
        self,
        tmp_path: pathlib.Path,
        option_configs: list[str],
        argument_configs: list[str],
    ) -> None:
        """
        Check that different configs passed through -c and argument-style config file raises an error.
        """
        mockconfig = mock_empty_config()
        option_configs = [
            config_file(tmp_path / f"config{c}", {"foo": 1}) for c in option_configs
        ]
        argument_configs = [
            config_file(tmp_path / f"config{c}", {"bar": 1}) for c in argument_configs
        ]
        assert option_configs != argument_configs
        with pytest.raises(
            ValueError, match=r"^Exactly one config file source option must be provided"
        ):
            parse_config_files(
                mockconfig, config_filepath=option_configs, config_files=argument_configs
            )

    @pytest.mark.parametrize(
        ("option_configs", "argument_configs"),
        [
            (["A"], ["A"]),
            (["X", "Y"], ["X", "Y"]),
            (["alpha", "beta", "gamma"], ["alpha", "beta", "gamma"]),
            (["only_opt"], None),
            (None, ["only_arg"]),
        ],
    )
    def test_resolve_same_config_given_via_arg_and_opts(
        self,
        tmp_path: pathlib.Path,
        option_configs: list[str] | None,
        argument_configs: list[str] | None,
    ) -> None:
        """
        Check that identical config paths passed through -c and argument-style config file doesn't raise an error.
        """
        mockconfig = mock_empty_config()
        if option_configs is not None:
            option_configs = [
                config_file(tmp_path / f"config_{c}", {f"shared_key_{c}": 1})
                for c in option_configs
            ]
        if argument_configs is not None:
            argument_configs = [
                config_file(tmp_path / f"config_{c}", {f"shared_key_{c}": 1})
                for c in argument_configs
            ]
        try:
            parse_config_files(
                mockconfig,
                config_filepath=option_configs,
                config_files=argument_configs,
            )
        except ValueError:
            pytest.fail(
                "shared_cli.parse_config_files() not resolving references to identical config paths."
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

    def test_multifile_config_collision(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Check that multiple config overlapping keys are warned."""
        testdict1 = {"foo": "notthis", "test": 123}
        testdict2 = {"foo": "this"}
        tmpconfigfile1 = config_file(tmp_path, testdict1, "config1.yaml")
        tmpconfigfile2 = config_file(tmp_path, testdict2, "config2.yaml")
        mockconfig = mock_empty_config()
        with pytest.raises(ValueError, match=r"foo"):
            parse_config_files(mockconfig, config_files=[tmpconfigfile1, tmpconfigfile2])
        for k, v in (testdict1 | testdict2).items():
            assert mockconfig[k].get(v) == v

    @pytest.mark.parametrize("opt", [(k) for k in other_single_opt_args])
    def test_other_opts(self, tmp_path: pathlib.Path, opt: str) -> None:
        """for the non-scenario modifier parameters, test default, envvar, invalid values, valid values, override"""

        goodopt = good_opt_args(opt)
        refopt = ref_cfg_kvs(opt)

        # the config file, with option set
        tmpconfigfile_wi_ref = config_file(tmp_path, refopt, "withref.yaml")
        # the config, without option set
        tmpconfigfile_wo_ref = config_file(tmp_path, filename="noref.yaml")
        mockconfig = mock_empty_config()

        for cfg in [tmpconfigfile_wi_ref, tmpconfigfile_wo_ref]:
            # both versions error on bad values
            # when supplied an override, both should have the override
            parse_config_files(mockconfig, config_files=cfg, **goodopt)
            for k, v in goodopt.items():
                if k == "method":
                    assert mockconfig["seir"]["integration"][k].get(v) == v
                else:
                    assert mockconfig[k].get(v) == v
            mockconfig.clear()

        # the config file with the option set should override the default
        parse_config_files(mockconfig, config_files=tmpconfigfile_wi_ref)
        for k, v in refopt.items():
            if k == "method":
                assert mockconfig["seir"]["integration"][k].get(v) == v
            elif k == "populations":
                assert mockconfig["subpop_setup"]["selected"].get(v) == v
            else:
                assert mockconfig[k].get(v) == v
        mockconfig.clear()

        # the config file without the option set should adopt the default
        parse_config_files(mockconfig, config_files=tmpconfigfile_wo_ref)
        defopt = config_file_options[opt].default
        if defopt is not None:
            assert mockconfig[opt].get(defopt) == defopt
