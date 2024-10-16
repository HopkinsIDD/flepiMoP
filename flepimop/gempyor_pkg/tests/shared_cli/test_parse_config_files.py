
import yaml
import pathlib
from typing import Callable, Any

import pytest

from gempyor.shared_cli import parse_config_files

class TestParseConfigFiles:
    
    def test_deprecated_config(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], Any],
    ) -> None:
        pass

    def test_preferred_config(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], Any],
    ) -> None:
        pass

    def test_conflict_config_opts_error(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], Any],
    ) -> None:
        pass

    def test_multifile_config(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], Any],
    ) -> None:
        pass

    # for all the options:
    # - test the default
    # - test the envvar
    # - test invalid values => error
    # - test valid values => present in config
    # - test override: if not present in config => assigned
    # - test override: if present in config => not default (i.e. actually provided) overridden
    # - test override: if present in config => default => not overridden
