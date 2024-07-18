from pathlib import Path

import pytest

from gempyor.io import resolve_paths
from gempyor.testing import *


class TestResolvePaths:
    """
    Unit tests for the `gempyor.io.resolve_paths` function.
    """

    @pytest.mark.usefixtures("change_directory_to_temp_directory")
    @pytest.mark.parametrize(
        "paths,expected_paths",
        [
            ("abc", Path("abc")),
            (b"def", Path("def")),
            (Path("ghi"), Path("ghi")),
            (["abc", b"def", Path("ghi")], [Path("abc"), Path("def"), Path("ghi")]),
            ("/abc/def/ghi", Path("/abc/def/ghi")),
            (b"/abc/def/ghi", Path("/abc/def/ghi")),
            (Path("/abc/def/ghi"), Path("/abc/def/ghi")),
            (
                ["/path/one", b"/path/two", Path("/path/three")],
                [Path("/path/one"), Path("/path/two"), Path("/path/three")],
            ),
        ],
    )
    @pytest.mark.parametrize("resolve", [True, False])
    def test_resolve_paths(self, paths, expected_paths, resolve) -> None:
        paths = resolve_paths(paths, resolve=resolve)
        if resolve:
            if isinstance(expected_paths, list):
                assert paths == [p.resolve() for p in expected_paths]
            else:
                assert paths == expected_paths.resolve()
        else:
            assert paths == expected_paths
