from pathlib import Path
from typing import Literal

from pydantic import BaseModel
import pytest
import yaml

from gempyor.info import _get_info


class NameOnly(BaseModel):
    name: str


@pytest.fixture
def create_mock_info_directory(tmp_path: Path) -> Path:
    for file, contents in (
        ("abc/def.yml", {"name": "Foobar"}),
        ("abc/ghi.yml", {"name": "Fizzbuzz"}),
    ):
        file = tmp_path / "info" / file
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open(mode="w") as f:
            yaml.dump(contents, f)
    return tmp_path.absolute()


def test_file_does_not_exist_value_error(
    monkeypatch: pytest.MonkeyPatch, create_mock_info_directory: Path
) -> None:
    monkeypatch.setenv("FLEPI_PATH", str(create_mock_info_directory.parent))
    with pytest.raises(ValueError):
        _get_info("does_not", "exist", object, None)


@pytest.mark.parametrize(
    ("category", "name", "model"), (("abc", "def", NameOnly), ("abc", "ghi", NameOnly))
)
def test_output_validation_with_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    create_mock_info_directory: Path,
    category: str,
    name: str,
    model: type[BaseModel],
) -> None:
    monkeypatch.chdir(create_mock_info_directory)
    _output_validation_test(create_mock_info_directory, category, name, model)


@pytest.mark.parametrize(
    ("category", "name", "model"), (("abc", "def", NameOnly), ("abc", "ghi", NameOnly))
)
@pytest.mark.parametrize("envvar", ("FLEPI_INFO_PATH", "FLEPI_PATH"))
def test_output_validation_with_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    create_mock_info_directory: Path,
    category: str,
    name: str,
    model: type[BaseModel],
    envvar: Literal["FLEPI_INFO_PATH", "FLEPI_PATH"],
) -> None:
    monkeypatch.setenv(envvar, str(create_mock_info_directory))
    _output_validation_test(create_mock_info_directory, category, name, model)


def _output_validation_test(
    path: Path,
    category: str,
    name: str,
    model: type[BaseModel],
) -> None:
    results = []
    for path in (None, path, str(path)):
        info = _get_info(category, name, model, path)
        assert isinstance(info, model)
        results.append(info)
    for i in range(len(results) - 1):
        assert results[i] == results[i + 1]
