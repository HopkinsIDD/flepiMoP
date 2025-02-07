import os
from typing import Any

import pytest

from gempyor._jinja import _jinja_environment
from gempyor.info import Cluster, Module, PathExport, get_cluster_info


@pytest.mark.parametrize("cluster", (None, "longleaf", "rockfish"))
@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
def test_output_validation(cluster: str | None) -> None:
    cluster = cluster if cluster is None else get_cluster_info(cluster).model_dump()
    rendered_template = _jinja_environment.get_template("cluster_setup.bash.j2").render(
        {"cluster": cluster}
    )
    lines = rendered_template.split("\n")
    if cluster is not None:
        assert "module purge" in lines
        assert len(lines) > 2
    else:
        assert lines == [""]


@pytest.mark.parametrize(
    ("cluster", "expected"),
    (
        (
            Cluster(name="foobar").model_dump(),
            [
                "# Purge/load modules",
                "module purge",
            ],
        ),
        (
            Cluster(name="foo", modules=[Module(name="fizz")]).model_dump(),
            [
                "# Purge/load modules",
                "module purge",
                "module load fizz",
            ],
        ),
        (
            Cluster(
                name="bar", modules=[Module(name="buzz", version="12.34")]
            ).model_dump(),
            [
                "# Purge/load modules",
                "module purge",
                "module load buzz/12.34",
            ],
        ),
        (
            Cluster(
                name="fizz", path_exports=[PathExport(path="/path/to/custom/bin")]
            ).model_dump(),
            [
                "# Purge/load modules",
                "module purge",
                "",
                "# Path modifications",
                'if [ -r "/path/to/custom/bin" ]; then',
                "    export PATH=/path/to/custom/bin:$PATH",
                'elif [ "False" = "True" ]; then',
                "    echo \"The path '/path/to/custom/bin' does not exist but is required.\"",
                "    exit 1",
                "fi",
            ],
        ),
        (
            Cluster(
                name="new_cluster",
                modules=[Module(name="abc", version="123"), Module(name="def")],
                path_exports=[
                    PathExport(path="~/helpful/bin", prepend=False),
                    PathExport(path="~/required/bin", error_if_missing=True),
                ],
            ).model_dump(),
            [
                "# Purge/load modules",
                "module purge",
                "module load abc/123",
                "module load def",
                "",
                "# Path modifications",
                'if [ -r "~/helpful/bin" ]; then',
                "    export PATH=$PATH:~/helpful/bin",
                'elif [ "False" = "True" ]; then',
                "    echo \"The path '~/helpful/bin' does not exist but is required.\"",
                "    exit 1",
                "fi",
                'if [ -r "~/required/bin" ]; then',
                "    export PATH=~/required/bin:$PATH",
                'elif [ "True" = "True" ]; then',
                "    echo \"The path '~/required/bin' does not exist but is required.\"",
                "    exit 1",
                "fi",
            ],
        ),
    ),
)
def test_exact_results_for_select_inputs(
    cluster: dict[str, Any], expected: list[str]
) -> None:
    lines = (
        _jinja_environment.get_template("cluster_setup.bash.j2")
        .render({"cluster": cluster})
        .split("\n")
    )
    assert lines == expected
