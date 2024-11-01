import os

import pytest

from gempyor.info import Cluster, get_cluster_info


@pytest.mark.parametrize("name", ("longleaf", "rockfish"))
@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
def test_output_validation(name: str) -> None:
    cluster = get_cluster_info(name)
    assert isinstance(cluster, Cluster)
