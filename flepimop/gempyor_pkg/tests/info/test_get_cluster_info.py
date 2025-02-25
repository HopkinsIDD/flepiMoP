import os
from unittest.mock import patch

import pytest

from gempyor.info import Cluster, get_cluster_info


@pytest.mark.parametrize("name", ("longleaf", "rockfish"))
@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
def test_exact_results_given_cluster_name(name: str) -> None:
    cluster = get_cluster_info(name)
    assert isinstance(cluster, Cluster)


@pytest.mark.parametrize("name", ("longleaf", "rockfish"))
@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
def test_exact_results_when_inferred_from_fqdn(name: str) -> None:
    def infer_cluster_from_fqdn_wraps() -> str:
        return name

    with patch(
        "gempyor.info._infer_cluster_from_fqdn", wraps=infer_cluster_from_fqdn_wraps
    ) as infer_cluster_from_fqdn_patch:
        cluster = get_cluster_info(None)
        assert isinstance(cluster, Cluster)
        infer_cluster_from_fqdn_patch.assert_called_once()
