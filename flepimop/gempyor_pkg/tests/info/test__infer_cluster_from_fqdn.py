from unittest.mock import patch

import pytest

from gempyor.info import _infer_cluster_from_fqdn


@pytest.mark.parametrize("fqdn", ("new.cluster.com", "unsupported.cluster"))
def test_no_matching_fqdn_found_value_error(fqdn: str) -> None:
    def socket_fqdn_wraps() -> str:
        return fqdn

    with patch("gempyor.info.getfqdn", wraps=socket_fqdn_wraps) as socket_fqdn_patch:
        with pytest.raises(
            ValueError,
            match=f"^The fqdn, '{fqdn}', does not match any of the expected clusters.$",
        ):
            _infer_cluster_from_fqdn()
        socket_fqdn_patch.assert_called_once()


@pytest.mark.parametrize(
    ("fqdn", "expected"),
    (
        ("login01.cm.cluster", "rockfish"),
        ("login3.cm.cluster", "rockfish"),
        ("longleaf-login1.its.unc.edu", "longleaf"),
        ("longleaf-login07.its.unc.edu", "longleaf"),
    ),
)
def test_exact_results_for_select_values(fqdn: str, expected: str) -> None:
    def socket_fqdn_wraps() -> str:
        return fqdn

    with patch("gempyor.info.getfqdn", wraps=socket_fqdn_wraps) as socket_fqdn_patch:
        assert _infer_cluster_from_fqdn() == expected
        socket_fqdn_patch.assert_called_once()
