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
    ("fqdn", "raise_error", "expected"),
    (
        ("login01.cm.cluster", True, "rockfish"),
        ("login01.cm.cluster", False, "rockfish"),
        ("login3.cm.cluster", True, "rockfish"),
        ("login3.cm.cluster", False, "rockfish"),
        ("longleaf-login1.its.unc.edu", True, "longleaf"),
        ("longleaf-login1.its.unc.edu", False, "longleaf"),
        ("longleaf-login07.its.unc.edu", True, "longleaf"),
        ("longleaf-login07.its.unc.edu", False, "longleaf"),
        ("longleaf-login07.its.unc.edu", True, "longleaf"),
        ("longleaf-login07.its.unc.edu", False, "longleaf"),
        ("longleaf-login.its.unc.edu", True, "longleaf"),
        ("longleaf-login.its.unc.edu", False, "longleaf"),
        ("longleaf.its.unc.edu", True, "longleaf"),
        ("longleaf.its.unc.edu", False, "longleaf"),
        ("epid-iss-MacBook-Pro.local", False, None),
    ),
)
def test_exact_results_for_select_values(
    fqdn: str, raise_error: bool, expected: str
) -> None:
    def socket_fqdn_wraps() -> str:
        return fqdn

    with patch("gempyor.info.getfqdn", wraps=socket_fqdn_wraps) as socket_fqdn_patch:
        assert _infer_cluster_from_fqdn(raise_error=raise_error) == expected
        socket_fqdn_patch.assert_called_once()
