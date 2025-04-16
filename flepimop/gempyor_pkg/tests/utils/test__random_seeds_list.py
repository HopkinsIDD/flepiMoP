"""Unit tests for `gempyor.utils._random_seeds_list`."""

import random

import pytest

from gempyor.utils import _random_seeds_list


@pytest.mark.parametrize(("a", "b"), [(4, 3), (8, 8), (38, 12)])
def test_lower_and_upper_bound_value_error(a: int, b: int) -> None:
    """Test that ValueError is raised when a > b."""
    assert not (a < b)
    with pytest.raises(
        ValueError, match=f"^The lower bound, {a}, must be less than the upper bound, {b}.$"
    ):
        _random_seeds_list(a, b, 1)


@pytest.mark.parametrize("n", [-25, -10, -1])
def test_n_negative_value_error(n: int) -> None:
    """Test that ValueError is raised when n is negative."""
    assert not (n >= 0)
    with pytest.raises(
        ValueError,
        match="^The number of random integers to generate must be non-negative.$",
    ):
        _random_seeds_list(1, 2, n)


@pytest.mark.parametrize(("a", "b", "n"), [(1, 5, 6), (1, 2, 3), (1, 10, 11)])
def test_n_not_large_enough_value_error(a: int, b: int, n: int) -> None:
    """Test that ValueError is raised when n is not large enough."""
    assert not (n > (b - a + 2))
    with pytest.raises(
        ValueError,
        match=rf"^Range \[{a}\, {b}\] is too small for {n} unique random integers\.$",
    ):
        _random_seeds_list(a, b, n)


@pytest.mark.parametrize(
    ("a", "b", "n"),
    [
        (1, 100, 20),
        (1, 500, 10),
        (1, 20, 0),
        (35, 50, 5),
        (100, 1000, 50),
        (1, 50, 50),
        (250, 500, 100),
    ],
)
def test_output_validation(a: int, b: int, n: int) -> None:
    random.seed(123)

    result1 = _random_seeds_list(a, b, n)
    assert isinstance(result1, list) and all(isinstance(i, int) for i in result1)
    assert len(result1) == len(set(result1)) == n
    assert all(a <= i <= b for i in result1)

    result2 = _random_seeds_list(a, b, n)
    assert isinstance(result2, list) and all(isinstance(i, int) for i in result2)
    assert len(result2) == len(set(result2)) == n
    assert all(a <= i <= b for i in result2)

    assert (result1 != result2) if n > 0 else (result1 == result2 == [])
