import pytest

from gempyor.batch import EstimationSettings


@pytest.mark.parametrize(
    "factors",
    (("not_a_valid_factor",), ("not_a_valid_factor", "time"), ("memory", "invalid")),
)
def test_invalid_factors_value_error(factors: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="Factors must be derived from JobSize: .*"):
        EstimationSettings(
            runs=10,
            interval=0.9,
            vary=("blocks", "chains", "simulations"),
            factors=factors,
            measurements=("cpu", "memory", "time"),
            scale_upper=3.0,
            scale_lower=10.0,
        )


@pytest.mark.parametrize(
    ("scale_lower", "scale_upper"), ((3, 3), (2.3, 2.3), (10, 20), (2.3, 2.4))
)
def test_lower_and_upper_scale_value_error(
    scale_lower: float | int, scale_upper: float | int
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"The lower scale, {scale_lower}.*, must be greater "
            f"than the upper scale, {scale_upper}.*."
        ),
    ):
        EstimationSettings(
            runs=10,
            interval=0.9,
            vary=("blocks", "chains", "simulations"),
            factors=("total_simulations",),
            measurements=("cpu", "memory", "time"),
            scale_upper=scale_upper,
            scale_lower=scale_lower,
        )
