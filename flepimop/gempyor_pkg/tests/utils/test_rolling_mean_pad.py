import numpy as np
import numpy.typing as npt
import pytest

from gempyor.utils import rolling_mean_pad


class TestRollingMeanPad:
    """Unit tests for the `gempyor.utils.rolling_mean_pad` function."""

    # Test data for various matrix configurations
    test_data = {
        # 1x1 matrices
        "one_by_one_const": np.array([[1.0]]),
        "one_by_one_nan": np.array([[np.nan]]),
        "one_by_one_rand": np.random.uniform(size=(1, 1)),
        # 1xN matrices
        "one_by_many_const": np.arange(start=1.0, stop=6.0).reshape((1, 5)),
        "one_by_many_nan": np.repeat(np.nan, 5).reshape((1, 5)),
        "one_by_many_rand": np.random.uniform(size=(1, 5)),
        # Mx1 matrices
        "many_by_one_const": np.arange(start=3.0, stop=9.0).reshape((6, 1)),
        "many_by_one_nan": np.repeat(np.nan, 6).reshape((6, 1)),
        "many_by_one_rand": np.random.uniform(size=(6, 1)),
        # MxN matrices
        "many_by_many_const": np.arange(start=1.0, stop=49.0).reshape((12, 4)),
        "many_by_many_nan": np.repeat(np.nan, 48).reshape((12, 4)),
        "many_by_many_rand": np.random.uniform(size=(12, 4)),
    }

    @pytest.mark.parametrize(
        "test_data_name,expected_shape,window,put_nans",
        [
            # 1x1 matrices
            ("one_by_one_const", (1, 1), 3, []),
            ("one_by_one_const", (1, 1), 4, []),
            ("one_by_one_nan", (1, 1), 3, []),
            ("one_by_one_nan", (1, 1), 4, []),
            ("one_by_one_rand", (1, 1), 3, []),
            ("one_by_one_rand", (1, 1), 4, []),
            ("one_by_one_rand", (1, 1), 5, []),
            ("one_by_one_rand", (1, 1), 6, []),
            # 1xN matrices
            ("one_by_many_const", (1, 5), 3, []),
            ("one_by_many_const", (1, 5), 4, []),
            ("one_by_many_nan", (1, 5), 3, []),
            ("one_by_many_nan", (1, 5), 4, []),
            ("one_by_many_rand", (1, 5), 3, []),
            ("one_by_many_rand", (1, 5), 4, []),
            ("one_by_many_rand", (1, 5), 5, []),
            ("one_by_many_rand", (1, 5), 6, []),
            # Mx1 matrices
            ("many_by_one_const", (6, 1), 3, []),
            ("many_by_one_const", (6, 1), 4, []),
            ("many_by_one_nan", (6, 1), 3, []),
            ("many_by_one_nan", (6, 1), 4, []),
            ("many_by_one_rand", (6, 1), 3, []),
            ("many_by_one_rand", (6, 1), 4, []),
            ("many_by_one_rand", (6, 1), 5, []),
            ("many_by_one_rand", (6, 1), 6, []),
            # MxN matrices
            ("many_by_many_const", (12, 4), 3, []),
            ("many_by_many_const", (12, 4), 4, []),
            ("many_by_many_const", (12, 4), 5, []),
            ("many_by_many_const", (12, 4), 6, []),
            ("many_by_many_nan", (12, 4), 3, []),
            ("many_by_many_nan", (12, 4), 4, []),
            ("many_by_many_nan", (12, 4), 5, []),
            ("many_by_many_nan", (12, 4), 6, []),
            ("many_by_many_rand", (12, 4), 3, []),
            ("many_by_many_rand", (12, 4), 4, []),
            ("many_by_many_rand", (12, 4), 5, []),
            ("many_by_many_rand", (12, 4), 6, []),
            ("many_by_many_rand", (12, 4), 7, []),
            ("many_by_many_rand", (12, 4), 8, []),
            ("many_by_many_rand", (12, 4), 9, []),
            ("many_by_many_rand", (12, 4), 10, []),
            ("many_by_many_rand", (12, 4), 11, []),
            ("many_by_many_rand", (12, 4), 12, []),
            ("many_by_many_rand", (12, 4), 13, []),
            ("many_by_many_rand", (12, 4), 14, []),
            ("many_by_many_rand", (12, 4), 3, [(2, 2), (4, 4)]),
            ("many_by_many_rand", (12, 4), 4, [(2, 2), (4, 4)]),
            ("many_by_many_rand", (12, 4), 5, [(2, 2), (4, 4)]),
            ("many_by_many_rand", (12, 4), 6, [(2, 2), (4, 4)]),
            ("many_by_many_rand", (12, 4), 7, [(2, 2), (4, 4)]),
            ("many_by_many_rand", (12, 4), 8, [(2, 2), (4, 4)]),
        ],
    )
    def test_rolling_mean_pad(
        self,
        test_data_name: str,
        expected_shape: tuple[int, int],
        window: int,
        put_nans: list[tuple[int, int]],
    ) -> None:
        """Tests `rolling_mean_pad` function with various configurations of input data.

        Args:
            test_data_name: The name of the test data set to use.
            expected_shape: The expected shape of the output array.
            window: The size of the rolling window.
            put_nans: A list of indices to insert NaNs into the input data.

        Raises:
            AssertionError: If the shape or contents of the output do not match the
                expected values.
        """
        test_data = self.test_data.get(test_data_name).copy()
        if put_nans:
            np.put(test_data, put_nans, np.nan)
        rolling_mean_data = rolling_mean_pad(test_data, window)
        rolling_mean_reference = self._rolling_mean_pad_reference(test_data, window)
        assert rolling_mean_data.shape == expected_shape
        assert np.isclose(
            rolling_mean_data, rolling_mean_reference, equal_nan=True
        ).all()

    def _rolling_mean_pad_reference(
        self, data: npt.NDArray[np.number], window: int
    ) -> npt.NDArray[np.number]:
        """Generates a reference rolling mean with padding.

        This implementation should match the `gempyor.utils.rolling_mean_pad`
        implementation, but is written for readability. As a result this
        reference implementation is extremely slow.

        Args:
            data: The input array for which to compute the rolling mean.
            window: The size of the rolling window.

        Returns:
            An array of the same shape as `data` containing the rolling mean values.
        """
        # Setup
        rows, cols = data.shape
        output = np.zeros((rows, cols), dtype=data.dtype)
        # Slow but intuitive triple loop
        for i in range(rows):
            for j in range(cols):
                # If the last row on an even window, change the window to be one less,
                # so 4 -> 3, but 5 -> 5.
                sub_window = window - 1 if window % 2 == 0 and i == rows - 1 else window
                weight = 1.0 / sub_window
                for l in range(-((sub_window - 1) // 2), 1 + (sub_window // 2)):
                    i_star = min(max(i + l, 0), rows - 1)
                    output[i, j] += weight * data[i_star, j]
        # Done
        return output
