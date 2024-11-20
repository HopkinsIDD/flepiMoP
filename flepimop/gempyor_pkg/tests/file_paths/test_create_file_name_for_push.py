from unittest.mock import patch
from gempyor.file_paths import create_file_name_for_push


class TestCreateFileNameForPush:
    # Mock implementation of create_file_name for testing
    def mocked_create_file_name(
        self,
        run_id,
        prefix,
        inference_filename_prefix,
        inference_filepath_suffix,
        index,
        ftype,
        extension,
    ):
        return f"{prefix}_{run_id}_{inference_filename_prefix}_{inference_filepath_suffix}_{index}_{ftype}.{extension}"

    # Test method for create_file_name_for_push
    @patch("gempyor.file_paths.create_file_name")
    def test_create_file_name_for_push(self, mock_create_file_name):
        mock_create_file_name.side_effect = self.mocked_create_file_name

        flepi_run_index = "run123"
        prefix = "testprefix"
        flepi_slot_index = "42"
        flepi_block_index = "3"

        expected_file_names = [
            f"testprefix_run123_000000042._chimeric/intermediate_3_seir.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_hosp.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_llik.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_spar.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_snpi.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_hnpi.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_hpar.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_init.parquet",
            f"testprefix_run123_000000042._chimeric/intermediate_3_seed.csv",
        ]

        result = create_file_name_for_push(
            flepi_run_index=flepi_run_index,
            prefix=prefix,
            flepi_slot_index=flepi_slot_index,
            flepi_block_index=flepi_block_index,
        )

        # Assert the result matches the expected file names
        assert result == expected_file_names

        # Assert that create_file_name was called the expected number of times
        assert mock_create_file_name.call_count == 9
