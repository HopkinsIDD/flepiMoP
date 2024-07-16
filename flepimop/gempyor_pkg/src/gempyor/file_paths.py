"""
This module contains utilities for interacting with and generating file paths
in the context of this package, which saves its output to a very particular
directory structure.

Functions:
    - create_file_name: Creates a full file name with extension.
    - create_file_name_without_extension: Creates a file name without extension.
    - run_id: Generates a run ID based on the current or provided timestamp.
    - create_dir_name: Creates a directory name based on given parameters.
"""

from datetime import datetime
import os
from pathlib import Path


def create_file_name(
    run_id: str,
    prefix: str,
    index: str | int,
    ftype: str,
    extension: str,
    inference_filepath_suffix: str = "",
    inference_filename_prefix: str = "",
    create_directory: bool = True,
) -> str:
    """
    Generates a full file name with the given parameters and extension.

    Args:
        run_id: The unique identifier for the run.
        prefix: A prefix for the file path.
        index: An index to include in the file name.
        ftype: The type of file being created.
        extension: The file extension, without the leading period.
        inference_filepath_suffix: Suffix for the inference file path. Defaults to "".
        inference_filename_prefix: Prefix for the inference file name. Defaults to "".
        create_directory: Whether to create the parent directory if it doesn't exist.
            Defaults to True.

    Returns:
        The full file name with extension.

    Examples:
        >>> from gempyor.file_paths import create_file_name
        >>> create_file_name(
        ...     "20240101_000000",
        ...     "abc",
        ...     1,
        ...     "hosp",
        ...     "parquet",
        ...     "global",
        ...     "jkl",
        ...     create_directory=False,
        ... )
        PosixPath('model_output/abc/hosp/global/jkl000000001.20240101_000000.hosp.parquet')
    """
    if create_directory:
        os.makedirs(
            create_dir_name(
                run_id,
                prefix,
                ftype,
                inference_filepath_suffix,
                inference_filename_prefix,
            ),
            exist_ok=True,
        )

    fn_no_ext = create_file_name_without_extension(
        run_id,
        prefix,
        index,
        ftype,
        inference_filepath_suffix,
        inference_filename_prefix,
        create_directory=create_directory,
    )
    return f"{fn_no_ext}.%s" % (extension,)


def create_file_name_without_extension(
    run_id: str,
    prefix: str,
    index: str | int,
    ftype: str,
    inference_filepath_suffix: str,
    inference_filename_prefix: str,
    create_directory: bool = True,
) -> Path:
    """
    Generates a file name without the extension.

    This function will return the file name to use, but does not actually create the
    file.

    Args:
        run_id: The unique identifier for the run.
        prefix: A prefix for the file path.
        index: An index to include in the file name.
        ftype: The type of file being created.
        inference_filepath_suffix: Suffix for the inference file path.
        inference_filename_prefix: Prefix for the inference file name.
        create_directory: Whether to create the file's parent directory if it doesn't
            exist. Defaults to True.

    Returns:
        The file name without extension as a Path object.

    Examples:
        >>> from gempyor.file_paths import create_file_name_without_extension
        >>> create_file_name_without_extension(
        ...     "20240101_000000",
        ...     "abc",
        ...     1,
        ...     "hosp",
        ...     "global",
        ...     "jkl",
        ...     create_directory=False,
        ... )
        PosixPath('model_output/abc/hosp/global/jkl000000001.20240101_000000.hosp')
    """
    if create_directory:
        os.makedirs(
            create_dir_name(
                run_id,
                prefix,
                ftype,
                inference_filepath_suffix,
                inference_filename_prefix,
            ),
            exist_ok=True,
        )
    filename = Path(
        "model_output",
        prefix,
        ftype,
        inference_filepath_suffix,
        f"{inference_filename_prefix}{index:>09}.{run_id}.{ftype}",
    )
    return filename


def run_id(timestamp: None | datetime = None) -> str:
    """
    Generates a run ID based on the current or provided timestamp.

    Args:
        timestamp: A specific timestamp to use. If `None` this function will use the
            current timestamp.

    Returns:
        The generated run ID.

    Examples:
        >>> from datetime import datetime, timezone
        >>> from gempyor.file_paths import run_id
        >>> run_id()
        '20240711_160059'
        >>> run_id(timestamp=datetime(2024, 1, 1))
        '20240101_000000'
        >>> run_id(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
        '20240101_000000UTC'
    """
    if not timestamp:
        timestamp = datetime.now()
    return datetime.strftime(timestamp, "%Y%m%d_%H%M%S%Z")


def create_dir_name(
    run_id: str,
    prefix: str,
    ftype: str,
    inference_filepath_suffix: str,
    inference_filename_prefix: str,
) -> str:
    """
    Generate a directory name based on the given parameters.

    This function will return the directory name to use, but does not actually create
    the directory.

    Args:
        run_id: The unique identifier for the run.
        prefix: A prefix for the file path.
        ftype: The type of file being created.
        inference_filepath_suffix: Suffix for the inference file path.
        inference_filename_prefix: Prefix for the inference file name.

    Returns:
        The directory name.

    Examples:
        >>> from gempyor.file_paths import create_dir_name
        >>> create_dir_name("20240101_000000", "abc", "hosp", "def", "jkl")
        'model_output/abc/hosp/def'
    """
    return os.path.dirname(
        create_file_name_without_extension(
            run_id,
            prefix,
            1,
            ftype,
            inference_filepath_suffix,
            inference_filename_prefix,
            create_directory=False,
        )
    )
