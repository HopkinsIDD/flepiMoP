#!/usr/bin/env python
"""
Script for fetching resume files based on various input parameters.

Overview:
This script is designed to fetch resume files based on various input parameters. 
It uses Click for command-line interface (CLI) options and handles the fetching process either by downloading from an S3 bucket or moving files locally.

Dependencies:
- click: A package for creating command-line interfaces.
- os: A module for interacting with the operating system.
- gempyor.utils: A module containing utility functions create_resume_file_names_map, download_file_from_s3, and move_file_at_local.

CLI Options:
The script uses Click to define the following command-line options:

--resume_location:
- Environment Variables: LAST_JOB_OUTPUT, RESUME_LOCATION
- Type: STRING
- Required: Yes
- Description: The path for the last run's output.

--discard_seeding:
- Environment Variable: RESUME_DISCARD_SEEDING
- Type: BOOL
- Required: Yes
- Description: Boolean value indicating whether to discard seeding or not.
- valid values: true, 1, y, yes, True, False, false, f, 0, no, n

--block_index:
- Environment Variable: FLEPI_BLOCK_INDEX
- Type: STRING
- Required: Yes
- Description: The block index for the FLEPI.

--resume_run_index:
- Environment Variable: RESUME_RUN_INDEX
- Type: STRING
- Required: Yes
- Description: The run index for resuming.

--flepi_run_index:
- Environment Variable: FLEPI_RUN_INDEX
- Type: STRING
- Required: Yes
- Description: The run index for the FLEPI.

--flepi_prefix:
- Environment Variable: FLEPI_PREFIX
- Type: STRING
- Required: Yes
- Description: The prefix for the FLEPI.

Function: fetching_resume_files

Parameters:
- resume_location (str): Path to the last run's output.
- discard_seeding (bool): Whether to discard seeding.
- flepi_block_index (str): Block index for FLEPI.
- resume_run_index (str): Run index for resuming.
- flepi_run_index (str): Run index for FLEPI.
- flepi_prefix (str): Prefix for FLEPI.

Description:
The function fetching_resume_files fetches resume files based on the provided parameters. It checks if the resume_location is an S3 path and decides to download from S3 or move files locally accordingly.

Workflow:
1. Retrieves the environment variable SLURM_ARRAY_TASK_ID for the slot index.
2. Converts the discard_seeding boolean to a string "true" if it is True.
3. Creates a resume file name map using the create_resume_file_names_map function.
4. Checks if resume_location starts with "s3://":
   - If yes, downloads the file from S3 using download_file_from_s3.
   - If no, moves the file locally using move_file_at_local.
5. After pulling the input files, it will do a check. If input src_file does not exist, it will output these files.
   If the input files exists, it is not pulled or copied to destination. It will raise FileExistsErrors. 

Example Usage:
To use this script, you can run it from the command line with the required options:
```sh
python script_name.py --resume_location "path/to/resume" --discard_seeding True --block_index "block123" --resume_run_index "run456" --flepi_run_index "run789" --flepi_prefix "prefix"
"""
import click
import os
import boto3
import botocore
from typing import Dict
from gempyor.utils import create_resume_file_names_map, download_file_from_s3, move_file_at_local


@click.command()
@click.option(
    "--resume_location",
    "resume_location",
    envvar=["LAST_JOB_OUTPUT", "RESUME_LOCATION"],
    type=click.STRING,
    required=True,
    help="the path for the last run's output",
)
@click.option(
    "--discard_seeding",
    "discard_seeding",
    envvar="RESUME_DISCARD_SEEDING",
    type=click.BOOL,
    required=True,
    help="required bool value for discarding seeding or not",
)
@click.option("--block_index", "flepi_block_index", envvar="FLEPI_BLOCK_INDEX", type=click.INT, required=True)
@click.option(
    "--resume_run_index", "resume_run_index", envvar="RESUME_RUN_INDEX", type=click.STRING, required=True,
)
@click.option("--flepi_run_index", "flepi_run_index", envvar="FLEPI_RUN_INDEX", type=click.STRING, required=True)
@click.option("--flepi_prefix", "flepi_prefix", envvar="FLEPI_PREFIX", type=click.STRING, required=True)
def fetching_resume_files(
    resume_location, discard_seeding, flepi_block_index, resume_run_index, flepi_run_index, flepi_prefix
):
    flepi_slot_index = os.environ["SLURM_ARRAY_TASK_ID"]
    if discard_seeding is True:
        discard_seeding = "true"

    resume_file_name_map = create_resume_file_names_map(
        resume_discard_seeding=discard_seeding,
        flepi_block_index=str(flepi_block_index),
        resume_run_index=resume_run_index,
        flepi_prefix=flepi_prefix,
        flepi_slot_index=flepi_slot_index,
        flepi_run_index=flepi_run_index,
        last_job_output=resume_location,
    )
    if resume_location.startswith("s3://"):
        download_file_from_s3(resume_file_name_map)
        pull_check_for_s3(resume_file_name_map)
    else:
        move_file_at_local(resume_file_name_map)
        pull_check(resume_file_name_map)


# Todo: Unit test
def pull_check_for_s3(file_name_map: Dict[str, str]) -> None:
    """
    Verifies the existence of specified files in an S3 bucket and checks if corresponding local files are present.
    If a file in the S3 bucket does not exist or the local file is missing, it raises appropriate errors or prints a message.

    Parameters:
    file_name_map (Dict[str, str]): A dictionary where the keys are S3 URIs (Uniform Resource Identifiers) and the values are the corresponding local file paths.

    Dependencies:
    - boto3: The AWS SDK for Python, used to interact with AWS services such as S3.
    - botocore: The low-level core functionality of boto3.
    - os: The standard library module for interacting with the operating system, used here to check for file existence.

    Functionality:
    1. Initialize S3 Client: The function initializes an S3 client using `boto3.client('s3')`.
    2. Iterate through S3 URIs: For each S3 URI in the `file_name_map` dictionary:
       - Parse the Bucket and Object Key: Extracts the bucket name and object key from the S3 URI.
       - Check if Object Exists in S3: Uses the `head_object` method to check if the object exists in the specified S3 bucket.
       - Check Local File Existence: If the object exists in S3, it checks if the corresponding local file exists using `os.path.exists`.
       - Handle Errors:
         - If the object does not exist in S3, it catches the `ClientError` and prints a message indicating the missing S3 object.
         - If the local file does not exist, it raises a `FileExistsError` indicating the local file is missing.

    Example Usage:
    file_name_map = {
        "s3://my-bucket/path/to/file1.txt": "/local/path/to/file1.txt",
        "s3://my-bucket/path/to/file2.txt": "/local/path/to/file2.txt"
    }

    pull_check_for_s3(file_name_map)

    Exceptions:
    - FileExistsError: Raised if the corresponding local file for an existing S3 object is missing.
    - botocore.exceptions.ClientError: Caught and handled to print a message if the S3 object does not exist. Other client errors are re-raised.

    Notes:
    - Ensure that AWS credentials are configured properly for boto3 to access the S3 bucket.
    - This function assumes that the S3 URIs provided are in the format `s3://bucket-name/path/to/object`.
    """
    s3 = boto3.client("s3")
    for s3_uri in file_name_map:
        bucket = s3_uri.split("/")[2]
        object = s3_uri[len(bucket) + 6 :]
        try:
            s3.head_object(Bucket=bucket, Key=object)
            if os.path.exists(file_name_map[s3_uri]) is False:
                raise FileExistsError(f"For {s3_uri}, it is not copied to {file_name_map[s3_uri]}.")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"Input {s3_uri} does not exist.")
            else:
                raise


# Todo: Unit Test
def pull_check(file_name_map: Dict[str, str]) -> None:
    """
    Verifies the existence of specified source files and checks if corresponding destination files are present.
    If a source file does not exist or the destination file is missing, it raises appropriate errors or prints a message.

    Parameters:
    file_name_map (Dict[str, str]): A dictionary where the keys are source file paths and the values are the corresponding destination file paths.

    Dependencies:
    - os: The standard library module for interacting with the operating system, used here to check for file existence.

    Functionality:
    1. Iterate through Source Files: For each source file path in the `file_name_map` dictionary:
       - Check if Source File Exists: Uses `os.path.exists` to check if the source file exists.
       - Check Destination File Existence: If the source file exists, it checks if the corresponding destination file exists using `os.path.exists`.
       - Handle Errors:
         - If the source file does not exist, it prints a message indicating the missing source file.
         - If the destination file does not exist, it raises a `FileExistsError` indicating the destination file is missing.

    Example Usage:
    file_name_map = {
        "/path/to/source1.txt": "/path/to/destination1.txt",
        "/path/to/source2.txt": "/path/to/destination2.txt"
    }

    pull_check(file_name_map)

    Exceptions:
    - FileExistsError: Raised if the corresponding destination file for an existing source file is missing.

    Notes:
    - Ensure that the paths provided are valid and accessible on the file system.
    """
    for src_file in file_name_map:
        if os.path.exists(src_file):
            if os.path.exists(file_name_map[src_file]) is False:
                raise FileExistsError(f"For {src_file}, it is not copied to {file_name_map[src_file]}.")
        else:
            print(f"Input {src_file} does not exist.")


if __name__ == "__main__":
    fetching_resume_files()
