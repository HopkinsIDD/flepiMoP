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

Example Usage:
To use this script, you can run it from the command line with the required options:
```sh
python script_name.py --resume_location "path/to/resume" --discard_seeding True --block_index "block123" --resume_run_index "run456" --flepi_run_index "run789" --flepi_prefix "prefix"
"""
import click
import os
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
    flep_slot_index = os.environ["SLURM_ARRAY_TASK_ID"]
    if discard_seeding is True:
        discard_seeding = "true"

    resume_file_name_map = create_resume_file_names_map(
        resume_discard_seeding=discard_seeding,
        flepi_block_index=str(flepi_block_index),
        resume_run_index=resume_run_index,
        flepi_prefix=flepi_prefix,
        flepi_slot_index=flep_slot_index,
        flepi_run_index=flepi_run_index,
        last_job_output=resume_location,
    )
    if resume_location.startswith("s3://"):
        download_file_from_s3(resume_file_name_map)
    else:
        move_file_at_local(resume_file_name_map)


if __name__ == "__main__":
    fetching_resume_files()
