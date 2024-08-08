import os
import click
import shutil
from file_paths import create_file_name_for_push


@click.command()
@click.option('--aws', is_flag=True, help='push files to aws', required=True)
@click.option('--flepi_run_index', 'flepi_run_index', envvar='FLEPI_RUN_INDEX', type=click.STRING, required=True)
@click.option('--flepi_prefix', 'flepi_prefix', envvar='FLEPI_PREFIX', type=click.STRING, required=True)
@click.option('--flepi_block_index', 'flepi_block_index', envvar='FLEPI_BLOCK_INDEX', type=click.STRING, required=True)
@click.option('--flepi_slot_index', 'flepi_slot_index', envvar='FLEPI_SLOT_INDEX', type=click.STRING, required=True)
@click.option('--s3_results_path', 's3_results_path', envvar='S3_RESULTS_PATH', type=click.STRING, required=False)
@click.option('--fs_results_path', 'fs_results_path', envvar="FS_RESULTS_PATH", type=click.STRING, required=False)
def flepimop_push(aws: bool,
                  flepi_run_index: str,
                  flepi_prefix: str,
                  flepi_slot_index: str,
                  flepi_block_index:str,
                  s3_results_path:str = "",
                  fs_results_path:str = "") -> None:
    file_name_list = create_file_name_for_push(flepi_run_index=flepi_run_index,
                                          prefix=flepi_prefix,
                                          flepi_slot_index=flepi_slot_index,
                                          flepi_block_index=flepi_block_index)
    exist_files = [f for f in file_name_list if os.path.exists(f) is True]
    if aws:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ModuleNotFoundError:
            raise ModuleNotFoundError((
                "No module named 'boto3', which is required for "
                "gempyor.utils.download_file_from_s3. Please install the aws target."
            ))
        if s3_results_path == "":
            raise ValueError("argument aws is setted to True, you must use --s3_results_path too or environment variable S3_RESULTS_PATH.")
        s3 = boto3.client("s3")
        for file in exist_files:
            s3_path = os.path.join(s3_results_path, file)
            bucket = s3_path.split("/")[2]
            object_name = s3_path[len(bucket) + 6: ]
            s3.upload_file(file, bucket, object_name)
    else:
        if fs_results_path == "":
            raise ValueError("argument aws is setted to False, you must use --fs_results_path or environment Variable FS_RESULTS_PATH.")
        for file in exist_files:
            dst = os.path.join(fs_results_path, file)
            os.path.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(file, dst)