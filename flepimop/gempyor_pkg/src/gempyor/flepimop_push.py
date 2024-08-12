import os
import click
import shutil
from gempyor.file_paths import create_file_name_for_push


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
    """
    Push files to either AWS S3 or the local filesystem.

    This function generates a list of file names based on the provided parameters and checks which files
    exist locally. It then uploads these files to either AWS S3 or the local filesystem based on the 
    specified options.

    Parameters:
    ----------
    aws : bool
        Flag indicating whether to push files to AWS S3. If set to True, files will be uploaded to S3.
    
    flepi_run_index : str
        The index of the FLEPI run. This is used to uniquely identify the run.
    
    flepi_prefix : str
        A prefix string to be included in the file names. This is typically used to categorize or 
        identify the files.
    
    flepi_slot_index : str
        The slot index used in the filename. This is formatted as a zero-padded nine-digit number.
    
    flepi_block_index : str
        The block index used in the filename. This typically indicates a specific block or segment 
        of the data being processed.
    
    s3_results_path : str, optional
        The S3 path where the results should be uploaded. This parameter is required if `aws` is set to True.
    
    fs_results_path : str, optional
        The local filesystem path where the results should be copied. This parameter is required if `aws` is set to False.

    Raises:
    ------
    ValueError
        If `aws` is set to True and `s3_results_path` is not provided.
        If `aws` is set to False and `fs_results_path` is not provided.
    
    ModuleNotFoundError
        If `boto3` is not installed when `aws` is set to True.
    """
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
                "gempyor.flepimop_push.flepimop_push. Please install the aws target."
            ))
        if s3_results_path == "":
            raise ValueError("argument aws is setted to True, you must use --s3_results_path or environment variable S3_RESULTS_PATH.")
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
    print("flepimop-push successfully push all existing files.")

            
if __name__ =="__main__":
    flepimop_push()