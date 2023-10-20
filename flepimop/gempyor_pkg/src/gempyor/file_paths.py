import os, pathlib, datetime


def create_file_name(
    run_id,
    prefix,
    index,
    ftype,
    extension,
    inference_filepath_suffix="",
    inference_filename_prefix="",
    create_directory=True,
):
    if create_directory:
        os.makedirs(
            create_dir_name(run_id, prefix, ftype, inference_filepath_suffix, inference_filename_prefix), exist_ok=True
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
    run_id, prefix, index, ftype, inference_filepath_suffix, inference_filename_prefix, create_directory=True
):
    if create_directory:
        os.makedirs(
            create_dir_name(run_id, prefix, ftype, inference_filepath_suffix, inference_filename_prefix), exist_ok=True
        )
    filename = pathlib.Path(
        "model_output",
        prefix,
        run_id,
        ftype,
        inference_filepath_suffix,
        f"{inference_filename_prefix}{index:>09}.{run_id}.{ftype}",
    )
    # old:  "model_output/%s/%s%09d.%s.%s" % (ftype, prefix, index, run_id, ftype)
    return filename


def run_id():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S%Z")


def create_dir_name(run_id, prefix, ftype, inference_filepath_suffix, inference_filename_prefix):
    return os.path.dirname(
        create_file_name_without_extension(
            run_id, prefix, 1, ftype, inference_filepath_suffix, inference_filename_prefix, create_directory=False
        )
    )
