import os
import datetime


def create_file_name(*, run_id, prefix, index, ftype, extension, model_output_path, create_directory=True):
    if create_directory:
        os.makedirs(create_dir_name(run_id=run_id, prefix=prefix, ftype=ftype, model_output_path=model_output_path), exist_ok=True)

    fn_no_ext = create_file_name_without_extension(run_id, prefix, index, ftype, model_output_path, create_directory=create_directory)
    return f"{fn_no_ext}.%s" % (extension,)


def create_file_name_without_extension(*, run_id, prefix, index, ftype, model_output_path, create_directory=True):
    if create_directory:
        os.makedirs(create_dir_name(run_id=run_id, prefix=prefix, ftype=ftype, model_output_path=model_output_path), exist_ok=True)
    if model_output_path[-1] == "/":
        model_output_path = model_output_path[:-1]
    return "%s/model_output/%s/%s%09d.%s.%s" % ( model_output_path, ftype, prefix, index, run_id, ftype)


def run_id():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y.%m.%d.%H:%M:%S.%Z")

def create_dir_name(*, run_id, prefix, ftype,  model_output_path):
    return os.path.dirname(create_file_name_without_extension(run_id=run_id, prefix=prefix, index=1, ftype=ftype, model_output_path=model_output_path, create_directory=False))
