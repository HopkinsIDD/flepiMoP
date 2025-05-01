"""
Project simulation scenarios from calibration results

Provides CLI options for users to specify simulation parameters.

Functions:
    project: Reads a configuration file for simulation settings.
"""

#!/usr/bin/env python
import os
import shutil
import copy
import pathlib
import multiprocessing
from abc import ABC, abstractmethod
from pathlib import Path

import click
import emcee
import numpy as np
from numpy import ndarray

import gempyor
from gempyor import model_info, file_paths, config, inference_parameter
from gempyor.inference import GempyorInference
from gempyor.utils import config, as_list
import gempyor.postprocess_inference

# from .profile import profile_options

# disable  operations using the MKL linear algebra.
os.environ["OMP_NUM_THREADS"] = "1"

class Backend(ABC):
    
    @abstractmethod
    def get_parameters(self) -> ndarray:
        raise NotImplementedError("Pure Backend is an abstract class.")

class ParquetBackend(Backend):
    def __init__(self, file : Path):
        super().__init__()
        self.rootdir = file
    
    def get_parameters(self):
        return super().get_parameters()

class HD5Backend(Backend):
    def __init__(self, file : Path):
        super().__init__()
        self.filename = file
    
    def get_parameters(self) -> ndarray:
        return emcee.backends.HDFBackend(self.filename).get_last_sample().coords

def backend_factory(file : Path) -> Backend:
    if file.is_dir:
        return ParquetBackend(file)
    else:
        return HD5Backend(file)

@click.command()
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar="CONFIG_PATH",
    type=click.Path(),
    required=True,
    help="configuration file for this simulation",
)
@click.option(
    "-p",
    "--project_path",
    "project_path",
    envvar="PROJECT_PATH",
    type=click.Path(exists=True),
    default=".",
    required=True,
    help="path to the flepiMoP directory",
)
@click.option(
    "-j",
    "--jobs",
    "ncpu",
    envvar="FLEPI_NJOBS",
    type=click.IntRange(min=1),
    default=multiprocessing.cpu_count(),
    show_default=True,
    help="the parallelization factor",
)
@click.option(
    "--id",
    "--id",
    "input_run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=None,
    help="Unique identifier for this run",
)
@click.option(
    "-prefix",
    "--prefix",
    "prefix",
    envvar="FLEPI_PREFIX",
    type=str,
    default=None,
    show_default=True,
    help="unique identifier for the run",
)
@click.option(
    ["-r", "--resume_location", "filename"],
    type=str,
    default=None,
    help="The location (folder or an S3 bucket) to use as the initial to the first block of the current run",
)
# @profile_options
# @profile()
def project(
    config_filepath: str | pathlib.Path,
    project_path: str,
    ncpu: int,
    input_run_id: int | None,
    filename: str | Path | None,
) -> None:
    """
    Project using sampler to initialize a model based on a config.
    """

    # Choose a run_id
    if input_run_id is None:
        base_run_id = pathlib.Path(config_filepath).stem.replace("config_", "")
        run_id = f"{base_run_id}-{file_paths.run_id()}"
        print(f"Auto-generating run_id: {run_id}")
    else:
        run_id = input_run_id

    gempyor_inference = GempyorInference(
        config_filepath=config_filepath,
        run_id=run_id,
        prefix=None,
        first_sim_index=1,
        rng_seed=None,
        nslots=1,
        inference_filename_prefix="global/final/",  # usually for {global or chimeric}/{intermediate or final}
        inference_filepath_suffix="",  # usually for the slot_id
        out_run_id=None,  # if out_run_id is different from in_run_id, fill this
        out_prefix=None,  # if out_prefix is different from in_prefix, fill this
        path_prefix=project_path,  # in case the data folder is on another directory
        autowrite_seir=True,
    )

    # Draw/get initial parameters:
    backend = backend_factory(filename)

    p0 = backend.get_parameters()

    with multiprocessing.Pool(ncpu) as pool:
        results = pool.starmap(
            gempyor_inference.simulate_proposal, [(p0[i],) for i in range(p0.shape[2])]
        )
    
    return 0

if __name__ == "__main__":
    project()

## @endcond
