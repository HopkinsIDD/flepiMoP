#!/usr/bin/env python

import click
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import tarfile
from datetime import datetime, timezone, date
import yaml
from gempyor import file_paths
import gempyor.utils


def user_confirmation(question="Continue?", default=False):
    if default:
        prompt = "[Y/n]"
    else:
        prompt = "[y/N]"
    while True:
        answer = input(f"{question} {prompt} ")
        if not answer:
            return default
        if answer.lower() in ("y", "yes"):
            return True
        if answer.lower() in ("n", "no"):
            return False


@click.command()
@click.option("--aws", "batch_system", flag_value="aws", default=True)
@click.option("--slurm", "batch_system", flag_value="slurm")
@click.option("--local", "batch_system", flag_value="local")
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar="CONFIG_PATH",
    type=click.Path(exists=True),
    required=True,
    help="configuration file for this run",
)
@click.option(
    "-p",
    "--flepi_path",
    "flepi_path",
    envvar="FLEPI_PATH",
    type=click.Path(exists=True),
    required=True,
    help="path to the flepiMoP directory",
)
@click.option(
    "--data-path",
    "--data-path",
    "data_path",
    envvar="PROJECT_PATH",
    type=click.Path(exists=True),
    default=".",
    help="path to the data directory",
)
@click.option(
    "--id",
    "--id",
    "run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=file_paths.run_id(),
    help="Unique identifier for this run",
)
@click.option(
    "-n",
    "--num-jobs",
    "num_jobs",
    type=click.IntRange(min=1, max=1000),
    default=None,
    help="number of output slots to generate",
)
@click.option(
    "-j",
    "--sims-per-job",
    "sims_per_job",
    type=click.IntRange(min=1),
    default=None,
    help="the number of sims to run on each child job",
)
@click.option(
    "-k",
    "--num-blocks",
    "num_blocks",
    type=click.IntRange(min=1),
    default=None,
    help="The number of sequential blocks of jobs to run; total sims per slot = sims-per-slot * num-blocks",
)
@click.option(
    "-o",
    "--output",
    "outputs",
    multiple=True,
    default=["model_output", "model_parameters", "importation", "hospitalization"],
    show_default=True,
    help="The output directories whose contents are captured and saved in S3",
)
@click.option(  # aws only option, or slurm if --upload-to-s3 is selected
    "-b",
    "--s3-bucket",
    "s3_bucket",
    type=str,
    default="idd-inference-runs",
    show_default=True,
    help="The S3 bucket to use for keeping state for the batch jobs",
)
@click.option(  # slurm only option
    "-f",
    "--fs-folder",
    "fs_folder",
    type=str,
    default="/scratch4/struelo1/flepimop-runs",
    show_default=True,
    help="The file system folder to use for keeping the job outputs",
)
@click.option(  # aws only option
    "-d",
    "--job-definition",
    "batch_job_definition",
    type=str,
    default="Batch-CovidPipeline-Job",
    show_default=True,
    help="The name of the AWS Batch Job Definition to use for the job",
)
@click.option(  # aws only option
    "-q",
    "--job-queue-prefix",
    "job_queue_prefix",
    type=str,
    default="Inference-JQ",
    show_default=True,
    help="The prefix string of the job queues we should use for this run",
)
@click.option(
    "-v",
    "--vcpus",
    "vcpus",
    type=click.IntRange(min=1, max=96),
    default=2,
    show_default=True,
    help="The number of CPUs to request for running jobs",
)
@click.option(
    "-m",
    "--memory",
    "memory",
    type=click.IntRange(min=1000, max=24000),
    default=8000,
    show_default=True,
    help="The amount of RAM in megabytes needed per CPU running simulations",
)
@click.option(
    "-t",
    "--time-per-sim",
    "time_per_sim",
    type=click.FloatRange(min=0.0, max=1000.0),
    default=3.0,
    show_default=True,
    help="The time (in minute) each simulation is expected to take, it is used to compute the time limit, so provide an upper-bound that accounts for downloading & uploading, initialization, etc.",
)
@click.option(
    "-r",
    "--restart-from-location",
    "restart_from_location",
    type=str,
    default=None,
    envvar="RESUME_LOCATION",
    help="The location (folder or an S3 bucket) to use as the initial to the first block of the current run",
)
@click.option(
    "-r",
    "--restart-from-run-id",
    "restart_from_run_id",
    type=str,
    default=None,
    help="The run_id of the run we are restarting from",
)
@click.option(
    "--resume-discard-seeding/--resume-carry-seeding",
    "--resume-discard-seeding/--resume-carry-seeding",
    "resume_discard_seeding",
    envvar="RESUME_DISCARD_SEEDING",
    type=bool,
    default=False,
    help="Flag determining whether to keep seeding in resume runs",
)
@click.option(
    "--stacked-max",
    "--stacked-max",
    "max_stacked_interventions",
    envvar="FLEPI_MAX_STACK_SIZE",
    type=click.IntRange(min=350),
    default=5000,
    help="Maximum number of interventions to allow in a stacked intervention",
)
@click.option(
    "--validation-end-date",
    "--validation-end-date",
    "last_validation_date",
    envvar="VALIDATION_DATE",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=str(date.today()),
    help="First date of projection/forecast -- first date without ground truth data",
)
@click.option(
    "--reset-chimerics-on-global-accept",
    "--reset-chimerics-on-global-accept",
    "reset_chimerics",
    envvar="FLEPI_RESET_CHIMERICS",
    type=bool,
    default=True,
    help="Flag determining whether to reset chimeric values on any global acceptances",
)
@click.option(  # slurm only option
    "--upload-to-s3",
    "--upload-to-s3",
    "s3_upload",
    type=bool,
    default=True,
    help="Flag determining whether we also save runs to s3 for slurm runs",
)
@click.option(
    "-s",
    "--slack-channel",
    "slack_channel",
    envvar="SLACK_CHANNEL",
    default="cspproduction",
    type=click.Choice(["cspproduction", "debug", "noslack"]),
    help="Slack channel, either 'csp-production' or 'debug', or 'noslack' to disable slack",
)
@click.option(
    "--continuation/--no-continuation",
    "--continuation/--no-continuation",
    "continuation",
    envvar="FLEPI_CONTINUATION",
    type=bool,
    default=False,
    help="Flag determining whether to use the resumed run /seir/files (or the provided /init/ial files bucket) as initial conditions for the next run",
)
@click.option(
    "--continuation-location",
    "--continuation-location",
    "continuation_location",
    type=str,
    default=None,
    envvar="FLEPI_CONTINUATION_LOCATION",
    help="The location (folder or an S3 bucket) from which to pull the /init/ files (if not set, uses the resume location seir files)",
)
@click.option(
    "--continuation-run-id",
    "--continuation-run-id",
    "continuation_run_id",
    type=str,
    default=None,
    envvar="FLEPI_CONTINUATION_RUN_ID",
    help="The run_id of the run we are continuing from",
)
def launch_batch(
    batch_system,
    config_filepath,
    flepi_path,
    data_path,
    run_id,
    num_jobs,
    sims_per_job,
    num_blocks,
    outputs,
    s3_bucket,
    fs_folder,
    batch_job_definition,
    job_queue_prefix,
    vcpus,
    memory,
    time_per_sim,
    restart_from_location,
    restart_from_run_id,
    resume_discard_seeding,
    max_stacked_interventions,
    last_validation_date,
    reset_chimerics,
    s3_upload,
    slack_channel,
    continuation,
    continuation_location,
    continuation_run_id,
):
    config = None
    with open(config_filepath) as f:
        config = yaml.full_load(f)

    # A unique name for this job run, based on the config name and current time
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_name = f"{config['name']}-{timestamp}"

    num_jobs, sims_per_job, num_blocks = autodetect_params(
        config,
        data_path=data_path,
        num_jobs=num_jobs,
        sims_per_job=sims_per_job,
        num_blocks=num_blocks,
        batch_system=batch_system,
    )

    # Update and save the config file with the number of sims to run
    # TODO: does this really save the config file?
    if "inference" in config:
        config["inference"]["iterations_per_slot"] = sims_per_job
        if not os.path.exists(pathlib.Path(data_path, config["inference"]["gt_data_path"])):
            print(
                f"ERROR: inference.data_path path {pathlib.Path(data_path, config['inference']['gt_data_path'])} does not exist!"
            )
            return 1
    else:
        print(f"WARNING: no inference section found in {config_filepath}!")

    if "s3://" in str(restart_from_location):  # ugly hack: str because it might be None
        restart_from_run_id = aws_countfiles_autodetect_runid(
            s3_bucket=s3_bucket,
            restart_from_location=restart_from_location,
            restart_from_run_id=restart_from_run_id,
            num_jobs=num_jobs,
            strict=False,
        )
    else:
        if restart_from_run_id is None and restart_from_location is not None:
            raise Exception(
                "No auto-detection of run_id from local folder, please specify --restart_from_run_id (or fixme)"
            )
    if "s3://" in str(continuation_location):
        continuation_run_id = aws_countfiles_autodetect_runid(
            s3_bucket=s3_bucket,
            restart_from_location=continuation_location,
            restart_from_run_id=continuation_run_id,
            num_jobs=num_jobs,
            strict=True,
        )
    else:
        if continuation_run_id is None and continuation_location is not None:
            raise Exception(
                "No auto-detection of run_id from local folder, please specify --continuation_run_id (or fixme)"
            )
    if continuation and continuation_location is None:
        continuation_location = restart_from_location
        continuation_run_id = restart_from_run_id
        print(
            "Continuation enabled but no continuation location provided. Assuming that continuation location is the same as resume location"
        )

    handler = BatchJobHandler(
        batch_system,
        flepi_path,
        data_path,
        run_id,
        num_jobs,
        sims_per_job,
        num_blocks,
        outputs,
        s3_bucket,
        fs_folder,
        batch_job_definition,
        job_queue_prefix,
        vcpus,
        memory,
        time_per_sim,
        restart_from_location,
        restart_from_run_id,
        resume_discard_seeding,
        max_stacked_interventions,
        last_validation_date,
        reset_chimerics,
        s3_upload,
        slack_channel,
        continuation,
        continuation_location,
        continuation_run_id,
    )

    seir_modifiers_scenarios = None
    outcome_modifiers_scenarios = None
    # here the config is a dict
    if "seir_modifiers" in config:
        if "scenarios" in config["seir_modifiers"]:
            seir_modifiers_scenarios = config["seir_modifiers"]["scenarios"]
    if "outcome_modifiers" in config:
        if "scenarios" in config["outcome_modifiers"]:
            outcome_modifiers_scenarios = config["outcome_modifiers"]["scenarios"]

    handler.launch(
        job_name, config_filepath, seir_modifiers_scenarios, outcome_modifiers_scenarios
    )

    # Set job_name as environmental variable so it can be pulled for pushing to git
    os.environ["job_name"] = job_name
    # Set run_id as environmental variable so it can be pulled for pushing to git TODO

    (rc, txt) = subprocess.getstatusoutput(
        f"git checkout -b run_{job_name}"
    )  # TODO: cd ...
    print(txt)
    return rc


def autodetect_params(
    config,
    data_path,
    *,
    num_jobs=None,
    sims_per_job=None,
    num_blocks=None,
    batch_system=None,
):
    if num_jobs and sims_per_job and num_blocks:
        return (num_jobs, sims_per_job, num_blocks)

    if "inference" not in config or "iterations_per_slot" not in config["inference"]:
        raise click.UsageError(
            "inference::iterations_per_slot undefined in config, can't autodetect parameters"
        )
    iterations_per_slot = int(config["inference"]["iterations_per_slot"])

    if num_jobs is None:
        num_jobs = config["nslots"]
        print(f"Setting number of output slots to {num_jobs} [via config file]")

    if sims_per_job is None:
        if num_blocks is not None:
            sims_per_job = int(math.ceil(iterations_per_slot / num_blocks))
            print(
                f"Setting number of blocks to {num_blocks} [via num_blocks (-k) argument]"
            )
            print(
                f"Setting sims per job to {sims_per_job} [via {iterations_per_slot} iterations_per_slot in config]"
            )
        else:
            if "data_path" in config:
                raise ValueError(
                    "The config has a data_path section. This is no longer supported."
                )
            geodata_fname = pathlib.Path(data_path) / config["subpop_setup"]["geodata"]
            with open(geodata_fname) as geodata_fp:
                num_subpops = sum(1 for line in geodata_fp)

            if batch_system == "aws":
                # formula based on a simple regression of subpops (based on known good performant params)
                sims_per_job = max(60 - math.sqrt(num_subpops), 10)
                sims_per_job = 5 * int(math.ceil(sims_per_job / 5))  # multiple of 5
                num_blocks = int(math.ceil(iterations_per_slot / sims_per_job))
            elif batch_system == "slurm" or batch_system == "local":
                # now launch full sims:
                sims_per_job = iterations_per_slot
                num_blocks = 1
            else:
                raise ValueError(f"Unknown batch submission system {batch_system}")

            print(
                f"Setting sims per job to {sims_per_job} "
                f"[estimated based on {num_subpops} subpop(s) and {iterations_per_slot} iterations_per_slot in config]"
            )
            print(f"Setting number of blocks to {num_blocks} [via math]")

    if num_blocks is None:
        num_blocks = int(math.ceil(iterations_per_slot / sims_per_job))
        print(
            f"Setting number of blocks to {num_blocks} [via {iterations_per_slot} iterations_per_slot in config]"
        )

    return (num_jobs, sims_per_job, num_blocks)


def get_aws_job_queues(job_queue_prefix):
    import boto3

    batch_client = boto3.client("batch")
    queues_with_jobs = {}
    resp = batch_client.describe_job_queues()
    for q in resp["jobQueues"]:
        queue_name = q["jobQueueName"]
        if queue_name.startswith(job_queue_prefix):
            job_list_resp = batch_client.list_jobs(jobQueue=queue_name, jobStatus="PENDING")
            queues_with_jobs[queue_name] = len(job_list_resp["jobSummaryList"])
    # Return the least-loaded queues first
    return sorted(queues_with_jobs, key=queues_with_jobs.get)


def aws_countfiles_autodetect_runid(
    s3_bucket, restart_from_location, restart_from_run_id, num_jobs, strict=False
):
    import boto3

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)
    prefix = restart_from_location.split("/")[3] + "/model_output/"
    all_files = list(bucket.objects.filter(Prefix=prefix))
    all_files = [f.key for f in all_files]
    if restart_from_run_id is None:
        print(
            "WARNING: no --restart_from_run_id specified, autodetecting... please wait querying S3 👀🔎..."
        )
        restart_from_run_id = all_files[0].split("/")[3]
        if user_confirmation(
            question=f"Auto-detected run_id {restart_from_run_id}. Correct ?", default=True
        ):
            print(f"great, continuing with run_id {restart_from_run_id}...")
        else:
            raise ValueError(f"Abording, please specify --restart_from_run_id manually.")

    final_llik = [f for f in all_files if ("llik" in f) and ("final" in f)]
    if (
        len(final_llik) == 0
    ):  # hacky: there might be a bucket with no llik files, e.g if init.
        final_llik = [f for f in all_files if ("init" in f) and ("final" in f)]

    if len(final_llik) != num_jobs:
        if strict:
            raise ValueError(
                f"number of good slots in resume_location: ({len(final_llik)}) does not match number of jobs ({num_jobs})."
            )
        else:
            print(
                f"WARNING: number of good slots in resume_location: ({len(final_llik)}) does not match number of jobs ({num_jobs})."
            )
            if (num_jobs - len(final_llik)) > 50:
                user_confirmation(question=f"Difference > 50. Should we continue ?")

    return restart_from_run_id


class BatchJobHandler(object):
    def __init__(
        self,
        batch_system,
        flepi_path,
        data_path,
        run_id,
        num_jobs,
        sims_per_job,
        num_blocks,
        outputs,
        s3_bucket,
        fs_folder,
        batch_job_definition,
        job_queue_prefix,
        vcpus,
        memory,
        time_per_sim,
        restart_from_location,
        restart_from_run_id,
        resume_discard_seeding,
        max_stacked_interventions,
        last_validation_date,
        reset_chimerics,
        s3_upload,
        slack_channel,
        continuation,
        continuation_location,
        continuation_run_id,
    ):
        self.batch_system = batch_system
        self.flepi_path = flepi_path
        self.data_path = data_path
        self.run_id = run_id
        self.num_jobs = num_jobs
        self.sims_per_job = sims_per_job
        self.num_blocks = num_blocks
        self.outputs = outputs
        self.s3_bucket = s3_bucket
        self.fs_folder = fs_folder
        self.batch_job_definition = batch_job_definition
        self.job_queue_prefix = job_queue_prefix
        self.vcpus = vcpus
        self.memory = memory
        self.time_per_sim = time_per_sim
        self.restart_from_location = restart_from_location
        self.restart_from_run_id = restart_from_run_id
        self.resume_discard_seeding = resume_discard_seeding
        self.max_stacked_interventions = max_stacked_interventions
        self.last_validation_date = last_validation_date
        self.reset_chimerics = reset_chimerics
        self.s3_upload = s3_upload
        self.slack_channel = slack_channel
        self.continuation = continuation
        self.continuation_location = continuation_location
        self.continuation_run_id = continuation_run_id

    def build_job_metadata(self, job_name):
        """
        Create a manifest file to preserve what is used for the current run.
        - For slurm: save this manifest into the fs_folder
        - For aws: save the manifest into the s3_bucket, but also upload the necessary files to run the job
        (inference_runner.sh, inference_runner.py, and the flepimop_ and data_ folders)
        TODO: should we save the tar file when doing the slurm as well in case the user pulls while the job is running?
        """
        manifest = {}
        manifest["cmd"] = " ".join(sys.argv[:])
        manifest["job_name"] = job_name
        manifest["data_sha"] = subprocess.getoutput(
            "cd {self.data_path}; git rev-parse HEAD"
        )
        manifest["flepimop_sha"] = subprocess.getoutput(
            f"cd {self.flepi_path}; git rev-parse HEAD"
        )

        # Save the manifest file to S3
        with open("manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        if self.batch_system == "aws":
            # need these to be uploaded so they can be executed.
            this_file_path = os.path.dirname(os.path.realpath(__file__))
            self.save_file(
                source=os.path.join(this_file_path, "AWS_inference_runner.sh"),
                destination=f"{job_name}-runner.sh",
            )
            self.save_file(
                source=os.path.join(this_file_path, "AWS_inference_copy.sh"),
                destination=f"{job_name}-copy.sh",
            )

            tarfile_name = f"{job_name}.tar.gz"
            self.tar_working_dir(tarfile_name=tarfile_name)
            self.save_file(
                source=tarfile_name, destination=f"{job_name}.tar.gz", remove_source=True
            )

        self.save_file(
            source="manifest.json",
            destination=f"{job_name}/manifest.json",
            remove_source=True,
        )

    def tar_working_dir(self, tarfile_name):
        # this tar file always has the structure:
        # where all data files are in the root of the tar file and the csp files are in a flepiMoP folder.
        tar = tarfile.open(tarfile_name, "w:gz", dereference=True)
        for q in os.listdir(self.flepi_path):
            if not (
                q == "packrat"
                or q == "covid-dashboard-app"
                or q == "renv.cache"
                or q == "sample_data"
                or q
                == "renv"  # joseph: I added this to fix a bug, hopefully it doesn't break anything
                or q.startswith(".")
            ):
                tar.add(
                    os.path.join(self.flepi_path, q), arcname=os.path.join("flepiMoP", q)
                )
            elif q == "sample_data":
                for r in os.listdir(os.path.join(self.flepi_path, "sample_data")):
                    if r != "united-states-commutes":
                        tar.add(
                            os.path.join(self.flepi_path, "sample_data", r),
                            arcname=os.path.join("flepiMoP", "sample_data", r),
                        )
                        # tar.add(os.path.join("flepiMoP", "sample_data", r))
        for p in os.listdir(self.data_path):
            if not (
                p.startswith(".")
                or p.endswith("tar.gz")
                or p in self.outputs
                or p == "flepiMoP"
            ):
                tar.add(
                    p,
                    filter=lambda x: (
                        None if os.path.basename(x.name).startswith(".") else x
                    ),
                )
        tar.close()

    def save_file(self, source, destination, remove_source=False, prefix=""):
        """
        Put a file to the appropriate location, either or s3 or both, in the right folder or both
        """
        if self.s3_upload or self.batch_system == "aws":
            import boto3

            s3_client = boto3.client("s3")
            s3_client.upload_file(source, self.s3_bucket, os.path.join(prefix, destination))

        if self.batch_system == "slurm":
            import shutil

            # Copy the tar'd contents of this directory and the runner script to the appropriate location
            # os.path.join makes sure that the / are correct whatever finishes fs_folder
            shutil.copy(source, os.path.join(self.fs_folder, prefix, destination))

        if remove_source:
            os.remove(source)

    def launch(
        self,
        job_name,
        config_filepath,
        seir_modifiers_scenarios,
        outcome_modifiers_scenarios,
    ):
        s3_results_path = f"s3://{self.s3_bucket}/{job_name}"

        if self.batch_system == "slurm":
            fs_results_path = os.path.join(self.fs_folder, job_name)
            os.makedirs(f"{fs_results_path}", exist_ok=True)
        else:
            fs_results_path = ""  # needs to be defined for the env_var

        self.build_job_metadata(job_name)

        if self.batch_system == "aws":
            import boto3

            job_queues = get_aws_job_queues(self.job_queue_prefix)
            batch_client = boto3.client("batch")

        ## TODO: check how each of these variables are used downstream
        base_env_vars = [
            {"name": "BATCH_SYSTEM", "value": self.batch_system},
            {
                "name": "S3_MODEL_PROJECT_PATH",
                "value": f"s3://{self.s3_bucket}/{job_name}.tar.gz",
            },
            {"name": "DVC_OUTPUTS", "value": " ".join(self.outputs)},
            {"name": "S3_RESULTS_PATH", "value": s3_results_path},
            {"name": "FS_RESULTS_PATH", "value": fs_results_path},
            {"name": "S3_UPLOAD", "value": str(self.s3_upload).lower()},
            {"name": "PROJECT_PATH", "value": str(self.data_path)},
            {"name": "FLEPI_PATH", "value": str(self.flepi_path)},
            {"name": "CONFIG_PATH", "value": config_filepath},
            {"name": "FLEPI_NUM_SLOTS", "value": str(self.num_jobs)},
            {
                "name": "FLEPI_MAX_STACK_SIZE",
                "value": str(self.max_stacked_interventions),
            },
            {"name": "VALIDATION_DATE", "value": str(self.last_validation_date.date())},
            {"name": "SIMS_PER_JOB", "value": str(self.sims_per_job)},
            {"name": "FLEPI_ITERATIONS_PER_SLOT", "value": str(self.sims_per_job)},
            {
                "name": "RESUME_DISCARD_SEEDING",
                "value": str(
                    self.resume_discard_seeding
                ).lower(),  # lower is import here, this is string-compared to "true" in the run script
            },
            {"name": "FLEPI_RESET_CHIMERICS", "value": str(self.reset_chimerics)},
            {
                "name": "FLEPI_MEM_PROFILE",
                "value": str(os.getenv("FLEPI_MEM_PROFILE", default="FALSE")),
            },
            {
                "name": "FLEPI_MEM_PROF_ITERS",
                "value": str(os.getenv("FLEPI_MEM_PROF_ITERS", default="50")),
            },
            {"name": "SLACK_CHANNEL", "value": str(self.slack_channel)},
        ]
        with open(config_filepath) as f:
            config = yaml.full_load(f)

        for ctr, (s, d) in enumerate(
            itertools.product(seir_modifiers_scenarios, outcome_modifiers_scenarios)
        ):
            cur_job_name = f"{job_name}_{s}_{d}"
            # Create first job
            cur_env_vars = base_env_vars.copy()
            cur_env_vars.append({"name": "FLEPI_SEIR_SCENARIOS", "value": s})
            cur_env_vars.append({"name": "FLEPI_OUTCOME_SCENARIOS", "value": d})
            cur_env_vars.append(
                {"name": "FLEPI_PREFIX", "value": f"{config['name']}_{s}_{d}"}
            )  # TODO: get it from gempyor and makes it contains run_id also in scripts
            cur_env_vars.append({"name": "FLEPI_BLOCK_INDEX", "value": "1"})
            cur_env_vars.append({"name": "FLEPI_RUN_INDEX", "value": f"{self.run_id}"})
            if not (self.restart_from_location is None):
                cur_env_vars.append(
                    {"name": "LAST_JOB_OUTPUT", "value": f"{self.restart_from_location}"}
                )
                cur_env_vars.append(
                    {
                        "name": "OLD_FLEPI_RUN_INDEX",
                        "value": f"{self.restart_from_run_id}",
                    }
                )
                cur_env_vars.append({"name": "RESUME_RUN", "value": f"TRUE"})
            else:
                cur_env_vars.append({"name": "RESUME_RUN", "value": f"FALSE"})

            if self.continuation:
                cur_env_vars.append({"name": "FLEPI_CONTINUATION", "value": f"TRUE"})
                cur_env_vars.append(
                    {
                        "name": "FLEPI_CONTINUATION_RUN_ID",
                        "value": f"{self.continuation_run_id}",
                    }
                )
                cur_env_vars.append(
                    {
                        "name": "FLEPI_CONTINUATION_LOCATION",
                        "value": f"{self.continuation_location}",
                    }
                )
                cur_env_vars.append(
                    {
                        "name": "FLEPI_CONTINUATION_FTYPE",
                        "value": f"{config['initial_conditions']['initial_file_type']}",
                    }
                )

            # First job:
            if self.batch_system == "aws":
                cur_env_vars.append({"name": "JOB_NAME", "value": f"{cur_job_name}_block0"})
                runner_script_path = f"s3://{self.s3_bucket}/{job_name}-runner.sh"
                s3_cp_run_script = f"aws s3 cp {runner_script_path} $PWD/run-flepimop-inference"  # line to copy the runner script in wd as ./run-covid-pipeline
                command = [
                    "sh",
                    "-c",
                    f"{s3_cp_run_script}; /bin/bash $PWD/run-flepimop-inference",
                ]  # execute copy line above and then run the script

                cur_job_queue = job_queues[ctr % len(job_queues)]
                last_job = batch_client.submit_job(
                    jobName=f"{cur_job_name}_block0",
                    jobQueue=cur_job_queue,
                    arrayProperties={"size": self.num_jobs},
                    jobDefinition=self.batch_job_definition,
                    containerOverrides={
                        "vcpus": self.vcpus,
                        "memory": self.vcpus * self.memory,
                        "environment": cur_env_vars,
                        "command": command,
                    },
                    retryStrategy={"attempts": 3},
                )
            elif self.batch_system == "slurm":
                cur_env_vars.append({"name": "JOB_NAME", "value": f"{cur_job_name}"})
                for envar in cur_env_vars:  # set env vars as enviroment variables
                    os.environ[envar["name"]] = envar["value"]
                    print(f"""{envar["name"]} = {envar["value"]}""")
                # add them to the export command of slurm
                export_str = "--export=ALL,"
                for envar in cur_env_vars:
                    export_str += f"""{envar["name"]}="{envar["value"]}","""
                export_str = export_str[:-1]

                # add 5 minutes of overhead
                time_limit = int(self.sims_per_job * self.time_per_sim) + 5
                # submit job (idea: use slumpy to get the "depend on")
                # command = [
                #    "sbatch",
                #    export_str,
                #    f"--array=1-{self.num_jobs}",
                #    f"--mem={self.memory}M",  # memory per node
                #    # use vcpu here ? no need afaik.
                #    # time:  Acceptable time formats include "minutes", ... "days-hours:minutes" or  #J-H:m:s.
                #    f"--time={time_limit}",¨
                #                    #f"--mem={self.memory}M",  # memory per node
                # use vcpu here ? no need afaik.
                # time:  Acceptable time formats include "minutes", ... "days-hours:minutes" or  #J-H:m:s.
                #    f"--job-name={cur_job_name}",
                #    f"{os.path.dirname(os.path.realpath(__file__))}/inference_job.run",
                # ]
                command = f"sbatch {export_str} --array=1-{self.num_jobs} --mem={self.memory}M --time={time_limit} --job-name={cur_job_name} --output=log_inference_{self.run_id}_{cur_job_name}_%a.txt {os.path.dirname(os.path.realpath(__file__))}/SLURM_inference_job.run"

                print("slurm command to be run >>>>>>>> ")
                print(command)
                print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")

                returncode, stdout, stderr = gempyor.utils.command_safe_run(
                    command, command_name="sbatch", fail_on_fail=True
                )
                slurm_job_id = stdout.decode().split(" ")[-1][:-1]
                print(f">>> SUCCESS SCHEDULING JOB. Slurm job id is {slurm_job_id}")

                postprod_command = f"""sbatch {export_str} --dependency=afterany:{slurm_job_id} --mem={12000}M --time={240} --job-name=post-{cur_job_name} --output=log_postprod_{self.run_id}_{cur_job_name}.txt {os.path.dirname(os.path.realpath(__file__))}/SLURM_postprocess_runner.run"""
                print("post-processing command to be run >>>>>>>> ")
                print(postprod_command)
                print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")
                returncode, stdout, stderr = gempyor.utils.command_safe_run(
                    postprod_command, command_name="sbatch postprod", fail_on_fail=True
                )
                postprod_job_id = stdout.decode().split(" ")[-1][:-1]
                print(
                    f">>> SUCCESS SCHEDULING POST-PROCESSING JOB. Slurm job id is {postprod_job_id}"
                )

            elif self.batch_system == "local":
                cur_env_vars.append({"name": "JOB_NAME", "value": f"{cur_job_name}"})
                print(f"--- env var to set ---")
                for envar in cur_env_vars:  # set env vars as enviroment variables
                    os.environ[envar["name"]] = envar["value"]
                    print(f"""export {envar["name"]}="{envar["value"]}" """)
                print(f"--- end env var to set ---")

            # On aws: create all other jobs + the copy job. slurm script is only one block and copies itself at the end.
            if self.batch_system == "aws":
                block_idx = 1
                while block_idx < self.num_blocks:
                    cur_env_vars = base_env_vars.copy()
                    cur_env_vars.append({"name": "FLEPI_SEIR_SCENARIOS", "value": s})
                    cur_env_vars.append({"name": "FLEPI_OUTCOME_SCENARIOS", "value": d})
                    cur_env_vars.append(
                        {"name": "FLEPI_PREFIX", "value": f"{config['name']}_{s}_{d}"}
                    )
                    cur_env_vars.append(
                        {"name": "FLEPI_BLOCK_INDEX", "value": f"{block_idx+1}"}
                    )
                    cur_env_vars.append(
                        {"name": "FLEPI_RUN_INDEX", "value": f"{self.run_id}"}
                    )
                    cur_env_vars.append(
                        {"name": "OLD_FLEPI_RUN_INDEX", "value": f"{self.run_id}"}
                    )
                    cur_env_vars.append(
                        {"name": "LAST_JOB_OUTPUT", "value": f"{s3_results_path}/"}
                    )
                    cur_env_vars.append(
                        {"name": "JOB_NAME", "value": f"{cur_job_name}_block{block_idx}"}
                    )
                    cur_job = batch_client.submit_job(
                        jobName=f"{cur_job_name}_block{block_idx}",
                        jobQueue=cur_job_queue,
                        arrayProperties={"size": self.num_jobs},
                        dependsOn=[{"jobId": last_job["jobId"], "type": "N_TO_N"}],
                        jobDefinition=self.batch_job_definition,
                        containerOverrides={
                            "vcpus": self.vcpus,
                            "memory": self.vcpus * self.memory,
                            "environment": cur_env_vars,
                            "command": command,
                        },
                        retryStrategy={"attempts": 3},
                    )
                    last_job = cur_job
                    block_idx += 1

                # Prepare and launch the num_jobs via AWS Batch.
                cp_env_vars = [
                    {"name": "S3_RESULTS_PATH", "value": s3_results_path},
                    {"name": "LAST_JOB_OUTPUT", "value": f"{s3_results_path}"},
                    {"name": "NSLOT", "value": str(self.num_jobs)},
                ]

                copy_script_path = f"s3://{self.s3_bucket}/{job_name}-copy.sh"
                s3_cp_run_script = f"aws s3 cp {copy_script_path} $PWD/run-flepimop-copy"
                cp_command = [
                    "sh",
                    "-c",
                    f"{s3_cp_run_script}; /bin/bash $PWD/run-flepimop-copy",
                ]

                run_id_restart = self.run_id

                # Joseph: I feel like inference_copy does not do anything, a
                # there is no folder in s3 that is called final_output...
                copy_job = batch_client.submit_job(
                    jobName=f"{cur_job_name}_copy",
                    jobQueue=cur_job_queue,
                    jobDefinition=self.batch_job_definition,
                    dependsOn=[{"jobId": last_job["jobId"]}],
                    containerOverrides={
                        "vcpus": 1,
                        "environment": cp_env_vars,
                        "command": cp_command,
                    },
                    retryStrategy={"attempts": 3},
                )

        print(f" --------- COPY TO #flepimop_production message below ---------")
        print(f"Launching {cur_job_name} on {self.batch_system}...")
        print(
            f" >> Job array: {self.num_jobs} slot(s) X {self.num_blocks} block(s) of {self.sims_per_job} simulation(s) each."
        )
        if not (self.restart_from_location is None):
            em = ""
            if self.resume_discard_seeding:
                em = f", discarding seeding results."
            print(
                f" >> Resuming from run id is {self.restart_from_run_id} located in {self.restart_from_location}{em}"
            )
        if self.batch_system == "aws":
            print(f" >> Final output will be: {s3_results_path}/model_output/")
        elif self.batch_system == "slurm":
            print(f" >> Final output will be: {fs_results_path}/model_output/")
            if self.s3_upload:
                print(
                    f" >> Final output will be uploaded to {s3_results_path}/model_output/"
                )
        if self.continuation:
            print(
                f" >> Continuing from run id is {self.continuation_run_id} located in {self.continuation_location}"
            )
        print(f" >> Run id is {self.run_id}")
        print(f" >> config is {config_filepath.split('/')[-1]}")
        flepimop_branch = subprocess.getoutput(
            f"cd {self.flepi_path}; git rev-parse --abbrev-ref HEAD"
        )
        data_branch = subprocess.getoutput(
            f"cd {self.data_path}; git rev-parse --abbrev-ref HEAD"
        )
        data_hash = subprocess.getoutput(f"cd {self.data_path}; git rev-parse HEAD")
        flepimop_hash = subprocess.getoutput(f"cd {self.flepi_path}; git rev-parse HEAD")
        print(f""" >> FLEPIMOP branch is {flepimop_branch} with hash {flepimop_hash}""")
        print(f""" >> DATA branch is {data_branch} with hash {data_hash}""")
        print(f" ------------------------- END -------------------------")
        # add in csp and data path branch.

        # TODO add if Flu or Not, add validation date


if __name__ == "__main__":
    launch_batch()
