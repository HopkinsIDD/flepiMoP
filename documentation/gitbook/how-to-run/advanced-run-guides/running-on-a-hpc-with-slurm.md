---
description: Tutorial on how to install and run flepiMoP on a supported HPC with slurm.
---

# Running On A HPC With Slurm

These details cover how to install and initialize `flepiMoP` on an HPC environment and submit a job with slurm.

{% hint style="warning" %}
Currently only JHU's Rockfish and UNC's Longleaf HPC clusters are supported. If you need support for a new HPC cluster please file an issue in [the `flepiMoP` GitHub repository](https://github.com/HopkinsIDD/flepiMoP/issues).
{% endhint %}

For getting access to one of the supported HPC environments please refer to the following documentation before continuing:

* [UNC's Longleaf Cluster](https://help.rc.unc.edu/getting-started-on-longleaf/) for UNC users, or
* [JHU's Rockfish Cluster](https://www.arch.jhu.edu/support/access/) for JHU users.

External users will need to consult with their PI contact at the respective institution.

## Installing `flepiMoP`

This task needs to be ran once to do the initial install of `flepiMoP`.

{% hint style="info" %}
On JHU's Rockfish you'll need to run these steps in a slurm interactive job. This can be launched with `/data/apps/helpers/interact -n 4 -m 12GB -t 4:00:00`, but please consult the [Rockfish user guide](https://www.arch.jhu.edu/guide/) for up to date information.
{% endhint %}

Download and run the the appropriate installation script with the following command:

```shell
$ curl -LsSf -o flepimop-install-<cluster-name> https://raw.githubusercontent.com/HopkinsIDD/flepiMoP/refs/heads/main/bin/flepimop-install-<cluster-name>
$ chmod +x flepimop-install-<cluster-name>
$ ./flepimop-install-<cluster-name>
```

Substituting `<cluster-name>` with either `rockfish` or `longleaf`. This script will install `flepiMoP` to the correct locations on the cluster. Once the installation is done the conda environment can be activated and the script can be removed with:

```shell
$ conda activate flepimop-env
$ rm flepimop-install-<cluster-name> flepimop-install
```

## Updating `flepiMoP`

Updating `flepiMoP` is designed to work just the same as installing `flepiMoP`. First change directory to your `flepiMoP` installation and then make sure that your clone of the `flepiMoP` repository is set to the branch you are working with (if doing development or operations work) and then run the `flepimop-install-<cluster-name>` script, substituting `<cluster-name>` with either `rockfish` or `longleaf`.

```
$ ./bin/flepimop-install-<cluster-name>
```

## Initialize The Created `flepiMoP` Environment

These steps to initialize the environment need to run on a per run or as needed basis.

Change directory to where a full clone of the `flepiMoP` repository was placed (it will state the location in the output of the script above). And then run the `hpc_init` script, substituting `<cluster-name>` with either `rockfish` or `longleaf`. This script will assume the same defaults as the script before for where the `flepiMoP` clone is and the name of the conda environment. This script will also ask about a project directory and config, if this is your first time initializing `flepiMoP` it might be helpful to use configs out of `flepiMoP/examples/tutorials` directory as a test.

```
$ ./batch/hpc_init <cluster-name>
```

Upon completing this script it will output a sample set of commands to run to quickly test if the installation/initialization has gone okay.

## Submitting A Batch Inference Job To Slurm

The main entry point for submitting batch inference jobs is the `flepimop batch-calibrate` action. This CLI tool will let you submit a job to slurm once logged into a cluster. For details on the available options please refer to `flepimop batch-calibrate --help`. As a quick example let's submit an R inference and EMCEE inference job. For the R inference run execute the following once logged into either longleaf or rockfish:

```
$ export PROJECT_PATH="$FLEPI_PATH/examples/tutorials/"
$ cd $PROJECT_PATH
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 100 \
    --time-limit 30min \
    --slurm \
    --nodes 4 \
    --cpus 1 \
    --memory 1G \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    -vvv \
    config_sample_2pop_inference.yml
```

This command will produce a large amount of output, due to `-vvv`. If you want to try the command without actually submitting the job you can pass the `--dry-run` option. This command will submit a job to calibrate the sample 2 population configuration which uses R inference. The R inference supports array jobs so each chain will be run on an individual node with 1 CPU and 1GB of memory a piece. Additionally the extra option allows you to provide additional info to the batch system, in this case what partition to submit the jobs to but email is also supported with slurm for notifications. After running this command you should notice the following output:

* `config_sample_2pop-YYYYMMDDTHHMMSS.yml`: This file contains the compiled config that is actually submitted for inference,
* `manifest.json`: This file contains a description of the submitted job with the command used, the job name, and `flepiMoP` and project git commit hashes,
* `slurm-*_*.out`: These files contain output from slurm for each of the array jobs submitted,
* `tmp*.sbatch`: Contains the generated file submitted to slurm with `sbatch`.

For operational runs these files should be committed to the checked out branch for archival/reproducibility reasons. Since this is just a test you can safely remove these files after inspecting them.

Now, let's submit an EMCEE inference job with the same tool. Importantly, the options we'll use won't change much because `flepimop batch-calibrate` is designed to provide a unified implementation independent interface.

```
$ export PROJECT_PATH="$FLEPI_PATH/examples/simple_usa_statelevel/"
$ cd $PROJECT_PATH
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 100 \
    --time-limit 30min \
    --slurm \
    --nodes 1 \
    --cpus 4 \
    --memory 8G \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    -vvv \
    simple_usa_statelevel.yml
```

One notable difference is, unlike R inference, EMCEE inference only supports running on 1 node so resources for this command are adjusted accordingly:

* Swapping 4 nodes with 1 cpu each to 1 node with 4 cpus, and
* Doubling the memory usage from 4 nodes with 1GB each for 4GB total to 1 node with 8GB for 8GB total.

The extra increase in memory is to run a configuration that is slightly more resource intense than the previous example. This command will also produce a similar set of record keeping files like before that you can safely remove after inspecting.

### Estimating Required Resources For A Batch Inference Job

When inspecting the output of `flepimop batch-calibrate --help` you may have noticed several options named `--estimate-*`. While not required for the smaller jobs above this tool has the ability to estimate the required resources to run a larger batch estimation job. The tool does this by running smaller jobs and then projecting the required resources for a large job from those smaller jobs. To use this feature provide the `--estimate` flag, a job size of the targeted job, resources for test jobs, and the following estimation settings:

* `--estimate-runs`: The number of smaller jobs to run to estimate the required resources from,
* `--estimate-interval`: The size of the prediction interval to use for estimating the resource/time limit upper bounds,
* `--estimate-vary`: The job size elements to vary when generating smaller jobs,
* `--estimate-factors`: The factors to use in projecting the larger scale estimation job,
* `--estimate-measurements`: The resources to estimate,
* `--estimate-scale-upper`: The scale factor to use to determine the largest sample job to generate, and
* `--estimate-scale-lower`: The scale factor to use to determine the smallest sample job to generate.

Effectively using these options requires some knowledge of the underlying inference method. Sticking with the simple usa state level example above try submitting the following command (after cleaning up the output from the previous example):

```
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 500 \
    --time-limit 2hr \
    --slurm \
    --nodes 1 \
    --cpus 4 \
    --memory 24GB \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    --estimate \
    --estimate-runs 6 \
    --estimate-interval 0.8 \
    --estimate-vary simulations \
    --estimate-factors simulations \
    --estimate-measurements time \
    --estimate-measurements memory \
    --estimate-scale-upper 5 \
    --estimate-scale-lower 10 \
    -vvv \
    simple_usa_statelevel.yml > simple_usa_statelevel_estimation.log 2>&1 & disown
```

In short, this command will submit 6 test jobs that will vary simulations and measure time and memory. The number of simulations will be used to project the required resources. The test jobs will range from 1/5 to 1/10 of the target job size. This command will take a bit to run because it needs to wait on these test jobs to finish running before it can do the analysis, so you can check on the progress by checking the output of the `simple_usa_statelevel_estimation.log` file.

Once this command finishes running you should notice a file called `USA_influpaint_resources.json`. This JSON file contains the estimated resources required to run the target job. You can submit the target job with the estimated resources by using the same command as before without the `--estimate-*` options and using the `--from-estimate` option to pull the information from the outputted file:

```
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 500 \
    --time-limit 2hr \
    --slurm \
    --nodes 1 \
    --cpus 4 \
    --memory 24GB \
    --from-estimate USA_influpaint_resources.json \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    -vvv \
    simple_usa_statelevel.yml
```

### Saving Model Outputs On Batch Inference Job Finish

For production runs it is particularly helpful to save the calibration results after a successful run to long term storage for safe keeping. To accomplish this `flepimop batch-calibrate` can chain a call to `flepimop sync` after a successful run via the `--sync-protocol` option. For more details on the `flepimop sync` command in general please refer to the [Synchronizing files: Syntax and Applications](../synchronization.md) guide.

For a quick demonstration of how to use this option start with the `config_sample_2pop_inference.yml` configuration file and add the following section:

```yaml
sync:
  rsync-model-output:
    type: rsync
    source: model_output
    target: /path/to/an/example-folder
  s3-model-output:
    type: s3sync
    source: model_output
    target: s3://my-bucket/and-sub-bucket
```

Where `/path/to/an/example-folder` and `s3://my-bucket/and-sub-bucket` are place holders for paths to your desired location. Importantly, note that there is no trailing slash on the `model_output` directory name. This will cause `flepimop sync` to sync the `model_output` directory itself and not just it's contents. You can also apply additional filters to the sync protocols here, say to limit the backed up model outputs to certain folders or exclude `llik` outputs, but the `--sync-protocol` option will add filters to limit the synced directories to those corresponding to the run submitted. Note that users do not need to specify run/job ids or configuration file names in the sync protocol. The `flepimop batch-calibrate` CLI will take advantage of `flepimop sync`'s options to set paths appropriately to accommodate for run/job ids.

Modifying the first `flepimop batch-calibrate` command from before:

```
$ export PROJECT_PATH="$FLEPI_PATH/examples/tutorials/"
$ cd $PROJECT_PATH
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 100 \
    --time-limit 30min \
    --slurm \
    --nodes 4 \
    --cpus 1 \
    --memory 1G \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    --sync-protocol <your sync protocol, either rsync-model-output or s3-model-output in this case> \
    -vvv \
    config_sample_2pop_inference.yml
```

This command will submit an array job just like before, but will also add a dependent job with the same name prefixed with 'sync_'. This should looks like:

```
[twillard@longleaf-login6 tutorials]$ squeue -p jlessler
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           2374868  jlessler sync_sam twillard PD       0:00      1 (Dependency)
         2374867_1  jlessler sample_2 twillard  R       2:26      1 g1803jles01
         2374867_2  jlessler sample_2 twillard  R       2:26      1 g1803jles01
         2374867_3  jlessler sample_2 twillard  R       2:26      1 g1803jles01
         2374867_4  jlessler sample_2 twillard  R       2:26      1 g1803jles01
```

After those jobs finish the results can be found in a subdirectory named after the job and whose contents will look like:

```
[twillard@longleaf-login6 sample_2pop-20250521T190823_Ro_all_test_limits]$ tree -L 4
.
├── manifest.json
└── model_output
    └── sample_2pop_Ro_all_test_limits
        └── sample_2pop-20250521T190823_Ro_all_test_limits
            ├── hnpi
            ├── hosp
            ├── hpar
            ├── init
            ├── llik
            ├── seir
            ├── snpi
            └── spar

11 directories, 1 file
```

Note that this contains the `model_output` directory but only limited to the batch run named 'sample_2pop-20250521T190823_Ro_all_test_limits' as well as a file called `manifest.json` which can be used to reproduce the run from scratch if needed. 

#### Saving Model Outputs To S3 For Hopkins Users

For Hopkins affiliated users there is a configuration file patch included with `flepiMoP` that can be used to add S3 syncing for model outputs to `s3://idd-inference-runs`. Taking the example before of running the `config_sample_2pop_inference.yml` configuration we can slightly modify the command to:

```
$ flepimop batch-calibrate \
    --blocks 1 \
    --chains 4 \
    --samples 20 \
    --simulations 100 \
    --time-limit 30min \
    --slurm \
    --nodes 4 \
    --cpus 1 \
    --memory 1G \
    --extra 'partition=<your partition, if relevant>' \
    --extra 'email=<your email, if relevant>' \
    --skip-checkout \
    --sync-protocol s3-idd-inference-runs \
    -vvv \
    config_sample_2pop_inference.yml $FLEPI_PATH/common/s3-idd-inference-runs.yml
```

This will take advantage of the patching abilities of the `flepimop batch-calibrate` to add a sync protocol named `s3-idd-inference-runs` that will save the results to the `s3://idd-inference-runs` bucket.
