# Running on UNC's Longleaf Cluster with Slurm

## Background

Please read: https://help.rc.unc.edu/getting-started-on-longleaf.


## Initial Setup

These steps are for getting the code and initializing the modules needed for UNC's longleaf. These only need to be run once.

1. `ssh` into longleaf via `ssh <onyen>@longleaf.unc.edu` where `<onyen>` is your onyen user id. I.e. `ssh twillard@longleaf.unc.edu`.
2. Load git so you can get a copy of `flepiMoP` on the longleaf cluster. Execute the following commands after `ssh`-ing in:

```bash
module load git
git clone --depth 1 https://github.com/HopkinsIDD/flepiMoP.git /users/<o>/<n>/<onyen>/flepiMoP
git clone --depth 1 https://github.com/HopkinsIDD/flepimop_sample.git /users/<o>/<n>/<onyen>/flepimop_sample
```
Where `<o>` is the first letter of your onyen and `<n>` is the second letter of your onyen (i.e. `/users/t/w/twillard/`). The `--depth 1` flag only clones the most recent commit. Since development work will likely not be occurring on the cluster having just the most recent version should be sufficient.

3. Create a `/users/<o>/<n>/<onyen>/slack_credentials.sh` file with the following in your terminal text editor of choice:

```bash
export SLACK_WEBHOOK="https://hooks.slack.com/services/*****/*****"
export SLACK_TOKEN="*****"
export DELPHI_API_KEY="*****"
export CENSUS_API_KEY="*****"
```

Replacing the `*****` with the appropriate secrets. This file contains sensitive credentials so make sure that the file has the appropriate permissions, typically `chmod 600 /users/<o>/<n>/<onyen>/slack_credentials.sh` will be sufficient.

4. Run the HPC installation script with the longleaf flag:

```bash
source /users/<o>/<n>/<onyen>/flepiMoP/build/hpc_install.sh longleaf
```

This script takes a while to run, but has some interactive components so make sure you have something else to do but don't walk away.


## Per Run Steps

These steps are needed prior to running `flepiMoP` and will need to be ran once per a session.

1. Run the pre-run Longleaf script:

```bash
source /users/<o>/<n>/<onyen>/flepiMoP/batch/slurm_prerun_longleaf.sh
```

If you just did the steps above you can skip this step, the init script calls the pre-run script for you.

2. Set the project path to the appropriate location, in this case the `flepimop_sample` directory:

```bash
export FLEPI_PATH=/users/<o>/<n>/<onyen>/flepiMoP
export PROJECT_PATH=/users/<o>/<n>/<onyen>/flepimop_sample
```

3. Set the configuration path environment variable. In this case using the `config_sample_2pop_inference.yml` file from the `flepimop_sample` reop.

```bash
export CONFIG_PATH=$PROJECT_PATH/config_sample_2pop_inference.yml
```

4. Now, let's test if this works. If you are just refreshing yourself on the steps to submit to slurm you can skip this.

``bash
cd $PROJECT_PATH
Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R -c $CONFIG_PATH -j 1 -n 1 -k 1
```

If you do run this command to test the installation up until this point make sure to delete the model output folder after successfully running.

```bash
rm -r model_output/
```
