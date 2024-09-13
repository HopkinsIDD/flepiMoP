# Running on UNC's Longleaf Cluster with Slurm

## Background

Please read: https://help.rc.unc.edu/getting-started-on-longleaf.


## Initial Setup

These steps are for getting the code and initializing the modules needed for UNC's longleaf. These only need to be run once.

1. `ssh` into longleaf via `ssh <onyen>@longleaf.unc.edu` where `<onyen>` is your onyen user id. I.e. `ssh twillard@longleaf.unc.edu`.
2. Load git so you can get a copy of `flepiMoP` on the longleaf cluster. Execute the following commands after `ssh`-ing in:

```bash
module load git
module load git-lfs
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

4. Run the Longleaf init script:

```bash
source /users/<o>/<n>/<onyen>/flepiMoP/batch/slurm_init_longleaf.sh
```

NOTE: That if a `module purge` command is run this init script will need to be run again.


## Per Run Setup

These steps are needed prior to running `flepiMoP` and will need to be ran once per a session.
