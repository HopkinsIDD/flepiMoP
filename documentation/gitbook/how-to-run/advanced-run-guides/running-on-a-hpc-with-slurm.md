---
description: Tutorial on how to install and run flepiMoP on a supported HPC with slurm.
---

# Running On A HPC With Slurm

These details cover how to install and initialize `flepiMoP` on an HPC environment and submit a job with slurm.

{% hint style="warning" %}
Currently only JHU's Rockfish and UNC's Longleaf HPC clusters are supported. If you need support for a new HPC cluster please file an issue in [the `flepiMoP` GitHub repository](https://github.com/HopkinsIDD/flepiMoP/issues).
{% endhint %}

## Installing `flepiMoP`

This task needs to be ran once to do the initial install of `flepiMoP`.

{% hint style="info" %}
On JHU's Rockfish you'll need to run these steps in a slurm interactive job. This can be launched with `/data/apps/helpers/interact -n 4 -m 12GB -t 4:00:00`, but please consult the [Rockfish user guide](https://www.arch.jhu.edu/guide/) for up to date information.
{% endhint %}

Obtain a temporary clone of the `flepiMoP` repository. The install script will place a permanent clone in the correct location once ran. You may need to take necessary steps to setup git on the HPC cluster being used first before running this step.

```
$ git clone git@github.com:HopkinsIDD/flepiMoP.git --depth 1
Cloning into 'flepiMoP'...
remote: Enumerating objects: 487, done.
remote: Counting objects: 100% (487/487), done.
remote: Compressing objects: 100% (424/424), done.
remote: Total 487 (delta 59), reused 320 (delta 34), pack-reused 0 (from 0)
Receiving objects: 100% (487/487), 84.04 MiB | 41.45 MiB/s, done.
Resolving deltas: 100% (59/59), done.
Updating files: 100% (411/411), done.
```

Run the `hpc_install_or_update.sh` script, substituting `<cluster-name>` with either `rockfish` or `longleaf`. This script will prompt the user asking for the location to place the `flepiMoP` clone and the name of the conda environment that it will create. If this is your first time using this script accepting the defaults is the quickest way to get started. Also, expect this script to take a while the first time that you run it.

```
$ ./flepiMoP/build/hpc_install_or_update.sh <cluster-name>
```

Remove the temporary clone of the `flepiMoP` repository created before. This step is not required, but does help alleviate confusion later.

```
$ rm -rf flepiMoP/
```

## Updating `flepiMoP`

Updating `flepiMoP` is designed to work just the same as installing `flepiMoP`. Make sure that your clone of the `flepiMoP` repository is set to the branch your working with (if doing development or operations work) and then run the `hpc_install_or_update.sh` script, substituting `<cluster-name>` with either `rockfish` or `longleaf`.

```
$ ./flepiMoP/build/hpc_install_or_update.sh <cluster-name>
```

## Initialize The Created `flepiMoP` Environment

These steps to initialize the environment need to run on a per run or as needed basis.

Change directory to where a full clone of the `flepiMoP` repository was placed (it will state the location in the output of the script above). And then run the `hpc_init.sh` script, substituting `<cluster-name>` with either `rockfish` or `longleaf`. This script will assume the same defaults as the script before for where the `flepiMoP` clone is and the name of the conda environment. This script will also ask about a project directory and config, if this is your first time initializing `flepiMoP` it might be helpful to clone [the `flepimop_sample` GitHub repository](https://github.com/HopkinsIDD/flepimop\_sample) to the same directory to use as a test.

```
$ source batch/hpc_init.sh <cluster-name>
```

Upon completing this script it will output a sample set of commands to run to quickly test if the installation/initialization has gone okay.&#x20;

## Submitting A Batch Inference Job To Slurm

When an inference batch job is launched, a few post processing scripts are called to run automatically `postprocessing-scripts.sh.` You can manually change what you want to run by editing this script.

A batch job can can be submitted after this by running the following:

<pre><code><strong>$ cd $PROJECT_PATH
</strong><strong>$ python $FLEPI_PATH/batch/inference_job_launcher.py --slurm 2>&#x26;1 | tee $FLEPI_RUN_INDEX_submission.log
</strong></code></pre>

This launches a batch job to your HPC, with each slot on a separate node. This command attempts to infer the required arguments from your environment variables (i.e. if there is a resume or not, what is the run\_id, etc.). The part after the "2" makes sure this file output is redirected to a script for logging, but has no impact on your submission.

If you'd like to have more control, you can specify the arguments manually:

```
$ python $FLEPI_PATH/batch/inference_job_launcher.py --slurm \
                    -c $CONFIG_PATH \
                    -p $FLEPI_PATH \
                    --data-path $DATA_PATH \
                    --upload-to-s3 True \
                    --id $FLEPI_RUN_INDEX \
                    --fs-folder /scratch4/primary-user/flepimop-runs \
                    --restart-from-location $RESUME_LOCATION
```

More detailed arguments and advanced usage of the `inference_job_launcher.py` script please refer to the `--help`.&#x20;

After the job is successfully submitted, you will now be in a new branch of the project repository. For documentation purposes, we recommend committing the ground truth data files to the branch on GitHub substituting `<your-commit-message>` with a description of the contents:

<pre><code><strong>$ git add data/ 
</strong>$ git commit -m "&#x3C;your-commit-message>" 
$ git push --set-upstream origin $( git rev-parse --abbrev-ref HEAD )
</code></pre>

## Monitoring Submitted Jobs

During an inference batch run, log files will show the progress of each array/slot. These log files will show up in your project directory and have the file name structure:

```
log_{scenario}_{FLEPI_RUN_INDEX}_{JOB_NAME}_{seir_modifier_scenario}_{outcome_modifiers_scenario}_{array number}.txt
```

To view these as they are being written, type:

```
cat log_{scenario}_{FLEPI_RUN_INDEX}_{JOB_NAME}_{seir_modifier_scenario}_{outcome_modifiers_scenario}_{array number}.txt
```

or your file viewing command of choice. Other commands that are helpful for monitoring the status of your runs (note that `<Job ID>` here is the SLURM job ID, _not_ the `JOB_NAME` set by flepiMoP):

| SLURM command      | What does it do?                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `squeue -u $USER`  | Displays the names and statuses of all jobs submitted by the user. Job status might be: R: running, P: pending. |
| `seff <Job ID>`    | Displays information related to the efficiency of resource usage by the job                                     |
| `sacct`            | Displays accounting data for all jobs and job steps                                                             |
| `scancel <Job ID>` | This cancels a job. If you want to cancel/kill all jobs submitted by a user, you can type `scancel -u $USER`    |



## Other Tips & Tricks

### Moving files to your local computer <a href="#moving-files-to-your-local-computer" id="moving-files-to-your-local-computer"></a>

Often you'll need to move files back and forth between your HPC and your local computer. To do this, your HPC might suggest [Filezilla](https://filezilla-project.org/) or [Globus file manager](https://www.globus.org/). You can also use commands `scp` or `rsync` (check what works for your HPC).

```
# To get files from HPC to local computer
scp -r <user>@<data transfer node>:"<file path of what you want>" <where you want to put it in your local>
rsync 

# To get files from local computer to HPC
rsync local-file user@remote-host:remote-file
```

### Other helpful commands <a href="#other-helpful-commands" id="other-helpful-commands"></a>

If your system is approaching a file number quota, you can find subfolders that contain a large number of files by typing:

```
find . -maxdepth 1 -type d | while read -r dir
 do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done 
```
