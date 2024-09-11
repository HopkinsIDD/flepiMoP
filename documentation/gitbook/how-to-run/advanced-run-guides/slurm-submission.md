---
description: HPC using the slurm workload manager
---

# Running on SLURM HPC

## 🧱 Setting up your environment (only once per user)

You will need to load some modules in order to run the code. These include `gcc`, `anaconda`, and `git`. You should be able to do this using the `module` commands.

* First purge the current modules:

```bash
module purge
```

* Now find the name of the available gcc 9.x module and load that:

{% code overflow="wrap" fullWidth="false" %}
```bash
module spider gc
module load <gcc 9.x version>
```
{% endcode %}

* Now load anaconda:

<pre class="language-bash"><code class="lang-bash"><strong>module spider anaconda
</strong>module load &#x3C;latest anaconda environment>
</code></pre>

Now you need to create the conda environment. You will create the environment in two shorter commands, installing the python and R stuff separately. This can be extremely long if done in one command, so doing it in two helps. This command is quite long you'll have the time to brew some nice coffee ☕️:

{% code overflow="wrap" %}
```bash
# install all python stuff first
conda create -c conda-forge -n flepimop-env numba pandas numpy seaborn tqdm matplotlib click confuse pyarrow sympy dask pytest scipy graphviz emcee xarray boto3 slack_sdk
```
{% endcode %}

The next step in preparing your environment is to install the necessary R packages. First, activate your environment, launch R and then install the following packages.

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong>conda activate flepimop-env # this launches the environment you just created
</strong>
R # to launch R from command line

<strong># while in R
</strong><strong>install.packages(c("readr","sf","lubridate","tidyverse","gridExtra","reticulate","truncnorm","xts","ggfortify","flextable","doParallel","foreach","optparse","arrow","devtools","cowplot","ggraph"))
</strong></code></pre>

You are now ready to run using SLURM!

## 🗂️ Files and folder organization

HPC administrators are likely to provide different partitions with different properties for your use. We recommend a partition that supports a **shared environment** and **storage intensive** needs.

For example, we use a scratch partition with 20T of space, which has a primary user, and other users share this storage. In our setup this looks like: `/scratch4/primary-user/` . We will describe this setup as an example, but note that your HPC setup might be different (if so, change the relevant paths).

We recommend setting up two folders: one containing the code, and one for storing the model output. Helper scripts are setup to use this code structure.

* **code folder:** `/scratch4/primary-user/flepimop-code`\
  Check the [Before any run](../before-any-run.md) page for how to set up the appropriate folders or repositories. For our purposes, a subfolder is also setup for each user. This allows users to be able to manage their own code and launch their own runs. For example, for a user, this might look like\
  `/scratch4/primary-user/flepimop-code/$USER/flepiMoP`\
  `/scratch4/primary-user/flepimop-code/$USER/flepimop-sample`

{% hint style="warning" %}
Note that the repository should be cloned **flat,** i.e the `flepiMoP` repository is at the same level as the data repository, not inside it.
{% endhint %}

* **output folder:**`/scratch4/primary-user/flepimop-runs`\
  After an inference run finishes, it's output and the logs files are copied from the project folder where the model is run from, to `scratch4/primary-user/flepimop-runs/$JOB_NAME` where `JOB_NAME` is an environmental variable set up within the submission script (described below; this is usually of the form `USA-DATE`).

<details>

<summary>Storing model outputs on Amazon Web Services s3 storage 🪣</summary>

We provide scripts that also store model outputs on Amazon Web Services (AWS) s3 storage. If you are storing on your HPC or locally, skip this step.\
\
In order to push and pull model outputs to and from s3, setup your AWS credentials by:

<pre class="language-bash"><code class="lang-bash">cd ~ # go in your home directory
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install -i ~/aws-cli -b ~/aws-cli/bin
<strong>./aws-cli/bin/aws --version
</strong></code></pre>

Then run `./aws-cli/bin/aws configure` and use the following :

```
# AWS Access Key ID [None]: YOUR ID
# AWS Secret Access Key [None]: YOUR SECRET ID
# Default region name [None]: us-west-2
# Default output format [None]: json
```

</details>

## 🚀 Run inference using slurm (do everytime)

In your HPC system, enter the following command:

```
source /scratch4/primary-user/flepimop-code/$USER/flepiMoP/batch/slurm_init.sh
```

This will prepare the environment and setup variables for the validation date, the location of the model output from which you want to resume (this can be an S3 bucket, or a local path) and the run index for this run. If you don't want to set a variable, just hit enter.

<details>

<summary>what does this do || help: it returns an error</summary>

This script runs the following commands to setup up the environment, which you can run individually as well.

```bash
module purge
module load gcc/9.3.0
module load git
module load git-lfs
module load slurm
module load anaconda3/2022.05
conda activate flepimop-env
export CENSUS_API_KEY={A CENSUS API KEY}
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export FLEPI_PATH=/scratch4/primary-user/flepimop-code/$USER/flepiMoP

# And then it asks you some questions to setup some environment variables
```

and then it does some prompts to fix the following 3 environment variables. You can skip this part and do it later manually.

```bash
export VALIDATION_DATE="2023-01-29"
export RESUME_LOCATION=s3://idd-inference-runs/USA-20230122T145824
export FLEPI_RUN_INDEX=inference_test
```

</details>

{% hint style="info" %}
Check that the conda environment is activated: you should see`(flepimop-env)` on the left of your command-line prompt.
{% endhint %}

Then prepare the pipeline directory (if you have already done that and the pipeline hasn't been updated (`git pull` says it's up to date) then you can skip these steps.

### Define environment variables

Create environmental variables for the paths to the flepiMoP code folder and the project folder:

```bash
cd /scratch4/primary-user/flepimop-code/$USER # move to the directory where all your code is stored
export FLEPI_PATH=$(pwd)/flepiMoP 
export DATA_PATH=$(pwd)/flepimop-sample  # whatever your project directory is called
```

Go into the code directory and do the installation the R and Python code packages:

```bash
cd $FLEPI_PATH # move to the flepimop directory
Rscript build/local_install.R # Install R packages
pip install --no-deps -e flepimop/gempyor_pkg/ # Install Python package gempyor
```

Each installation step may take a few minutes to run.

### Run the code

Everything is now ready 🎉 The next step depends on what sort of simulation you want to run: One that includes inference (fitting model to data) or only a forward simulation (non-inference). Inference is run from R, while forward-only simulations are run directly from the Python package `gempyor`.

In either case, navigate to the project directory and make sure to delete any old model output files that are there. Note that in the example config provided, the output is saved to `model_output`, but this might be otherwise defined in `config::model_output_dirname.`

```bash
cd $DATA_PATH       # goes to your project repository
rm -r model_output/ # delete the outputs of past run if there are
```

Set the path to your config

```bash
export CONFIG_PATH=config_example.yml # TO DO: ADD AN EXAMPLE
```

You may want to test that it works before launching a full batch:

```bash
Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R -c $CONFIG_PATH -j 1 -n 1 -k 1 
```

If this fails, you may want to investigate this error. In case this succeeds, then you can proceed (but remember to delete the existing model output).

```bash
rm -r model_output/ # delete the outputs of past run if they exist
```

### Launch your inference batch job

When an inference batch job is launched, a few post processing scripts are called to run automatically `postprocessing-scripts.sh.` You can manually change what you want to run by editing this script.

To launch the whole inference batch job, type the following command:

```bash
python $FLEPI_PATH/batch/inference_job_launcher.py --slurm 2>&1 | tee $FLEPI_RUN_INDEX_submission.log
```

This command infers everything from you environment variables, if there is a resume or not, what is the run\_id, etc. The part after the "2" makes sure this file output is redirected to a script for logging, but has no impact on your submission.

This launches a batch job to your HPC, with each slot on a separate node.

If you'd like to have more control, you can specify the arguments manually:

<pre class="language-bash"><code class="lang-bash"><strong>python $FLEPI_PATH/batch/inference_job_launcher.py --slurm \
</strong><strong>                    -c $CONFIG_PATH \
</strong><strong>                    -p $FLEPI_PATH \
</strong><strong>                    --data-path $DATA_PATH \
</strong><strong>                    --upload-to-s3 True \
</strong><strong>                    --id $FLEPI_RUN_INDEX \
</strong><strong>                    --fs-folder /scratch4/primary-user/flepimop-runs \
</strong><strong>                    --restart-from-location $RESUME_LOCATION
</strong></code></pre>

**Commit files to Github.** After the job is successfully submitted, you will now be in a new branch of the project repository. For documentation purposes, we recommend committing the ground truth data files to the branch on github:

<pre class="language-bash"><code class="lang-bash"><strong>git add data/ 
</strong>git commit -m "scenario run initial" 
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
git push --set-upstream origin $branch
</code></pre>

{% hint style="danger" %}
**DO NOT** move to a different git branch after this step, as the run will use data in the current directory.
{% endhint %}

## 🛠 Helpful tools and other notes

### Monitor your run

During an inference batch run, log files will show the progress of each array/slot. These log files will show up in your project directory and have the file name structure:

{% code overflow="wrap" %}
```
log_{scenario}_{FLEPI_RUN_INDEX}_{JOB_NAME}_{seir_modifier_scenario}_{outcome_modifiers_scenario}_{array number}.txt
```
{% endcode %}

To view these as they are being written, type

{% code overflow="wrap" %}
```bash
cat log_{scenario}_{FLEPI_RUN_INDEX}_{JOB_NAME}_{seir_modifier_scenario}_{outcome_modifiers_scenario}_{array number}.txt
```
{% endcode %}

Other commands that are helpful for monitoring the status of your runs (note that `<Job ID>` here is the SLURM job ID, _not_ the `JOB_NAME` set by flepiMoP):

<table><thead><tr><th width="250">SLURM command</th><th>What does it do?</th></tr></thead><tbody><tr><td><code>squeue -u $USER</code></td><td>Displays the names and statuses of all jobs submitted by the user. Job status might be: R: running, P: pending.</td></tr><tr><td><code>seff &#x3C;Job ID></code></td><td>Displays information related to the efficiency of resource usage by the job</td></tr><tr><td><code>sacct</code></td><td>Displays accounting data for all jobs and job steps</td></tr><tr><td><code>scancel &#x3C;Job ID></code></td><td>This cancels a job. If you want to cancel/kill all jobs submitted by a user, you can type <code>scancel -u $USER</code></td></tr></tbody></table>

### Running an interactive session

To check your code prior to submitting a large batch job, it's often helpful to run an interactive session to debug your code and check everything works as you want. On 🪨🐠 this can be done using `interact` like the below line, which requests an interactive session with 4 cores, 24GB of memory, for 12 hours.

```
interact -p defq -n 4 -m 24G -t 12:00:00
```

The options here are `[-n tasks or cores]`, `[-t walltime]`, `[-p partition]` and `[-m memory]`, though other options can also be included or modified to your requirements. More details can be found on the [ARCH User Guide](https://marcc.readthedocs.io/Slurm.html#request-interactive-jobs).

### Moving files to your local computer

Often you'll need to move files back and forth between your HPC and your local computer. To do this, your HPC might suggest [Filezilla](https://filezilla-project.org/) or [Globus file manager](https://www.globus.org/). You can also use commands `scp` or `rsync` (check what works for your HPC).

<pre class="language-bash"><code class="lang-bash"><strong># To get files from HPC to local computer
</strong><strong>scp -r &#x3C;user>@&#x3C;data transfer node>:"&#x3C;file path of what you want>" &#x3C;where you want to put it in your local>
</strong>rsync 

# To get files from local computer to HPC
rsync local-file user@remote-host:remote-file
</code></pre>

### Other helpful commands

If your system is approaching a file number quota, you can find subfolders that contain a large number of files by typing:

```bash
find . -maxdepth 1 -type d | while read -r dir
 do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done 
```
