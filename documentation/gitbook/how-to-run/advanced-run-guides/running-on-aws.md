---
description: using Docker container
---

# Running on AWS 🌳

## 🖥 Start and access AWS submission box

**Spin up an Ubuntu submission box if not already running**. To do this, log onto AWS Console and start the EC2 instance.

**Update IP address in .ssh/config file.** To do this, open a terminal and type the command below. This will open your config file where you can change the IP to the IP4 assigned to the AWS EC2 instance (see AWS Console for this):

```
notepad .ssh/config
```

**SSH into the box.** In the terminal, SSH into your box. Typically we name these instances "staging", so usually the command is:

```
ssh staging
```

## 🧱 Setup

Now you should be logged onto the AWS submission box. If you haven't yet, set up your directory structure.

### 🗂 Create the directory structure (ONCE PER USER)

Type the following commands:

```bash
git clone https://github.com/HopkinsIDD/flepiMoP.git
git clone https://github.com/HopkinsIDD/Flu_USA.git
git clone https://github.com/HopkinsIDD/COVID19_USA.git
cd COVID19_USA
git clone https://github.com/HopkinsIDD/flepiMoP.git
cd ..
# or any other data directories
```

{% hint style="warning" %}
Note that the repository is cloned **nested,** i.e the `flepiMoP` repository is _INSIDE_ the data repository.
{% endhint %}

Have your Github ssh key passphrase handy so you can paste it when prompted (possibly multiple times) with the git pull command. _Alternatively, you can add your github key to your batch box so you don't have to enter your token 6 times per day._

```bash
git config --global credential.helper store
git config --global user.name "{NAME SURNAME}"
git config --global user.email YOUREMAIL@EMAIL.COM
git config --global pull.rebase false # so you use merge as the default reconciliation method
```

{% code overflow="wrap" %}
```
cd COVID19_USA
git config --global credential.helper cache
git pull 
git checkout main
git pull

cd flepiMoP
git pull	
git checkout main
git pull
cd .. 
```
{% endcode %}

## 🚀 Run inference using AWS (do everytime)

### 🛳 Initiate the Docker

Start up and log into the docker container, and run setup scripts to setup the environment. This setup code links the docker directories to the existing directories on your box. As this is the case, you should not run job submission simultaneously using this setup, as one job submission might modify the data for another job submission.

{% code overflow="wrap" %}
```
sudo docker pull hopkinsidd/flepimop:latest
sudo docker run -it \
  -v /home/ec2-user/COVID19_USA:/home/app/drp/COVID19_USA \
  -v /home/ec2-user/flepiMoP:/home/app/drp/flepiMoP \
  -v /home/ec2-user/.ssh:/home/app/.ssh \
hopkinsidd/flepimop:latest 
```
{% endcode %}

### Setup environment

To set up the environment for your run, run the following commands. These are specific to _your run_, i.e., change `VALIDATION_DATE`, `FLEPI_RUN_INDEX` and `RESUME_LOCATION` as required. If submitting multiple jobs, it is recommended to split jobs between 2 queues: `Compartment-JQ-1588569569` and `Compartment-JQ-1588569574`.

NOTE: If you are not running a _resume run_, DO NOT export the environmental variable `RESUME_LOCATION`.

<pre class="language-bash"><code class="lang-bash">cd ~/drp
export CENSUS_API_KEY={A CENSUS API KEY}
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export COMPUTE_QUEUE="Compartment-JQ-1588569574"

export VALIDATION_DATE="2023-01-29"
<strong>export RESUME_LOCATION=s3://idd-inference-runs/USA-20230122T145824
</strong>export FLEPI_RUN_INDEX=FCH_R16_lowBoo_modVar_ContRes_blk4_Jan29_tsvacc

export CONFIG_PATH=config_FCH_R16_lowBoo_modVar_ContRes_blk4_Jan29_tsvacc.yml
</code></pre>

Additionally, if you want to profile how the model is using your memory resources during the run, run the following commands

```bash
export FLEPI_MEM_PROFILE=TRUE
export FLEPI_MEM_PROF_ITERS=50
```

Then prepare the pipeline directory (if you have already done that and the pipeline hasn't been updated (`git pull` says it's up to date). You need to set $DATA\_PATH to your data folder. For a COVID-19 run, do:

```bash
cd ~/drp
export DATA_PATH=$(pwd)/COVID19_USA
export GT_DATA_SOURCE="csse_case, fluview_death, hhs_hosp"
```

for Flu do:

```bash
cd ~/drp
export DATA_PATH=$(pwd)/Flu_USA
```

Now for any type of run:

```bash
cd $DATA_PATH
export FLEPI_PATH=$(pwd)/flepiMoP
cd $FLEPI_PATH
git checkout main
git pull
git config --global credential.helper 'cache --timeout 300000'

#install gempyor and the R modules. There should be no error, please report if not.
# Sometimes you might need to run the next line two times because inference depends
# on report.generation, which is installed later, in alphabetical order.
# (or if you know R well enough to fix that 😊)

Rscript build/local_install.R # warnings are ok; there should be no error.
   python -m pip install --upgrade pip &
   pip install -e flepimop/gempyor_pkg/ &
   pip install boto3 &
   cd ..

```

For now, just in case: update the `arrow` package from 8.0.0 in the docker to 11.0.3 ;

Now flepiMoP is ready 🎉 ;

```bash
cd $DATA_PATH
git pull 
git checkout main
```

Do some clean-up before your run. The fast way is to restore the `$DATA_PATH` git repository to its blank states (⚠️ removes everything that does not come from git):

<pre class="language-bash"><code class="lang-bash"><strong>git reset --hard &#x26;&#x26; git clean -f -d  # this deletes everything that is not on github in this repo !!!
</strong></code></pre>

<details>

<summary>I want more control over what is deleted</summary>

if you prefer to have more control, delete the files you like, e.g

If you still want to use git to clean the repo but want finer control or to understand how dangerous is the command, [read this](https://stackoverflow.com/questions/1090309/git-undo-all-working-dir-changes-including-new-files).

```bash
rm -rf model_output data/us_data.csv data-truth &&
   rm -rf data/mobility_territories.csv data/geodata_territories.csv &&
   rm -rf data/seeding_territories.csv && 
   rm -rf data/seeding_territories_Level5.csv data/seeding_territories_Level67.csv

# don't delete model_output if you have another run in //
rm -rf $DATA_PATH/model_output
```

</details>

Then run the preparatory data building scripts and you are good

```bash
export CONFIG_PATH=config_FCH_R16_lowBoo_modVar_ContRes_blk4_Jan29_tsvacc.yml # if you haven't already done this
Rscript $FLEPI_PATH/datasetup/build_US_setup.R

# For covid do
Rscript $FLEPI_PATH/datasetup/build_covid_data.R

# For Flu do
Rscript $FLEPI_PATH/datasetup/build_flu_data.R
```

Now you may want to test that it works :

```bash
$ flepimop-inference-main -c $CONFIG_PATH -j 1 -n 1 -k 1 
```

If this fails, you may want to investigate this error. In case this succeeds, then you can proceed by first deleting the model\_output:

```
rm -r model_output
```

### Launch your inference batch job on AWS

Assuming that the initial test simulation finishes successfully, you will now enter credentials and submit your job onto AWS batch. Enter the following command into the terminal:

```
aws configure
```

You will be prompted to enter the following items. These can be found in a file you received from Shaun called `new_user_credentials.csv`.

* Access key ID when prompted
* Secret access key when prompted
* Default region name: us-west-2
* Default output: Leave blank when this is prompted and press enter (The Access Key ID and Secret Access Key will be given to you once in a file)

Now you're fully set to go 🎉

To launch the whole inference batch job, type the following command:

{% code overflow="wrap" %}
```bash
python $FLEPI_PATH/batch/inference_job_launcher.py --aws -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic 
```
{% endcode %}

This command infers everything from you environment variables, if there is a resume or not, what is the run\_id, etc., and the default is to carry seeding if it is a resume (see below for alternative options).

If you'd like to have more control, you can specify the arguments manually:

<pre class="language-bash"><code class="lang-bash"><strong>python $FLEPI_PATH/batch/inference_job_launcher.py --aws \ ## FIX THIS TO REFLECT AWS OPTIONS
</strong><strong>                    -c $CONFIG_PATH \
</strong><strong>                    -p $FLEPI_PATH \
</strong><strong>                    --data-path $DATA_PATH \
</strong><strong>                    --upload-to-s3 True \
</strong><strong>                    --id $FLEPI_RUN_INDEX \
</strong><strong>                    --restart-from-location $RESUME_LOCATION
</strong></code></pre>

We allow for a number of different jobs, with different setups, e.g., you may _not_ want to carry seeding. Some examples of appropriate setups are given below. No modification of these code chunks should be required ;

{% tabs %}
{% tab title="Standard" %}
<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong>cd $DATA_PATH 
</strong><strong>
</strong>$FLEPI_PATH/batch/inference_job_launcher.py --aws -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic
</code></pre>
{% endtab %}

{% tab title="Non-inference" %}
{% code overflow="wrap" %}
```bash
cd $DATA_PATH 

$FLEPI_PATH/batch/inference_job_launcher.py --aws -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic -j 1 -k 1
```
{% endcode %}
{% endtab %}

{% tab title="Resume" %}
> NOTE: _Resume_ and _Continuation Resume_ runs are currently submitted the same way, resuming from an S3 that was generated manually. Typically we will also submit any _Continuation Resume_ run specifying `--resume-carry-seeding` as starting seeding conditions will be manually constructed and put in the S3.

**Carrying seeding** (_do this to use seeding fits from resumed run_):

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong>cd $DATA_PATH 
</strong><strong>
</strong>$FLEPI_PATH/batch/inference_job_launcher.py --aws -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-carry-seeding --restart-from-location $RESUME_LOCATION
</code></pre>

**Discarding seeding** (_do this to refit seeding again_)_:_

{% code overflow="wrap" %}
```bash
cd $DATA_PATH 

$COVID_PATH/batch/inference_job_launcher.py --aws -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-discard-seeding --restart-from-location $RESUME_LOCATION
```
{% endcode %}

**Single Iteration + Carry seeding** (_do this to produce additional scenarios where no fitting is required_)_:_

{% code overflow="wrap" %}
```bash
cd $DATA_PATH 

$COVID_PATH/batch/inference_job_launcher.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-carry-seeding --restart-from-location $RESUME_LOCATION
```
{% endcode %}
{% endtab %}
{% endtabs %}

### Document the submission

After the job is successfully submitted, you will now be in a new branch of the data repo. Commit the ground truth data files to the branch on github and then return to the main branch:

<pre><code>git add data/ 
git config --global user.email "[email]" 
git config --global user.name "[github username]" 
git commit -m"scenario run initial" 
branch=$(git branch | sed -n -e 's/^\* \(.*\)/\1/p')
<strong>git push --set-upstream origin $branch
</strong>
git checkout main
git pull
</code></pre>

Send the submission information to slack so we can identify the job later. Example output:

```
Launching USA-20230426T135628_inference_med on aws...
 >> Job array: 300 slot(s) X 5 block(s) of 55 simulation(s) each.
 >> Final output will be: s3://idd-inference-runs/USA-20230426T135628/model_output/
 >> Run id is SMH_R17_noBoo_lowIE_phase1_blk1
 >> config is config_SMH_R17_noBoo_lowIE_phase1_blk1.yml
 >> FLEPIMOP branch is main with hash 3773ed8a20186e82accd6914bfaf907fd9c52002
 >> DATA branch is R17 with hash 6f060fefa9784d3f98d88a313af6ce433b1ac913
```
