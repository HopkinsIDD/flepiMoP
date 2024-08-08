# AWS Submission Instructions: COVID-19

{% hint style="warning" %}
This page, along with the other AWS run guides, are not deprecated in case we need to run `flepiMoP` on AWS again in the future, but also are not maintained as other platforms (such as longleaf and rockfish) are preferred for running production jobs.
{% endhint %}

### Step 1. Create the configuration file.

_see Building a configuration file_

### Step 2. Start and access AWS submission box

**Spin up an Ubuntu submission box if not already running**. To do this, log onto AWS Console and start the EC2 instance.

**Update IP address in .ssh/config file.** To do this, open a terminal and type the command below. This will open your config file where you can change the IP to the IP4 assigned to the AWS EC2 instance (see AWS Console for this):

```
notepad .ssh/config
```

**SSH into the box.** In the terminal, SSH into your box. Typically we name these instances "staging", so usually the command is:

```
ssh staging
```

### Step 3. Setup the environment

Now you should be logged onto the AWS submission box.

**Update the github repositories.** In the below example we assume you are running `main`branch in Flu\_USA and`main`branch in COVIDScenarioPipeline. This assumes you have already loaded the appropriate repositories on your EC2 instance. Have your Github ssh key passphrase handy so you can paste it when prompted (possibly multiple times) with the git pull command. _Alternatively, you can add your github key to your batch box so you do not have to log in repeated (see X)._

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

**Initiate the docker.** Start up and log into the docker container, pull the repos from Github, and run setup scripts to setup the environment. This setup code links the docker directories to the existing directories on your box. As this is the case, you should not run job submission simultaneously using this setup, as one job submission might modify the data for another job submission.

<pre data-overflow="wrap"><code>sudo docker pull hopkinsidd/flepimop:latest-dev
sudo docker run -it \
  -v /home/ec2-user/COVID19_USA:/home/app/drp/COVID19_USA \
  -v /home/ec2-user/flepiMoP:/home/app/drp/flepiMoP \
  -v /home/ec2-user/.ssh:/home/app/.ssh \
hopkinsidd/flepimop:latest-dev  
    
cd ~/drp/COVID19_USA
<strong>git config credential.helper store 
</strong>git pull 
git checkout main
git pull
git config --global credential.helper 'cache --timeout 300000'

<strong>cd ~/drp/flepiMoP 
</strong>git pull 
git checkout main
git pull 

Rscript build/local_install.R &#x26;&#x26; 
   python -m pip install --upgrade pip &#x26;&#x26;
   pip install -e flepimop/gempyor_pkg/ &#x26;&#x26; 
   pip install boto3 &#x26;&#x26; 
   cd ..
</code></pre>

### Step 4. Model Setup

To run the via AWS, we first run a setup run locally (in docker on the submission EC2 box).&#x20;

**Setup environment variables.** Modify the code chunk below and submit in the terminal. We also clear certain files and model output that get generated in the submission process. If these files exist in the repo, they may not get cleared and could cause issues. You need to modify the variable values in the _first 4 lines_ below. These include the `SCENARIO`, `VALIDATION_DATE`, `COVID_MAX_STACK_SIZE`, and `COMPUTE_QUEUE`. If submitting multiple jobs, it is recommended to split jobs between 2 queues: `Compartment-JQ-1588569569` and `Compartment-JQ-1588569574`.

{% tabs %}
{% tab title="Non-resume run" %}
If not resuming off previous run:

```
export FLEPI_RUN_INDEX=FCH_R16_lowBoo_modVar_ContRes_blk4_FCH_Dec11_tsvacc && 
   export VALIDATION_DATE="2022-12-11" && 
   export COVID_MAX_STACK_SIZE=1000 && 
   export COMPUTE_QUEUE="Compartment-JQ-1588569574" &&
   export CENSUS_API_KEY=c235e1b5620232fab506af060c5f8580604d89c1 && 
   export FLEPI_RESET_CHIMERICS=TRUE &&
   rm -rf model_output data/us_data.csv data-truth &&
   rm -rf data/mobility_territories.csv data/geodata_territories.csv &&
   rm -rf data/seeding_territories.csv && 
   rm -rf data/seeding_territories_Level5.csv data/seeding_territories_Level67.csv
```
{% endtab %}

{% tab title="Resume run" %}
If resuming from a previous run, there are an additional couple variables to set. This is the same for a _regular resume_ or _continuation resume._ Specifically:

* `RESUME_ID` - the `COVID_RUN_INDEX` from the run resuming from.
* `RESUME_S3` - the S3 bucket where this previous run is stored



```
export FLEPI_RUN_INDEX=FCH_R16_lowBoo_modVar_ContRes_blk4_Dec18_tsvacc && 
   export VALIDATION_DATE="2022-12-18" && 
   export COVID_MAX_STACK_SIZE=1000 && 
   export COMPUTE_QUEUE="Compartment-JQ-1588569574" &&
   export CENSUS_API_KEY=c235e1b5620232fab506af060c5f8580604d89c1 && 
   export FLEPI_RESET_CHIMERICS=TRUE &&
   rm -rf model_output data/us_data.csv data-truth &&
   rm -rf data/mobility_territories.csv data/geodata_territories.csv &&
   rm -rf data/seeding_territories.csv && 
   rm -rf data/seeding_territories_Level5.csv data/seeding_territories_Level67.csv
   
export RESUME_LOCATION=s3://idd-inference-runs/USA-20230423T235232
```
{% endtab %}
{% endtabs %}

**Preliminary model run.** We do a setup run with 1 to 2 iterations to make sure the model runs and setup input data. This takes several minutes to complete, depending on how complex the simulation will be. To do this, run the following code chunk, with no modification of the code required:

{% code overflow="wrap" %}
```
export CONFIG_NAME=config_$SCENARIO.yml && 
   export CONFIG_PATH=/home/app/drp/COVID19_USA/$CONFIG_NAME && 
   export FLEPI_PATH=/home/app/drp/flepiMoP && 
   export DATA_PATH=/home/app/drp/COVID19_USA && 
   export INTERVENTION_NAME="med" && 
   export FLEPI_STOCHASTIC=FALSE && 
   rm -rf $DATA_PATH/model_output DATA_PATH/us_data.csv && 
   cd $DATA_PATH && 
   Rscript $FLEPI_PATH/R/scripts/build_US_setup.R -c $CONFIG_NAME && 
   Rscript $FLEPI_PATH/R/scripts/build_covid_data.R -c $CONFIG_NAME && 
   Rscript $FLEPI_PATH/R/scripts/full_filter.R -c $CONFIG_NAME -j 1 -n 1 -k 1 && 
   printenv CONFIG_NAME
```
{% endcode %}

### Step 5. Launch job on AWS batch

**Configure AWS.** Assuming that the simulations finish successfully, you will now enter credentials and submit your job onto AWS batch. Enter the following command into the terminal:&#x20;

```
aws configure
```

You will be prompted to enter the following items. These can be found in a file called `new_user_credentials.csv`.&#x20;

* Access key ID when prompted
* Secret access key when prompted
* Default region name: us-west-2
* Default output: Leave blank when this is prompted and press enter (The Access Key ID and Secret Access Key will be given to you once in a file)

**Launch the job.** To launch the job, use the appropriate setup based on the type of job you are doing. No modification of these code chunks should be required.

{% tabs %}
{% tab title="Standard" %}
<pre><code><strong>export CONFIG_PATH=$CONFIG_NAME &#x26;&#x26;
</strong><strong>cd $DATA_PATH &#x26;&#x26;
</strong>$FLEPI_PATH/batch/inference_job.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic &#x26;&#x26;
printenv CONFIG_NAME
</code></pre>
{% endtab %}

{% tab title="Non-inference" %}
```
export CONFIG_PATH=$CONFIG_NAME &&
cd $DATA_PATH &&
$FLEPI_PATH/batch/inference_job.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic -j 1 -k 1 &&
printenv CONFIG_NAME
```
{% endtab %}

{% tab title="Resume" %}
> NOTE: _Resume_ and _Continuation Resume_ runs are currently submitted the same way,  resuming from an S3 that was generated manually. Typically we will also submit any _Continuation Resume_ run specifying `--resume-carry-seeding` as starting seeding conditions will be manually constructed and put in the S3.



**Carrying seeding**  (_do this to use seeding fits from resumed run_):

<pre><code>export CONFIG_PATH=$CONFIG_NAME &#x26;&#x26;
<strong>cd $DATA_PATH &#x26;&#x26;
</strong>$FLEPI_PATH/batch/inference_job.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-carry-seeding --restart-from-location=s3://idd-inference-runs/$RESUME_S3 --restart-from-run-id=$RESUME_ID &#x26;&#x26;
printenv CONFIG_NAME
</code></pre>



**Discarding seeding**  (_do this to refit seeding again_)_:_

```
export CONFIG_PATH=$CONFIG_NAME &&  
cd $DATA_PATH &&
$COVID_PATH/batch/inference_job.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-discard-seeding --restart-from-location=s3://idd-inference-runs/$RESUME_S3 --restart-from-run-id=$RESUME_ID &&
printenv CONFIG_NAME
```



**Single Iteration + Carry seeding**  (_do this to produce additional scenarios where no fitting is required_)_:_

```
export CONFIG_PATH=$CONFIG_NAME &&
cd $DATA_PATH &&
$COVID_PATH/batch/inference_job.py -c $CONFIG_PATH -q $COMPUTE_QUEUE --non-stochastic --resume-carry-seeding --restart-from-location=s3://idd-inference-runs/$RESUME_S3 --restart-from-run-id=$RESUME_ID -j 1 -k 1 &&
printenv CONFIG_NAME
```
{% endtab %}
{% endtabs %}

### Step 6. Document the Submission

**Commit files to GitHub.** After the job is successfully submitted, you will now be in a new branch of the population repo. Commit the ground truth data files to the branch on GitHub and then return to the main branch:

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

**Save submission info to slack.** We use a slack channel to save the submission information that gets outputted. Copy this to slack so you can identify the job later. Example output:

```
Setting number of output slots to 300 [via config file]
Launching USA-20220923T160106_inference_med...
Resuming from run id is SMH_R1_lowVac_optImm_2018 located in s3://idd-inference-runs/USA-20220913T000opt
Discarding seeding results
Final output will be: s3://idd-inference-runs/USA-20220923T160106/model_output/
Run id is SMH_R1_highVac_optImm_2022
Switched to a new branch 'run_USA-20220923T160106'
config_SMH_R1_highVac_optImm_2022.yml
```

