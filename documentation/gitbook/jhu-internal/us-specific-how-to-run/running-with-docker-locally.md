---
description: Short internal tutorial on running locally using a "Docker" container.
---

# Running with Docker locally (outdated/US specific) 🛳

###

{% hint style="danger" %}
There are more comprehensive directions in the How to run -> Running with Docker locally section, but this section has some specifics required to do US-specific, COVID-19 and flu-specific runs
{% endhint %}

### Setup

**Run Docker Image**

{% hint style="info" %}
Current Docker image: `/hopkinsidd/flepimop:latest-dev`
{% endhint %}

Docker is a software platform that allows you to build, test, and deploy applications quickly. Docker packages software into standardized units called **containers** that have everything the software needs to run including libraries, system tools, code, and runtime. This means you can run and install software without installing the dependencies in the system.

A docker container is an environment which is isolated from the rest of the operating system i.e. you can create files, programs, delete and everything but that will not affect your OS. It is a local virtual OS within your OS ;

For flepiMoP, we have a docker container that will help you get running quickly ;

```
docker pull hopkinsidd/flepimop:latest-dev
docker run -it \
  -v <dir1>:/home/app/flepimop \
  -v <dir2>:/home/app/drp \
hopkinsidd/flepimop:latest-dev  
```

In this command we run the docker image `hopkinsidd/flepimop`. The `-v` command is used to allocate space from Docker and mount it at the given location ;

This mounts the data folder `<dir1>` to a path called `drp` within the docker environment, and the COVIDScenarioPipeline `<dir2>` in `flepimop` ;

## 🚀 Run inference

#### Fill the environment variables (do this every time)

First, populate the folder name variables:

```bash
export FLEPI_PATH=/home/app/csp/
export DATA_PATH=/home/app/drp/
```

Then, export variables for some flags and the census API key (you can use your own):

{% code overflow="wrap" %}
```bash
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export CENSUS_API_KEY="6a98b751a5a7a6fc365d14fa8e825d5785138935"
```
{% endcode %}

<details>

<summary>Where do I get a census key API?</summary>

The Census Data Application Programming Interface (API) is an API that gives the public access to raw statistical data from various Census Bureau data programs.  To acquire your own API Key, click [here](https://api.census.gov/data/key\_signup.html).

After you enter your details, you should receive an email using which you can activate your key and then use it.

_Note: Do not enter the API Key in quotes, copy the key as it is._

</details>

Go into the Pipeline repo (making sure it is up to date on your favorite branch) and do the installation required of the repository:

<pre class="language-bash"><code class="lang-bash">cd $FLEPI_PATH   # it'll move to the flepiMoP/ directory
Rscript local_install.R               # Install the R stuff
<strong>pip install --no-deps -e gempyor_pkg/ # install gempyor
</strong>git lfs install
git lfs pull
</code></pre>

{% hint style="info" %}
Note: These installations take place in the docker container and not the Operating System. They must be made once while starting the container and need not be done for every time you are running tests, provided they have been installed once.
{% endhint %}

### Run the code

Everything is now ready. 🎉 Let's do some clean-up in the data folder (these files might not exist, but it's good practice to make sure your simulation isn't re-using some old files) ;

```bash
cd $DATA_PATH       # goes to Flu_USA
git restore data/
rm -rf data/mobility_territories.csv data/geodata_territories.csv data/us_data.csv
rm -r model_output/ # delete the outputs of past run if there are
```

Stay in `$DATA_PATH`, select a config, and build the setup. The setup creates the population seeding file (geodata) and the population mobility file (mobility). Then, run inference:

```bash
export CONFIG_PATH=config_SMH_R1_lowVac_optImm_2022.yml
Rscript $FLEPI_PATH/datasetup/build_US_setup.R
Rscript $FLEPI_PATH/datasetup/build_flu_data.R
Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R.R -j 1 -n 1 -k 1
```

where:

* `n` is the number of parallel inference slots,
* `j` is the number of CPU cores it'll use in your machine,
* `k` is the number of iterations per slots.

It should run successfully and create a lot of files in `model_output/` ;

The last few lines visible on the command prompt should be:

> \[\[1]]
>
> \[\[1]]\[\[1]]
>
> \[\[1]]\[\[1]]\[\[1]]
>
> NULL

{% hint style="info" %}
**Other helpful tools**

To understand the basics of docker refer to the following: [Docker Basics](https://www.docker.com/)

To install docker for windows refer to the following link: [Installing Docker](https://docs.docker.com/desktop/windows/install/)

The following is a good tutorial for introduction to docker: [Docker Tutorial](https://www.youtube.com/watch?v=gFjxB0Jn8Wo\&list=PL6gx4Cwl9DGBkvpSIgwchk0glHLz7CQ-7)

To run the entire pipeline we use the command prompt. To open the command prompt type “Command Prompt" in the search bar and open the command prompt. Here is a tutorial video for navigating through the command prompt: [Command Prompt Tutorial](https://www.youtube.com/watch?v=A3nwRCV-bTU)
{% endhint %}

To test, we use the test folder (test\_documentation\_inference\_us in this case) in the CovidScenariPipeline as the data repository folder. We run the docker container and set the paths.
