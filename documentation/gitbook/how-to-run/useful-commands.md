# Useful commands

### Git setup

Type the following line so git remembers your credential and you don't have to enter your token 6 times per day:

```bash
git config --global credential.helper store
git config --global user.name "{NAME SURNAME}"
git config --global user.email YOUREMAIL@EMAIL.COM
git config --global pull.rebase false # so you use merge as the default reconciliation method
```

### Get a notification on your phone/mail when a run is done

We use [ntfy.sh](https://ntfy.sh) for notification. Install ntfy on your Iphone or Android device. Then subscribe to the channel `ntfy.sh/flepimop_alerts` where you'll receive notifications when runs are done.

* End of job notifications goes as urgent priority.

### Install slack integration

Within included example postprocessing scripts, we include a helper script that sends a slack message with some output snapshots of our model output. So our 🤖-friend can send us some notifications once a run is done.

```
cd /scratch4/struelo1/flepimop-code/
nano slack_credentials.sh
# and fill the file:
export SLACK_WEBHOOK="{THE SLACK WEBHOOK FOR CSP_PRODUCTION}"
export SLACK_TOKEN="{THE SLACK TOKEN}"

```

### Delphi **Epidata** API

{% hint style="info" %}
If you are using the **Delph Epidata API**, first [register for a key](https://cmu-delphi.github.io/delphi-epidata/). Once you have a key, add that below where you see \[YOUR API KEY]. Alternatively, you can put that key in your config file in the `inference` section as `gt_api_key: "YOUR API KEY"`.

```bash
export DELPHI_API_KEY="[YOUR API KEY]"
```
{% endhint %}

## 🚀 Run inference using slurm (do everytime)



TODO: add how to run test, and everything

{% hint style="danger" %}
Don't paste them if you don't know what they do
{% endhint %}

### Filepaths structure

in configs with a setup `name: USA`

```
model_output/{FileType}/{Prefix}{Index}.{run_id}.{FileType}.{Extension}
                           ^ 
                          setup name(USA)/scenario(inference/med)/run_id/{Inference stuff}
                                                                           ^ global/{final, intermediate}/slot#.
```

where, eg:

* the index is `1`
* the run\_id is `2021.12.14.23:56:12.CET`
* the prefix is `USA/inference/med/2021.12.14.23:56:12.CET/global/intermediate/000000001.`

### Steps to first local run

```bash
export COVID_PATH=$(pwd)/COVIDScenarioPipeline
export DATA_PATH=$(pwd)/COVID19_USA
conda activate covidSP
cd $COVID_PATH
Rscript local_install.R
pip install --no-deps -e gempyor_pkg # before: python setup.py develop --no-deps
git lfs install
git lfs pull
export CENSUS_API_KEY=YOUR_KEY
cd $DATA_PATH
git restore data/
export CONFIG_PATH=config_smh_r11_optsev_highie_base_deathscases_blk1.yml
Rscript $COVID_PATH/R/scripts/build_US_setup.R
Rscript $COVID_PATH/R/scripts/create_seeding.R
Rscript $COVID_PATH/R/scripts/full_filter.R -j 1 -n 1 -k 1
```

where:

* $$n$$ is slots
* $$j$$ is core
* $$k$$ is iteration per slot

### Launch the docker locally

```bash
docker pull hopkinsidd/covidscenariopipeline:latest-dev
docker run -it -v "$(pwd)":/home/app/covidsp hopkinsidd/covidscenariopipeline:latest-dev
```

### Pipeline git-fu (dealing with the commute\_data)

because a big file get changed and added automatically. Since Git 2.13 (Q2 2017), you can stash individual files, with [_git stash push_](https://git-scm.com/docs/git-stash#git-stash-push-p--patch-k--no-keep-index-u--include-untracked-a--all-q--quiet-m--messageltmessagegt--ltpathspecgt82308203). One of these should work.

```bash
git restore --staged sample_data/united-states-commutes/commute_data.csv
git stash push sample_data/united-states-commutes/commute_data.csv
git reset sample_data/united-states-commutes/commute_data.csv
```
