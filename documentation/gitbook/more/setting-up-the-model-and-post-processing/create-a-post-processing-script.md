---
description: These scripts are run automatically after an inference run
---

# Create a post-processing script

Some information to consider if you'd like your script to be run automatically after an inference run:&#x20;

* Most R/python packages are installed already installed. Try to run your script on the conda environment defined on the [submission page](../../how-to-run/advanced-run-guides/slurm-submission-on-marcc.md) (or easier if you are not set up on MARCC, ask me)
* There will be some variables set in the environment. These variables are:
  * `$CONFIG_PATH` the path to the configuration file&#x20;
  * `$FLEPI_RUN_INDEX` the run id for this run (e.g \``CH_R3_highVE_pesImm_2022_Jan29`\`
  * `$JOB_NAME` this job name (e.g `USA-20230130T163847_inference_med`)
  * `$FS_RESULTS_PATH` the path where lies the model results. It's a folder that contains the model\_ouput/ as a subfolder
  * `$FLEPI_PATH` path of the flepiMoP repository.
  * `$DATA_PATH` path of the Data directory (e.g Flu\_USA or COVID19\_USA).
  * Anything you ask can theoretically be provided here.
* The script must run without any user intervention.
* The script is run from $DATA\_PATH.
* Your script lies in the flepiMoP directory (preferably) or it's ok if it is in a data directory if it makes sense.&#x20;
* It is run on a 64Gb of RAM multicore machine. All scripts combined must complete under 4 hours, and you can use multiprocessing (48 cores)
* Outputs (pdf, csv, html, txt, png ...) must be saved in a directory named `pplot/` (you can assume that it exists) in order to be sent to slack by FlepiBot 🤖 after the run.
* an example postprocessing script (in python) is [here](https://github.com/HopkinsIDD/COVIDScenarioPipeline/blob/main-flu-subfix2/scripts/postprocess\_auto.py).
* You can test your script on MARCC on a run that is already saved in `/data/struelo1/flepimop-runs` or I can do it for you.
* Once your script works, add (or ask to add) the command line to run in file `batch/postprocessing_scripts.sh` [(here)](https://github.com/HopkinsIDD/COVIDScenarioPipeline/blob/main-flu-subfix2/batch/postprocessing-scripts.sh) between the START and END lines, with a little comment about what your script does.
