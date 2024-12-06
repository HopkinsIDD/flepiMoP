# Inference with EMCEE

### Config Changes Relative To Classical Inference

The major changes are:

1. Under the 'inference' section add `method: emcee` entry, and
2. Under the 'statistics' section move the resample specific configuration under a 'resample' subsection as show bellow:

<figure><img src="../.gitbook/assets/Screenshot 2024-10-25 at 15.19.02.png" alt=""><figcaption><p>left: classical inference config, right: new EMCEE config</p></figcaption></figure>

In addition to those configuration changes there are now new likelihood statistics offered: `pois`, `norm`/`norm_homoskedastic`, `norm_cov`/`norm_heteroskedastic`, `nbinom`, `rmse`, `absolute_error`. As well as new regularizations: `forecast` and `allsubpops`.

### Running Locally

You can test your updated config by running:

```bash
flepimop-calibrate -c config_emcee.yml --nwalkers 5  --jobs 5 --niterations 10 --nsamples 5 --id my_run_id
```

If it works, it should produce:

* Plots of simulation directly from your config,
* Plots after the fits with the fits and the parameter chains,
* An h5 file with all the chains, and
* The usual `model_output/` directory.

It will also immediately produce standard out that is similar to (dependent on config):

```
  gempyor >> Running ***DETERMINISTIC*** simulation;
  gempyor >> ModelInfo USA_inference_all; index: 1; run_id: SMH_Rdisparity_phase_one_phase1_blk1_fixprojnpis_CA-NC_emcee,
  gempyor >> prefix: USA_inference_all/SMH_Rdisparity_phase_one_phase1_blk1_fixprojnpis_CA-NC_emcee/;
Loaded subpops in loaded relative probablity file: 51 Intersect with seir simulation:  2 kept
Running Gempyor Inference

LogLoss: 6 statistics and 92 data points,number of NA for each statistic: 
incidD_latino    46
incidD_other      0
incidD_asian      0
incidD_black      0
incidD_white      0
incidC_white     24
incidC_black     24
incidC_other     24
incidC_asian     24
incidC_latino    61
incidC           24
incidD            0
dtype: int64
InferenceParameters: with 92 parameters: 
    seir_modifiers: 84 parameters
    outcome_modifiers: 8 parameters
```

Here, it says the config fits 92 parameters, we'll keep that in mind and choose a number of walkers greater than (ideally 2 times) this number of parameters.

### Running On An HPC Environment With Slurm

First, install `flepiMoP` on the cluster following the [Running On A HPC With Slurm](./../how-to-run/advanced-run-guides/running-on-a-hpc-with-slurm.md) guide. Then manually create a batch file to submit to slurm like so:

```bash
#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --mem 450g
#SBATCH --cpus-per-task 256
#SBATCH --time 20:00:00
flepimop-calibrate --config config_NC_emcee.yml \
  --nwalkers 500  \
  --jobs 256 \
  --niterations 2000 \
  --nsamples 250 \
  --id my_id  > out_fit256.out 2>&1
```

Breaking down what each of these lines does:

* `#SBATCH --ntasks 1`: Requests that this be run as a single job,
* `#SBATCH --nodes 1`: Requests that the job be run on 1 node, as of right now EMCEE only supports single nodes,
* `#SBATCH --mem 450g`: Requests that the whole job get 405GB of memory should be ~2-3GB per a walker,
* `#SBATCH --cpus-per-task 256`: Requests that the whole job get 256 CPUs (technically 256 per a task by `ntasks` should be set to 1 for EMCEE),
* `#SBATCH --time 20:00:00`: Specifies a time limit of 20hrs for this job to complete in, and
* `flepimop-calibrate ...`:
  - `--config config_NC_emcee.yml`: Use the `config_NC_emcee.yml` for this calibration run,
  - `--nwalkers 500`: Use 500 walkers (or chains) for this calibration, should be about 2x the number of parameters,
  - `--jobs 256`: The number of parallel walkers to run, should be either 1x or 0.5x the number of cpus,
  - `--niterations`: The number of iterations to run for for each walker,
  - `--nsamples`: The number of posterier samples (taken from the end of each walker) to save to the `model_output/` directory, and
  - `--id`: An optional short but unique job name, if not explicitly provided one will be generated from the config.

For more details on other options provided by gempyor for calibration please see `flepimop-calibrate --help`.

### Postprocessing EMCEE

At this stage postprocessing for EMCEE outputs is fairly manual. A good starting point can be found in `postprocessing/emcee_postprocess.ipynb` which plots the chains and can run forward projections from the sample drawn from calibration.
