# Inference with EMCEE

{% hint style="warning" %}
For now this only work from branch emcee\_batch
{% endhint %}

### Config changes w.r.t classical inference

You need, under inference, to add `method: emcee` and modify the `statistics:` as shown in the diff below (basically: all resampling goes to one subsection, with some minor changes to names).&#x20;

<figure><img src="../.gitbook/assets/Screenshot 2024-10-25 at 15.19.02.png" alt=""><figcaption><p>left: classical inference config, right: new EMCEE config</p></figcaption></figure>

To see which llik options and regularization (e.g do you want to weigh more the last weeks for forecasts, or do you want to add the sum of all subpop) see files `statistics.py.`

### Test on your computer

Install gempyor from branch emcee\_batch . Test your config by running:

```bash
flepimop-calibrate -c config_emcee.yml --nwalkers 5  --jobs 5 --niterations 10 --nsamples 5 --id my_rim_id
```

on your laptop. If it works, it should produce:

* plots of simulation directly from your config
* &#x20;plots after the fits with the fits and the parameter chains
* and h5 file with all the chains
* and in model\_output, the final hosp/snpi/seir/... files in the flepiMoP structure.

It will output something like

\`\`\`

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

### Run on cluster

Install gempyor on the cluster. test it with the above line, then modify this script:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=450g
#SBATCH -c 256
#SBATCH -t 00-20:00:00
flepimop-calibrate -c config_NC_emcee.yml --nwalkers 500  --jobs 256 --niterations 2000 --nsamples 250 --id my_id  > out_fit256.out 2>&1
```

so you need to have:

* &#x20;`-c` (number of core) equal to **roughly half the number of walkers** (slots/parallel chains)
* mem to be around two times the number of walkers. Look at the computes nodes you have access to and make something that can be prioritized fast enough.&#x20;
* nsamples is the number of final results you want, but it's fine not to care about it, I rerun the sampling from my computer.
* To resume from an existing run, add the previous line `--resume` and it 'll start from the last parameter values in the h5 files.

### Postprocess EMCEE

To analyze run `postprocessing/emcee_postprocess.ipynb`\
First, this plots the chains and then it runs nsamples (you can choose it) projection with the end of the chains and does the plot of the fit, with and without projections
