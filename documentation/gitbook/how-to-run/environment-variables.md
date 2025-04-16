--- 
description: >-
    A library of environment variables in the flepiMoP codebase.
    These variables may be updated or deprecated as the project evolves.
---

# Environment Variables

Below you will find a list of environment variables (envvars) defined throughout the flepiMoP codebase. Often, these variables are set in response to command-line argument input. Though, some are set by `flepiMoP` without direct user input (these are denoted by a 'Not a CLI option' note in the **Argument** column.)

<table>
    <thead>
        <tr>
            <th width="135">Envvar.</th>
            <th width="100">Argument</th>
            <th width="300">Description</th>
            <th width="115">Default</th>
            <th width="150">Valid values</th> 
            <th width="150">Key file locations (inexhaustive)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>BATCH_SYSTEM</code></td>
            <td>Not a CLI option.</td>
            <td>System you are running on (e.g., aws, SLURM, local).</td>
            <td>N/A</td>
            <td>e.g., <code>aws</code>, <code>slurm</code></td>
            <td><code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>CENSUS_API_KEY</code></td>
            <td>Not a CLI option.</td>
            <td>A unique key to the API for census data.</td>
            <td>N/A</td>
            <td><a href="https://api.census.gov/data/key_signup.html">Get your own API key</a></td>
            <td><code>slurm_init.sh</code>, <code>build_US_setup.R</code></td>
        </tr>
        <tr>
            <td><code>CONFIG_PATH</code></td>
            <td><code>-c</code>, <code>--config</code></td>
            <td>Path to a configuration file.</td>
            <td>--</td>
            <td><code>your/path/to/config_file</code></td>
            <td><code>build_covid_data.R</code>, <code>build_US_setup.R</code>, <code>build_initial_seeding.R</code>, <code>build_flu_data.R</code>, <code>config.R</code>, <code>preprocessing/</code> files</td>
        </tr>
        <tr>
            <td><code>DELPHI_API_KEY</code></td>
            <td><code>-d</code>, <code>--delhpi_api_key</code></td>
            <td>Your personalized key for the Delphi Epidata API. Alternatively, this key can go in the config inference section as <code>gt_api_key</code>.</td>
            <td>--</td>
            <td><a href="https://cmu-delphi.github.io/delphi-epidata/api/api_keys.html">Get your own API key</a></td>
            <td><code>build_covid_data.R</code></td>
        </tr>
        <tr>
            <td><code>DIAGNOSTICS</code></td>
            <td><code>-n</code>, <code>--run-diagnostics</code></td>
            <td>Flag for whether or not diagnostic tests should be run during execution.</td>
            <td><code>TRUE</code></td>
            <td><code>--run-diagnostics FALSE</code> for <code>FALSE</code>, <code>--run-diagnostics</code> or no mention for <code>TRUE</code></td>
            <td><code>run_sim_processing_SLURM.R</code></td>
        </tr>
        <tr>
            <td><code>DISEASE</code></td>
            <td><code>-i</code>, <code>--disease</code></td>
            <td>Which disease is being simulated in the prsent run.</td>
            <td><code>flu</code></td>
            <td>e.g., <code>rsv</code>, <code>covid</code></td>
            <td><code>run_sim_processing_SLURM.R</code>/td>
        </tr>
        <tr>
            <td><code>DVC_OUTPUTS</code></td>
            <td>Not a CLI option, but defined using <code>--output</code></td>
            <td>The names of the directories with outputs to save in S3 (separated by a space).</td>
            <td><code>model_output model_parameters importation hospitalization</code></td>
            <td>e.g., <code>model_output model_parameters importation hospitalization</code></td>
            <td><code>scenario_job.py</code>, <code>AWS_scenario_runner.sh</code></td>
        </tr>
        <tr>
            <td><code>FILENAME</code></td>
            <td>Not a CLI option.</td>
            <td>Filenames for output files, determined dynamically during inference.</td>
            <td>N/A</td>
            <td><code>file.parquet</code>, <code>plot.pdf</code></td>
            <td><code>AWS_postprocess_runner.sh</code>, <code>SLURM_inference_job.run</code>, <code>AWS_inference_runner.sh</code></td>
        </tr>
        <tr>
            <td><code>FIRST_SIM_INDEX</code></td>
            <td><code>-i</code>, <code>--first_sim_index</code></td>
            <td>The index of the first simulation.</td>
            <td><code>1</code></td>
            <td><code>int</code></td>
            <td><code>shared_cli.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_BLOCK_INDEX</code></td>
            <td><code>-b</code>, <code>--this_block</code></td>
            <td>Index of current block.</td>
            <td><code>1</code></td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-main.R</code>, <code>utils.py</code>, <code>AWS_postprocess_runner.sh</code>, <code>AWS_inference_runner.sh</code>, <code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION</code></td>
            <td><code>--continuation</code>/<code>--no-continuation</code></td>
            <td>Flag for whether or not to use the resumed run seir files (or provided initial files bucket) as initial conditions for the next run.</td>
            <td><code>FALSE</code></td>
            <td><code>--continuation TRUE</code> for <code>TRUE</code>, <code>--continuation</code> or no mention for <code>FALSE</code></td>
            <td><code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td> 
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_FTYPE</code></td>
            <td>Not a CLI option.</td>
            <td>If running a continuation, the file type of the initial condition files.</td>
            <td><code>config['initial_conditions']['initial_file_type']</code></td>
            <td>e.g., <code>.csv</code></td>
            <td><code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_LOCATION</code></td>
            <td><code>--continuation-location</code></td>
            <td>The location (folder or an S3 bucket) from which to pull the /init/ files (if not set, uses the resume location seir files).</td>
            <td>--</td>
            <td><code>path/to/your/location</code></td>
            <td><code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_RUN_ID</code></td>
            <td><code>--continuation-run-id</code></td>
            <td>The ID of run to continue at, if doing a continuation.</td>
            <td>--</td>
            <td><code>int</code></td>
            <td><code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_INFO_PATH</code></td>
            <td>Not a CLI option.</td>
            <td><i>pending</i></td>
            <td><i>pending</i></td>
            <td><i>pending</i></td>
            <td><code>info.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_ITERATIONS_PER_SLOT</code></td>
            <td><code>-k</code>, <code>--iterations_per_slot</code></td>
            <td>Number of iterations to run per slot.</td>
            <td>--</td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_MAX_STACK_SIZE</code></td>
            <td><code>--stacked-max</code></td>
            <td>Maximum number of iterventions to allow in a stacked intervention.</td>
            <td><code>5000</code></td>
            <td><code>int >=350</code></td>
            <td><code>StackedModifier.py</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_MEM_PROFILE</code></td>
            <td><code>-M</code>, <code>--memory_profiling</code></td>
            <td>Flag for whether or not memory profile should be run during iterations.</td>
            <td><code>FALSE</code></td>
            <td><code>--memory_profiling TRUE</code> for <code>TRUE</code>, <code>--memory_profiling</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_MEM_PROF_ITERS</code></td>
            <td><code>-P</code>, <code>--memory_profiling_iters</code></td>
            <td>If doing memory profiling, after every X iterations, run the profiling.</td>
            <td><code>100</code></td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_NJOBS</code></td>
            <td><code>-j</code>, <code>--jobs</code></td>
            <td>Number of parallel processors used to run the simulation. If there are more slots than jobs, slots will be divided up between processors and run in series on each.</td>
            <td>Number of cores detected as available at computing cluster.</td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>calibrate.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_NUM_SLOTS</code></td>
            <td><code>-n</code>, <code>--slots</code></td>
            <td>Number of independent simulations of the model to be run.</td>
            <td>--</td>
            <td><code>int >=1</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>calibrate.py</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_OUTCOME_SCENARIOS</code></td>
            <td><code>-d</code>, <code>--outcome_modifiers_scenarios</code></td>
            <td>Name of the outcome scenario to run.</td>
            <td><code>'all'</code></td>
            <td><i>pending</i></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_PATH</code></td>
            <td><code>-p</code>, <code>--flepi_path</code></td>
            <td>Path to the flepiMoP directory.</td>
            <td><code>'flepiMoP'</code></td>
            <td><code>path/to/flepiMoP</code></td>
            <td>several <code>postprocessing/</code> files, several <code>batch/</code> files, several <code>preprocessing/</code> files, <code>info.py</code>, <code>utils.py</code>, <code>_cli.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_PREFIX</code></td>
            <td><code>--in-prefix</code></td>
            <td>Unique name for the run.</td>
            <td>--</td>
            <td>e.g., <code>project_scenario1_outcomeA</code>, etc.</td>
            <td><code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code>, <code>AWS_postprocess_runner.sh</code>, <code>calibrate.py</code>, several <code>preprocessing/</code> files, several <code>postprocessing/</code> files, several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>FLEPI_RESET_CHIMERICS</code></td>
            <td><code>-L</code>, <code>--reset_chimeric_on_accept</code></td>
            <td>Flag for whether or not chimeric parameters should be reset to global parameters whena  global acceptance occurs.</td>
            <td><code>TRUE</code></td>
            <td><code>--reset_chimeric_on_accept FALSE</code> for <code>FALSE</code>, <code>--reset_chimeric_on_accept</code> or no mention for <code>TRUE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>slurm_init.sh</code>, <code>hpc_init</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_RESUME</code></td>
            <td><code>--resume</code>/<code>--no-resume</code></td>
            <td>Flag for whether or not to resume the current calibration.</td>
            <td><code>FALSE</code></td>
            <td><code>--resume TRUE</code> for <code>TRUE</code>, <code>--resume</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>slurm_init.sh</code>, <code>hpc_init</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_RUN_INDEX</code></td>
            <td><code>-u</code>, <code>--run_id</code></td>
            <td>Unique ID given to the model run. If the same config is run multiple times, you can avoid the output being overwritten by using unique model run IDs.</td>
            <td>Auto-assigned run ID</td>
            <td><code>int</code></td>
            <td><code>copy_for_continuation.py</code>, <code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>shared_cli.py</code>, <code>base.py</code>, <code>calibrate.py</code>, several <code>batch/</code> files, several <code>postprocessing/</code> files</td>
        </tr>
        <tr>
            <td><code>FLEPI_SEIR_SCENARIOS</code></td>
            <td><code>-s</code>, <code>--seir_modifier_scenarios</code></td>
            <td>Names of the intervention scenarios to run.</td>
            <td><code>'all'</code></td>
            <td><i>pending</i></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_SLOT_INDEX</code></td>
            <td><code>-i</code>, <code>--this_slot</code></td>
            <td>Index for current slots.</td>
            <td><code>1</code></td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-slot.R</code>, several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>FS_RESULTS_PATH</code></td>
            <td><code>-R</code>, <code>--results-path</code></td>
            <td>A path to the model results.</td>
            <td>--</td>
            <td><code>your/path/to/model_results</code></td>
            <td><code>prune_by_llik.py</code>, <code>prune_by_llik_and_proj.py</code>, several <code>postprocessing/</code> files, several <code>batch/</code> files, <code>model_output_notebook.Rmd</code></td>
        </tr>
        <tr>
            <td><code>FULL_FIT</code></td>
            <td><code>-F</code>, <code>--full-fit</code></td>
            <td>Whether or not to process the full fit.</td>
            <td><code>FALSE</code></td>
            <td><code>--full-fit TRUE</code> for <code>TRUE</code>, <code>--full-fit</code> or no mention for <code>FALSE</code></td>
            <td><code>run_sim_processing_SLURM.R</code></td>
        </tr>
        <tr>
            <td><code>GT_DATA_SOURCE</code></td>
            <td><code>-s</code>, <code>--gt_data_source</code></td>
            <td>Sources of groundtruth data.</td>
            <td><code>'csse_case, fluview_death, hhs_hosp'</code></td>
            <td>See default</td>
            <td><code>build_covid_data.R</code</td>
        </tr>
        <tr>
            <td><code>GT_END_DATE</code></td>
            <td><code>--ground_truth_end</code></td>
            <td>Last date to include ground truth for.</td>
            <td>--</td>
            <td><code>YYYY-MM-DD</code> format</td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>GT_START_DATE</code></td>
            <td><code>--ground_truth_start</code></td>
            <td>First date to include ground truth for.</td>
            <td>--</td>
            <td><code>YYYY-MM-DD</code> format</td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>IMM_ESC_PROP</code></td>
            <td><code>--imm_esc_prop</code></td>
            <td>Annual percent of immune escape.</td>
            <td><code>0.35</code></td>
            <td><code>float</code> between <code>0.00 - 1.00</code></td>
            <td>several <code>preprocessing/</code> files</td> <!-- Start here -->
        </tr>
        <tr>
            <td><code>INCL_AGGR_LIKELIHOOD</code></td>
            <td><code>-a</code>, <code>--incl_aggr_likelihood</code></td>
            <td>Whether or not the likelihood should be calculated with aggregate estimates.</td>
            <td><code>FALSE</code></td>
            <td><code>--incl_aggr_likelihood TRUE</code> for <code>TRUE</code>, <code>--incl_aggr_likelihood</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code></td>
        </tr>
        <tr>
            <td><code>IN_FILENAME</code></td>
            <td>Not a CLI option.</td>
            <td>Name of input files.</td>
            <td>N/A</td>
            <td><code>file_1.csv</code> <code>file_2.csv</code>, etc.</td> 
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>INIT_FILENAME</code></td>
            <td><code>--init_file_name</code></td>
            <td>Initial file global intermediate name.</td>
            <td>--</td>
            <td><code>file.csv</code></td>
            <td><code>seir_init_immuneladder.R</code>, <code>inference_job.run</code>, several <code>preprocessing/</code> files</td>
        </tr>
        <tr>
            <td><code>INTERACTIVE_RUN</code></td>
            <td><code>-I</code>, <code>--is-interactive</code></td>
            <td>Whether or not the current run is interactive.</td>
            <td><code>FALSE</code></td>
            <td><code>--is-interactive TRUE</code> for <code>TRUE</code>, <code>--is-interactive</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>JOB_NAME</code></td>
            <td><code>--job-name</code></td>
            <td>Unique job name (intended for use when submitting to SLURM).</td>
            <td>--</td>
            <td>Convention: <code>{config['name']}-{timestamp}</code> (str)</td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>LAST_JOB_OUTPUT</code></td>
            <td>Not a CLI option.</td>
            <td>Path to output of last job.</td>
            <td>N/A</td>
            <td><code>path/to/last_job/output</code></td>
            <td><code>utils.py</code>, several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>OLD_FLEPI_RUN_INDEX</code></td>
            <td>Not a CLI option.</td>
            <td>Run ID of old flepiMoP run.</td>
            <td>N/A</td>
            <td><code>int</code></td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>OUT_FILENAME</code></td>
            <td>Not a CLI option.</td>
            <td>Name of output files.</td>
            <td>N/A</td>
            <td><code>file_1.csv</code> <code>file_2.csv</code>, etc.</td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>OUT_FILENAME_DIR</code></td>
            <td>Not a CLI option.</td>
            <td>Directory for output files.</td>
            <td>N/A</td>
            <td><code>path/to/output/files</code></td>
            <td><code>SLURM_inference_job.run</code></td>
        </tr>
        <tr>
            <td><code>OUTPUTS</code></td>
            <td><code>-o</code>, <code>--select-outputs</code></td>
            <td>A list of outputs to plot.</td>
            <td><code>'hosp, hnpi, snpi, llik'</code></td>
            <td><code>hosp, hnpi, snpi, llik</code></td>
            <td><code>postprocess_snapshot.R</code></td>
        </tr>
        <tr>
            <td><code>PARQUET_TYPES</code></td>
            <td>Not a CLI option.</td>
            <td>Parquet files.</td>
            <td><code>'seed spar snpi seir hpar hnpi hosp llik init'</code></td>
            <td><code>seed spar snpi seir hpar hnpi hosp llik init</code></td>
            <td><code>AWS_postprocess_runner.sh</code>, <code>SLURM_inference_job.run</code>, <code>AWS_inference_runner.sh</code></td>
        </tr>
        <tr>
            <td><code>PATH</code></td>
            <td>Not a CLI option.</td>
            <td>Path relating to AWS installation. Used during SLURM runs.</td>
            <td>N/A</td>
            <td>set with <code>export PATH=~/aws-cli/bin:$PATH</code> in <code>SLURM_inference_job.run</code></td>
            <td><code>schema.yml</code>, <code>utils.py</code>, <code>info.py</code>, <code>AWS_postprocess_runner.sh</code>, <code>SLURM_inference_job.run</code></td>
        </tr>
        <tr>
            <td><code>PROCESS</code></td>
            <td><code>-r</code>, <code>--run-processing</code></td>
            <td>Whether or not to process the run.</td>
            <td><code>FALSE</code></td>
            <td><code>--run-processing TRUE</code> for <code>TRUE</code>, <code>--run-processing</code> or no mention for <code>FALSE</code></td>
            <td><code>run_sim_processing_SLURM.R</code></td>
        </tr>
        <tr>
            <td><code>PROJECT_PATH</code></td>
            <td><code>-d</code>, <code>--data_path</code></td>
            <td>Path to the folder with configs and model output.</td>
            <td>--</td>
            <td><code>path/to/configs_and_model-output</code></td>
            <td><code>base.py</code>, <code>_cli.py</code>, <code>calibrate.py</code>, several <code>postprocessing/</code> files, several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>PULL_GT</code></td>
            <td><code>-g</code>, <code>--pull-gt</code></td>
            <td>Whether or not to pull ground truth data.</td>
            <td><code>FALSE</code></td>
            <td><code>--pull-gt TRUE</code> for <code>TRUE</code>, <code>--pull-gt</code> or no mention for <code>FALSE</code></td>
            <td><code>run_sm_processing_SLURM.R</code></td>
        </tr>
        <tr>
            <td><code>PYTHON_PATH</code></td>
            <td><code>-y</code>, <code>--python</code></td>
            <td>Path to Python executable.</td>
            <td><code>'python3'</code></td>
            <td><code>path/to/your_python</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>RESUMED_CONFIG_PATH</code></td>
            <td><code>--res_config</code></td>
            <td>Path to previous config file, if using resumes.</td>
            <td><code>NA</code></td>
            <td><code>path/to/past_config</code></td>
            <td><code>seir_init_immuneladder.R</code>, several <code>preprocessing/</code> files</td>
        </tr>
        <tr>
            <td><code>RESUME_DISCARD_SEEDING</code></td>
            <td><code>--resume-discard-seeding</code>, <code>--resume-carry-seeding</code></td>
            <td>Whether or not to keep seeding in resume runs.</td>
            <td><code>FALSE</code></td>
            <td><code>--resume-carry-seeding TRUE</code> for <code>TRUE</code>, <code>--resume-carry-seeding</code> or no mention for <code>FALSE</code></td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>RESUME_LOCATION</code></td>
            <td><code>-r</code>, <code>--restart-from-location</code></td>
            <td>The location (folder or an S3 bucket) where the previous run is stored.</td>
            <td>--</td>
            <td><code>path/to/last_job/output</code></td>
            <td><code>built_initial_seeding.R</code>, <code>calibrate.py</code>, <code>slurm_init.sh</code>, <code>hpc_init</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>RESUME_RUN</code></td>
            <td><code>-R</code>, <code>--is-resume</code></td>
            <td>Whether or not this run is a resume.</td>
            <td><code>FALSE</code></td>
            <td><code>--is-a-resume TRUE</code> for <code>TRUE</code>, <code>--is-a-resume</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>RESUME_RUN_INDEX</code></td>
            <td>Not a CLI option.</td>
            <td>Index of resumed run.</td>
            <td>set by <code>OLD_FLEPI_RUN_INDEX</code></td>
            <td><code>int</code></td>
            <td><code>SLURM_inference_job.run</code></td>
        </tr>
        <tr>
            <td><code>RSCRIPT_PATH</code></td>
            <td><code>-r</code>, <code>--rpath</code></td>
            <td>Path to R executable.</td>
            <td><code>'Rscript'</code></td>
            <td><code>path/to/your_R</code></td>
            <td><code>build_initial_seeding.R</code>, <code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>RUN_INTERACTIVE</code></td>
            <td><code>-I</code>, <code>--is-interactive</code></td>
            <td>Whether or not the current run is interactive.</td>
            <td><code>FALSE</code></td>
            <td><code>--is-interactive</code> for <code>TRUE</code>, <code>--is-interactive</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>SAVE_HOSP</code></td>
            <td><code>-H</code>, <code>--save_hosp</code></td>
            <td>Whether or not the HOSP output files should be saved for each iteration.</td>
            <td><code>TRUE</code></td>
            <td><code>--save_hosp FALSE</code> for <code>FALSE</code>, <code>--save_hosp</code> or no mention for <code>TRUE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>SAVE_SEIR</code></td>
            <td><code>-S</code>, <code>--save_seir</code></td>
            <td>Whether or not the SEIR output files should be saved for each iteration.</td>
            <td><code>FALSE</code></td>
            <td><code>--save_seir TRUE</code> for <code>TRUE</code>, <code>--save_seir</code> or no mention for <code>FALSE</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>SEED_VARIANTS</code></td>
            <td><code>-s</code>, <code>--seed_variants</code></td>
            <td>Whether or not to add variants/subtypes to outcomes in seeding.</td>
            <td>--</td>
            <td><code>FALSE</code>, <code>TRUE</code></td>
            <td><code>create_seeding.R</code></td>
        </tr>
        <tr>
            <td><code>SIMS_PER_JOB</code></td>
            <td>Not a CLI option.</td>
            <td>Simulations per job.</td>
            <td>N/A</td>
            <td><code>int >=1</code></td>
            <td><code>AWS_postprocess_runner.sh</code>, <code>inference_job_launcher.py</code>, <code>AWS_inference_runner.sh</code></td>
        </tr>
        <tr>
            <td><code>SLACK_CHANNEL</code></td>
            <td><code>-s</code>, <code>--slack-channel</code></td>
            <td>Slack channel, either 'csp-production' or 'debug'; or 'noslack' to disable slack.</td>
            <td>--</td>
            <td><code>csp-production</code>, <code>debug</code>, or <code>noslack</code></td>
            <td><code>postrpocess_auto.py</code>, <code>postprocessing-scripts.sh</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>SLACK_TOKEN</code></td>
            <td><code>-s</code>, <code>--slack-token</code></td>
            <td>Slack token.</td>
            <td>--</td>
            <td><a href="https://api.slack.com/tutorials/tracks/getting-a-token">How to get a Slack token</a></td>
            <td><code>postprocess_auto.py</code>, <code>SLURM_postprocess_runner.run</code></td>
        </tr>
        <tr>
            <td><code>SUBPOP_LENGTH</code></td>
            <td><code>-g</code>, <code>--subpop_len</code></td>
            <td>Number of digits in subpops.</td>
            <td><code>5</code></td>
            <td><code>int</code></td>
            <td><code>flepimop-inference-slot.R</code>, <code>flepimop-inference-main.R</code></td>
        </tr>
        <tr>
            <td><code>S3_MODEL_PROJECT_PATH</code></td>
            <td>Not a CLI option.</td>
            <td>Location in S3 bucket with the code, data, and dvc pipeline.</td>
            <td>N/A</td>
            <td><code>path/to/code_data_dvc</code></td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>S3_RESULTS_PATH</code></td>
            <td>Not a CLI option.</td>
            <td>Location in S3 to store results.</td>
            <td>N/A</td>
            <td><code>path/to/s3/results</code></td>
            <td>several <code>batch/</code> files</td>
        </tr>
        <tr>
            <td><code>S3_UPLOAD</code></td>
            <td>Not a CLI option.</td>
            <td>Whether or not we also save runs to S3 for slurm runs</td>
            <td><code>TRUE</code></td>
            <td><code>TRUE</code>, <code>FALSE</code></td>
            <td><code>SLURM_postprocess_runner.run</code>, <code>SLURM_inference_job.run</code>, <code>inference_job_launcher.py</code></td>
        </tr>
        <tr>
            <td><code>VALIDATION_DATE</code></td>
            <td><code>--validation-end-date</code></td>
            <td>First date of projection/forecast (first date without ground truth data).</td>
            <td><code>date.today()</code></td>
            <td><code>YYYY-MM-DD</code> format</td>
            <td><code>data_setup_source.R</code>, <code>DataUtils.R</code>, <code>groundtruth_source.R</code>, <code>slurm_init.sh</code>, <code>hpc_init</code>, <code>inference_job_launcher.py</code></td>
        </tr>
    </tbody>
</table>
