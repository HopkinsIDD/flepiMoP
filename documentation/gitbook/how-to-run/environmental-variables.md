--- 
description: >-
    A library of environmental variables in the flepiMoP codebase.
    These variables may be updated or deprecated as the project evolves.
---

# Environmental Variables 🌍

Below you will find a list of environmental variables (envvars) defined throughout the flepiMoP codebase. Often, these variables are set in reponse to command-line argument input. 

<table>
    <thead>
        <tr>
            <th width="135">Envvar.</th>
            <th width="100">Argument</th>
            <th width="300">Description</th>
            <th width="115">Default</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>BATCH_SYSTEM</code></td>
            <td>N/A</td>
            <td>System you are running on (e.g., aws, SLURM, local).</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>CENSUS_API_KEY</code></td>
            <td>N/A</td>
            <td>A unique key to the API for census data.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>CONDA_PREFIX</code></td>
            <td>N/A</td>
            <td>Holds the path to the active conda environment. Used to determine the installation path for CLI installation.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>CONFIG_PATH</code></td>
            <td><code>-c</code>, <code>--config</code></td>
            <td>Path to a configuration file.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>CONTINUATION_LOCATION</code></td>
            <td><code>--continuation-location</code></td>
            <td>The location (folder or an S3 bucket) from which to pull the init files (if not set, uses the resume location seir files)</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>DELPHI_API_KEY</code></td>
            <td><code>-d</code>, <code>--delhpi_api_key</code></td>
            <td>Your personalized key for the Delphi Epidata API. Alternatively, this key can go in the config inference section as <code>gt_api_key</code>.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>DIAGNOSTICS</code></td>
            <td><code>-n</code>, <code>--run-diagnostics</code></td>
            <td>Flag for whether or not diagnostic tests should be run during execution.</td>
            <td><code>True</code></td>
        </tr>
        <tr>
            <td><code>DISEASE</code></td>
            <td><code>-i</code>, <code>--disease</code></td>
            <td>Which disease is being simulated in the prsent run.</td>
            <td><code>'flu'</code></td>
        </tr>
        <tr>
            <td><code>DVC_OUTPUTS</code></td>
            <td>N/A</td>
            <td>The names of the directories with outputs to save in S3 (separated by a space).</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>FILENAME</code></td>
            <td>N/A</td>
            <td>Filenames for output files, determined dynamically during inference.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>FIRST_SIM_INDEX</code></td>
            <td><code>-i</code>, <code>--first_sim_index</code></td>
            <td>The index of the first simulation.</td>
            <td><code>1</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_BLOCK_INDEX</code></td>
            <td><code>-b</code>, <code>--this_block</code></td>
            <td>Index of current block.</td>
            <td><code>1</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION</code></td>
            <td><code>--continuation</code>/<code>--no-continuation</code></td>
            <td>Flag for whether or not to use the resumed run seir files (or provided initial files bucket) as initial conditions for the next run.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_FTYPE</code></td>
            <td>N/A</td>
            <td>If running a continuation, the file type of the initial condition files.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_LOCATION</code></td>
            <td><code>--continuation-location</code></td>
            <td>The location (folder or an S3 bucket) from which to pull the /init/ files (if not set, uses the resume location seir files).</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>FLEPI_CONTINUATION_RUN_ID</code></td>
            <td><code>--continuation-run-id</code></td>
            <td>The ID of run to continue at, if doing a continuation.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>FLEPI_INFO_PATH</code></td>
            <td><code></code></td>
            <td>See <code>info.py</code></td>
            <td><code></code></td>
        </tr>
        <tr>
            <td><code>FLEPI_ITERATIONS_PER_SLOT</code></td>
            <td><code>-k</code>, <code>--iterations_per_slot</code></td>
            <td>Number of iterations to run per slot.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>FLEPI_MAX_STACK_SIZE</code></td>
            <td><code>--stacked-max</code></td>
            <td>Maximum number of iterventions to allow in a stacked intervention.</td>
            <td><code>5000</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_MEM_PROFILE</code></td>
            <td><code>-M</code>, <code>--memory_profiling</code></td>
            <td>Flag for whether or not memory profile should be run during iterations.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_MEM_PROFILE_ITERS</code></td>
            <td><code>-P</code>, <code>--memory_profiling_iters</code></td>
            <td>If doing memory profiling, after every X iterations, run the profiling.</td>
            <td><code>100</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_NJOBS</code></td>
            <td><code>-j</code>, <code>--jobs</code></td>
            <td>Number of parallel processors used to run the simulation. If there are more slots than jobs, slots will be divided up between processors and run in series on each.</td>
            <td>Number of cores detected as available at computing cluster.</td>
        </tr>
        <tr>
            <td><code>FLEPI_NUM_SLOTS</code></td>
            <td><code>-n</code, <code>--slots</code></td>
            <td>Number of independent simulations of the model to be run.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>FLEPI_OUTCOME_SCENARIOS</code></td>
            <td><code>-d</code>, <code>--outcome_modifiers_scenarios</code></td>
            <td>Name of the outcome scenario to run.</td>
            <td><code>'all'</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_PATH</code></td>
            <td><code>-p</code>, <code>--flepi_path</code></td>
            <td>Path to the flepiMoP directory.</td>
            <td><code>'flepiMoP'</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_PREFIX</code></td>
            <td><code>--in-prefix</code></td>
            <td>Unique identifier for the run.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>FLEPI_RESET_CHIMERICS</code></td>
            <td><code>-L</code>, <code>--reset_chimeric_on_accept</code></td>
            <td>Flag for whether or not chimeric parameters should be reset to global parameters whena  global acceptance occurs.</td>
            <td><code>True</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_RESUME</code></td>
            <td><code>--resume</code>/<code>--no-resume</code></td>
            <td>Flag for whether or not to resume the current calibration.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_RUN_INDEX</code></td>
            <td><code>-u</code>, <code>--run_id</code></td>
            <td>Unique ID given to the model run. If the same config is run multiple times, you can avoid the output being overwritten by using unique model run IDs.</td>
            <td>Auto-assigned run ID</td>
        </tr>
        <tr>
            <td><code>FLEPI_SEIR_SCENARIOS</code></td>
            <td><code>-s</code>, <code>--seir_modifier_scenarios</code></td>
            <td>Names of the intervention scenarios to run.</td>
            <td><code>'all'</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_SLOT_INDEX</code></td>
            <td><code>-i</code>, <code>--this_slot</code></td>
            <td>Index for current slots.</td>
            <td><code>1</code></td>
        </tr>
        <tr>
            <td><code>FLEPI_STOCHASTIC_RUN</code></td>
            <td><code>-t</code>, <code>--stoch_traj_flag</code></td>
            <td>Whether or not the model should be run stochastically or non-stochastically (deterministic numerical integration of equations using the RK4 algorithm)</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>FS_RESULTS_PATH</code></td>
            <td><code>-R</code>, <code>--results-path</code></td>
            <td>A path to the model results.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>FULL_FIT</code></td>
            <td><code>-F</code>, <code>--full-fit</code></td>
            <td>Whether or not to process the full fit.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>GT_DATA_SOURCE</code></td>
            <td><code>-s</code>, <code>--gt_data_source</code></td>
            <td>Sources of groundtruth data.</td>
            <td><code>'csse_case, fluview_death, hhs_hosp'</code></td>
        </tr>
        <tr>
            <td><code>GT_END_DATE</code></td>
            <td><code>--ground_truth_end</code></td>
            <td>Last date to include ground truth for.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>GT_START_DATE</code></td>
            <td><code>--ground_truth_start</code></td>
            <td>First date to include ground truth for.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>IMM_ESC_PROP</code></td>
            <td><code>--imm_esc_prop</code></td>
            <td>Annual percent of immune escape.</td>
            <td><code>0.35</code></td>
        </tr>
        <tr>
            <td><code>INCL_AGGR_LIKELIHOOD</code></td>
            <td><code>-a</code>, <code>--incl_aggr_likelihood</code></td>
            <td>Whether or not the likelihood should be calculated with aggregate estimates.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>IN_FILENAME</code></td>
            <td>N/A</td>
            <td>Name of input files.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>INIT_FILENAME</code></td>
            <td><code>--init_file_name</code></td>
            <td>Initial file global intermediate name.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>INTERACTIVE_RUN</code></td>
            <td><code>-I</code>, <code>--is-interactive</code></td>
            <td>Whether or not the current run is interactive.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>JOB_NAME</code></td>
            <td><code>--job-name</code></td>
            <td>Unique job name (intended for use when submitting to SLURM).</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>LAST_JOB_OUTPUT</code></td>
            <td>N/A</td>
            <td>Path to output of last job.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>OLD_FLEPI_RUN_INDEX</code></td>
            <td>N/A</td>
            <td>Run ID of old flepiMoP run.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>OUT_FILENAME</code></td>
            <td>N/A</td>
            <td>Name of output files.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>OUT_FILENAME_DIR</code></td>
            <td>N/A</td>
            <td>Directory for output files.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>OUTPUTS</code></td>
            <td><code>-o</code>, <code>--select-outputs</code></td>
            <td>A list of outputs to plot.</td>
            <td><code>'hosp, hnpi, snpi, llik'</code></td>
        </tr>
        <tr>
            <td><code>PARQUET_TYPES</code></td>
            <td>N/A</td>
            <td>Parquet files.</td>
            <td><code>'seed spar snpi seir hpar hnpi hosp llik init'</code></td>
        </tr>
        <tr>
            <td><code>PATH</code></td>
            <td>N/A</td>
            <td>Path relating to AWS installation. Used during SLURM runs.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>PROCESS</code></td>
            <td><code>-r</code>, <code>--run-processing</code></td>
            <td>Whether or not to process the run.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>PROJECT_PATH</code></td>
            <td><code>-d</code>, <code>--data_path</code></td>
            <td>Path to the folder with configs and model output.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>PULL_GT</code></td>
            <td><code>-g</code>, <code>--pull-gt</code></td>
            <td>Whether or not to pull ground truth data.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>PYTHON_PATH</code></td>
            <td><code>-y</code>, <code>--python</code></td>
            <td>Path to Python executable.</td>
            <td><code>'python3'</code></td>
        </tr>
        <tr>
            <td><code>RESUMED_CONFIG_PATH</code></td>
            <td><code>--res_config</code></td>
            <td>Path to previous config file, if using resumes.</td>
            <td><code>NA</code></td>
        </tr>
        <tr>
            <td><code>RESUME_DISCARD_SEEDING</code></td>
            <td><code>--resume-discard-seeding</code>, <code>--resume-carry-seeding</code></td>
            <td>Whether or not to keep seeding in resume runs.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>RESUME_LOCATION</code></td>
            <td><code>-r</code>, <code>--restart-from-location</code></td>
            <td>The location (folder or an S3 bucket) where the previous run is stored.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>RESUME_RUN</code></td>
            <td><code>-R</code>, <code>--is-resume</code></td>
            <td>Whether or not this run is a resume.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>RSCRIPT_PATH</code></td>
            <td><code>-r</code>, <code>--rpath</code></td>
            <td>Path to R executable.</td>
            <td><code>'Rscript'</code></td>
        </tr>
        <tr>
            <td><code>RUN_INTERACTIVE</code></td>
            <td><code>-I</code>, <code>--is-interactive</code></td>
            <td>Whether or not the current run is interactive.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>SAVE_HOSP</code></td>
            <td><code>-H</code>, <code>--save_hosp</code></td>
            <td>Whether or not the HOSP output files should be saved for each iteration.</td>
            <td><code>True</code></td>
        </tr>
        <tr>
            <td><code>SAVE_SEIR</code></td>
            <td><code>-S</code>, <code>--save_seir</code></td>
            <td>Whether or not the SEIR output files should be saved for each iteration.</td>
            <td><code>False</code></td>
        </tr>
        <tr>
            <td><code>SEED_VARIANTS</code></td>
            <td><code>-s</code>, <code>--seed_variants</code></td>
            <td>Whether or not to add variants/subtypes to outcomes in seeding.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>SIMS_PER_JOB</code></td>
            <td>N/A</td>
            <td>Simulations per job.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>SLACK_CHANNEL</code></td>
            <td><code>-s</code>, <code>--slack-channel</code></td>
            <td>Slack channel, either 'csp-production' or 'debug'; or 'noslack' to disable slack.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>SLACK_TOKEN</code></td>
            <td><code>-s</code>, <code>--slack-token</code></td>
            <td>Slack token.</td>
            <td>--</td>
        </tr>
        <tr>
            <td><code>SUBPOP_LENGTH</code></td>
            <td><code>-g</code>, <code>--subpop_len</code></td>
            <td>Number of digits in subpops.</td>
            <td><code>5</code></td>
        </tr>
        <tr>
            <td><code>S3_MODEL_PROJECT_PATH</code></td>
            <td>N/A</td>
            <td>Location in S3 bucket with the code, data, and dvc pipeline.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>S3_RESULTS_PATH</code></td>
            <td>N/A</td>
            <td>Location in S3 to store results.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>S3_UPLOAD</code></td>
            <td>N/A</td>
            <td>Whether or not to upload S3 buckets.</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td><code>VALIDATION_DATE</code></td>
            <td><code>--validation-end-date</code></td>
            <td>First date of projection/forecast (first date without ground truth data).</td>
            <td><code>date.today()</code></td>
        </tr>
    </tbody>
</table>









