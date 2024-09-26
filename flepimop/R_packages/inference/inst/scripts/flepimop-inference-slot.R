#! /usr/bin/env Rscript

# About

## This script runs a single slot (MCMC chain) of an inference run. It can be called directly, but is often called from flepimop-inference-main if multiple slots are run.

# Run Options ---------------------------------------------------------------------

suppressMessages(library(readr))
suppressWarnings(suppressMessages(library(flepicommon)))
suppressMessages(library(stringr))
suppressMessages(library(foreach))
suppressMessages(library(magrittr))
suppressMessages(library(xts))
suppressMessages(library(reticulate))
suppressMessages(library(truncnorm))
suppressMessages(library(parallel))
suppressMessages(library(purrr))
options(warn = 1)
options(readr.num_columns = 0)

required_packages <- c("dplyr", "magrittr", "xts", "zoo", "stringr")

#Temporary
#print("Setting random number seed")
#set.seed(1) # set within R
#reticulate::py_run_string(paste0("rng_seed = ", 1)) #set within Python

# There are multiple ways to specify options when flepimop-inference-slot is run, which take the following precedence:
#  1) (optional) options called along with the script at the command line (ie > flepimop-inference-slot -c my_config.yml)
#  2) (optional) environmental variables set by the user (ie user could set > export CONFIG_PATH = ~/flepimop_sample/my_config.yml to not have t specify it each time the script is run)
# If neither are specified, then a default value is used, given by the second argument of Sys.getenv() commands below.
#  *3) For some options, a default doesn't exist, and the value specified in the config will be used if the option is not specified at the command line or by an environmental variable (iterations_per_slot, slots)

option_list = list(
  optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
  optparse::make_option(c("-u","--run_id"), action="store", type='character', help="Unique identifier for this run", default = Sys.getenv("FLEPI_RUN_INDEX",flepicommon::run_id())),
  optparse::make_option(c("-s", "--seir_modifiers_scenarios"), action="store", default=Sys.getenv("FLEPI_SEIR_SCENARIOS", 'all'), type='character', help="name of the intervention to run, or 'all' to run all of them"),
  optparse::make_option(c("-d", "--outcome_modifiers_scenarios"), action="store", default=Sys.getenv("FLEPI_OUTCOME_SCENARIOS", 'all'), type='character', help="name of the outcome scenarios to run, or 'all' to run all of them"),
  optparse::make_option(c("-j", "--jobs"), action="store", default=Sys.getenv("FLEPI_NJOBS", parallel::detectCores()), type='integer', help="Number of jobs to run in parallel"),
  optparse::make_option(c("-k", "--iterations_per_slot"), action="store", default=Sys.getenv("FLEPI_ITERATIONS_PER_SLOT", NA), type='integer', help = "number of iterations to run for this slot"),
  optparse::make_option(c("-i", "--this_slot"), action="store", default=Sys.getenv("FLEPI_SLOT_INDEX", 1), type='integer', help = "id of this slot"),
  optparse::make_option(c("-b", "--this_block"), action="store", default=Sys.getenv("FLEPI_BLOCK_INDEX",1), type='integer', help = "id of this block"),
  optparse::make_option(c("-t", "--stoch_traj_flag"), action="store", default=Sys.getenv("FLEPI_STOCHASTIC_RUN",FALSE), type='logical', help = "Stochastic SEIR and outcomes trajectories if true"),
  optparse::make_option(c("--ground_truth_start"), action = "store", default = Sys.getenv("GT_START_DATE", ""), type = "character", help = "First date to include groundtruth for"),
  optparse::make_option(c("--ground_truth_end"), action = "store", default = Sys.getenv("GT_END_DATE", ""), type = "character", help = "Last date to include groundtruth for"),
  optparse::make_option(c("-p", "--flepi_path"), action="store", type='character', help="path to the flepiMoP directory", default = Sys.getenv("FLEPI_PATH", "flepiMoP/")),
  optparse::make_option(c("-y", "--python"), action="store", default=Sys.getenv("PYTHON_PATH","python3"), type='character', help="path to python executable"),
  optparse::make_option(c("-r", "--rpath"), action="store", default=Sys.getenv("RSCRIPT_PATH","Rscript"), type = 'character', help = "path to R executable"),
  optparse::make_option(c("-R", "--is-resume"), action="store", default=Sys.getenv("RESUME_RUN",FALSE), type = 'logical', help = "Is this run a resume"),
  optparse::make_option(c("-I", "--is-interactive"), action="store", default=Sys.getenv("RUN_INTERACTIVE",Sys.getenv("INTERACTIVE_RUN", FALSE)), type = 'logical', help = "Is this run an interactive run"),
  optparse::make_option(c("-L", "--reset_chimeric_on_accept"), action = "store", default = Sys.getenv("FLEPI_RESET_CHIMERICS", TRUE), type = 'logical', help = 'Should the chimeric parameters get reset to global parameters when a global acceptance occurs'),
  optparse::make_option(c("-S","--save_seir"), action = "store", default = Sys.getenv("SAVE_SEIR", FALSE), type = 'logical', help = 'Should the SEIR output files be saved for each iteration'),
  optparse::make_option(c("-H","--save_hosp"), action = "store", default = Sys.getenv("SAVE_HOSP", TRUE), type = 'logical', help = 'Should the HOSP output files be saved for each iteration'),
  optparse::make_option(c("-M", "--memory_profiling"), action = "store", default = Sys.getenv("FLEPI_MEM_PROFILE", FALSE), type = 'logical', help = 'Should the memory profiling be run during iterations'),
  optparse::make_option(c("-P", "--memory_profiling_iters"), action = "store", default = Sys.getenv("FLEPI_MEM_PROF_ITERS", 100), type = 'integer', help = 'If doing memory profiling, after every X iterations run the profiler'),
  optparse::make_option(c("-g", "--subpop_len"), action="store", default=Sys.getenv("SUBPOP_LENGTH", 5), type='integer', help = "number of digits in subpop"),
  optparse::make_option(c("-a", "--incl_aggr_likelihood"), action = "store", default = Sys.getenv("INCL_AGGR_LIKELIHOOD", FALSE), type = 'logical', help = 'Should the likelihood be calculated with the aggregate estiamtes.')
)

parser=optparse::OptionParser(option_list=option_list)
opt = optparse::parse_args(parser)


if (opt[["is-interactive"]]) {
  options(error=recover)
} else {
  options(
    error = function(...) {
      quit(..., status = 2)
    }
  )
}
flepicommon::prettyprint_optlist(opt)

# Simulation set up ---------------------------------------------------------------------

## Load Python ---------------------------------------------------------------------


# load Python to use via R
reticulate::use_python(Sys.which(opt$python), required = TRUE)

# Load gempyor module
gempyor <- reticulate::import("gempyor")

# Loads the config file
if (opt$config == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a config YAML file with either -c option or CONFIG_PATH environment variable."
  ))
}
config = flepicommon::load_config(opt$config)

opt$total_ll_multiplier <- 1
if (!is.null(config$inference$incl_aggr_likelihood)){
    print("Using config option for `incl_aggr_likelihood`.")
    opt$incl_aggr_likelihood <- config$inference$incl_aggr_likelihood
    if (!is.null(config$inference$total_ll_multiplier)){
        print("Using config option for `total_ll_multiplier`.")
        opt$total_ll_multiplier <- config$inference$total_ll_multiplier
    } else {
        opt$total_ll_multiplier <- 1
    }
}

## Check for errors in config ---------------------------------------------------------------------

##  seeding section
if (!is.null(config$seeding)){
  if (('perturbation_sd' %in% names(config$seeding))) {
    if (('date_sd' %in% names(config$seeding))) {
      stop("Both the key seeding::perturbation_sd and the key seeding::date_sd are present in the config file, but only one allowed.")
    }
    config$seeding$date_sd <- config$seeding$perturbation_sd
  }
  if (!('date_sd' %in% names(config$seeding))) {
    stop("Neither the key seeding::perturbation_sd nor the key seeding::date_sd are present in the config file, but one is required. They can be set to zero if desired.")
  }
  if (!('amount_sd' %in% names(config$seeding))) {
    config$seeding$amount_sd <- 1
  }
  if (!(config$seeding$method %in% c('FolderDraw'))){
    stop("Inference requires the seeding method be 'FolderDraw' if seeding section is included")
  }
} else {
  print("⚠️ No seeding: section found in config >> not using or fitting seeding.")
}

##  initial condition section
infer_initial_conditions <- FALSE
if (!is.null(config$initial_conditions)){
  if (('perturbation' %in% names(config$initial_conditions))) {
    infer_initial_conditions <- TRUE
    if (!(config$initial_conditions$method %in% c('SetInitialConditionsFolderDraw'))){
      stop("If initial conditions are being inferred (if perturbation column exists), then the initial_condition method must be 'SetInitialConditionsFolderDraw'")
    }
    if (!(config$initial_conditions$proportional)){
      stop("If initial conditions are being inferred (if perturbation column exists), then initial_condition$proportional must be set to TRUE")
    }
  }
} else {
  print("⚠️ No initial_conditions: section found in config >> not starting with or fitting initial_conditions.")
}


## Data options + loading ---------------------------------------------------------------------

### Data on subpopulations (geodata)

# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$subpop_setup$state_level) && config$subpop_setup$state_level, TRUE, FALSE)

##Load information on geographic locations from geodata file.
suppressMessages(
  geodata <- flepicommon::load_geodata_file(
    paste(
      config$subpop_setup$geodata, sep = "/"
    ),
    subpop_len = ifelse(config$name == "USA", opt$subpop_len, 0),
    state_name = ifelse(config$name == "USA" & state_level == TRUE, TRUE, FALSE)
  )
)
obs_subpop <- "subpop"

### Ground truth data (observations of disease)

##Define data directory and create if it does not exist
gt_data_path <- config$inference$gt_data_path

## backwards compatibility with configs that don't have inference$gt_source parameter will use the previous default data source (USA Facts)
if (is.null(config$inference$gt_source)){
  gt_source <- "usafacts"
} else{
  gt_source <- config$inference$gt_source
}

gt_scale <- ifelse(state_level, "US state", "US county")
subpops_ <- geodata[[obs_subpop]]

gt_start_date <- lubridate::ymd(config$start_date)
if (opt$ground_truth_start != "") {
  gt_start_date <- lubridate::ymd(opt$ground_truth_start)
} else if (!is.null(config$start_date_groundtruth)) {
  gt_start_date <- lubridate::ymd(config$start_date_groundtruth)
}
if (gt_start_date < lubridate::ymd(config$start_date)) {
  gt_start_date <- lubridate::ymd(config$start_date)
}

gt_end_date <- lubridate::ymd(config$end_date)
if (opt$ground_truth_end != "") {
  gt_end_date <- lubridate::ymd(opt$ground_truth_end)
} else if (!is.null(config$end_date_groundtruth)) {
  gt_end_date <- lubridate::ymd(config$end_date_groundtruth)
}
if (gt_end_date > lubridate::ymd(config$end_date)) {
  gt_end_date <- lubridate::ymd(config$end_date)
}


## Scenario Options ----------------------------------------------


##Load simulations per slot from config if not defined on command line
##command options take precedence
if (is.na(opt$iterations_per_slot)){
  opt$iterations_per_slot <- config$inference$iterations_per_slot
}
print(paste("Running",opt$iterations_per_slot,"simulations for slot ",opt$this_slot))

# if opt$outcome_modifiers_scenarios is specified
#  --> run only those scenarios
#  If it is not or is "all"

##If intervention scenarios are specified check their existence

seir_modifiers_scenarios <- opt$seir_modifiers_scenarios
if (all(seir_modifiers_scenarios == "all")) {
  if (!is.null(config$seir_modifiers$scenarios)){
    seir_modifiers_scenarios <- config$seir_modifiers$scenarios
  } else {
    seir_modifiers_scenarios <- "all"
  }
} else if (!all(seir_modifiers_scenarios %in% config$seir_modifiers$scenarios)) {
  message(paste("Invalid intervention scenario arguments: [", paste(setdiff(seir_modifiers_scenarios, config$seir_modifiers$scenarios)),
                "] did not match any of the named args in ", paste(config$seir_modifiers$scenarios, collapse = ", "), "\n"))
  quit("yes", status=1)
}

##If outcome scenarios are specified check their existence
outcome_modifiers_scenarios <- opt$outcome_modifiers_scenarios
if (all(outcome_modifiers_scenarios == "all")) {
  if (!is.null(config$outcome_modifiers$scenarios)){
    outcome_modifiers_scenarios <- config$outcome_modifiers$scenarios
  } else {
    outcome_modifiers_scenarios <- "all"
  }
} else if (!all(outcome_modifiers_scenarios %in% config$outcome_modifiers$scenarios)) {
  message(paste("Invalid outcome scenario arguments: [",paste(setdiff(outcome_modifiers_scenarios, config$outcome_modifiers$scenarios)),
                "] did not match any of the named args in", paste(config$outcome_modifiers$scenarios, collapse = ", "), "\n"))
  quit("yes", status=1)
}


## Other Stats + Inference Options ----------------------------------------

##Create heirarchical stats object if specified
hierarchical_stats <- list()
if ("hierarchical_stats_geo" %in% names(config$inference)) {
  hierarchical_stats <- config$inference$hierarchical_stats_geo
}

##Create priors if specified
defined_priors <- list()
if ("priors" %in% names(config$inference)) {
  defined_priors <- config$inference$priors
}


# Setup Obs, Initial Stats, and Likelihood fn -----------------------------

# ~ WITH Inference ----------------------------------------------------

if (config$inference$do_inference){

  ## Load ground truth data

  obs <- suppressMessages(
    readr::read_csv(config$inference$gt_data_path,
                    col_types = readr::cols(date = readr::col_date(),
                                            # source = readr::col_character(),
                                            subpop = readr::col_character(),
                                            .default = readr::col_double()), )) %>%
    dplyr::filter(subpop %in% subpops_, date >= gt_start_date, date <= gt_end_date) %>%
    dplyr::right_join(tidyr::expand_grid(subpop = unique(.$subpop), date = unique(.$date))) #%>%
    # dplyr::mutate_if(is.numeric, dplyr::coalesce, 0)


  # add aggregate groundtruth to the obs data for the likelihood calc
  if (opt$incl_aggr_likelihood){
      obs <- obs %>%
          dplyr::bind_rows(
              obs %>%
                  dplyr::select(date, where(is.numeric)) %>%
                  dplyr::group_by(date) %>%
                  dplyr::summarise(across(everything(), sum)) %>% # no likelihood is calculated for time periods with missing data for any subpop
                  dplyr::mutate(source = "Total",
                                subpop = "Total")
          )
  }

  subpopnames <- unique(obs[[obs_subpop]])


  ## Compute statistics

  ##   for each subpopulation, processes the data as specified in config - finds data types of interest, aggregates by specified period (e.g week), deals with zeros and NAs, etc
  data_stats <- lapply(
    subpopnames,
    function(x) {
      df <- obs[obs[[obs_subpop]] == x, ]
      inference::getStats(
        df,
        "date",
        "data_var",
        stat_list = config$inference$statistics,
        start_date = gt_start_date,
        end_date = gt_end_date
      )
    }) %>%
    set_names(subpopnames)


  # function to calculate the likelihood when comparing simulation output (sim_hosp) to ground truth data
  likelihood_calculation_fun <- function(sim_hosp){

    sim_hosp <- dplyr::filter(sim_hosp, sim_hosp$date >= min(obs$date), sim_hosp$date <= max(obs$date))
    lhs <- unique(sim_hosp[[obs_subpop]])
    rhs <- unique(names(data_stats))
    all_locations <- rhs[rhs %in% lhs]

    ## No references to config$inference$statistics
    inference::aggregate_and_calc_loc_likelihoods(
      all_locations = all_locations, # technically different
      modeled_outcome = sim_hosp, # simulation output
      obs_subpop = obs_subpop,
      targets_config = config[["inference"]][["statistics"]],
      obs = obs,
      ground_truth_data = data_stats,
      hosp_file = first_global_files[['llik_filename']],
      hierarchical_stats = hierarchical_stats,
      defined_priors = defined_priors,
      geodata = geodata,
      snpi = flepicommon::read_parquet_with_check(first_global_files[['snpi_filename']]),
      hnpi = flepicommon::read_parquet_with_check(first_global_files[['hnpi_filename']]),
      hpar = dplyr::mutate(flepicommon::read_parquet_with_check(first_global_files[['hpar_filename']]),parameter=paste(quantity,!!rlang::sym(obs_subpop),outcome,sep='_')),
      start_date = gt_start_date,
      end_date = gt_end_date
    )
  }
  print("Running WITH inference")


  # ~ WITHOUT Inference ---------------------------------------------------

} else {

  subpopnames <- obs_subpop

  likelihood_calculation_fun <- function(sim_hosp){

    all_locations <- unique(sim_hosp[[obs_subpop]])

    ## No references to config$inference$statistics
    inference::aggregate_and_calc_loc_likelihoods(
      all_locations = all_locations, # technically different
      modeled_outcome = sim_hosp,
      obs_subpop = obs_subpop,
      targets_config = config[["inference"]][["statistics"]],
      obs = sim_hosp,
      ground_truth_data = sim_hosp,
      hosp_file = first_global_files[['llik_filename']],
      hierarchical_stats = hierarchical_stats,
      defined_priors = defined_priors,
      geodata = geodata,
      snpi = flepicommon::read_parquet_with_check(first_global_files[['snpi_filename']]),
      hnpi = flepicommon::read_parquet_with_check(first_global_files[['hnpi_filename']]),
      hpar = dplyr::mutate(flepicommon::read_parquet_with_check(first_global_files[['hpar_filename']]),
                           parameter=paste(quantity,!!rlang::sym(obs_subpop),outcome,sep='_')),
      start_date = gt_start_date,
      end_date = gt_end_date
    )
  }
  print("Running WITHOUT inference")
}





# Run Model Looping through Scenarios -------------------------------------

print(paste("Chimeric reset is", (opt$reset_chimeric_on_accept)))
print(names(opt))

if (!opt$reset_chimeric_on_accept) {
  warning("We recommend setting reset_chimeric_on_accept TRUE, since reseting chimeric chains on global acceptances more closely matches normal MCMC behaviour")
}

if(!opt$save_seir){
  warning("To save space, intermediate SEIR files will not be saved for every iteration of the MCMC inference procedure. To save these files, set option save_seir TRUE.")
}

if(!opt$save_hosp){
  warning("To save space, intermediate HOSP files will not be saved for every iteration of the MCMC inference procedure. To save these files, set option save_hosp TRUE.")
}

if(opt$memory_profiling){
  print(paste("Inference will run memory profiling every",opt$memory_profiling_iters,"iterations"))
}


for(seir_modifiers_scenario in seir_modifiers_scenarios) {

  if (!is.null(config$seir_modifiers)){
    print(paste0("Running seir modifier scenario: ", seir_modifiers_scenario))
  } else {
    print(paste0("No seir modifier scenarios"))
    seir_modifiers_scenario <- NULL
  }

  for(outcome_modifiers_scenario in outcome_modifiers_scenarios) {

    if (!is.null(config$outcome_modifiers)){
      print(paste0("Running outcome modifier scenario: ", outcome_modifiers_scenario))
    } else {
      print(paste0("No outcome modifier scenarios"))
      outcome_modifiers_scenario <- NULL
    }

    # If no seir or outcome scenarios, instead pass py_none() to Gempyor (which assigns no value to the scenario)
    if (is.null(seir_modifiers_scenario)){
      seir_modifiers_scenario <- reticulate::py_none()
    }
    if (is.null(outcome_modifiers_scenario)){
      outcome_modifiers_scenario <- reticulate::py_none()
    }

    reset_chimeric_files <- FALSE # this turns on whenever a global acceptance occurs



# ~ Set up first iteration of chain ---------------------------------------

    ###  Create python simulator object

    # Create parts of filename pieces for simulation to save output with
    # flepicommon::create_prefix is roughly equivalent to paste(...) with some specific formatting rule
    chimeric_intermediate_filepath_suffix <- flepicommon::create_prefix(prefix="",'chimeric','intermediate',sep='/',trailing_separator='')
    global_intermediate_filepath_suffix <- flepicommon::create_prefix(prefix="",'global','intermediate',sep='/',trailing_separator='')
    slot_filename_prefix <- flepicommon::create_prefix(slot=list(opt$this_slot,"%09d"), sep='.', trailing_separator='.')
    slotblock_filename_prefix <- flepicommon::create_prefix(slot=list(opt$this_slot,"%09d"), block=list(opt$this_block,"%09d"), sep='.', trailing_separator='.')

    ## python configuration: build simulator model specified in config
    tryCatch({
      gempyor_inference_runner <- gempyor$GempyorInference(
        config_filepath=opt$config,
        stoch_traj_flag=opt$stoch_traj_flag,
        run_id=opt$run_id,
        prefix=reticulate::py_none(), # we let gempyor create setup prefix
        inference_filepath_suffix=global_intermediate_filepath_suffix,
        inference_filename_prefix=slotblock_filename_prefix,
        autowrite_seir = TRUE
      )
    }, error = function(e) {
      print("GempyorInference failed to run (call on l. 443 of flepimop-inference-slot).")
      print("Here is all the debug information I could find:")
      for(m in reticulate::py_last_error()) print(m)
      stop("GempyorInference failed to run... stopping")
    })
    setup_prefix <- gempyor_inference_runner$modinf$get_setup_name() # file name piece of the form [config$name]_[seir_modifier_scenario]_[outcome_modifier_scenario]
    print("gempyor_inference_runner created successfully.")


    # Get names of files where output from the initial simulation will be saved
    ## {prefix}/{run_id}/{type}/{suffix}/{prefix}.{index = block-1}.{run_id}.{type}.{ext}
    ## N.B.: prefix should end in "{slot}." NOTE: Potential problem. Prefix is {slot}.{block} but then "index" includes block also??
    first_global_files <- inference::create_filename_list(run_id=opt$run_id,
                                                          prefix=setup_prefix,
                                                          filepath_suffix=global_intermediate_filepath_suffix,
                                                          filename_prefix=slotblock_filename_prefix,
                                                          index=opt$this_block - 1)
    first_chimeric_files <- inference::create_filename_list(run_id=opt$run_id,
                                                            prefix=setup_prefix,
                                                            filepath_suffix=chimeric_intermediate_filepath_suffix,
                                                            filename_prefix=slotblock_filename_prefix,
                                                            index=opt$this_block - 1)

    print("RUNNING: MCMC initialization for the first block")
    # Output saved to files of the form {setup_prefix}/{run_id}/{type}/global/intermediate/{slotblock_filename_prefix}.(block-1).{run_id}.{type}.{ext}
    # also copied into the /chimeric/ version, which are referenced by first_global_files and first_chimeric_files
    inference::initialize_mcmc_first_block(
      run_id = opt$run_id,
      block = opt$this_block,
      setup_prefix = setup_prefix,
      global_intermediate_filepath_suffix = global_intermediate_filepath_suffix,
      chimeric_intermediate_filepath_suffix = chimeric_intermediate_filepath_suffix,
      filename_prefix = slotblock_filename_prefix, # might be wrong, maybe should just be slot_filename_prefix
      gempyor_inference_runner = gempyor_inference_runner,
      likelihood_calculation_function = likelihood_calculation_fun,
      is_resume = opt[['is-resume']]
    )
    print("First MCMC block initialized successfully.")

    # So far no acceptances have occurred
    last_accepted_index <- 0

    # get filenames of last accepted files (copy these files when rejections occur)
    last_accepted_global_files <- inference::create_filename_list(run_id=opt$run_id,
                                                                  prefix=setup_prefix,
                                                                  filepath_suffix=global_intermediate_filepath_suffix,
                                                                  filename_prefix=slotblock_filename_prefix,
                                                                  index=last_accepted_index)

    # Load files with the output of initialize_mcmc_first_block

    # load those files (chimeric currently identical to global)
    initial_spar <- flepicommon::read_parquet_with_check(first_chimeric_files[['spar_filename']])
    initial_hpar <- flepicommon::read_parquet_with_check(first_chimeric_files[['hpar_filename']])
    initial_snpi <- flepicommon::read_parquet_with_check(first_chimeric_files[['snpi_filename']])
    initial_hnpi <- flepicommon::read_parquet_with_check(first_chimeric_files[['hnpi_filename']])
    if (!is.null(config$initial_conditions)){
      initial_init <- flepicommon::read_parquet_with_check(first_chimeric_files[['init_filename']])
    }
    if (!is.null(config$seeding)){
      seeding_col_types <- NULL
      suppressMessages(initial_seeding <- readr::read_csv(first_chimeric_files[['seed_filename']], col_types=seeding_col_types))

      if (opt$stoch_traj_flag) {
        initial_seeding$amount <- as.integer(round(initial_seeding$amount))
      }
    }else{
      initial_seeding <- NULL
    }
    chimeric_current_likelihood_data <- flepicommon::read_parquet_with_check(first_chimeric_files[['llik_filename']])
    global_current_likelihood_data <- flepicommon::read_parquet_with_check(first_global_files[['llik_filename']]) # they are the same ... don't need to load both


    ##### Get the full likelihood (WHY IS THIS A DATA FRAME?)
    # Compute total loglik for each sim
    global_current_likelihood_total <- sum(global_current_likelihood_data$ll)

    #####LOOP NOTES
    ### this_index is the current MCMC iteration
    ### last_accepted_index is the index of the most recent globally accepted iternation

    startTimeCount=Sys.time()



# ~ Loop through Simulations ----------------------------------------------

    # keep track of running average global acceptance rate, since old global likelihood data not kept in memory. Each geoID has same value for acceptance rate in global case, so we just take the 1st entry
    old_avg_global_accept_rate <- global_current_likelihood_data$accept_avg[1]
    old_avg_chimeric_accept_rate <- chimeric_current_likelihood_data$accept_avg

    for (this_index in seq_len(opt$iterations_per_slot)) {

      print(paste("Running iteration", this_index))

      startTimeCountEach = Sys.time()

      ## Create filenames to save output from each iteration
      this_global_files <- inference::create_filename_list(run_id=opt$run_id,  prefix = setup_prefix, filepath_suffix=global_intermediate_filepath_suffix, filename_prefix=slotblock_filename_prefix, index=this_index)
      this_chimeric_files <- inference::create_filename_list(run_id=opt$run_id, prefix = setup_prefix, filepath_suffix=chimeric_intermediate_filepath_suffix, filename_prefix=slotblock_filename_prefix, index=this_index)

      ### Perturb accepted parameters to get proposed parameters ----

      # since the first iteration is accepted by default, we don't perturb it, so proposed = initial
      if ((opt$this_block == 1) && (last_accepted_index == 0)) {

        proposed_spar <- initial_spar
        proposed_hpar <- initial_hpar
        proposed_snpi <- initial_snpi
        proposed_hnpi <- initial_hnpi
        if (!is.null(config$initial_conditions)){
          proposed_init <- initial_init
        }
        if (!is.null(config$seeding)){
          proposed_seeding <- initial_seeding
        }

      } else { # perturb each parameter type

        proposed_spar <- initial_spar # currently no function to perturb
        proposed_hpar <- inference::perturb_hpar(initial_hpar, config$outcomes$outcomes) # NOTE: Deprecated?? ?no scenarios possible right now?

        if (!is.null(config$seir_modifiers$modifiers)){
          proposed_snpi <- inference::perturb_snpi(initial_snpi, config$seir_modifiers$modifiers)
        }
        if (!is.null(config$outcome_modifiers$modifiers)){
          proposed_hnpi <- inference::perturb_hnpi(initial_hnpi, config$outcome_modifiers$modifiers)
        }

        if (!is.null(config$seeding)){
          proposed_seeding <- inference::perturb_seeding(
            seeding = initial_seeding,
            date_sd = config$seeding$date_sd,
            date_bounds = c(gt_start_date, gt_end_date),
            amount_sd = config$seeding$amount_sd,
            continuous = !(opt$stoch_traj_flag)
          )
        } else {
          proposed_seeding <- initial_seeding
        }
        if (!is.null(config$initial_conditions)){
          if (infer_initial_conditions) {
            proposed_init <- inference::perturb_init(initial_init, config$initial_conditions$perturbation)
          } else {
            proposed_init <- initial_init
          }
        }

      }

      # Write proposed parameters to files for other code to read.
      # Temporarily stored in global files, which are eventually overwritten with global accepted values
      arrow::write_parquet(proposed_spar,this_global_files[['spar_filename']])
      arrow::write_parquet(proposed_hpar,this_global_files[['hpar_filename']])
      arrow::write_parquet(proposed_snpi,this_global_files[['snpi_filename']])
      arrow::write_parquet(proposed_hnpi,this_global_files[['hnpi_filename']])
      if (!is.null(config$seeding)){
        readr::write_csv(proposed_seeding, this_global_files[['seed_filename']])
      }
      if (!is.null(config$initial_conditions)){
        arrow::write_parquet(proposed_init, this_global_files[['init_filename']])
      }

      ## Run the simulator with proposed parameters -------------------

      # create simulator
      tryCatch({
        gempyor_inference_runner$one_simulation(
          sim_id2write=this_index,
          load_ID=TRUE,
          sim_id2load=this_index)
      }, error = function(e) {
        print("GempyorInference failed to run (call on l. 575 of flepimop-inference-sl).")
        print("Here is all the debug information I could find:")
        for(m in reticulate::py_last_error()) print(m)
        stop("GempyorInference failed to run... stopping")
      })

      # run
      if (config$inference$do_inference){
        sim_hosp <- flepicommon::read_file_of_type(gsub(".*[.]","",this_global_files[['hosp_filename']]))(this_global_files[['hosp_filename']]) %>%
             dplyr::filter(date >= min(obs$date),date <= max(obs$date))

        # add aggregate groundtruth to the obs data for the likelihood calc
        if (opt$incl_aggr_likelihood){
            sim_hosp <- sim_hosp %>%
                dplyr::bind_rows(
                    sim_hosp %>%
                        dplyr::select(-tidyselect::all_of(obs_subpop)) %>%
                        dplyr::group_by(date) %>%
                        dplyr::summarise(dplyr::across(tidyselect::everything(), sum)) %>% # no likelihood is calculated for time periods with missing data for any subpop
                        dplyr::mutate(!!obs_subpop := "Total")
                )
        }
        lhs <- unique(sim_hosp[[obs_subpop]])
        rhs <- unique(names(data_stats))
        all_locations <- rhs[rhs %in% lhs]
      } else {
        sim_hosp <- flepicommon::read_file_of_type(gsub(".*[.]","",this_global_files[['hosp_filename']]))(this_global_files[['hosp_filename']])
        all_locations <- unique(sim_hosp[[obs_subpop]])
        obs <- sim_hosp
        data_stats <- sim_hosp
      }

      ## Compare model output to data and calculate likelihood ----
      proposed_likelihood_data <- inference::aggregate_and_calc_loc_likelihoods(
        all_locations = all_locations,
        modeled_outcome = sim_hosp,
        obs_subpop = obs_subpop,
        targets_config = config[["inference"]][["statistics"]],
        obs = obs,
        ground_truth_data = data_stats,
        hosp_file = this_global_files[["llik_filename"]],
        hierarchical_stats = hierarchical_stats,
        defined_priors = defined_priors,
        geodata = geodata,
        snpi = proposed_snpi,
        hnpi = proposed_hnpi,
        hpar = dplyr::mutate(
          proposed_hpar,
          parameter = paste(quantity, !!rlang::sym(obs_subpop), outcome, sep = "_")
        ),
        start_date = gt_start_date,
        end_date = gt_end_date
      )

      rm(sim_hosp)

      # multiply aggregate likelihood by a factor if specified in config
      if (opt$incl_aggr_likelihood){
        proposed_likelihood_data$ll[proposed_likelihood_data$subpop == "Total"] <- proposed_likelihood_data$ll[proposed_likelihood_data$subpop == "Total"] * opt$total_ll_multiplier
      }


      # write proposed likelihood to global file
      arrow::write_parquet(proposed_likelihood_data, this_global_files[['llik_filename']])

      ## UNCOMMENT TO DEBUG
      # print('current global likelihood')
      # print(global_current_likelihood_data)
      # print('current chimeric likelihood')
      # print(chimeric_current_likelihood_data)
      #print('proposed likelihood')
      #print(proposed_likelihood_data)

      ## Compute total loglik for each sim
      proposed_likelihood_total <- sum(proposed_likelihood_data$ll)
      ## For logging
      print(paste("Current likelihood",formatC(global_current_likelihood_total,digits=2,format="f"),"Proposed likelihood",
                  formatC(proposed_likelihood_total,digits=2,format="f")))


      ## Global likelihood acceptance or rejection decision -----------

      # Compare total likelihood (product of all subpopulations) in current vs proposed likelihood.
      # Accept if MCMC acceptance decision = 1 or it's the first iteration of the first block
      # note - we already have a catch for the first block thing earlier (we set proposed = initial likelihood) - shouldn't need 2!
      global_accept <- ifelse(  #same value for all subpopulations
        inference::iterateAccept(global_current_likelihood_total, proposed_likelihood_total) ||
          ((last_accepted_index == 0) && (opt$this_block == 1)) ||
          ((this_index == opt$iterations_per_slot && !opt$reset_chimeric_on_accept))
        ,1,0
      )

      # only do global accept if all subpopulations accepted?
      if (global_accept == 1 | config$inference$do_inference == FALSE) {

        print("**** GLOBAL ACCEPT (Recording) ****")

        if ((opt$this_block == 1) && (last_accepted_index == 0)) {
          print("by default because it's the first iteration of a block 1")
        } else {
          # gempyor_inference_runner$write_last_seir(sim_id2write=this_index)
        }

        # delete previously accepted files if using a space saving option
        if(!opt$save_seir){
          file.remove(last_accepted_global_files[['seir_filename']]) # remove proposed SEIR file
        }
        if(!opt$save_hosp){
          file.remove(last_accepted_global_files[['hosp_filename']]) # remove proposed HOSP file
        }

        # Update the index of the most recent globally accepted parameters
        last_accepted_index <- this_index

        # update filenames of last accepted files
        last_accepted_global_files <- inference::create_filename_list(run_id=opt$run_id,
                                                                      prefix=setup_prefix,
                                                                      filepath_suffix=global_intermediate_filepath_suffix,
                                                                      filename_prefix=slotblock_filename_prefix,
                                                                      index=last_accepted_index)

        if (opt$reset_chimeric_on_accept) {
          reset_chimeric_files <- TRUE # triggers globally accepted parameters to push back to chimeric
        }

        # Update current global likelihood to proposed likelihood and record some acceptance statistics

        #acceptance probability for this iteration
        this_accept_prob <- exp(min(c(0, proposed_likelihood_total - global_current_likelihood_total)))

        global_current_likelihood_data <- proposed_likelihood_data # this is used for next iteration
        global_current_likelihood_total <- proposed_likelihood_total # this is used for next iteration

        global_current_likelihood_data$accept <- 1 # global acceptance decision (0/1), same for each geoID
        global_current_likelihood_data$accept_prob <- this_accept_prob

        # File saving: If global accept occurs, the global parameter files are already correct as they contain the proposed values

      } else {
        print("**** GLOBAL REJECT (Recording) ****")

        # File saving: If global reject occurs, remove "proposed" parameters from global files and instead replacing with the last accepted values

        # Update current global likelihood to last accepted one, and record some acceptance statistics

        # Replace current global files with last accepted values

        # If save_seir = FALSE, don't copy intermediate SEIR files because they aren't being saved
        # If save_hosp = FALSE, don't copy intermediate HOSP files because they aren't being saved
        for (type in names(this_global_files)) {
          if((!opt$save_seir & type!='seir_filename') & (!opt$save_hosp & type!='hosp_filename')){
          # copy if (save_seir = FALSE OR type is not SEIR) AND (save_hosp = FALSE OR type is not HOSP)
          file.copy(last_accepted_global_files[[type]],this_global_files[[type]], overwrite = TRUE)
          }
        }
        if(!opt$save_seir){
          file.remove(this_global_files[['seir_filename']]) # remove proposed SEIR file
        }
        if(!opt$save_hosp){
          file.remove(this_global_files[['hosp_filename']]) # remove proposed HOSP file
        }

        #acceptance probability for this iteration
        this_accept_prob <- exp(min(c(0, proposed_likelihood_total - global_current_likelihood_total)))

        #NOTE: Don't technically need the next 2 lines, as the values saved to memory are last accepted values, but confusing to track these variable names if we skip this
        #global_current_likelihood_data <- flepicommon::read_parquet_with_check(this_global_files[['llik_filename']])
        #global_current_likelihood_total <- sum(global_current_likelihood_data$ll)

        global_current_likelihood_data$accept <- 0 # global acceptance decision (0/1), same for each geoID
        global_current_likelihood_data$accept_prob <- this_accept_prob

      }

      # Calculate more acceptance statistics for the global chain. Same value to each subpopulation
      effective_index <- (opt$this_block - 1) * opt$iterations_per_slot + this_index # index after all blocks
      avg_global_accept_rate <- ((effective_index-1)*old_avg_global_accept_rate + global_accept)/(effective_index)
      global_current_likelihood_data$accept_avg <-avg_global_accept_rate # update running average acceptance probability
      old_avg_global_accept_rate <- avg_global_accept_rate # keep track, since old global likelihood data not kept in memory

      # print(paste("Average global acceptance rate: ",formatC(100*avg_global_accept_rate,digits=2,format="f"),"%"))

      # Update global likelihood files
      arrow::write_parquet(global_current_likelihood_data, this_global_files[['llik_filename']]) # update likelihood saved to file

      ## Chimeric likelihood acceptance or rejection decisions (one round) ---------------------------------------------------------------------------

      if (!reset_chimeric_files) { # will make separate acceptance decision for each subpop

        #  "Chimeric" means GeoID-specific
        print("Making chimeric acceptance decision")

        if (is.null(config$initial_conditions)){
          initial_init <- NULL
          proposed_init <- NULL
        }
        if (is.null(config$seeding)){
          initial_seeding <- NULL
          proposed_seeding <- NULL
        }

        chimeric_acceptance_list <- inference::accept_reject_proposals( # need to rename this function!!
          init_orig = initial_init,
          init_prop = proposed_init,
          seeding_orig = initial_seeding,
          seeding_prop = proposed_seeding,
          snpi_orig = initial_snpi,
          snpi_prop = proposed_snpi,
          hnpi_orig = initial_hnpi,
          hnpi_prop = proposed_hnpi,
          hpar_orig = initial_hpar,
          hpar_prop = proposed_hpar,
          orig_lls = chimeric_current_likelihood_data,
          prop_lls = proposed_likelihood_data
        )

        # Update accepted parameters to start next simulation
        if (!is.null(config$initial_conditions)){
          new_init <- chimeric_acceptance_list$init
        }
        if (!is.null(config$seeding)){
          new_seeding <- chimeric_acceptance_list$seeding
        }
        new_spar <- initial_spar
        new_hpar <- chimeric_acceptance_list$hpar
        new_snpi <- chimeric_acceptance_list$snpi
        new_hnpi <- chimeric_acceptance_list$hnpi
        chimeric_current_likelihood_data <- chimeric_acceptance_list$ll

      } else { # Proposed values were globally accepted and will be copied to chimeric

        print("Resetting chimeric values to global due to global acceptance")
        if (!is.null(config$initial_conditions)){
          new_init <- proposed_init
        }
        if (!is.null(config$seeding)){
          new_seeding <- proposed_seeding
        }
        new_spar <- initial_spar
        new_hpar <- proposed_hpar
        new_snpi <- proposed_snpi
        new_hnpi <- proposed_hnpi
        chimeric_current_likelihood_data <- proposed_likelihood_data

        reset_chimeric_files <- FALSE

        chimeric_current_likelihood_data$accept <- 1
      }

      # Calculate acceptance statistics of the chimeric chain

      effective_index <- (opt$this_block - 1) * opt$iterations_per_slot + this_index
      avg_chimeric_accept_rate <- ((effective_index - 1) * old_avg_chimeric_accept_rate + chimeric_current_likelihood_data$accept) / (effective_index) # running average acceptance rate
      chimeric_current_likelihood_data$accept_avg <- avg_chimeric_accept_rate
      chimeric_current_likelihood_data$accept_prob <- exp(min(c(0, proposed_likelihood_data$ll - chimeric_current_likelihood_data$ll))) #acceptance probability
      old_avg_chimeric_accept_rate <- avg_chimeric_accept_rate

      ## Write accepted chimeric parameters to file
      if (!is.null(config$seeding)){
        readr::write_csv(new_seeding,this_chimeric_files[['seed_filename']])
      }
      if (!is.null(config$initial_conditions)){
        arrow::write_parquet(new_init, this_chimeric_files[['init_filename']])
      }
      arrow::write_parquet(new_spar,this_chimeric_files[['spar_filename']])
      arrow::write_parquet(new_hpar,this_chimeric_files[['hpar_filename']])
      arrow::write_parquet(new_snpi,this_chimeric_files[['snpi_filename']])
      arrow::write_parquet(new_hnpi,this_chimeric_files[['hnpi_filename']])
      arrow::write_parquet(chimeric_current_likelihood_data, this_chimeric_files[['llik_filename']])

      print(paste("Last accepted index is ",last_accepted_index))


      # set initial values to start next iteration
      if (!is.null(config$initial_conditions)){
        initial_init <- new_init
      }
      if (!is.null(config$seeding)){
        initial_seeding<- new_seeding
      }
      initial_spar <- new_spar
      initial_hpar <- new_hpar
      initial_snpi <- new_snpi
      initial_hnpi <- new_hnpi

      # remove "new" and "proposed" values from memory
      rm(proposed_spar, proposed_hpar, proposed_snpi,proposed_hnpi)
      rm(new_spar, new_hpar, new_snpi,new_hnpi)
      if (!is.null(config$initial_conditions)){
        rm(proposed_init)
        rm(new_init)
      }
      if (!is.null(config$seeding)){
        rm(proposed_seeding)
        rm(new_seeding)
      }

      endTimeCountEach=difftime(Sys.time(), startTimeCountEach, units = "secs")
      print(paste("Time to run MCMC iteration",this_index,"of slot",opt$this_slot," is ",formatC(endTimeCountEach,digits=2,format="f")," seconds"))

      # memory profiler to diagnose memory creep

      if (opt$memory_profiling){

        if (this_index %% opt$memory_profiling_iters == 0 | this_index == 1){
          print('doing memory profiling')
          tot_objs_ <- as.numeric(object.size(x=lapply(ls(all.names = TRUE), get)) * 9.31e-10)
          tot_mem_ <- sum(gc()[,2]) / 1000
          curr_obj_sizes <- data.frame('object' = ls()) %>%
            dplyr::mutate(size_unit = object %>% sapply(. %>% get() %>% object.size %>% format(., unit = 'Mb')),
                          size = as.numeric(sapply(strsplit(size_unit, split = ' '), FUN = function(x) x[1])),
                          unit = factor(sapply(strsplit(size_unit, split = ' '), FUN = function(x) x[2]), levels = c('Gb', 'Mb', 'Kb', 'bytes'))) %>%
            dplyr::arrange(unit, dplyr::desc(size)) %>%
            dplyr::select(-size_unit) %>% dplyr::as_tibble() %>%
            dplyr::mutate(unit = as.character(unit))
          curr_obj_sizes <- curr_obj_sizes %>%
            dplyr::add_row(object = c("TOTAL_MEMORY", "TOTAL_OBJECTS"),
                           size = c(tot_mem_, tot_objs_),
                           unit = c("Gb", "Gb"),
                           .before = 1)

          this_global_memprofile <- inference::create_filename_list(run_id=opt$run_id,
                                                                    prefix=setup_prefix, filepath_suffix=global_intermediate_filepath_suffix,
                                                                    filename_prefix=slotblock_filename_prefix, index=this_index,types = "memprof",
                                                                    extensions = "parquet")
          arrow::write_parquet(curr_obj_sizes, this_global_memprofile[['memprof_filename']])
          rm(curr_obj_sizes)
        }

      }

      ## Run garbage collector to clear memory and prevent memory leakage
      # gc_after_a_number <- 1 ## # Garbage collection every 1 iteration
      if (this_index %% 1 == 0){
        gc()
      }

      # Ending this MCMC iteration

    }

    # Ending this MCMC chain (aka "slot")

    # Create "final" files after MCMC chain is completed
    #   Will fail if unsuccessful

    # moves the most recently globally accepted parameter values from global/intermediate file to global/final
    # all file types
    print("Copying latest global files to final")
    cpy_res_global <- inference::perform_MCMC_step_copies_global(current_index = last_accepted_index,
                                                                 slot = opt$this_slot,
                                                                 block = opt$this_block,
                                                                 run_id = opt$run_id,
                                                                 global_intermediate_filepath_suffix = global_intermediate_filepath_suffix,
                                                                 slotblock_filename_prefix = slotblock_filename_prefix,
                                                                 slot_filename_prefix = slot_filename_prefix)
    # if (!prod(unlist(cpy_res_global))) {stop("File copy failed:", paste(unlist(cpy_res_global),paste(names(cpy_res_global),"|")))}

    # moves the most recent chimeric parameter values from chimeric/intermediate file to chimeric/final
    # all file types except seir and hosp
    print("Copying latest chimeric files to final")
    cpy_res_chimeric <- inference::perform_MCMC_step_copies_chimeric(current_index = this_index,
                                                                     slot = opt$this_slot,
                                                                     block = opt$this_block,
                                                                     run_id = opt$run_id,
                                                                     chimeric_intermediate_filepath_suffix = chimeric_intermediate_filepath_suffix,
                                                                     slotblock_filename_prefix = slotblock_filename_prefix,
                                                                     slot_filename_prefix = slot_filename_prefix)
    # if (!prod(unlist(cpy_res_chimeric))) {stop("File copy failed:", paste(unlist(cpy_res_chimeric),paste(names(cpy_res_chimeric),"|")))}

    warning("Chimeric hosp and seir files not yet supported, just using the most recently globally accepted file of each type")

        #NOTE: Don't understand why we write these files that don't have an iteration index. Not sure what used for
    #files of the form ../chimeric/intermediate/{slot}.{block}.{run_id}.{variable}.parquet
    output_chimeric_files <- inference::create_filename_list(run_id=opt$run_id,prefix=setup_prefix,  filepath_suffix=chimeric_intermediate_filepath_suffix, filename_prefix=slot_filename_prefix, index=opt$this_block)

    #files of the form .../global/intermediate/{slot}.{block}.{iteration}.{run_id}.{variable}.parquet
    this_index_global_files <- inference::create_filename_list(run_id=opt$run_id,prefix=setup_prefix, filepath_suffix=global_intermediate_filepath_suffix, filename_prefix=slotblock_filename_prefix, index=last_accepted_index)

    # copy files from most recent global to end of block chimeric??
    file.copy(this_index_global_files[['hosp_filename']],output_chimeric_files[['hosp_filename']])
    file.copy(this_index_global_files[['seir_filename']],output_chimeric_files[['seir_filename']])

    # if using space-saving options, delete the last accepted global intermediate giles
    if(!opt$save_seir){
      file.remove(last_accepted_global_files[['seir_filename']]) # remove proposed SEIR file
    }
    if(!opt$save_hosp){
      file.remove(last_accepted_global_files[['hosp_filename']]) # remove proposed HOSP file
    }

    endTimeCount=difftime(Sys.time(), startTimeCount, units = "secs")
    print(paste("Time to run all MCMC iterations of slot ",opt$this_slot," is ",formatC(endTimeCount,digits=2,format="f")," seconds"))

  }
}
