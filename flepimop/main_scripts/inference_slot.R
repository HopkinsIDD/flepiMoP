## Preamble ---------------------------------------------------------------------
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

# Load gempyor module
gempyor <- reticulate::import("gempyor")

#Temporary
#print("Setting random number seed")
#set.seed(1) # set within R
#reticulate::py_run_string(paste0("rng_seed = ", 1)) #set within Python

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
    optparse::make_option(c("-L", "--reset_chimeric_on_accept"), action = "store", default = Sys.getenv("FLEPI_RESET_CHIMERICS", FALSE), type = 'logical', help = 'Should the chimeric parameters get reset to global parameters when a global acceptance occurs'),
    optparse::make_option(c("-M", "--memory_profiling"), action = "store", default = Sys.getenv("FLEPI_MEM_PROFILE", FALSE), type = 'logical', help = 'Should the memory profiling be run during iterations'),
    optparse::make_option(c("-P", "--memory_profiling_iters"), action = "store", default = Sys.getenv("FLEPI_MEM_PROF_ITERS", 100), type = 'integer', help = 'If doing memory profiling, after every X iterations run the profiler'),
    optparse::make_option(c("-g", "--subpop_len"), action="store", default=Sys.getenv("SUBPOP_LENGTH", 5), type='integer', help = "number of digits in subpop")
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

reticulate::use_python(Sys.which(opt$python), required = TRUE)
## Block loads the config file and geodata
if (opt$config == ""){
    optparse::print_help(parser)
    stop(paste(
        "Please specify a config YAML file with either -c option or CONFIG_PATH environment variable."
    ))
}
config = flepicommon::load_config(opt$config)


if (!is.null(config$seeding)){
    if (('perturbation_sd' %in% names(config$seeding))) {
        if (('date_sd' %in% names(config$seeding))) {
            stop("Both the key seeding::perturbation_sd and the key seeding::date_sd are present in the config file, but only one allowed.")
        }
        config$seeding$date_sd <- config$seeding$perturbation_sd
    }
    if (!('date_sd' %in% names(config$seeding))) {
        stop("Neither the key seeding::perturbation_sd nor the key seeding::date_sd are present in the config file, but one is required.")
    }
    if (!('amount_sd' %in% names(config$seeding))) {
        config$seeding$amount_sd <- 1
    }

    if (!(config$seeding$method %in% c('FolderDraw','InitialConditionsFolderDraw'))){
        stop("This filtration method requires the seeding method 'FolderDraw'")
    }
} else {
    print("⚠️ No seeding: section found in config >> not fitting seeding.")
}


infer_initial_conditions <- FALSE
if (!is.null(config$initial_conditions)){
    if (('perturbation' %in% names(config$initial_conditions))) {
        infer_initial_conditions <- TRUE
        if (!(config$initial_conditions$method %in% c('SetInitialConditionsFolderDraw'))){
            stop("This filtration method requires the initial_condition method 'SetInitialConditionsFolderDraw'")
        }
        if (!(config$initial_conditions$proportional)){
            stop("This filtration method requires the initial_condition to be set proportional'")
        }
    }
} else {
    print("⚠️ No initial_conditions: section found in config >> not fitting initial_conditions.")
}


#if (!('lambda_file' %in% names(config$seeding))) {
#  stop("Despite being a folder draw method, filtration method requires the seeding to provide a lambda_file argument.")
#}

# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$subpop_setup$state_level) && config$subpop_setup$state_level, TRUE, FALSE)


##Load information on geographic locations from geodata file.
suppressMessages(
    geodata <- flepicommon::load_geodata_file(
        paste(
            config$data_path,
            config$subpop_setup$geodata, sep = "/"
        ),
        subpop_len = opt$subpop_len
    )
)
obs_subpop <- "subpop"

##Load simulations per slot from config if not defined on command line
##command options take precedence
if (is.na(opt$iterations_per_slot)){
    opt$iterations_per_slot <- config$inference$iterations_per_slot
}
print(paste("Running",opt$iterations_per_slot,"simulations"))

##Define data directory and create if it does not exist
gt_data_path <- config$inference$gt_data_path
data_dir <- dirname(config$data_path)
if (!dir.exists(data_dir)){
    suppressWarnings(dir.create(data_dir, recursive = TRUE))
}



# ~ Parse Scenario Arguments ----------------------------------------------

# if opt$outcome_modifiers_scenarios is specified
#  --> run only those scenarios
#  If it is not or is "all"


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



# ~ Other Stats and Inference Args ----------------------------------------

##Creat heirarchical stats object if specified
hierarchical_stats <- list()
if ("hierarchical_stats_geo" %in% names(config$inference)) {
    hierarchical_stats <- config$inference$hierarchical_stats_geo
}

##Create priors if specified
defined_priors <- list()
if ("priors" %in% names(config$inference)) {
    defined_priors <- config$inference$priors
}

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



# Setup Obs, Initial Stats, and Likelihood fn -----------------------------

# ~ WITH Inference ----------------------------------------------------

if (config$inference$do_inference){

    # obs <- inference::get_ground_truth(
    #     data_path = data_path,
    #     fips_codes = fips_codes_,
    #     fips_column_name = obs_subpop,
    #     start_date = gt_start_date,
    #     end_date = gt_end_date,
    #     gt_source = gt_source,
    #     gt_scale = gt_scale,
    #     variant_filename = config$seeding$variant_filename
    # )

    obs <- suppressMessages(
        readr::read_csv(config$inference$gt_data_path,
                        col_types = readr::cols(date = readr::col_date(),
                                                source = readr::col_character(),
                                                subpop = readr::col_character(),
                                                .default = readr::col_double()), )) %>%
        dplyr::filter(subpop %in% subpops_, date >= gt_start_date, date <= gt_end_date) %>%
        dplyr::right_join(tidyr::expand_grid(subpop = unique(.$subpop), date = unique(.$date))) %>%
        dplyr::mutate_if(is.numeric, dplyr::coalesce, 0)

    subpopnames <- unique(obs[[obs_subpop]])


    ## Compute statistics
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


    likelihood_calculation_fun <- function(sim_hosp){

        sim_hosp <- dplyr::filter(sim_hosp,sim_hosp$time >= min(obs$date),sim_hosp$time <= max(obs$date))
        lhs <- unique(sim_hosp[[obs_subpop]])
        rhs <- unique(names(data_stats))
        all_locations <- rhs[rhs %in% lhs]

        ## No references to config$inference$statistics
        inference::aggregate_and_calc_loc_likelihoods(
            all_locations = all_locations, # technically different
            modeled_outcome = sim_hosp,
            obs_subpop = obs_subpop,
            targets_config = config[["inference"]][["statistics"]],
            obs = obs,
            ground_truth_data = data_stats,
            hosp_file = first_global_files[['llik_filename']],
            hierarchical_stats = hierarchical_stats,
            defined_priors = defined_priors,
            geodata = geodata,
            snpi = arrow::read_parquet(first_global_files[['snpi_filename']]),
            hnpi = arrow::read_parquet(first_global_files[['hnpi_filename']]),
            hpar = dplyr::mutate(arrow::read_parquet(first_global_files[['hpar_filename']]),parameter=paste(quantity,!!rlang::sym(obs_subpop),outcome,sep='_')),
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
            snpi = arrow::read_parquet(first_global_files[['snpi_filename']]),
            hnpi = arrow::read_parquet(first_global_files[['hnpi_filename']]),
            hpar = dplyr::mutate(arrow::read_parquet(first_global_files[['hpar_filename']]),parameter=paste(quantity,!!rlang::sym(obs_subpop),outcome,sep='_')),
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

for(seir_modifiers_scenario in seir_modifiers_scenarios) {

    print(paste0("Running intervention scenario: ", seir_modifiers_scenario))

    for(outcome_modifiers_scenario in outcome_modifiers_scenarios) {

        print(paste0("Running outcome scenario: ", outcome_modifiers_scenario))

        reset_chimeric_files <- FALSE

        # Data -------------------------------------------------------------------------
        # Load

        ## file name prefixes for this seir_modifiers_scenario + outcome_modifiers_scenario combination
        ## Create prefix is roughly equivalent to paste(...) so
        ## create_prefix("USA", "inference", "med", "2022.03.04.10.18.42.CET", sep='/')
        ## would be "USA/inference/med/2022.03.04.10.18.42.CET"
        ## There is some fanciness about formatting though so
        ## create_prefix(("43", "%09d"))
        ## would be "000000043"
        ## if a prefix argument is explicitly specified, the separator between it and the rest is skipped instead of sep so
        ## trailing separator is always added at the end of the string if specified.
        ## create_prefix(prefix="USA/", "inference", "med", "2022.03.04.10.18.42.CET", sep='/', trailing_separator='.')
        ## would be "USA/inference/med/2022.03.04.10.18.42.CET."

        setup_prefix <- flepicommon::create_setup_prefix(config$setup_name,
                                                         seir_modifiers_scenario, outcome_modifiers_scenario,
                                                         trailing_separator='')
        slot_prefix <- file.path(setup_prefix, opt$run_id)


        gf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','final',sep='/',trailing_separator='/')
        cf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','final',sep='/',trailing_separator='/')
        ci_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','intermediate',sep='/',trailing_separator='/')
        gi_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','intermediate',sep='/',trailing_separator='/')

        chimeric_block_prefix <- flepicommon::create_prefix(prefix=ci_prefix, slot=list(opt$this_slot,"%09d"), sep='.', trailing_separator='.')
        chimeric_local_prefix <- flepicommon::create_prefix(prefix=chimeric_block_prefix, slot=list(opt$this_block,"%09d"), sep='.', trailing_separator='.')
        global_block_prefix <- flepicommon::create_prefix(prefix=gi_prefix, slot=list(opt$this_slot,"%09d"), sep='.', trailing_separator='.')
        global_local_prefix <- flepicommon::create_prefix(prefix=global_block_prefix, slot=list(opt$this_block,"%09d"), sep='.', trailing_separator='.')

        print("prefixes created successfully.")


        ### Set up initial conditions ----------
        ## python configuration: build simulator model initialized with compartment and all.
        gempyor_inference_runner <- gempyor$GempyorSimulator(
            config_path=opt$config,
            run_id=opt$run_id,
            slot_prefix=global_block_prefix,
            block_info=file.path("global", "intermediate", flepicommon::create_prefix(slot=list(opt$this_slot,"%09d"), sep='.', trailing_separator='.')),
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            stoch_traj_flag=opt$stoch_traj_flag,
            initialize=TRUE  # Shall we pre-compute now things that are not pertubed by inference
        )
        print("gempyor_inference_runner created successfully.")


        ## Using the prefixes, create standardized files of each type (e.g., seir) of the form
        ## {variable}/{prefix}{block-1}.{run_id}.{variable}.{ext}
        ## N.B.: prefix should end in "{slot}."
        first_global_files <- inference::create_filename_list(opt$run_id, global_block_prefix, opt$this_block - 1)
        first_chimeric_files <- inference::create_filename_list(opt$run_id, chimeric_block_prefix, opt$this_block - 1)
        ## print("RUNNING: initialization of first block")
        ## Functions within this function save variables to files of the form variable/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.(block-1),run_id.variable.ext and also copied into the /chimeric/ version, which are referenced by first_global_files and first_chimeric_files
        inference::initialize_mcmc_first_block(
            opt$run_id,
            opt$this_block,
            global_block_prefix,
            chimeric_block_prefix,
            gempyor_inference_runner,
            likelihood_calculation_fun,
            is_resume = opt[['is-resume']]
        )
        print("First MCMC block initialized successfully.")

        ## So far no acceptances have occurred
        current_index <- 0

        ### Load initial files (were created within function initialize_mcmc_first_block)
        # if (!is.null(config$seeding)){
            seeding_col_types <- NULL
            suppressMessages(initial_seeding <- readr::read_csv(first_chimeric_files[['seed_filename']], col_types=seeding_col_types))

            if (opt$stoch_traj_flag) {
                initial_seeding$amount <- as.integer(round(initial_seeding$amount))
            }
        # }
        initial_init <- arrow::read_parquet(first_chimeric_files[['init_filename']])
        initial_snpi <- arrow::read_parquet(first_chimeric_files[['snpi_filename']])
        initial_hnpi <- arrow::read_parquet(first_chimeric_files[['hnpi_filename']])
        initial_spar <- arrow::read_parquet(first_chimeric_files[['spar_filename']])
        initial_hpar <- arrow::read_parquet(first_chimeric_files[['hpar_filename']])
        if (!is.null(config$initial_conditions)){
            initial_init <- arrow::read_parquet(first_global_files[['init_filename']])
        }
        chimeric_likelihood_data <- arrow::read_parquet(first_chimeric_files[['llik_filename']])
        global_likelihood_data <- arrow::read_parquet(first_global_files[['llik_filename']])

        ##Add initial perturbation sd values to parameter files----
        initial_snpi <- inference::add_perturb_column_snpi(initial_snpi,config$seir_modifiers$modifiers)
        initial_hnpi <- inference::add_perturb_column_hnpi(initial_hnpi,config$outcome_modifiers$modifiers)

        #Need to write these parameters back to the SAME chimeric file since they have a new column now
        arrow::write_parquet(initial_snpi,first_chimeric_files[['snpi_filename']])
        arrow::write_parquet(initial_hnpi,first_chimeric_files[['hnpi_filename']])

        # Also need to add this column to the global file (it will always be equal in the first block) (MIGHT NOT BE WORKING)
        arrow::write_parquet(initial_snpi,first_global_files[['snpi_filename']])
        arrow::write_parquet(initial_hnpi,first_global_files[['hnpi_filename']])

        #####Get the full likelihood (WHY IS THIS A DATA FRAME)
        # Compute total loglik for each sim
        global_likelihood <- sum(global_likelihood_data$ll)

        #####LOOP NOTES
        ### initial means accepted/current
        ### current means proposed

        startTimeCount=Sys.time()
        ##Loop over simulations in this block ----

        # keep track of running average global acceptance rate, since old global likelihood data not kept in memory. Each geoID has same value for acceptance rate in global case, so we just take the 1st entry
        old_avg_global_accept_rate <- global_likelihood_data$accept_avg[1]

        for (this_index in seq_len(opt$iterations_per_slot)) {
            print(paste("Running simulation", this_index))

            startTimeCountEach = Sys.time()

            ## Create filenames

            ## Using the prefixes, create standardized files of each type (e.g., seir) of the form
            ## {variable}/{prefix}{block-1}.{run_id}.{variable}.{ext}
            ## N.B.: prefix should end in "{block}."
            this_global_files <- inference::create_filename_list(opt$run_id, global_local_prefix, this_index)
            this_chimeric_files <- inference::create_filename_list(opt$run_id, chimeric_local_prefix, this_index)

            ### Do perturbations from accepted parameters to get proposed parameters ----

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
            if (infer_initial_conditions) {
                proposed_init <- inference::perturb_init(initial_init, config$initial_conditions$perturbation)
            } else {
                proposed_init <- initial_init
            }
            proposed_snpi <- inference::perturb_snpi(initial_snpi, config$seir_modifiers$modifiers)
            proposed_hnpi <- inference::perturb_hnpi(initial_hnpi, config$outcome_modifiers$modifiers)  # NOTE: no scenarios possible right now
            proposed_spar <- initial_spar
            proposed_hpar <- inference::perturb_hpar(initial_hpar, config$outcomes$outcomes) # NOTE: no scenarios possible right now
            if (!is.null(config$initial_conditions)){
                proposed_init <- initial_init
            }

            # since the first iteration is accepted by default, we don't perturb it
            if ((opt$this_block == 1) && (current_index == 0)) {
                proposed_init <- initial_init
                proposed_snpi <- initial_snpi
                proposed_hnpi <- initial_hnpi
                proposed_spar <- initial_spar
                proposed_hpar <- initial_hpar
                if (!is.null(config$initial_conditions)){
                    proposed_init <- initial_init
                }
                proposed_seeding <- initial_seeding
            }

            # proposed_snpi <- inference::perturb_snpi_from_file(initial_snpi, config$seir_modifiers$settings, chimeric_likelihood_data)
            # proposed_hnpi <- inference::perturb_hnpi_from_file(initial_hnpi, config$seir_modifiers$settings, chimeric_likelihood_data)
            # proposed_spar <- inference::perturb_spar_from_file(initial_spar, config$seir_modifiers$settings, chimeric_likelihood_data)
            # proposed_hpar <- inference::perturb_hpar_from_file(initial_hpar, config$seir_modifiers$settings, chimeric_likelihood_data)


            ## Write files that need to be written for other code to read
            # writes to file  of the form variable/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.block.iter.run_id.variable.ext
            # if (!is.null(config$seeding)){
                write.csv(proposed_seeding, this_global_files[['seed_filename']], row.names = FALSE)
            # }

            arrow::write_parquet(proposed_init,this_global_files[['init_filename']])
            arrow::write_parquet(proposed_snpi,this_global_files[['snpi_filename']])
            arrow::write_parquet(proposed_hnpi,this_global_files[['hnpi_filename']])
            arrow::write_parquet(proposed_spar,this_global_files[['spar_filename']])
            arrow::write_parquet(proposed_hpar,this_global_files[['hpar_filename']])
            if (!is.null(config$initial_conditions)){
                arrow::write_parquet(proposed_init,this_global_files[['init_filename']])
            }

            ## Update the prefix
            gempyor_inference_runner$update_prefix(new_prefix=global_local_prefix)
            ## Run the simulator
            err <- gempyor_inference_runner$one_simulation(
                sim_id2write=this_index,
                load_ID=TRUE,
                sim_id2load=this_index)
            if (err != 0){
                stop("GempyorSimulator failed to run")
            }

            if (config$inference$do_inference){
                sim_hosp <- flepicommon::read_file_of_type(gsub(".*[.]","",this_global_files[['hosp_filename']]))(this_global_files[['hosp_filename']]) %>%
                    dplyr::filter(time >= min(obs$date),time <= max(obs$date))

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

            ## UNCOMMENT TO DEBUG
            ## print(global_likelihood_data)
            ## print(chimeric_likelihood_data)
            ## print(proposed_likelihood_data)

            ## Compute total loglik for each sim
            proposed_likelihood <- sum(proposed_likelihood_data$ll)

            ## For logging
            print(paste("Current likelihood",formatC(global_likelihood,digits=2,format="f"),"Proposed likelihood",formatC(proposed_likelihood,digits=2,format="f")))

            ## Global likelihood acceptance or rejection decision ----


            proposed_likelihood_data$accept <- ifelse(inference::iterateAccept(global_likelihood, proposed_likelihood) || ((current_index == 0) && (opt$this_block == 1)),1,0)
            if (all(proposed_likelihood_data$accept == 1) | config$inference$do_inference == FALSE) {

                print("**** ACCEPT (Recording) ****")
                if ((opt$this_block == 1) && (current_index == 0)) {
                    print("by default because it's the first iteration of a block 1")
                }

                old_global_files <- inference::create_filename_list(opt$run_id, global_local_prefix, current_index)
                old_chimeric_files <- inference::create_filename_list(opt$run_id, chimeric_local_prefix, current_index)

                #IMPORTANT: This is the index of the most recent globally accepted parameters
                current_index <- this_index

                proposed_likelihood_data$accept <- 1 # global acceptance decision (0/1), same recorded for each geoID

                #This carries forward to next iteration as current global likelihood
                global_likelihood <- proposed_likelihood
                #This carries forward to next iteration as current global likelihood data
                global_likelihood_data <- proposed_likelihood_data

                if (opt$reset_chimeric_on_accept) {
                    reset_chimeric_files <- TRUE
                }

                warning("Removing unused files")
                sapply(old_global_files, file.remove)

            } else {
                print("**** REJECT (Recording) ****")
                warning("Removing unused files")
                if (this_index < opt$iterations_per_slot) {
                    sapply(this_global_files, file.remove)
                }
            }

            effective_index <- (opt$this_block - 1) * opt$iterations_per_slot + this_index
            avg_global_accept_rate <- ((effective_index-1)*old_avg_global_accept_rate + proposed_likelihood_data$accept)/(effective_index) # update running average acceptance probability
            proposed_likelihood_data$accept_avg <-avg_global_accept_rate
            proposed_likelihood_data$accept_prob <- exp(min(c(0, proposed_likelihood - global_likelihood))) #acceptance probability


            old_avg_global_accept_rate <- avg_global_accept_rate # keep track, since old global likelihood data not kept in memory

            ## Print average global acceptance rate
            # print(paste("Average global acceptance rate: ",formatC(100*avg_global_accept_rate,digits=2,format="f"),"%"))

            # prints to file of the form llik/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.block.iter.run_id.llik.ext
            arrow::write_parquet(proposed_likelihood_data, this_global_files[['llik_filename']])

            # keep track of running average chimeric acceptance rate, for each geoID, since old chimeric likelihood data not kept in memory
            old_avg_chimeric_accept_rate <- chimeric_likelihood_data$accept_avg

            if (!reset_chimeric_files) {
                ## Chimeric likelihood acceptance or rejection decisions (one round) -----
                #  "Chimeric" means GeoID-specific

                seeding_npis_list <- inference::accept_reject_new_seeding_npis(
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
                    orig_lls = chimeric_likelihood_data,
                    prop_lls = proposed_likelihood_data
                )


                # Update accepted parameters to start next simulation
                initial_init <- seeding_npis_list$init
                initial_seeding <- seeding_npis_list$seeding
                initial_snpi <- seeding_npis_list$snpi
                initial_hnpi <- seeding_npis_list$hnpi
                initial_hpar <- seeding_npis_list$hpar
                chimeric_likelihood_data <- seeding_npis_list$ll
            } else {
                print("Resetting chimeric files to global")
                initial_init <- proposed_init
                initial_seeding <- proposed_seeding
                initial_snpi <- proposed_snpi
                initial_hnpi <- proposed_hnpi
                initial_hpar <- proposed_hpar
                chimeric_likelihood_data <- global_likelihood_data
                reset_chimeric_files <- FALSE
            }

            # Update running average acceptance rate
            # update running average acceptance probability. CHECK, this depends on values being in same order in both dataframes. Better to bind??
            effective_index <- (opt$this_block - 1) * opt$iterations_per_slot + this_index
            chimeric_likelihood_data$accept_avg <- ((effective_index - 1) * old_avg_chimeric_accept_rate + chimeric_likelihood_data$accept) / (effective_index)

            ## Write accepted parameters to file
            # writes to file of the form variable/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/chimeric/intermediate/slot.block.iter.run_id.variable.ext
            write.csv(initial_seeding,this_chimeric_files[['seed_filename']], row.names = FALSE)
            arrow::write_parquet(initial_init,this_chimeric_files[['init_filename']])
            arrow::write_parquet(initial_snpi,this_chimeric_files[['snpi_filename']])
            arrow::write_parquet(initial_hnpi,this_chimeric_files[['hnpi_filename']])
            arrow::write_parquet(initial_spar,this_chimeric_files[['spar_filename']])
            arrow::write_parquet(initial_hpar,this_chimeric_files[['hpar_filename']])
            arrow::write_parquet(chimeric_likelihood_data, this_chimeric_files[['llik_filename']])

            print(paste("Current index is ",current_index))

            ###Memory management
            rm(proposed_init)
            rm(proposed_snpi)
            rm(proposed_hnpi)
            rm(proposed_hpar)
            rm(proposed_seeding)

            endTimeCountEach=difftime(Sys.time(), startTimeCountEach, units = "secs")
            print(paste("Time to run this MCMC iteration is ",formatC(endTimeCountEach,digits=2,format="f")," seconds"))

            # memory profiler to diagnose memory creep

            if (opt$memory_profiling){

                if (this_index %% opt$memory_profiling_iters == 0 | this_index == 1){
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

                    this_global_memprofile <- inference::create_filename_list(opt$run_id, global_local_prefix, this_index,
                                                                              types = "memprof", extensions = "parquet")
                    arrow::write_parquet(curr_obj_sizes, this_global_memprofile[['memprof_filename']])
                    rm(curr_obj_sizes)
                }

            }

            ## Run garbage collector to clear memory and prevent memory leakage
            # gc_after_a_number <- 1 ## # Garbage collection every 1 iteration
            if (this_index %% 1 == 0){
                gc()
            }

        }

        endTimeCount=difftime(Sys.time(), startTimeCount, units = "secs")
        # print(paste("Time to run all MCMC iterations is ",formatC(endTimeCount,digits=2,format="f")," seconds"))

        #####Do MCMC end copy. Fail if unsucessfull
        # moves the most recently globally accepted parameter values from global/intermediate file to global/final
        cpy_res_global <- inference::perform_MCMC_step_copies_global(current_index,
                                                                     opt$this_slot,
                                                                     opt$this_block,
                                                                     opt$run_id,
                                                                     global_local_prefix,
                                                                     gf_prefix,
                                                                     global_block_prefix)
        if (!prod(unlist(cpy_res_global))) {stop("File copy failed:", paste(unlist(cpy_res_global),paste(names(cpy_res_global),"|")))}
        # moves the most recently chimeric accepted parameter values from chimeric/intermediate file to chimeric/final

        cpy_res_chimeric <- inference::perform_MCMC_step_copies_chimeric(this_index,
                                                                         opt$this_slot,
                                                                         opt$this_block,
                                                                         opt$run_id,
                                                                         chimeric_local_prefix,
                                                                         cf_prefix,
                                                                         chimeric_block_prefix)
        if (!prod(unlist(cpy_res_chimeric))) {stop("File copy failed:", paste(unlist(cpy_res_chimeric),paste(names(cpy_res_chimeric),"|")))}
        #####Write currently accepted files to disk
        #files of the form variables/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/chimeric/intermediate/slot.block.run_id.variable.parquet
        output_chimeric_files <- inference::create_filename_list(opt$run_id, chimeric_block_prefix, opt$this_block)
        #files of the form variables/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.block.run_id.variable.parquet
        output_global_files <- inference::create_filename_list(opt$run_id, global_block_prefix, opt$this_block)

        warning("Chimeric hosp and seir files not yet supported, just using the most recently generated file of each type")
        this_index_global_files <- inference::create_filename_list(opt$run_id, global_local_prefix, this_index)
        file.copy(this_index_global_files[['hosp_filename']],output_chimeric_files[['hosp_filename']])
        file.copy(this_index_global_files[['seir_filename']],output_chimeric_files[['seir_filename']])
    }
}
