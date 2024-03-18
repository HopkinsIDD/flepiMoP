# About ------------------------------------------------------------------------

## This script processes the options for an inference run and then creates a separate parallel processing job for each combination of SEIR parameter modification scenario, outcome parameter modification scenario, and independent MCMC chain ("slot")


# Run Options ---------------------------------------------------------------------

suppressMessages(library(parallel))
suppressMessages(library(foreach))
suppressMessages(library(parallel))
suppressMessages(library(doParallel))
options(readr.num_columns = 0)

# There are multiple ways to specify options when inference_main.R is run, which take the following precedence:
#  1) (optional) options called along with the script at the command line (ie > Rscript inference_main.R -c my_config.yml)
#  2) (optional) environmental variables set by the user (ie user could set > export CONFIG_PATH = ~/flepimop_sample/my_config.yml to not have t specify it each time the script is run)
# If neither are specified, then a default value is used, given by the second argument of Sys.getenv() commands below. 
#  *3) For some options, a default doesn't exist, and the value specified in the config will be used if the option is not specified at the command line or by an environmental variable (iterations_per_slot, slots)

option_list = list(
  optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
  optparse::make_option(c("-u","--run_id"), action="store", type='character', help="Unique identifier for this run", default = Sys.getenv("FLEPI_RUN_INDEX",flepicommon::run_id())),
  optparse::make_option(c("-s", "--seir_modifiers_scenarios"), action="store", default=Sys.getenv("FLEPI_SEIR_SCENARIOS", 'all'), type='character', help="name of the intervention scenario to run, or 'all' to run all of them"),
  optparse::make_option(c("-d", "--outcome_modifiers_scenarios"), action="store", default=Sys.getenv("FLEPI_OUTCOME_SCENARIOS", 'all'), type='character', help="name of the outcome scenario to run, or 'all' to run all of them"),
  optparse::make_option(c("-j", "--jobs"), action="store", default=Sys.getenv("FLEPI_NJOBS", parallel::detectCores()), type='integer', help="Number of jobs to run in parallel"),
  optparse::make_option(c("-k", "--iterations_per_slot"), action="store", default=Sys.getenv("FLEPI_ITERATIONS_PER_SLOT", NA), type='integer', help = "number of iterations to run for this slot"),
  optparse::make_option(c("-n", "--slots"), action="store", default=Sys.getenv("FLEPI_NUM_SLOTS", as.numeric(NA)), type='integer', help = "Number of slots to run."),
  optparse::make_option(c("-b", "--this_block"), action="store", default=Sys.getenv("FLEPI_BLOCK_INDEX",1), type='integer', help = "id of this block"),
  optparse::make_option(c("-t", "--stoch_traj_flag"), action="store", default=Sys.getenv("FLEPI_STOCHASTIC_RUN",FALSE), type='logical', help = "Stochastic SEIR and outcomes trajectories if true"),
  optparse::make_option(c("--ground_truth_start"), action = "store", default = Sys.getenv("GT_START_DATE", ""), type = "character", help = "First date to include groundtruth for"),
  optparse::make_option(c("--ground_truth_end"), action = "store", default = Sys.getenv("GT_END_DATE", ""), type = "character", help = "Last date to include groundtruth for"),
  optparse::make_option(c("-p", "--flepi_path"), action="store", type='character', help="path to the flepiMoP directory", default = Sys.getenv("FLEPI_PATH", "flepiMoP")),
  optparse::make_option(c("-y", "--python"), action="store", default=Sys.getenv("PYTHON_PATH","python3"), type='character', help="path to python executable"),
  optparse::make_option(c("-r", "--rpath"), action="store", default=Sys.getenv("RSCRIPT_PATH","Rscript"), type = 'character', help = "path to R executable"),
  optparse::make_option(c("-R", "--is-resume"), action="store", default=Sys.getenv("RESUME_RUN",FALSE), type = 'logical', help = "Is this run a resume"),
  optparse::make_option(c("-I", "--is-interactive"), action="store", default=Sys.getenv("RUN_INTERACTIVE",Sys.getenv("INTERACTIVE_RUN", FALSE)), type = 'logical', help = "Is this run an interactive run"),
  optparse::make_option(c("-L", "--reset_chimeric_on_accept"), action = "store", default = Sys.getenv("FLEPI_RESET_CHIMERICS", TRUE), type = 'logical', help = 'Should the chimeric parameters get reset to global parameters when a global acceptance occurs')
)

parser=optparse::OptionParser(option_list=option_list)
opt = optparse::parse_args(parser)

print("Starting")
if(opt$config == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a config YAML file with either -c option or CONFIG_PATH environment variable."
  ))
}

print(paste('Running ',opt$j,' jobs in parallel'))

config <- flepicommon::load_config(opt$config)

# Slots +  Iteration Options -----------------------------------------------------------

if(is.na(opt$iterations_per_slot)) {
  opt$iterations_per_slot <- config$inference$iterations_per_slot
}

if(is.na(opt$slots)) {
  opt$slots <- config$nslots
}



# Scenario Options  ------------------------------------------------------

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



# Run Scenarios and Slots in Parallel -------------------------------------


cl <- parallel::makeCluster(opt$j)
doParallel::registerDoParallel(cl)
print(paste0("Making cluster with ", opt$j, " cores."))

flepicommon::prettyprint_optlist(list(seir_modifiers_scenarios=seir_modifiers_scenarios,outcome_modifiers_scenarios=outcome_modifiers_scenarios,slots=seq_len(opt$slots)))
foreach(seir_modifiers_scenario = seir_modifiers_scenarios) %:%
foreach(outcome_modifiers_scenario = outcome_modifiers_scenarios) %:%
foreach(flepi_slot = seq_len(opt$slots)) %dopar% {
  print(paste("Slot", flepi_slot, "of", opt$slots))

  ground_truth_start_text <- NULL
  ground_truth_end_text <- NULL
  if (nchar(opt$ground_truth_start) > 0) {
    ground_truth_start_text <- c("--ground_truth_start", opt$ground_truth_start)
  }
  if (nchar(opt$ground_truth_start) > 0) {
    ground_truth_end_text <- c("--ground_truth_end", opt$ground_truth_end)
  }

  err <- system(
    paste(
      opt$rpath,
      file.path(opt$flepi_path, "flepimop", "main_scripts","inference_slot.R"),
        "-c", opt$config,
        "-u", opt$run_id,
        "-s", opt$seir_modifiers_scenarios,
        "-d", opt$outcome_modifiers_scenarios,
        "-j", opt$jobs,
        "-k", opt$iterations_per_slot,
        "-i", flepi_slot,
        "-b", opt$this_block,
        "-t", opt$stoch_traj_flag,
        ground_truth_start_text,
        ground_truth_end_text,
        "-p", opt$flepi_path,
        "-y", opt$python,
        "-r", opt$rpath,
        "-R", opt[["is-resume"]],
        "-I", opt[["is-interactive"]],
        "-L", opt$reset_chimeric_on_accept,
      #paste("2>&1 | tee log_inference_slot_",config$name,"_",opt$run_id, "_", flepi_slot, ".txt", sep=""), # works
      #paste("2>&1 | tee model_output/",config$name,"/",opt$run_id,"/log/log_inference_slot", flepi_slot, ".txt", sep=""), # doesn't work because config$name needs to be combined with scenarios to generate the folder name, and, because this command seems to only be able to pipe output to pre-existing folders
      sep = " ")
    )
  if(err != 0){quit("no")}
}
parallel::stopCluster(cl)
