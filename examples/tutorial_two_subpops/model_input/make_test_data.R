# Script to take output from a model simulated with another config and turn it into fake "data" that can be used as ground truth input to a different inference config file
library(dplyr)
library(data.table)
library(reticulate)
library(readr)
gempyor <- import("gempyor")

# FUNCTIONS ---------------------------------------------------------------

import_model_outputs <- function(scn_dir, outcome, global_opt, final_opt, run_id = opt$run_id,
                                 lim_hosp = c("date", 
                                              sapply(1:length(names(config$inference$statistics)), function(i) purrr::flatten(config$inference$statistics[i])$sim_var),
                                              "subpop")){
  # model_output/USA_inference_fake/20231016_204739CEST/hnpi/global/intermediate/000000001.000000001.000000030.20231016_204739CEST.hnpi.parquet
  dir_ <- file.path(scn_dir, 
                    paste0(config$name, "_", config$seir_modifiers$scenarios[scenario_num], "_", config$outcome_modifiers$scenarios[scenario_num]),
                    run_id, 
                    outcome)
  subdir_ <- paste0(dir_, "/",
                    "/",
                    global_opt,
                    "/",
                    final_opt)
  subdir_list <- list.files(subdir_)
  
  out_ <- NULL
  total <- length(subdir_list)
  pb <- txtProgressBar(min=0, max=total, style = 3)
  
  print(paste0("Importing ", outcome, " files (n = ", total, "):"))
  
  for (i in 1:length(subdir_list)) {
    if(any(grepl("parquet", subdir_list))){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/"))
    }
    if(outcome == "hosp"){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/")) %>%
        select(all_of(lim_hosp))
    }
    if(any(grepl("csv", subdir_list))){
      dat <- read.csv(paste(subdir_, subdir_list[i], sep = "/"))
    }
    if(final_opt == "final"){
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
    }
    if(final_opt == "intermediate"){
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
      dat$block <- as.numeric(str_sub(subdir_list[i], start = 11, end = 19))
    }
    out_ <- rbind(out_, dat)
    
    # Increase the amount the progress bar is filled by setting the value to i.
    setTxtProgressBar(pb, value = i)
  }
  close(pb)
  return(out_)
}

# Setup files ----------

# config to take output from
config <- flepicommon::load_config("config_sample_2pop_modifiers.yml")

# config that will run inference
config_inference <- flepicommon::load_config("config_sample_2pop_inference.yml")

res_dir <- "model_output"

# IMPORT OUTCOMES ---------------------------------------------------------
# output location
scenario_num <- 1
setup_prefix <- paste0(config$name,
                       ifelse(is.null(config$seir_modifiers$scenarios),"",paste0("_",config$seir_modifiers$scenarios[scenario_num])),
                       ifelse(is.null(config$outcome_modifiers$scenarios),"",paste0("_",config$outcome_modifiers$scenarios[scenario_num])))

res_dir <- file.path(ifelse(is.null(config$model_output_dirname),"model_output", config$model_output_dirname))
print(res_dir)

results_filelist <- file.path(res_dir, 
                              paste0(config$name, "_", config$seir_modifiers$scenarios[scenario_num], "_", config$outcome_modifiers$scenarios[scenario_num]))
results_filelist <- file.path(results_filelist, list.files(results_filelist))
model_outputs <- "hosp"

# outcomes variables to choose -------
# get hosp values
hosp_file <- list.files(file.path(results_filelist,"hosp"))
output_hosp <- setDT(arrow::read_parquet(file.path(results_filelist,"hosp",hosp_file)))

# filter these outcome variables for desired dates then aggregate to desired level -------
outcome_hosp_ <- output_hosp %>% 
  .[date >= config_inference$start_date & date <= config_inference$end_date] 
# add filter line here to aggregate to desired level 


# output to correct file form -------
# format to groundtruth format
outcome_hosp_formatted <- outcome_hosp_ %>% 
  .[, date := lubridate::as_date(date)] %>%
  .[, .(date, subpop, incidH = incidHosp)]

readr::write_csv(outcome_hosp_formatted, file = "data/sample_2pop_cases.csv")

