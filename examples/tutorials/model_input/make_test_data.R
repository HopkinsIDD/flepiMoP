# Script to take output from a model simulated with another config and turn it into fake "data" that can be used as ground truth input to a different inference config file
library(dplyr)
library(data.table)
library(reticulate)
library(readr)
library(stringr)
gempyor <- import("gempyor")


# INPUT FILES AND PARAMETERS  ----------

input_config = "config_sample_2pop_modifiers.yml" # config to take output from (forward simulation)
input_inference_config = "config_sample_2pop_inference.yml"
input_seir_modifier_scenario = NULL # which SEIR modifier scenario to use. If null, will take the first. Not required if only 1 scenario. 
input_outcome_modifier_scenario = NULL # which SEIR modifier scenario to use. If null, will take the first. Not required if only 1 scenario. 
input_run_id = NULL # which RUNID to use results from. If null, will take the first. Nor required if only 1 output run. 
input_slot = NULL 


# FUNCTIONS ---------------------------------------------------------------

# Function to read in any model output file type for inference or non-inference run. Taken from model_output_notebook.Rmd
import_model_outputs <- function(scn_run_dir, inference, outcome, global_opt = NULL, final_opt = NULL){
  
  if(inference){ 
    
    if(is.null(global_opt) | is.null(final_opt)){
      stop("Inference run, must specify global_opt and final_opt")
    }else{
      inference_filepath_suffix <-paste0("/",global_opt,"/",final_opt)
      print(paste0('Assuming inference run with files in',inference_filepath_suffix))
    }
    
  }else{ # non inference run
    
    inference_filepath_suffix <-""
    print('Assuming non-inference run. Ignoring values of global_opt and final_opt if specified') 
    
  }
  
  subdir <- paste0(scn_run_dir,"/", outcome,"/",inference_filepath_suffix, "/")
  #print(subdir)
  subdir_list <- list.files(subdir)
  #print(subdir_list)
  
  out <- NULL
  total <- length(subdir_list)
  
  print(paste0("Importing ", outcome, " files (n = ", total, "):"))
  
  for (i in 1:length(subdir_list)) {
    
    # read in parquet or csv files
    if (any(grepl("parquet", subdir_list))) {
      dat <-
        arrow::read_parquet(paste(subdir, subdir_list[i], sep = "/"))
    } else if (any(grepl("csv", subdir_list))) {
      dat <- read.csv(paste(subdir, subdir_list[i], sep = "/"))
    }
    
    if(inference == TRUE & identical(final_opt,"intermediate")){ # if an 'intermediate inference run', filename prefix will include slot, (block), and iteration number
      
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
      dat$block <-as.numeric(str_sub(subdir_list[i], start = 11, end = 19))
      dat$iter <-as.numeric(str_sub(subdir_list[i], start = 21, end = 29))
      
    }else{ # if a non-inference run or a 'final' inference run, filename prefix will only contain slot #. Each file is a separate slot
      
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
      
    }
    
    out <- rbind(out, dat)
    
  }
  return(out)
  
}

  
# IMPORT AND PERTURB SIMULATION DATA ------------------------


config <- flepicommon::load_config(input_config)
config_inference <- flepicommon::load_config(input_inference_config)

# location of output files
res_dir <- file.path(ifelse(is.null(config$model_output_dirname),"model_output", config$model_output_dirname))
print(res_dir)

# get the directory of the results for this config + scenario: {config$name}_{seir_modifier_scenario}_{outcome_modifier_scenario}
#setup_prefix <- paste0(config$name,ifelse(is.null(config$seir_modifiers$scenarios),"",paste0("_",input_seir_modifier_scenario)),ifelse(is.null(config$outcome_modifiers$scenarios),"",paste0("_",input_outcome_modifier_scenario)))
# NEEDS TO BE FIXED
setup_prefix <- paste0(config$name,
                       ifelse(is.null(config$seir_modifiers$scenarios),"",
                              ifelse(length(config$seir_modifiers$scenarios)==1,paste0("_",config$seir_modifiers$scenarios),
                                     ifelse(is.null(input_seir_modifier_scenario),paste0("_",config$seir_modifiers$scenarios[1]),paste0("_",input_seir_modifier_scenario)))),
                       ifelse(is.null(config$outcome_modifiers$scenarios),"",
                              ifelse(length(config$outcome_modifiers$scenarios)==1,paste0("_",config$outcome_modifiers$scenarios),
                                     ifelse(is.null(input_outcome_modifier_scenario),paste0("_",config$outcome_modifiers$scenarios[1]),paste0("_",input_outcome_modifier_scenario)))))
print(setup_prefix)

scenario_dir <-file.path(res_dir,setup_prefix)
print(scenario_dir)

# find all unique run_ids within model_output. Must choose one only for plotting
run_ids <- list.files(scenario_dir)
print(run_ids)

this_run_id <- ifelse(length(run_ids)==1,run_ids[1],ifelse(is.null(input_run_id),stop(paste0('There are multiple run_ids within ',scenario_dir,'/, you must specify which one to plot the results for in the notebook header using input_run_id')),input_run_id))
print(this_run_id)

# entire path to the directory for each type of model output
scenario_run_dir <- file.path(scenario_dir,this_run_id)

# import outcomes
hosp_outputs <- setDT(import_model_outputs(scenario_run_dir, 0,"hosp"))

# choose slot
choose_slot <- ifelse(is.null(input_slot),1,input_slot)

# get outcomes that will be fit during inference, and apply desired aggregation and date range
fit_stats <- names(config_inference$inference$statistics)
outcome_vars_sim <- sapply(1:length(fit_stats), function(j) config_inference$inference$statistics[[j]]$sim_var) #name of model variables
outcome_vars_data <- sapply(1:length(fit_stats), function(j) config_inference$inference$statistics[[j]]$data_var) #name of data variable

# This is not yet working/implemented so it's not doing this aggregation or reformatting automatically yet

# df_data <- lapply(subpop_names, function(x) {
#   purrr::flatten_df(
#     inference::getStats(
#       gt_data %>% .[subpop == x,..cols_data],
#       "date",
#       "data_var",
#       stat_list = config$inference$statistics[i],
#       start_date = config$start_date_groundtruth,
#       end_date = config$end_date_groundtruth
#     )) %>% dplyr::mutate(subpop = x) %>% 
#     mutate(data_var = as.numeric(data_var)) }) %>% dplyr::bind_rows()



# results_filelist <- file.path(res_dir, 
#                               paste0(config$name, "_", config$seir_modifiers$scenarios[scenario_num], "_", config$outcome_modifiers$scenarios[scenario_num]))
# results_filelist <- file.path(results_filelist, list.files(results_filelist))
# model_outputs <- "hosp"
# 
# # outcomes variables to choose -------
# # get hosp values
# hosp_file <- list.files(file.path(results_filelist,"hosp"))
# output_hosp <- setDT(arrow::read_parquet(file.path(results_filelist,"hosp",hosp_file)))

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

