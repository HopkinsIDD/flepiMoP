## Script runs automatically after batch run: for an inference run, compares outcomes to data

suppressMessages(library(parallel))
suppressMessages(library(foreach))
suppressMessages(library(inference))
suppressMessages(library(tidyverse))
suppressMessages(library(doParallel))
suppressMessages(library(dplyr))
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(ggforce))

options(readr.num_columns = 0)

option_list = list(
  optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH", Sys.getenv("CONFIG_PATH")), type='character', help="path to the config file"),
  optparse::make_option(c("-u","--run-id"), action="store", dest = "run_id", type='character', help="Unique identifier for this run", default = Sys.getenv("FLEPI_RUN_INDEX",covidcommon::run_id())),
  optparse::make_option(c("-R", "--results-path"), action="store", dest = "results_path",  type='character', help="Path for model output", default = Sys.getenv("FS_RESULTS_PATH", Sys.getenv("FS_RESULTS_PATH"))),
  optparse::make_option(c("-p", "--flepimop-repo"), action="store", dest = "flepimop_repo", default=Sys.getenv("FLEPI_PATH", Sys.getenv("FLEPI_PATH")), type='character', help="path to the flepimop repo"),
  optparse::make_option(c("-o", "--select-outputs"), action="store", dest = "select_outputs", default=Sys.getenv("OUTPUTS",c("hosp", "snpi", "llik")), type='character', help="path to the flepimop repo")
)

parser=optparse::OptionParser(option_list=option_list)
opt = optparse::parse_args(parser, convert_hyphens_to_underscores = TRUE)

print("Generating plots")
print(paste("Config:", opt$config, 'for', opt$run_id, 'saved in', opt$results_path))

if(opt$config == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a config YAML file with either -c option or CONFIG_PATH environment variable."
  ))
}

if(opt$results_path == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a results path with either -P option or FS_RESULTS_PATH environment variable."
  ))
}

if(opt$flepimop_repo == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a flepiMoP path with -p option or FLEPI_PATH environment variable."
  ))
}

print(paste('Processing run ',opt$results_path))

## SETUP -----------------------------------------------------------------------

config <- covidcommon::load_config(opt$config)

# Pull in geoid data
geodata <- read.csv(file.path(config$data_path, config$spatial_setup$geodata))

## gt_data MUST exist directly after a run
gt_data <- data.table::fread(config$inference$gt_data_path) %>%
  .[, geoid := stringr::str_pad(FIPS, width = 5, side = "left", pad = "0")]

# FUNCTIONS ---------------------------------------------------------------

import_model_outputs <- function(scn_dir, outcome, global_opt, final_opt){
  dir_ <- paste0(scn_dir, "/",
                 outcome, "/",
                 config$name, "/",
                 config$interventions$scenarios, "/",
                 config$outcomes$scenarios)
  subdir_ <- paste0(dir_, "/", list.files(dir_),
                    "/",
                    global_opt,
                    "/",
                    final_opt)
  subdir_list <- list.files(subdir_)
  
  out_ <- NULL
  total <- length(subdir_list)
  
  print(paste0("Importing ", outcome, " files (n = ", total, "):"))
  
  for (i in 1:length(subdir_list)) {
    if(any(grepl("parquet", subdir_list))){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/"))
    }
    if(outcome == "hosp"){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/")) 
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
    
  }
  return(out_)
}

# IMPORT OUTCOMES ---------------------------------------------------------

res_dir <- opt$results_path

model_outputs <- list.files(res_dir)[match(opt$select_outputs,list.files(res_dir))]
model_outputs <- ifelse(!("llik" %in% opt$select_outputs), c(model_outputs, "llik"), model_outputs)

outputs_global <- lapply(model_outputs, function(i) setDT(import_model_outputs(res_dir, i, 'global', 'final')))
names(outputs_global) <- model_outputs

if("llik" %in% model_outputs){
  opts <- c("global", "chimeric")
  int_llik <- lapply(opts, function(i) setDT(import_model_outputs(res_dir, "llik", i, "intermediate")))
}

pdf(paste0("mod_outputs_", opt$run_id,".pdf"), width = 15, height = 12)

## HOSP --------------------------------------------------------------------
# Compare inference statistics sim_var to data_var
if("hosp" %in% model_outputs){
  fit_stats <- names(config$inference$statistics)
  hosp_plots <- list()
  hosp_quants <- list()
  hosp_allslots <- list()
  for(i in 1:length(fit_stats)){
    statistics <- purrr::flatten(config$inference$statistics[i])
    cols_sim <- c("date", statistics$sim_var, config$spatial_setup$nodenames,"slot")
    cols_data <- c("date", config$spatial_setup$nodenames, statistics$data_var)
    ## summarize slots 
    print(outputs_global$hosp %>%
      .[, ..cols_sim] %>%
      .[, date := lubridate::as_date(date)] %>%
      .[, as.list(quantile(get(statistics$sim_var), c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", config$spatial_setup$nodenames)] %>%
      ggplot() + 
      geom_ribbon(aes(x = date, ymin = V1, ymax = V5), alpha = 0.1) +
      geom_ribbon(aes(x = date, ymin = V2, ymax = V4), alpha = 0.1) +
      geom_line(aes(x = date, y = V3)) + 
      geom_point(data = gt_data %>%
                   .[, ..cols_data] ,
                 aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
      facet_wrap(~get(config$spatial_setup$nodenames), scales = 'free') +
      labs(x = 'date', y = fit_stats[i]) +
      theme_classic()
    )
    
    # ## plot all slots
    # print(outputs_global$hosp %>% 
    #   ggplot() +
    #   geom_line(aes(lubridate::as_date(date), get(sim_var), group = as.factor(slot)), alpha = 0.1) + 
    #   facet_wrap(~get(config$spatial_setup$nodenames), scales = 'free') +
    #   geom_point(data = gt_data %>%
    #                .[, ..cols_data],
    #              aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
    #   theme_classic() + 
    #   labs(x = 'date', y = fit_stats[i]) +
    #   theme(legend.position="none")
    # )
  }
}


## HNPI --------------------------------------------------------------------
if("hnpi" %in% model_outputs){
  print("TO DO")
}

## HPAR --------------------------------------------------------------------
if("hpar" %in% model_outputs){
  print("TO DO")
}

## LLIK --------------------------------------------------------------------
if("llik" %in% model_outputs){
  print("TO DO")
  
}

## SEED --------------------------------------------------------------------
if("seed" %in% model_outputs){
  print("TO DO")
  
}

## SEIR --------------------------------------------------------------------
if("seir" %in% model_outputs){
  print("TO DO")
  
}

## SNPI --------------------------------------------------------------------
if("snpi" %in% model_outputs){
  node_names <- unique(outputs_global$snpi %>% .[ , get(config$spatial_setup$nodenames)])
  num_nodes <- length(node_names)
  pgs <- ceiling(num_nodes / 6)
  
  for(i in 1:pgs){
    print(outputs_global$snpi %>%
            .[outputs_global$llik, on = c(config$spatial_setup$nodenames, "slot")] %>%
            ggplot(aes(npi_name,reduction)) + 
            geom_violin() + 
            geom_jitter(aes(group = npi_name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.4) +
            facet_wrap_paginate(~get(config$spatial_setup$nodenames),
                                scales = 'free', drop = TRUE, 
                                ncol = 2, nrow = 3, page = i) +
            theme_bw(base_size = 10) +
            theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
            scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
            labs(x = "parameter")
    )
  }
}

## SPAR --------------------------------------------------------------------
if("spar" %in% model_outputs){
  print("TO DO")
  
}

dev.off()
