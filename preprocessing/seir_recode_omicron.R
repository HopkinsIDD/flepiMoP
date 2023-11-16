##
# @file
# @brief Creates an init file converting all previous variants into Omicron 
#
# @details
# ```

# SETUP -------------------------------------------------------------------

library(flepicommon)
library(magrittr)
library(dplyr)
library(readr)
library(tidyr)
# library(purrr)

select <- dplyr::select

### ---> need to move whatever functions we need from this to flepicommon
# install_configwriter <- FALSE
#source("R/scripts/config_writers/config_writer_setup_flepimop.R")
#####


option_list <- list(
  optparse::make_option(c("--res_config"), action = "store", default = Sys.getenv("RESUMED_CONFIG_PATH", NA), type = "character", help = "path to the previous config file"),
  optparse::make_option(c("-c", "--config"), action = "store", default = Sys.getenv("CONFIG_PATH"), type = "character", help = "path to the config file"),
  optparse::make_option(c("-p", "--flepi_path"), action="store", type='character', default = Sys.getenv("FLEPI_PATH", "flepiMoP/"), help="path to the flepiMoP directory"),
  optparse::make_option(c("-v", "--recode_variant"), action="store", type='character', default = Sys.getenv("RECODE_VAR", "DELTA"), help="variant you want to change all variant_types to"),
  optparse::make_option(c("--in_filename"), action="store", type='character', default = Sys.getenv("IN_FILENAME"), help="seir file global intermediate name"), # This is the CONTINUED SEIR filename
  optparse::make_option(c("--in_seed_filename"), action="store", type='character', default = Sys.getenv("IN_SEED_FILENAME"), help="seed file global intermediate name"), # This is the CONTINUED SEED filename
  optparse::make_option(c("--init_filename"), action="store", type='character', default = Sys.getenv("INIT_FILENAME"), help="init file global intermediate name") # This is the new init filename
  )
opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))

print(paste0("Using config files: ", opt$config, " and ", opt$res_config))
config <- flepicommon::load_config(opt$config)

if (!is.null(config$initial_conditions$resumed_config)){
  opt$res_config <- config$initial_conditions$resumed_config
}

res_config <- flepicommon::load_config(opt$res_config)

if (length(config) == 0) {
  stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}
if (length(res_config) == 0) {
  stop("no resumed configuration found -- please set RESUMED_CONFIG_PATH environment variable or use the -r command flag")
}


# INPUT -------------------------------------------------------------------

# variant_compartments <- config$compartments$variant_type
# age_strat <- config$compartments$age_strata

compartment_tracts <- names(res_config$compartments)


# PULL SEIR FILES -------------------------------------------------------------------
## Read in SEIR files from resume, get the correct date and rewrite all variant types 

# new config start date
transition_date <- lubridate::as_date(config$start_date) 
print(opt$init_filename)

# seir file from continued run
continued_seir <- arrow::read_parquet(opt$init_filename)
head(continued_seir)
unique(continued_seir$mc_variant_type)

# subset at start date, smoosh all to Omicron
seir_dt <- continued_seir %>% mutate(date = lubridate::as_date(date))  %>%
  select(-mc_name) %>%
  filter(mc_value_type == "prevalence") %>%
  filter(date == transition_date) %>%
  pivot_longer(cols = -c(starts_with("mc_"), date), names_to = "geoid", values_to = "value") %>%
  #mutate(mc_variant_type = opt$recode_variant) %>%
  mutate(mc_variant_type = ifelse(mc_variant_type %in% c("WILD","ALPHA"),"DELTA",mc_variant_type)) %>% 
  group_by(across(c(-value))) %>%
  summarise(value = sum(value, na.rm = TRUE)) %>%  
  dplyr::mutate(mc_name = paste(mc_infection_stage, mc_vaccination_stage, mc_variant_type, mc_age_strata, sep = "_")) %>%
  pivot_wider(names_from = geoid, values_from = value) 

# SAVE --------------------------------------------------------------------
  
# rewrite to new init file
new_init_file <- opt$init_filename

arrow::write_parquet(seir_dt, new_init_file)

cat(paste0("\nWriting modified init file to: \n  -- ", new_init_file, "\n\n"))

  

# PULL SEED FILES -------------------------------------------------------------------
## Read in SEED files from resume, and remove other variants

# variants to keep 
variant_comp <- config$compartments$variant_type

# seed file from continued run
continued_seed <- readr::read_csv(opt$in_seed_filename) 

# remove all but Omicron
seed_dt <- continued_seed %>% mutate(date = lubridate::as_date(date))  %>%
  filter(destination_variant_type %in% variant_comp) %>%
  mutate(source_variant_type = opt$recode_variant)  


# SAVE --------------------------------------------------------------------
  
# rewrite to SAME seed file
rewrite_seed_filename <- opt$in_seed_filename

write.csv(seed_dt, rewrite_seed_filename,row.names = FALSE)

cat(paste0("\nWriting modified seed file to: \n  -- ", rewrite_seed_filename, "\n\n"))



