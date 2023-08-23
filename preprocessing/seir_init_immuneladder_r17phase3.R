##
# @file
# @brief Creates a seir_init file to transistion to a immune ladder structure
#
# @details
#
#
# ## Configuration Items
#
# ```yaml
# setup_name: <string>
# start_date: <date>
# end_date: <date>
# data_path: <path to directory>

# spatial_setup:
#   geodata: <path to file>
#   subpop: <string>
#
# seeding:
#   lambda_file: <path to file>

# ```
#
# ## Input Data
#
# * <b>{data_path}/{spatial_setup::geodata}</b> is a csv with column {spatial_setup::subpop} that denotes the subpop
#
# ## Output Data
#
# * <b>data/case_data/USAFacts_case_data.csv</b> is the case csv downloaded from USAFacts
# * <b>data/case_data/USAFacts_death_data.csv</b> is the death csv downloaded from USAFacts
# * <b>{seeding::lambda_file}</b>: filter file
#

## @cond

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
    optparse::make_option(c("--init_file_name"), action="store", type='character', default = Sys.getenv("INIT_FILENAME"), help="init file global intermediate name"),
    optparse::make_option(c("-e", "--imm_esc_prop"), action="store", type='numeric', default = Sys.getenv("IMM_ESC_PROP", .35), help="annual percent of immune escape")
)
opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))

print(paste0("Using config files: ", opt$config, " and ", opt$res_config))
config <- flepicommon::load_config(opt$config)

if (exists(config$initial_conditions$resumed_config)){
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


# # Geodata
# geodata <- read_csv("data/geodata_2019_statelevel_agestrat.csv")


# Vaccine Effectiveness  --------------------------------------------------------------------

# Extract thetas from Config
seir_thetas <- res_config$seir$parameters
seir_thetas <- seir_thetas[which(grepl("theta", names(seir_thetas)))]
seir_thetas <- tibble::tibble(theta = names(seir_thetas), value = sapply(X = 1:length(seir_thetas), function(x=X) seir_thetas[x][[1]]$value$value))
seir_thetas <- seir_thetas %>% dplyr::mutate(value = as.numeric(gsub("1 - ", "", value)))
seir_thetas <- seir_thetas %>%
    tidyr::separate(col = theta, into = c("param", "sourcevar", "destvar"), sep = "_", remove = FALSE, extra = "drop", fill = "right") %>%
    dplyr::mutate(sourcevar = ifelse(is.na(destvar), "WILD", sourcevar))



# Extract the seir structure of interest (only those transitions moving to "E")

compartment_tracts <- names(res_config$compartments)
seir_struct <- res_config$seir$transitions
# dest_E <- sapply(X = 1:length(seir_struct), function(x = X) seir_struct[x][[1]]$destination[[1]][1] == "E")
# seir_struct <- seir_struct[which(sapply(X = 1:length(seir_struct), function(x = X) seir_struct[x][[1]]$destination[[1]][1] == "E"))]

seir_struct_tab <- seir_struct %>%
    purrr::transpose() %>%
    tibble::as_tibble() %>%
    dplyr::filter(proportional_to != "source") %>%
    dplyr::mutate(source = purrr::map(source, ~tibble::as_tibble(t(.x)))) %>%
    dplyr::mutate(source = purrr::map(source, ~set_names(.x, paste0("source_", compartment_tracts)))) %>%
    dplyr::mutate(destination = purrr::map(destination, ~tibble::as_tibble(t(.x)))) %>%
    dplyr::mutate(destination = purrr::map(destination, ~rlang::set_names(.x, paste0("destination_", compartment_tracts)))) %>%
    dplyr::mutate(rate = purrr::map(rate, ~tibble::as_tibble(t(.x)))) %>%
    dplyr::mutate(rate = purrr::map(rate, ~rlang::set_names(.x, paste0("rate_", compartment_tracts)))) %>%
    tidyr::unnest(c(source, destination, rate)) %>%
    dplyr::select(starts_with("source"), starts_with("destination"), starts_with("rate"))

# tab1$source_infection_stage[1]
# tab1 %>% filter(source_infection_stage =="S") %>% pull(rate_age_strata)
# tab1 %>% filter(source_infection_stage =="W") %>% pull(rate_vaccination_stage)
#
#
# # limit to transitions going to OMICRON
# tab1 <- tab1 %>% filter(destination_variant_type == "OMICRON")
# # limit to transitions going to "E"
# tab1 <- tab1 %>% filter(destination_infection_stage == "E")
#
# tab1 %>%
#     dplyr::mutate(source = purrr::map(source, ~tibble::as_tibble(t(.x)))) %>%
#     dplyr::mutate(source = purrr::map(source, ~set_names(.x, paste0("source_", compartment_tracts))))
#
#
# tab_S <- tab1 %>%
#     dplyr::filter(source_infection_stage =="S") %>%
#     # dplyr::select(source_vaccination_stage, rate_vaccination_stage) %>%
#     dplyr::mutate(rate_vaccination_stage = purrr::map(rate_vaccination_stage, ~tibble::as_tibble(t(.x)))) %>%
#     dplyr::mutate(mc_vaccination_stage = purrr::map2(rate_vaccination_stage, source_vaccination_stage, ~set_names(.x, paste0("vaccrate_", .y)))) %>%
#     dplyr::mutate(rate_infection_stage = purrr::map(rate_infection_stage, ~tibble::as_tibble(t(.x)))) %>%
#     dplyr::mutate(mc_infection_stage = purrr::map2(rate_vaccination_stage, source_vaccination_stage, ~set_names(.x, paste0("vaccrate_", .y)))) %>%
#     tidyr::unnest(mc_vaccination_stage) %>%
#     select(starts_with("mcvacc"))



tab2 <- seir_struct_tab %>%
    mutate(transition = 1:nrow(.)) %>%
    pivot_longer(cols=-transition, names_pattern = "(.*)_(.*)_(.*)", names_to = c("type", "tract", "stage"), values_to = "values") %>%
    mutate(tract = paste(tract, stage, sep = "_")) %>%
    select(-stage)

tab4 <- tibble()
for (i in 1:nrow(seir_struct_tab)){
    tmp_ <- tab2 %>% filter(transition == i)
    tracts_ <- unique(tmp_$tract)
    tab3 <- NULL
    for (x in 1:length(tracts_)){
        tmp2_ <- tmp_ %>% filter(tract == tracts_[x])
        tmp3_ <- tibble(transition = tmp2_$transition[1],
                        tract = tmp2_$tract[1],
                        source = tmp2_ %>% filter(type == "source") %>% pull(values) %>% unlist(),
                        destination = tmp2_ %>% filter(type == "destination") %>% pull(values) %>% unlist(),
                        rate = tmp2_ %>% filter(type == "rate") %>% pull(values) %>% unlist())
        colnames(tmp3_)[colnames(tmp3_) %in% c("source", "destination", "rate")] <- paste(c("source", "destination", "rate"), tmp3_$tract[1], sep = "_")
        tmp3_ <- tmp3_ %>% select(-tract)

        if (x == 1){
            tab3 = tmp3_
        } else {
            tab3 <- full_join(tab3, tmp3_, relationship = "many-to-many")
        }
    }
    tab4 <- tab4 %>% bind_rows(tab3)
}



# Attempt with nesting - difficult....
# tmp_ <- tab2 %>%
#     dplyr::filter(transition == i) %>%
#     dplyr::mutate(values = purrr::map(values, ~tibble::as_tibble(t(.x)))) %>%
#     unnest(values) %>%
#     pivot_longer(cols = starts_with("V"), names_to = "v", values_to = "value") %>%
#     dplyr::filter(!is.na(value)) %>%
#     dplyr::select(-v) %>%
#     dplyr::mutate(new_col = paste(type, tract, sep = "_")) %>%
#     dplyr::mutate(new_col = gsub("source_age_|destination_age_", "age_", new_col)) %>%
#     filter(!(type == "destination" & tract == "age_strata")) %>%
#     distinct()
#
# expand_nests <- function(data){
#     data %>%
#         mutate(id = 1:nrow(.)) %>%
#         pivot_wider(names_from = new_col, values_from = value) %>%
#         select(-id)
# }
#
# tmpX <-
#     tmp_ %>%
#     nest(.by = c(tract, type)) %>%
#     mutate(data = map(data, expand_nests)) %>%
#     select(-type, -tract) %>%
#     pull(data) %>%
#     reduce(inner_join, by = c("transition")) %>%
#     distinct()

# #$data[[2]]
#     nest(.by = type)
#     mutate(data = map(data, ~pivot_wider(names_from = new_col, values_from = value)))
#
# tmpX$data


ve_data <- tab4 %>%
    dplyr::filter(grepl("theta", rate_vaccination_stage), destination_variant_type == "OMICRON") %>%
    dplyr::distinct() %>%
    left_join(
        seir_thetas %>%
            rename(prob_immune = value,
                   rate_vaccination_stage = theta) %>%
            dplyr::select(-c(param, sourcevar, destvar))
    )





# Map immunity level to each combination ----------------------------------

# ~ 1. Format to apply to SEIR -------------------------------------------

ve_data <- ve_data %>%
    filter(destination_variant_type == "OMICRON") %>%
    select(mc_infection_stage = source_infection_stage,
           mc_vaccination_stage = source_vaccination_stage,
           mc_variant_type = source_variant_type,
           mc_age_strata = source_age_strata,
           prob_immune)


# ~ 2. Add fully susceptibles so there is a value --------

ve_data <- ve_data %>%
    bind_rows(
        tibble(
            mc_infection_stage = "S",
            mc_vaccination_stage = "unvaccinated",
            mc_variant_type = "WILD",
            mc_age_strata = unique(ve_data$mc_age_strata),
            prob_immune  = 0))


# ~ 3. Reduce immunity to account for X months of immune escape -----------

imm_esc_months <- as.numeric((lubridate::as_date(config$start_date) - lubridate::as_date("2022-01-01"))) / 30  # months since omicron # average of 2 assuming beginning of Jan 2022 peak.
imm_esc_rate <- opt$imm_esc_prop # annual amount
imm_esc_ammount <- imm_esc_rate * (imm_esc_months / 12)

ve_data <- ve_data %>% mutate(prob_immune = prob_immune * (1-imm_esc_ammount))



# ~ 4. Apply to SEIR files -------------------------------------------------

# -- PULL SEIR FILES

seir_resume_file <- opt$init_file_name
transition_date <- lubridate::as_date(config$start_date)

seir_dat <- arrow::read_parquet(seir_resume_file)
seir_dat <- seir_dat %>% dplyr::select(tidyselect::starts_with("date"), tidyselect::starts_with("mc_"), tidyselect::everything())
seir_dat_cols <- colnames(seir_dat)

seir_dat <- seir_dat %>%
    filter(mc_value_type == "prevalence") %>%
    mutate(date = lubridate::as_date(date)) %>%
    filter(date == transition_date) %>%
    # group_by(across(c(-date))) %>%
    # filter(date == max(date)) %>%
    # ungroup() %>%
    filter(mc_variant_type != "VARIANTX")


# separate out files in to those that will change and those that dont much
seir_dat_static <- seir_dat %>%
    filter(mc_infection_stage %in% c("E", "I1", "I2", "I3"))

seir_dat_changing <- seir_dat %>%
    filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3")))


# SEIR static (where we are not messing with the SEIR compartments)

seir_dat_static <- seir_dat_static %>%
    select(-mc_name) %>%
    filter(mc_value_type == "prevalence") %>%
    mutate(mc_vaccination_stage = ifelse(mc_vaccination_stage == "3dose", "vaccinated", "unvaccinated")) %>%
    mutate(mc_variant_type = "ALL") %>%
    pivot_longer(cols = -c(starts_with("mc_"), date), names_to = "subpop", values_to = "value") %>%
    group_by(across(c(-value))) %>%
    summarise(value = sum(value, na.rm = TRUE)) %>%
    mutate(mc_name = paste(mc_infection_stage, mc_vaccination_stage, mc_variant_type, mc_age_strata, sep = "_")) %>%
    pivot_wider(names_from = subpop, values_from = value) %>%
    dplyr::select(all_of(seir_dat_cols))



# SEIR changing
seir_dat_changing <- seir_dat_changing %>%
    filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3"))) %>%
    left_join(ve_data) %>%
    select(prob_immune, everything()) %>%
    mutate(prob_immune = ifelse(is.na(prob_immune), 0, prob_immune))


# this is not done
gradual_waning = FALSE
if (gradual_waning){

    # Waned
    seir_dat_W <- seir_dat_changing %>%
        filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3")) & (mc_infection_stage == "W"))
    seir_dat_W <- seir_dat_W %>%
        left_join(ve_data %>% filter(mc_vaccination_stage  == "waned") %>% select(-mc_vaccination_stage)) %>%
        select(prob_immune, everything())

    # try a gradual waning calc
    seir_dat_Wgrad <- seir_dat_W %>%
        mutate(mc_vaccination_stage = ifelse(mc_vaccination_stage == "3dose", "vaccinated", "unvaccinated")) %>%
        mutate(mc_variant_type = "ALL")

    seir_dat_Wgrad <- seir_dat_Wgrad %>%
        select(-mc_infection_stage, -mc_name) %>%
        as_tibble() %>%
        pivot_longer(cols = -c(prob_immune:mc_age_strata), names_to = "loc", values_to = "n") %>%
        group_by(prob_immune, date, mc_value_type, mc_variant_type, mc_vaccination_stage, mc_age_strata, loc) %>%
        summarise(n = sum(n, na.rm=TRUE)) %>%
        as_tibble() %>%
        left_join(geodata %>% rename(loc = subpop) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
        mutate(prop = n / pop_agestrata) %>%
        group_by(USPS, loc, date, mc_value_type, mc_variant_type, mc_vaccination_stage, mc_age_strata) %>%
        summarise(prop_immune = sum((n * prob_immune) / sum(n, na.rm = TRUE), na.rm = TRUE))  %>%
        as_tibble()

    # Put them back together
    seir_dat_changing <- seir_dat_S %>%
        bind_rows(seir_dat_W) %>%
        bind_rows(seir_dat_R) %>%
        mutate(prob_immune = ifelse(is.na(prob_immune), 0, prob_immune))
}



# round immunity to boxes
seir_dat_changing <- seir_dat_changing %>%
    mutate(prob_immune_nom = as.numeric(as.character(
        # cut(prob_immune, breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1), right = TRUE, include.lowest = FALSE)
        cut(prob_immune, breaks = seq(-.1, 1.1, 0.1), labels = c(0, seq(0, .9, 0.1)+.05, 1), right = FALSE, include.lowest = FALSE)
        ))) %>%
    mutate(prob_immune_nom = ifelse(prob_immune == 0, 0, prob_immune_nom)) %>%
    dplyr::select(prob_immune_nom, prob_immune, everything())

# Make everyone unvaccinated -- before bivalent was out, so all are eligible
# dropping any vaccination for dose other than bivalent after this.
# Make everyone a single "ALL" variant

seir_dat_changing %>%
    dplyr::group_by(mc_vaccination_stage, mc_infection_stage, date) %>%
    summarise(incidI = sum(`06000`)) %>%
    as_tibble() %>%
    mutate(p = incidI/sum(incidI))

seir_dat_changing %>%
    dplyr::group_by(mc_vaccination_stage, mc_infection_stage, mc_variant_type, date) %>%
    summarise(incidI = sum(`06000`)) %>%
    as_tibble() %>%
    mutate(p = incidI/sum(incidI))

seir_dat_changing <- seir_dat_changing %>%
    mutate(mc_vaccination_stage = ifelse(mc_vaccination_stage == "3dose" | (mc_vaccination_stage == "2dose" & mc_infection_stage == "S"),
                                         "vaccinated", "unvaccinated")) %>%
    mutate(mc_variant_type = "ALL")

# aggregate
seir_dat_changing <- seir_dat_changing %>%
    select(-mc_infection_stage, -mc_name, -prob_immune) %>%
    pivot_longer(cols = -c(prob_immune_nom:mc_age_strata, date), names_to = "loc", values_to = "n") %>%
    group_by(prob_immune_nom, date, mc_value_type, mc_vaccination_stage, mc_age_strata, loc) %>%
    summarise(n = sum(n, na.rm=TRUE)) %>%
    as_tibble()

# # check
# library(ggplot2)
# # Geodata
# geodata <- read_csv("data/geodata_2019_statelevel_agestrat.csv")
#
# seir_dat_changing %>% filter(mc_age_strata == "age18to64") %>%
#     left_join(geodata %>% rename(loc = subpop) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
#     mutate(prop = n / pop_agestrata) %>%
#     ggplot(aes(x = prob_immune_nom, y = prop, color = USPS)) +
#     geom_point() +
#     facet_wrap(~USPS, nrow = 9, scales = "free_y") +
#     theme_bw() +
#     ylab("Propotion of Population") +
#     xlab("Probability Immune") +
#     coord_cartesian(ylim = c(0,0.7)) +
#     theme(legend.position = "none", axis.text.x = element_text(angle = 90))
#
# seir_dat_changing %>%
#     left_join(geodata %>% rename(loc = subpop) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
#     mutate(prop = n / pop_agestrata) %>%
#     group_by(USPS, loc, date, mc_value_type, mc_vaccination_stage, mc_age_strata) %>%
#     summarise(prop_immune = sum((n * prob_immune_nom) / sum(n, na.rm = TRUE), na.rm = TRUE)) %>%
#     ggplot(aes(x = USPS, y = prop_immune, fill = USPS)) +
#     geom_bar(stat = "identity") +
#     theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
#     theme_bw() +
#     facet_wrap(~mc_age_strata, ncol = 1)

# Move all the S, R, W to the immune ladder Xs
seir_dat_changing <- seir_dat_changing %>%
    dplyr::mutate(mc_infection_stage = paste0("X", round(prob_immune_nom*10)))

# add any missing compartments
seir_dat_changing <- seir_dat_changing %>%
    full_join(
            expand_grid(date = unique(seir_dat_changing$date),
                        mc_value_type = unique(seir_dat_changing$mc_value_type),
                        mc_vaccination_stage = config$compartments$vaccination_stage,
                        mc_age_strata = config$compartments$age_strata,
                        loc = unique(seir_dat_changing$loc),
                        mc_infection_stage = config$compartments$infection_stage[!(config$compartments$infection_stage %in% c("E", "I1", "I2", "I3"))])
        ) %>%
            dplyr::arrange(date, mc_age_strata, loc, mc_infection_stage) %>%
    mutate(n = ifelse(is.na(n), 0, n))


seir_dat_changing_final <- seir_dat_changing %>%
    dplyr::mutate(mc_variant_type = "ALL") %>%
    dplyr::select(-prob_immune_nom) %>%
    dplyr::group_by(dplyr::across(c(-n))) %>%
    dplyr::summarise(n = sum(n, na.rm = TRUE)) %>%
    dplyr::as_tibble() %>%
    dplyr::mutate(mc_name = paste(mc_infection_stage, mc_vaccination_stage, mc_variant_type, mc_age_strata, sep = "_")) %>%
    tidyr::pivot_wider(names_from = loc, values_from = n) %>%
    dplyr::select(tidyselect::any_of(seir_dat_cols))


# # CHECK
# seir_dat_changing_final %>%
#     filter(mc_age_strata == "age18to64") %>%
#     pivot_longer(cols=-c(mc_value_type:mc_name, date), names_to = "subpop", values_to = "n") %>%
#     left_join(geodata %>% rename(subpop = subpop) %>% select(USPS, subpop, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
#     mutate(prop = n / pop_agestrata) %>%
#     ggplot(aes(x = mc_infection_stage, y = n, color = USPS)) +
#     geom_point() +
#     facet_wrap(~USPS, nrow = 9, scales = "free_y") +
#     theme_bw() +
#     ylab("Propotion of Population") +
#     xlab("Probability Immune") +
#     # coord_cartesian(ylim = c(0,0.7)) +
#     theme(legend.position = "none", axis.text.x = element_text(angle = 90))




# ~ 5. Combine Again -----------------------------------------------------------

# Combine back with the seir that was not changed

seir_dat_final <- seir_dat_changing_final %>%
    dplyr::bind_rows(seir_dat_static) %>%
    dplyr::filter(mc_value_type == "prevalence")

# #check all
# apply(seir_dat_final %>% dplyr::select(starts_with("mc_")), 2, unique)



# SAVE --------------------------------------------------------------------

arrow::write_parquet(seir_dat_final, seir_resume_file)

cat(paste0("\nWriting modified init file to: \n  -- ", seir_resume_file, "\n\n"))




