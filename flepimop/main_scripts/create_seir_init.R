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
#   nodenames: <string>
#
# seeding:
#   lambda_file: <path to file>

# ```
#
# ## Input Data
#
# * <b>{data_path}/{spatial_setup::geodata}</b> is a csv with column {spatial_setup::nodenames} that denotes the geoids
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
install_configwriter <- FALSE
source("R/scripts/config_writers/config_writer_setup_flepimop.R")
#####


option_list <- list(
    optparse::make_option(c("-c", "--config"), action = "store", default = Sys.getenv("CONFIG_PATH"), type = "character", help = "path to the config file")
    # optparse::make_option(c("-k", "--keep_all_seeding"), action="store",default=TRUE,type='logical',help="Whether to filter away seeding prior to the start date of the simulation.")
)
opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))

print(paste0("Using config file: ", opt$config))
config <- flepicommon::load_config(opt$config)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}





# INPUT -------------------------------------------------------------------
# config_file <- "config_SMH_R17_noBoo_lowIE_phase2_blk1.yml"
seir_csv_file <- paste0("config/seir_structure/", "seir_R17_phase", 2, ".csv") #SEIR structure file





cross_imm_file <- "data/vaccination/Round17/cross_protection.csv"
ve_input_file <- "data/vaccination/Round17/ve_data.csv"
imm_escape_file <- "data/vaccination/Round17/immune_escape.csv"


variant_compartments <- config$compartments$variant_type
age_strat <- config$compartments$age_strata


# Geodata
geodata <- read_csv("data/geodata_2019_statelevel_agestrat.csv")

# Vaccine Effectiveness  --------------------------------------------------------------------




#
# ## ~ VE Waning Scenario   --------------------------------------------------------------------
# # See https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1054071/vaccine-surveillance-report-week-6.pdf
#
#
# ve_waning_scenarios <- list(
#     bind_rows(
#         tibble(outcome = c("infection", "case", "hosp","death"),
#                VEprop = 1-c(.5, .5, .25, .15))),
#     bind_rows(
#         tibble(outcome = c("infection", "case", "hosp","death"),
#                VEprop = 1-c(.5, .5, .25, .15)))) # same for both, easier to leave 2 levels for future rounds
#
#
#
# # ~ Cross Protection     --------------------------------------------------
#
#
# cross_imm_orig <- read_csv(cross_imm_file) %>% as_tibble() %>%
#     filter(!grepl("imm_esc", variant))
#
# cross_imm <- cross_imm_orig %>%
#     rename(variant_source=variant) %>%
#     pivot_longer(cols=-variant_source, names_to = "variant_dest", values_to = "cross_imm") %>%
#     mutate(variant_dest = toupper(variant_dest),
#            variant_source = toupper(variant_source)) %>%
#     mutate(age_group = "all") %>%
#     mutate(outcome = 'infection',
#            dose = "2dose",
#            imm_escape = 0,
#            imm_escape_variant = (cross_imm<1),
#            variant = variant_dest) %>%
#     rename(VE = cross_imm)
#
# cross_imm <- cross_imm %>%
#     bind_rows(cross_imm %>% mutate(outcome = 'case', VE = 1-((1-VE)*.70))) %>%
#     bind_rows(cross_imm %>% mutate(outcome = 'hosp', VE = 1-((1-VE)*.20))) %>%
#     bind_rows(cross_imm %>% mutate(outcome = 'death', VE = 1-((1-VE)*.10))) %>%
#     arrange(variant, variant_source, age_group, outcome, dose)
#
#
#
#
# # ~ Calculate VE parameters -----------------------------------------------
#
#
# # VE Vaccination
# VE_data <- read_csv(ve_input_file) %>% as_tibble() %>%
#     pivot_longer(cols=starts_with("ve_"),
#                  names_to = "dose",
#                  values_to = "VE",
#                  names_pattern = "ve_(.*)") %>%
#     select(age_group, variant, outcome, dose, VE) %>%
#     mutate(variant = toupper(variant))
#
# imm_escape_data <- read_csv(imm_escape_file)%>%
#     pivot_longer(cols=starts_with("ve_"),
#                  names_to = "dose",
#                  values_to = "imm_escape",
#                  names_pattern = "ve_(.*)") %>%
#     select(age_group, variant, outcome, dose, imm_escape) %>%
#     mutate(variant = toupper(variant))
#
# # Immune escape
# VE_data <- VE_data %>%
#     left_join(imm_escape_data) %>%
#     mutate(VE = VE * (1-imm_escape),
#            imm_escape_variant = imm_escape>0)
#
# # Cross protection (already has immune escape built in)
# VE_data <- VE_data %>%
#     mutate(type = "vacc_imm",
#            variant_source = "?",
#            variant_dest = variant,
#            note = paste0("S/R to E, ", variant_source, " to ", variant_dest)) %>%
#     bind_rows(
#         cross_imm %>%
#             mutate(type = "cross_imm",
#                    note = paste0("R to E, ", variant_source, " to ", variant_dest)))
#
# # VE after waning
# ve_waning <- ve_waning_scenarios[[1]] %>%
#     left_join(VE_data) %>%
#     mutate(VE = VE*VEprop) %>% select(-VEprop, -age_group)
# VE_data <- VE_data %>%
#     mutate(waning = FALSE) %>%
#     bind_rows(
#         ve_waning %>% filter(dose %in% c("2dose")) %>%
#             mutate(dose = recode(dose, "2dose"="waned"),
#                    age_group="all",
#                    waning = TRUE,
#                    note = paste0("W to E, ", variant_source, " to ", variant_dest))) %>%
#     bind_rows(
#         ve_waning %>% filter(dose %in% c("3dose")) %>%
#             mutate(dose = recode(dose, "3dose"="waned3rd"),
#                    age_group="all",
#                    waning = TRUE,
#                    note = paste0("W to E, ", variant_source, " to ", variant_dest)))
#
# VE_data %>% filter(variant=="OMICRON", dose=="3dose")
#
#
#
# # ~ Tranform to Conditional VEs and Format for the model     -----------------------------------------------
#
#
# # Tranform to conditional VEs for the model
# VE_data <- VE_data %>%
#     distinct() %>%
#     group_by(variant_source, variant_dest, type, age_group, dose, waning) %>%
#     mutate(VE_cond = ifelse(outcome=="infection", VE,
#                             trans_to_condPr(ve_inf=VE[outcome=="infection"], ve_outcome = VE))) %>%
#     ungroup()
#
# # Remove unnecessary variants
# VE_data <- VE_data %>%
#     filter((variant_source %in% variant_compartments | is.na(variant_source) | variant_source=="?") &
#                variant_dest %in% variant_compartments)
#
# # arrange to match function
# VE_data <- VE_data %>%
#     mutate(variant = factor(toupper(variant), levels = variant_compartments),
#            variant_source = factor(toupper(variant_source), levels = c(variant_compartments, "?")),
#            variant_dest = factor(toupper(variant_dest), levels = variant_compartments),
#            outcome = factor(outcome, levels = c("infection", "case", "hosp","death")),
#            age_group = factor(age_group, levels = age_strat)) %>%
#     arrange(outcome, type, variant_source, variant_dest, age_group) %>%
#     mutate(imm_escape_variant = ifelse(is.na(imm_escape_variant), FALSE, imm_escape_variant))
#
# VE_data <- VE_data %>%
#     mutate(VE_cond = ifelse(VE==1, 1, VE_cond))
#


# ~ Additional VE adjustment ---------------------------------------------------
# age_seperator <- "to"
# ve_data <- VE_data %>% filter(outcome=="infection")
# ve_data <- ve_data %>%
#     select(-VE_cond) %>%
#     mutate(variant = as.character(variant)) %>%
#     #filter(!(variant %in% c("ALPHA","WILD") & dose=="waned")) %>%
#     #mutate(variant = ifelse(variant=="DELTA" & dose=="waned", NA, variant)) %>%
#     mutate(age_group = ifelse(is.na(age_group), "all", age_group)) %>%
#     mutate(age_group = paste0("age", age_group)) %>%
#     mutate(age_group = gsub("_",age_seperator,age_group)) %>%
#     mutate(theta_name = ifelse(type=="vacc_imm",
#                                paste0("theta", ifelse(waning, paste0("W", ifelse(dose=="waned3rd", "3", "2")), gsub("dose", "",dose)), "_",
#                                       ifelse(age_group=="ageall","",paste0(age_group,"_")),
#                                       variant_dest),
#                                paste0("theta", ifelse(waning, "W",""), "_", variant_source, "_",
#                                       ifelse(age_group=="ageall","",paste0(age_group,"_")),
#                                       variant_dest))) %>%
#     filter(!grepl("thetaW1", theta_name)) %>%
#     distinct()
#


# Extract thetas from Config
seir_thetas <- config$seir$parameters
seir_thetas <- seir_thetas[which(grepl("theta", names(seir_thetas)))]
seir_thetas <- tibble::tibble(theta = names(seir_thetas), value = sapply(X = 1:length(seir_thetas), function(x=X) seir_thetas[x][[1]]$value$value))
seir_thetas <- seir_thetas %>% dplyr::mutate(value = as.numeric(gsub("1 - ", "", value)))
seir_thetas <- seir_thetas %>%
    tidyr::separate(col = theta, into = c("param", "sourcevar", "destvar"), sep = "_", remove = FALSE, extra = "drop", fill = "right") %>%
    dplyr::mutate(sourcevar = ifelse(is.na(destvar), "WILD", sourcevar)) %>%
    dplyr::mutate(mc_infection_stage = ifelse(grepl("thetaW", theta), "W", NA))



# Extract the seir structure of interest (only those transitions moving to "E")

compartment_tracts <- names(config$compartments)
seir_struct <- config$seir$transitions
# dest_E <- sapply(X = 1:length(seir_struct), function(x = X) seir_struct[x][[1]]$destination[[1]][1] == "E")
# seir_struct <- seir_struct[which(sapply(X = 1:length(seir_struct), function(x = X) seir_struct[x][[1]]$destination[[1]][1] == "E"))]

seir_struct_tab <- seir_struct %>%
    purrr::list_transpose() %>%
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
for (i in 1:length(seir_struct_tab)){
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


# ~ 1. Start with VE data (which include vacc and previous infection) --------


# add fully susceptibles so there is a value

ve_data <- ve_data %>%
    select(age_group, outcome, dose, type, variant_source, variant_dest, theta_name, VE) %>%
    bind_rows(
        tibble(
            age_group = "ageall", outcome = "infection", dose = "unvaccinated", type = "cross_imm", variant_source = "WILD", variant_dest="OMICRON", theta_name = "theta_NONE_OMICRON", VE = 0)
        )



# ~ 2. Reduce immunity to account for X months of immune escape -----------

imm_esc_months <- 2 # average of 2 assuming beginning of Jan 2022 peak.
imm_esc_rate <- .35 # annual amount
imm_esc_ammount <- imm_esc_rate * (imm_esc_months / 12)

ve_data <- ve_data %>% mutate(VE_rev = VE * (1-imm_esc_ammount))




# ~ 3. Format to apply to SEIR -------------------------------------------

ve_data <- ve_data %>%
    mutate(mc_infection_stage = ifelse(dose == "unvaccinated", "S",
                                ifelse(dose == "waned", "W", NA))) %>%
    select(mc_infection_stage, mc_variant_type = variant_source, mc_vaccination_stage = dose, prob_immune = VE_rev)




# ~ 4. Apply to SEIR files -------------------------------------------------

# -- PULL SEIR FILES

seir_file_path <- "data_other/output/R17_output_test/model_output/seir/USA/inference/med/SMH_R13_pessWan_Var_blk1/global/final/000000001.SMH_R13_pessWan_Var_blk1.seir.parquet"
transition_date <- lubridate::as_date("2022-03-06")

seir_dat <- arrow::read_parquet(seir_file_path)
seir_dat_cols <- colnames(seir_dat)
seir_dat <- seir_dat %>%
    filter(mc_value_type == "prevalence") %>%
    mutate(date_cl = lubridate::as_date(date)) %>%
    filter(date_cl == transition_date) %>%
    select(date, everything())
seir_dat <- seir_dat %>%
    group_by(across(c(-date))) %>%
    filter(date == max(date)) %>%
    ungroup()

seir_dat <- seir_dat %>%
    filter(mc_variant_type != "VARIANTX")




seir_dat_static <- seir_dat %>%
    filter(mc_infection_stage %in% c("E", "I1", "I2", "I3"))

seir_dat_changing <- seir_dat %>%
    filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3")))


# SEIR static (where we are not messing with the SEIR compartments)
seir_dat_static <- seir_dat_static %>%
    mutate(mc_vaccination_stage = ifelse(mc_vaccination_stage == "3dose", "vaccinated", "unvaccinated")) %>%
    mutate(mc_variant_type = "ALL")



# Susceptible

seir_dat_S <- seir_dat_changing %>%
    filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3")) & (mc_infection_stage == "S"))
seir_dat_S <- seir_dat_S %>%
    left_join(ve_data %>% filter(mc_variant_type  == "?" | mc_vaccination_stage == "unvaccinated") %>% select(-mc_variant_type, -mc_infection_stage) %>% distinct()) %>%
    select(prob_immune, everything()) %>%
    mutate(prob_immune = ifelse(is.na(prob_immune), 0, prob_immune))

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
    select(-date_cl, -mc_infection_stage, -mc_name) %>%
    as_tibble() %>%
    pivot_longer(cols = -c(prob_immune:mc_age_strata), names_to = "loc", values_to = "n") %>%
    group_by(prob_immune, date, mc_value_type, mc_variant_type, mc_vaccination_stage, mc_age_strata, loc) %>%
    summarise(n = sum(n, na.rm=TRUE)) %>%
    as_tibble() %>%
    left_join(geodata %>% rename(loc = geoid) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
    mutate(prop = n / pop_agestrata) %>%
    group_by(USPS, loc, date, mc_value_type, mc_variant_type, mc_vaccination_stage, mc_age_strata) %>%
    summarise(prop_immune = sum((n * prob_immune) / sum(n, na.rm = TRUE), na.rm = TRUE))  %>%
    as_tibble()




# Recovered

seir_dat_R <- seir_dat_changing %>%
    filter(!(mc_infection_stage %in% c("E", "I1", "I2", "I3")) & (mc_infection_stage == "R"))
seir_dat_R <- seir_dat_R %>%
    left_join(ve_data %>% filter(mc_variant_type  != "?" & mc_vaccination_stage == "2dose") %>% select(-mc_vaccination_stage, -mc_infection_stage)) %>%
    select(prob_immune, everything())


# Put them back together

seir_immladder <- seir_dat_S %>%
    bind_rows(seir_dat_W) %>%
    bind_rows(seir_dat_R) %>%
    mutate(prob_immune = ifelse(is.na(prob_immune), 0, prob_immune))

# round immunity to boxes
seir_immladder <- seir_immladder %>%
    mutate(prob_immune_nom = as.numeric(as.character(
        # cut(prob_immune, breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1), right = TRUE, include.lowest = FALSE)
        cut(prob_immune, breaks = seq(-.1, 1.1, 0.1), labels = c(0, seq(0, .9, 0.1)+.05, 1), right = FALSE, include.lowest = FALSE)
        ))) %>%
    mutate(prob_immune_nom = ifelse(prob_immune == 0, 0, prob_immune_nom))

# Make everyone unvaccinated -- before bivalent was out, so all are eligible
# dropping any vaccination for dose other than bivalent after this.
# Make everyone a single "ALL" variant
seir_immladder <- seir_immladder %>%
    mutate(mc_vaccination_stage = ifelse(mc_vaccination_stage == "3dose", "vaccinated", "unvaccinated")) %>%
    mutate(mc_variant_type = "ALL")

# aggregate
seir_immladder <- seir_immladder %>%
    select(-date_cl, -mc_infection_stage, -mc_name) %>%
    pivot_longer(cols = -c(prob_immune:mc_age_strata, prob_immune_nom), names_to = "loc", values_to = "n") %>%
    group_by(prob_immune_nom, date, mc_value_type, mc_vaccination_stage, mc_age_strata, loc) %>%
    summarise(n = sum(n, na.rm=TRUE)) %>%
    as_tibble()






# check
seir_immladder %>% filter(mc_age_strata == "age18to64") %>%
    left_join(geodata %>% rename(loc = geoid) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
    mutate(prop = n / pop_agestrata) %>%
    ggplot(aes(x = prob_immune_nom, y = prop, color = USPS)) +
    geom_point() +
    facet_wrap(~USPS, nrow = 9, scales = "free_y") +
    theme_bw() +
    ylab("Propotion of Population") +
    xlab("Probability Immune") +
    coord_cartesian(ylim = c(0,0.7)) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90))


seir_immladder %>%
    left_join(geodata %>% rename(loc = geoid) %>% select(USPS, loc, mc_age_strata = age_strata, pop_agestrata) %>% distinct()) %>%
    mutate(prop = n / pop_agestrata) %>%
    group_by(USPS, loc, date, mc_value_type, mc_vaccination_stage, mc_age_strata) %>%
    summarise(prop_immune = sum((n * prob_immune_nom) / sum(n, na.rm = TRUE), na.rm = TRUE)) %>%
    ggplot(aes(x = USPS, y = prop_immune, fill = USPS)) +
    geom_bar(stat = "identity") +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90)) +
    theme_bw() +
    facet_wrap(~mc_age_strata, ncol = 1)


# Move all the S, R, W to the immune ladder Xs

seir_immladder_final <- seir_immladder %>%
    mutate(mc_infection_stage = paste0("X", round(prob_immune_nom*10))) %>%
    mutate(mc_variant_type = "ALL") %>%
    mutate(mc_name = paste("mc", mc_vaccination_stage, mc_variant_type, mc_age_strata, sep = "_")) %>%
    pivot_wider(names_from = loc, values_from = n)

seir_immladder_final <- seir_immladder_final %>% select(any_of(seir_dat_cols))




# Combine back with the seir that was not changed

seir_dat_final <- seir_immladder_final %>%
    bind_rows(seir_dat_static)
