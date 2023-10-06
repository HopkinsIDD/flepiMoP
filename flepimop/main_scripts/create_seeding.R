##
# @file
# @brief Creates a seeding file
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

# subpop_setup:
#   geodata: <path to file>
#   subpop: <string>
#
# seeding:
#   lambda_file: <path to file>

# ```
#
# ## Input Data
#
# * <b>{data_path}/{subpop_setup::geodata}</b> is a csv with column {subpop_setup::subpop} that denotes the subpop
#
# ## Output Data
#
# * <b>data/case_data/USAFacts_case_data.csv</b> is the case csv downloaded from USAFacts
# * <b>data/case_data/USAFacts_death_data.csv</b> is the death csv downloaded from USAFacts
# * <b>{seeding::lambda_file}</b>: filter file
#

## @cond

library(flepicommon)
library(magrittr)
library(dplyr)
library(readr)
library(tidyr)
library(purrr)


option_list <- list(
    optparse::make_option(c("-c", "--config"), action = "store", default = Sys.getenv("CONFIG_PATH"), type = "character", help = "path to the config file"),
    optparse::make_option(c("-s", "--seed_variants"), action="store", default = Sys.getenv("SEED_VARIANTS"), type='logical',help="Whether to add variants/subtypes to outcomes in seeding."),
    optparse::make_option(c("-k", "--keep_all_seeding"), action="store",default=TRUE,type='logical',help="Whether to filter away seeding prior to the start date of the simulation.")
)

opt <- optparse::parse_args(optparse::OptionParser(option_list = option_list))

gt_source <- NULL
print(paste0("Using config file: ", opt$config))
config <- flepicommon::load_config(opt$config)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}

if (is.null(config$subpop_setup$us_model)) {
    config$subpop_setup$us_model <- FALSE
    if ("modeled_states" %in% names(config$subpop_setup)) {
        config$subpop_setup$us_model <- TRUE
    }
}

is_US_run <- config$subpop_setup$us_model
seed_variants <- "variant_filename" %in% names(config$seeding)

if (!is.na(opt$seed_variants)){
    seed_variants <- opt$seed_variants
}


## backwards compatibility with configs that don't have inference$gt_source
## parameter will use the previous default data source (USA Facts)
if (is.null(config$inference$gt_source)) {
    if (is_US_run) {
        gt_source <- "usafacts"
    } else {
        gt_source <- NULL
    }
} else{
    gt_source <- config$inference$gt_source
}
if (is.null(config$seeding$delay_incidC)) {
    config$seeding$delay_incidC <- 5
}
if (is.null(config$seeding$ratio_incidC)) {
    config$seeding$ratio_incidC <- 10
}



# ~ Load ground truth data ------------------------------------------------
#  -- this is already saved from running the `build_[X]_data.R` script at the model setup stage.
#
data_path <- config$inference$gt_data_path
if (is.null(data_path)) {
    data_path <- config$seeding$casedata_file
    if (is.null(data_path)) {
        stop(paste(
            "Please provide a ground truth file",
            " as inference::gt_data_path or seeding::casedata_file"
        ))
    }
}
cases_deaths <- readr::read_csv(data_path)
print(paste("Successfully loaded data from ", data_path, "for seeding."))

if (is_US_run) {
    cases_deaths <- cases_deaths %>%
        mutate(subpop = stringr::str_pad(subpop, width = 5, side = "right", pad = "0"))
}

print(paste("Successfully pulled", gt_source, "data for seeding."))





# ~ Seeding Variants ------------------------------------------------------

if (seed_variants) {

    variant_data <- readr::read_csv(config$seeding$variant_filename)

    # rename date columns in data for joining
    colnames(variant_data)[colnames(variant_data) == "Update"] ="date"
    colnames(cases_deaths)[colnames(cases_deaths) == "Update"] ="date"

    if (any(grepl(paste(unique(variant_data$variant), collapse = "|"), colnames(cases_deaths)))){

        if (!is.null(config$seeding$seeding_outcome)){
            if (config$seeding$seeding_outcome=="incidH"){
                cases_deaths <- cases_deaths %>%
                    dplyr::select(date, subpop, paste0("incidH_", names(config$seeding$seeding_compartments)))
                colnames(cases_deaths) <- gsub("incidH_", "", colnames(cases_deaths))
                cases_deaths <- cases_deaths %>%
                    dplyr::mutate(dplyr::across(tidyselect::any_of(unique(names(config$seeding$seeding_compartments))), ~ tidyr::replace_na(.x, 0)))
            } else {
                stop(paste(
                    "Currently only incidH is implemented for config$seeding$seeding_outcome."
                ))
            }
        } else {
            cases_deaths <- cases_deaths %>%
                dplyr::select(date, subpop, paste0("incidC_", names(config$seeding$seeding_compartments)))
            colnames(cases_deaths) <- gsub("incidC_", "", colnames(cases_deaths))
            cases_deaths <- cases_deaths %>%
                dplyr::mutate(dplyr::across(tidyselect::any_of(unique(names(config$seeding$seeding_compartments))), ~ tidyr::replace_na(.x, 0)))
        }

    } else {

        if (!is.null(config$seeding$seeding_outcome)){
            if (config$seeding$seeding_outcome=="incidH"){
                cases_deaths <- cases_deaths %>%
                    dplyr::select(date, subpop, incidH) %>%
                    dplyr::left_join(variant_data) %>%
                    dplyr::mutate(incidI = incidH * prop) %>%
                    dplyr::select(-prop, -incidH) %>%
                    tidyr::pivot_wider(names_from = variant, values_from = incidI) %>%
                    dplyr::mutate(dplyr::across(tidyselect::any_of(unique(variant_data$variant)), ~ tidyr::replace_na(.x, 0)))
            } else {
                stop(paste(
                    "Currently only incidH is implemented for config$seeding$seeding_outcome."
                ))
            }
        } else {
            cases_deaths <- cases_deaths %>%
                dplyr::select(date, subpop, incidC) %>%
                dplyr::left_join(variant_data) %>%
                dplyr::mutate(incidI = incidC * prop) %>%
                dplyr::select(-prop, -incidC) %>%
                tidyr::pivot_wider(names_from = variant, values_from = incidI) %>%
                dplyr::mutate(dplyr::across(tidyselect::any_of(unique(variant_data$variant)), ~ tidyr::replace_na(.x, 0)))
        }
    }
}

## Check some data attributes:
## This is a hack:
if ("FIPS" %in% names(cases_deaths)) {
    cases_deaths$subpop <- cases_deaths$FIPS
    warning("Changing FIPS name in seeding. This is a hack")
}
if ("Update" %in% names(cases_deaths)) {
    cases_deaths$date <- cases_deaths$Update
    warning("Changing Update name in seeding. This is a hack")
}
obs_subpop <- config$subpop_setup$subpop
required_column_names <- NULL

check_required_names <- function(df, cols, msg) {
    if (!all(cols %in% names(df))) {
        stop(msg)
    }
}



# ~ Seeding Compartments --------------------------------------------------

if ("compartments" %in% names(config)) {

    if (all(names(config$seeding$seeding_compartments) %in% names(cases_deaths))) {
        required_column_names <- c("subpop", "date", names(config$seeding$seeding_compartments))
        check_required_names(
            cases_deaths,
            required_column_names,
            paste(
                "To create the seeding, we require the following columns to exist in the case data",
                paste(required_column_names, collapse = ", ")
            )
        )
        incident_cases <- cases_deaths[, required_column_names] %>%
            tidyr::pivot_longer(!!names(config$seeding$seeding_compartments), names_to = "seeding_group") %>%
            dplyr::mutate(
                source_column = sapply(
                    config$seeding$seeding_compartments[seeding_group],
                    function(x){
                        paste(x$source_compartment, collapse = "_")
                    }
                ),
                destination_column = sapply(
                    config$seeding$seeding_compartments[seeding_group],
                    function(x){
                        paste(x$destination_compartment, collapse = "_")
                    }
                )
            ) %>%
            tidyr::separate(source_column, paste("source", names(config$compartments), sep = "_")) %>%
            tidyr::separate(destination_column, paste("destination", names(config$compartments), sep = "_"))
        required_column_names <- c("subpop", "date", "value", paste("source", names(config$compartments), sep = "_"), paste("destination", names(config$compartments), sep = "_"))
        incident_cases <- incident_cases[, required_column_names]

        # if (!is.null(config$smh_round)) {
        #     if (config$smh_round=="R11"){
        #         incident_cases_om <- incident_cases %>%
        #             dplyr::filter(Update==lubridate::as_date("2021-12-01")) %>%
        #             dplyr::group_by(FIPS, Update, source_infection_stage, source_vaccination_stage, source_age_strata,
        #                             destination_vaccination_stage, destination_age_strata, destination_infection_stage) %>%
        #             dplyr::summarise(value = sum(value, na.rm=TRUE)) %>%
        #             dplyr::mutate(source_variant_type = "WILD", destination_variant_type = "OMICRON") %>%
        #             dplyr::mutate(value = round(ifelse(FIPS %in% c("06000","36000"), 10,
        #                                                ifelse(FIPS %in% c("53000","12000"), 5, 1)))) %>%
        #             tibble::as_tibble()
        #     }
        # }


    } else if ("seeding_compartments" %in% names(config$seeding) ) {
        stop(paste(
            "Could not find all compartments.  Looking for",
            paste(names(config$seeding$seeding_compartments), collapse = ", "),
            "from selection",
            paste(names(cases_deaths), collapse = ", ")
        ))
    } else {
        stop("Please add a seeding_compartments section to the config")
    }
} else {
    required_column_names <- c("subpop", "date", "incidI")
    check_required_names(
        cases_deaths,
        required_column_names,
        paste(
            "To create the seeding, we require the following columns to exist in the case data",
            paste(required_column_names, collapse = ", ")
        )
    )
    incident_cases <- cases_deaths[, required_column_names] %>%
        tidyr::pivot_longer(cols = "incidI", names_to = "source_infection_stage", values_to = "value")
    incident_cases$destination_infection_stage <- "E"
    incident_cases$source_infection_stage <- "S"
    required_column_names <- c("subpop", "date", "value", "source_infection_stage", "destination_infection_stage")

    if ("parallel_structure" %in% names(config[["seir"]][["parameters"]])) {
        parallel_compartments <- config[["seir"]][["parameters"]][["parallel_structure"]][["compartments"]]
    } else {
        parallel_compartments <- setNames(NA, "unvaccinated")
    }
    incident_cases[["source_vaccination_stage"]] <- names(parallel_compartments)[[1]]
    incident_cases[["destination_vaccination_stage"]] <- names(parallel_compartments)[[1]]
    required_column_names <- c(required_column_names, "source_vaccination_stage", "destination_vaccination_stage")
}

print(required_column_names)
incident_cases <- incident_cases[, required_column_names]


all_times <- lubridate::ymd(config$start_date) +
    seq_len(lubridate::ymd(config$end_date) - lubridate::ymd(config$start_date))

geodata <- flepicommon::load_geodata_file(
    file.path(config$data_path, config$subpop_setup$geodata),
    5,
    "0",
    TRUE
)

all_subpop <- geodata[["subpop"]]



incident_cases <- incident_cases %>%
    dplyr::filter(subpop %in% all_subpop) %>%
    dplyr::select(!!!required_column_names)
incident_cases <- incident_cases %>% filter(value>0)

incident_cases[["date"]] <- as.Date(incident_cases$date)

if (is.null(config[["seeding"]][["seeding_inflation_ratio"]])) {
    config[["seeding"]][["seeding_inflation_ratio"]] <- 10
}
if (is.null(config[["seeding"]][["seeding_delay"]])) {
    config[["seeding"]][["seeding_delay"]] <- 5
}

grouping_columns <- required_column_names[!required_column_names %in% c("date", "value")]
incident_cases <- incident_cases %>%
    dplyr::group_by(!!!rlang::syms(grouping_columns)) %>%
    dplyr::group_modify(function(.x, .y) {
        .x %>%
            dplyr::arrange(date) %>%
            dplyr::filter(value > 0) %>%
            .[seq_len(min(nrow(.x), 5)), ] %>%
            dplyr::mutate(
                date = date - lubridate::days(config[["seeding"]][["seeding_delay"]]),
                value = config[["seeding"]][["seeding_inflation_ratio"]] * value + .05
            ) %>%
            return
    }) %>%
    dplyr::ungroup() %>%
    dplyr::select(!!!rlang::syms(required_column_names))

names(incident_cases)[1:3] <- c("subpop", "date", "amount")

incident_cases <- incident_cases %>%
    dplyr::filter(!is.na(amount) | !is.na(date))

lambda_dir <- dirname(config$seeding$lambda_file)
if (!dir.exists(lambda_dir)) {
    suppressWarnings(dir.create(lambda_dir, recursive = TRUE))
}

# Add "no_perturb" flag
if (!("no_perturb" %in% colnames(incident_cases))){
    incident_cases$no_perturb <- FALSE
}

# Combine with population seeding for compartments (current hack to get population in)

if ("compartments" %in% names(config) & "pop_seed_file" %in% names(config[["seeding"]])) {
    seeding_pop <- readr::read_csv(config$seeding$pop_seed_file)

    # Add "no_perturb" flag
    if (!("no_perturb" %in% colnames(seeding_pop))){
        seeding_pop$no_perturb <- TRUE
    }
    seeding_pop <- seeding_pop %>%
        dplyr::filter(subpop %in% all_subpop) %>%
        dplyr::select(!!!colnames(incident_cases))

    incident_cases <- incident_cases %>%
        dplyr::bind_rows(seeding_pop) %>%
        dplyr::arrange(subpop, date)
}



# Limit seeding to on or after the config start date and before the config end date
if (max(incident_cases$date) < lubridate::as_date(config$start_date)){

    incident_cases <- incident_cases %>%
        group_by(subpop) %>%
        filter(date == min(date)) %>%
        distinct() %>%
        ungroup() %>%
        mutate(date = lubridate::as_date(config$start_date),
               amount = 0)
} else {

    # keep all seeding -- dont filter away before sim date
    if (opt$keep_all_seeding){
        incident_cases <- incident_cases %>%
            mutate(date = lubridate::as_date(ifelse(date < lubridate::as_date(config$start_date),
                                                    lubridate::as_date(config$start_date), lubridate::as_date(date)))) %>%
            filter(date <= config$end_date) %>%
            # group_by(across(c(-amount))) %>%
            # summarise(amount = sum(amount, na.rm = TRUE)) %>%
            as_tibble()
    } else {
        incident_cases <- incident_cases %>%
            filter(date >= config$start_date & date <= config$end_date)
    }
}




# Save it

write.csv(
    incident_cases,
    file = file.path(config$seeding$lambda_file),
    row.names = FALSE
)

print(paste("Saved seeding to", config$seeding$lambda_file))
head(incident_cases)

## @endcond

