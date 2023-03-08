



# SETUP -------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(tidycensus)
library(readr)
library(lubridate)


option_list = list(
    optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
    optparse::make_option(c("-p", "--path"), action="store", default=Sys.getenv("FLEPI_PATH", "flepiMoP"), type='character', help="path to the flepiMoP directory"),
    optparse::make_option(c("-w", "--wide_form"), action="store",default=FALSE,type='logical',help="Whether to generate the old wide format mobility or the new long format"),
    optparse::make_option(c("-s", "--gt_data_source"), action="store",default=Sys.getenv("GT_DATA_SOURCE", "csse_case, nchs_death, hhs_hosp"),type='character',help="sources of gt data")
)
opt = optparse::parse_args(optparse::OptionParser(option_list=option_list))

config <- flepicommon::load_config(opt$c)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}

if (exists("config$inference$gt_source")) {
    opt$gt_data_source <- config$inference$gt_source
}

outdir <- config$spatial_setup$base_path
filterUSPS <- config$spatial_setup$modeled_states
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$spatial_setup$state_level) && config$spatial_setup$state_level, TRUE, FALSE)

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# source data functions
source(file.path(opt$path, "datasetup/data_setup_source.R"))




# PULL DATA ---------------------------------------------------------------

end_date_ <- config$end_date_groundtruth
if (is.null(end_date_)) end_date_ <- config$end_date

gt_data <- list()

# ~ Pull Data from Covidcast -------------------

if (any(grepl("csse", opt$gt_data_source))){
    gt_source <- "covidcast"
    gt_scale <- "US state"

    csse_target <- unlist(strsplit(opt$gt_data_source, ", "))
    csse_target <- tolower(gsub("csse_", "", csse_target[grepl("csse", csse_target)]))


    us_data <- flepicommon::get_groundtruth_from_source(source = gt_source, scale = gt_scale,
                                                        incl_unass = TRUE,
                                                        variables = c("Confirmed", "Deaths", "incidI", "incidDeath"),
                                                        adjust_for_variant = TRUE,
                                                        variant_props_file = config$seeding$variant_filename)
    us_data <- us_data %>%
        mutate(FIPS = stringr::str_pad(FIPS, width=5, side="right", pad="0")) %>%
        filter(Update >= as_date(config$start_date) & Update <= as_date(end_date_)) %>%
        # mutate(gt_source = "csse") %>%
        filter()
    colnames(us_data) <- gsub("Deaths", "cumD", colnames(us_data))
    colnames(us_data) <- gsub("incidDeath", "incidD", colnames(us_data))
    colnames(us_data) <- gsub("Confirmed", "cumC", colnames(us_data))
    colnames(us_data) <- gsub("incidI", "incidC", colnames(us_data))

    if (!any(grepl("case", csse_target))){
        us_data <- us_data %>% select(-c(starts_with("incidC"), starts_with("cumC")))
    }
    if (!any(grepl("death", csse_target))){
        us_data <- us_data %>% select(-c(starts_with("incidD"), starts_with("cumD")))
    }
    gt_data <- append(gt_data, list(us_data))
}



# ~ Pull Deaths from NCHS -------------------------------------------------

if (any(grepl("nchs", opt$gt_data_source))){

    nchs_data <- get_covidcast_deaths(scale = "US state",
                                      source = "nchs-mortality",
                                      fix_negatives = TRUE,
                                      adjust_for_variant = FALSE,
                                      variant_props_file = config$seeding$variant_filename)

    # Distribute from weekly to daily
    # -- do this mainly for seeding. it gets re-aggregated for fitting
    # -- tbis is implemented as a spline fit to cumulative data, from which daily cum and incident are calculated.

    nchs_data <- make_daily_data(data = nchs_data %>% dplyr::select(-signal), current_timescale = "week") #%>%
        # mutate(gt_source = "nchs")

    gt_data <- append(gt_data, list(nchs_data))

}



# ~ Pull HHS hospitalization  -------------------

if (any(grepl("hhs", opt$gt_data_source))){

    us_hosp <- flepicommon::get_hhsCMU_incidH_st_data()
    us_hosp <- us_hosp %>%
        dplyr::select(-incidH_all) %>%
        rename(incidH = incidH_confirmed) %>%
        mutate(FIPS = stringr::str_pad(FIPS, width=5, side="right", pad="0")) %>%
        filter(Update >= as_date(config$start_date) & Update <= as_date(end_date_))

    # Apply variants
    variant_props_file <- config$seeding$variant_filename
    adjust_for_variant <- !is.null(variant_props_file)

    head(read_csv(variant_props_file))

    if (adjust_for_variant) {

        tryCatch({
            us_hosp <- flepicommon::do_variant_adjustment(us_hosp, variant_props_file)
        }, error = function(e) {
            stop(paste0("Could not use variant file |", variant_props_file,
                        "|, with error message", e$message))
        })
    }

    # us_hosp <- us_hosp %>% mutate(gt_source = "csse")

    gt_data <- append(gt_data, list(us_hosp))
}





# ~ Combine ---------------------------------------------------------------

us_data <- gt_data %>%
    purrr::reduce(full_join) %>%
    filter(source != "US", source != "USA") %>%
    mutate(FIPS = stringr::str_pad(FIPS, width=5, side="right", pad="0"))



# ~ Filter ----------------------------------------------------------------

# Filter to dates we care about for speed and space
us_data <- us_data %>%
    filter(Update >= lubridate::as_date(config$start_date) & Update <= lubridate::as_date(end_date_))

# Filter to states we care about
locs <- config$spatial_setup$modeled_states
us_data <- us_data %>%
    filter(source %in% locs) %>%
    filter(!is.na(source)) %>%
    rename(date = Update)




# ~ Fix Zeros -------------------------------------------------------------

us_data <- us_data %>%
    mutate(across(starts_with("incid"), ~ replace_na(.x, 0))) %>%
    mutate(across(starts_with("incid"), ~ as.numeric(.x)))


# Save
write_csv(us_data, config$filtering$data_path)



cat(paste0("Ground truth data saved\n",
           "  -- file:      ", config$filtering$data_path,".\n",
           "  -- outcomes:  ", paste(grep("incid", colnames(us_data), value = TRUE), collapse = ", ")))


# END
