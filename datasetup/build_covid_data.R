



# SETUP -------------------------------------------------------------------

library(dplyr)
library(tidyr)
# library(tidycensus)
library(readr)
library(lubridate)
library(flepicommon)


option_list = list(
    optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
    optparse::make_option(c("-p", "--path"), action="store", default=Sys.getenv("FLEPI_PATH", "flepiMoP"), type='character', help="path to the flepiMoP directory"),
    optparse::make_option(c("-w", "--wide_form"), action="store",default=FALSE,type='logical',help="Whether to generate the old wide format mobility or the new long format"),
    optparse::make_option(c("-s", "--gt_data_source"), action="store",default=Sys.getenv("GT_DATA_SOURCE", "csse_case, fluview_death, hhs_hosp"),type='character',help="sources of gt data"),
    optparse::make_option(c("-d", "--delphi_api_key"), action="store",default=Sys.getenv("DELPHI_API_KEY"),type='character',help="API key for Delphi Epidata API (see https://cmu-delphi.github.io/delphi-epidata/)")
)
opt = optparse::parse_args(optparse::OptionParser(option_list=option_list))

config <- flepicommon::load_config(opt$c)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}

if (exists("config$inference$gt_source")) {
    opt$gt_data_source <- config$inference$gt_source
}

outdir <- config$data_path
# filterUSPS <- config$subpop_setup$modeled_states
filterUSPS <- c("WY","VT","DC","AK","ND","SD","DE","MT","RI","ME","NH","HI","ID","WV","NE","NM",
                "KS","NV","MS","AR","UT","IA","CT","OK","OR","KY","LA","AL","SC","MN","CO","WI",
                "MD","MO","IN","TN","MA","AZ","WA","VA","NJ","MI","NC","GA","OH","IL","PA","NY","FL","TX","CA")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$subpop_setup$state_level) && config$subpop_setup$state_level, TRUE, FALSE)

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# source data functions
source(file.path(opt$path, "datasetup/data_setup_source.R"))





# SET DELPHI API KEY ------------------------------------------------------

if (any(grepl("nchs|hhs", opt$gt_data_source))){
    if (!is.null(config$inference$gt_api_key)){
        cat(paste0("Using Config variable for Delphi API key: ", config$inference$gt_api_key))
        options(covidcast.auth = config$inference$gt_api_key)
    } else if (!is.null(opt$delphi_api_key)){
        cat(paste0("Using Environment variable for Delphi API key: ", opt$delphi_api_key))
        options(covidcast.auth = opt$delphi_api_key)
    } else {
        newkey <- readline(prompt = "Please enter your Delphi API key before proceeding:")
        #check
        key_correct_len <- nchar(newkey) > 10 & nchar(newkey) < 20
        # cli <- covidcast::covidcast_signal(data_source = "fb-survey", signal = "smoothed_cli",
        #                 start_day = "2020-05-01", end_day = "2020-05-01",
        #                 geo_type = "state")
        if (!key_correct_len){
            cat(paste0("**Incorrect API Key.**\n
                       Please register for a Delphi Epidata API key before proceeding.\n
                       Go to `https://cmu-delphi.github.io/delphi-epidata/` to register."))
            stop()
        } else {
            cat(paste0("Using Input variable for Delphi API key: ", newkey))
            options(covidcast.auth = newkey)
        }
    }
}



# PULL DATA ---------------------------------------------------------------

end_date_ <- config$end_date_groundtruth
if (is.null(end_date_)) end_date_ <- config$end_date

# add CSSE case data to every pull for use as seeding data
if (!grepl("case", opt$gt_data_source)){
    opt$gt_data_source <- paste0("csse_case, ", opt$gt_data_source)
}

# whether to adjust for variants
adjust_for_variant <- !is.null(config$seeding$variant_filename)


gt_data <- list()

# ~ Pull Data from Covidcast -------------------

if (any(grepl("csse", opt$gt_data_source))){
    gt_source <- "covidcast"
    gt_scale <- "US state"

    csse_target <- unlist(strsplit(opt$gt_data_source, ", "))
    csse_target <- tolower(gsub("csse_", "", csse_target[grepl("csse", csse_target)]))

    csse_data <- flepicommon::get_groundtruth_from_source(source = gt_source, scale = gt_scale,
                                                          incl_unass = TRUE,
                                                          variables = c("incidC", "cumC", "incidD", "cumD"),
                                                          adjust_for_variant = FALSE,
                                                          variant_props_file = NULL)
    csse_data <- csse_data %>%
        mutate(FIPS = stringr::str_pad(FIPS, width=5, side="right", pad="0")) %>%
        filter(Update >= as_date(config$start_date) & Update <= as_date(end_date_)) %>%
        # mutate(gt_source = "csse") %>%
        filter()
    colnames(csse_data) <- gsub("Deaths", "cumD", colnames(csse_data))
    colnames(csse_data) <- gsub("incidDeath", "incidD", colnames(csse_data))
    colnames(csse_data) <- gsub("Confirmed", "cumC", colnames(csse_data))
    colnames(csse_data) <- gsub("incidI", "incidC", colnames(csse_data))

    if (!any(grepl("case", csse_target))){
        csse_data <- csse_data %>% select(-c(starts_with("incidC"), starts_with("cumC")))
    }
    if (!any(grepl("death", csse_target))){
        csse_data <- csse_data %>% select(-c(starts_with("incidD"), starts_with("cumD")))
    }

    # Apply variants
    if (!is.null(config$seeding$variant_filename)){
        variant_props_file <- config$seeding$variant_filename
        adjust_for_variant <- !is.null(variant_props_file)
        head(read_csv(variant_props_file))

        if (adjust_for_variant) {

            tryCatch({
                csse_data_vars <- flepicommon::do_variant_adjustment(csse_data, variant_props_file)
            }, error = function(e) {
                stop(paste0("Could not use variant file |", variant_props_file,
                            "|, with error message", e$message))
            })
        }
        csse_data <- csse_data_vars
    }

    gt_data <- append(gt_data, list(csse_data))
}



# ~ Pull Deaths from NCHS -------------------------------------------------

if (any(grepl("nchs", opt$gt_data_source))){

    nchs_data <- get_covidcast_deaths(scale = "US state",
                                      source = "nchs-mortality",
                                      fix_negatives = FALSE,
                                      adjust_for_variant = FALSE,
                                      variant_props_file = config$seeding$variant_filename)

    # Distribute from weekly to daily
    # -- do this mainly for seeding. it gets re-aggregated for fitting
    # -- tbis is implemented as a spline fit to cumulative data, from which daily cum and incident are calculated.


    # Limit the data to X weeks before the pulled date
    if (!exists("config$inference$nchs_weeklag")) { config$inference$nchs_weeklag <- 2}
    nchs_data <- nchs_data %>% filter(Update < lubridate::floor_date(Sys.Date() - config$inference$nchs_weeklag*7, "weeks"))


    nchs_data <- make_daily_data(data = nchs_data, current_timescale = "week") #%>%
    # mutate(gt_source = "nchs")

    gt_data <- append(gt_data, list(nchs_data))

}





# devtools::install_github("https://github.com/shauntruelove/cdcfluview")
# install.packages("cdcfluview") # original does not have covid19 data. Shaun's forked version does.

# ~ Pull Deaths from NCHS via FluView -------------------------------------------------

if (any(grepl("fluview", opt$gt_data_source))){

    library(cdcfluview)
    # cdcfluview::
    fluview_data <- cdcfluview::pi_mortality(coverage_area = "state", years = 2019:2023)

    # if (file.exists(file.path("data_other/nchs_deaths.csv"))){

        # downloaded data for deaths from https://gis.cdc.gov/grasp/fluview/mortality.html
        # fluview_data2 <- read_csv(file.path("../../ncov/covid19_usa4/data_other/nchs_deaths.csv"))
        # fluview_data2 <- read_csv(file.path("data_other/nchs_deaths.csv"))
        colnames(fluview_data) <- tolower(colnames(fluview_data))

        fluview_data <- fluview_data %>% select(state = region_name, seasonid, season,
                                                week = year_week_num, Update = week_start,
                                                percent_complete,
                                                incidD = number_covid19) %>%
            mutate(data_source = "fluview") %>%
            mutate(incidD = gsub(",", "", incidD)) %>%
            mutate(incidD = as.integer(incidD)) %>%
            mutate(year = ifelse(week >= 40, as.integer(substr(season, 1, 4)), as.integer(paste0("20", substr(season, 6,9))))) %>%
            left_join(
                tibble(Update = seq.Date(from = as_date("2019-12-01"), to=as_date(Sys.Date() + 21), by = "1 weeks")) %>%
                    mutate(week = lubridate::epiweek(Update),
                           year = lubridate::epiyear(Update),
                           Update = MMWRweek::MMWRweek2Date(MMWRyear = year, MMWRweek = week, MMWRday = 1))) %>%
            filter(!is.na(Update)) %>%
            mutate(state = ifelse(grepl("New York", state), "New York", state)) %>%
            left_join(
                tibble(state = c(state.name, "District of Columbia"),
                       source = c(state.abb, "DC"))) %>%
            group_by(across(-incidD)) %>%
            summarise(incidD = sum(incidD))

    max(fluview_data$Update)

    census_data <- read_csv(file = file.path(config$data_path, config$subpop_setup$geodata))
    fluview_data <- fluview_data %>%
        dplyr::inner_join(census_data %>% dplyr::select(source = USPS, FIPS = subpop)) %>%
        dplyr::select(Update, source, FIPS, incidD)


    # Distribute from weekly to daily
    # -- do this mainly for seeding. it gets re-aggregated for fitting
    # -- tbis is implemented as a spline fit to cumulative data, from which daily cum and incident are calculated.

    # Limit the data to X weeks before the pulled date
    if (!exists("config$inference$nchs_weeklag")) { config$inference$nchs_weeklag <- 2}
    fluview_data <- fluview_data %>% filter(Update < lubridate::floor_date(Sys.Date() - config$inference$nchs_weeklag*7, "weeks"))

    fluview_data <- make_daily_data(data = fluview_data, current_timescale = "week") #%>%
    # mutate(gt_source = "nchs")
    # fluview_data <- fluview_data %>%
    # filter(source %in% config$subpop_setup$modeled_states)
    # Update >= config$start_date,
    # Update <= config$end_date_groundtruth)
    gt_data <- append(gt_data, list(fluview_data))

}

#  OLD CODE TO USE MANUALLY DOWNLOADED DATA
# if (any(grepl("fluview", opt$gt_data_source))){
#
#     # cdcfluview::
#     fluview_data <- cdcfluview::pi_mortality(coverage_area = "state", years = 2019:2023)
#
#     if (file.exists(file.path("data_other/nchs_deaths.csv"))){
#
#         # downloaded deaths from https://gis.cdc.gov/grasp/fluview/mortality.html
#         fluview_data <- read_csv(file.path("data_other/nchs_deaths.csv"))
#         colnames(fluview_data) <- tolower(colnames(fluview_data))
#         fluview_data <- fluview_data %>% select(state = `sub area`, season, week, incidD = `num covid-19 deaths`) %>%
#             mutate(data_source = "fluview") %>%
#             mutate(incidD = gsub(",", "", incidD)) %>%
#             mutate(incidD = as.integer(incidD)) %>%
#             mutate(year = ifelse(week >= 40, as.integer(substr(season, 1, 4)), as.integer(paste0("20", substr(season, 6,9))))) %>%
#             left_join(
#                 tibble(Update = seq.Date(from = as_date("2019-12-01"), to=as_date(Sys.Date() + 21), by = "1 weeks")) %>%
#                     mutate(week = lubridate::epiweek(Update),
#                            year = lubridate::epiyear(Update),
#                            Update = MMWRweek::MMWRweek2Date(MMWRyear = year, MMWRweek = week, MMWRday = 1))) %>%
#             filter(!is.na(Update)) %>%
#             mutate(state = ifelse(grepl("New York", state), "New York", state)) %>%
#             left_join(
#                 tibble(state = c(state.name, "District of Columbia"),
#                        source = c(state.abb, "DC"))) %>%
#             group_by(across(-incidD)) %>%
#             summarise(incidD = sum(incidD))
#
#     } else {
#
#         stop(cat(paste0(
#             "STOP! FluView NCHS data is not found.\n",
#             "  (1) Please download from `https://gis.cdc.gov/grasp/fluview/mortality.htmland`, and \n",
#             "  (2) Save as `data_other/nchs_data.csv`, \n",
#             "  (3) In a directory called `data_other` in your project directory.")
#         ))
#     }
#
#     max(fluview_data$Update)
#
#     census_data <- read_csv(file = file.path(config$data_path, config$subpop_setup$geodata))
#     fluview_data <- fluview_data %>%
#         left_join(census_data %>% dplyr::select(source = USPS, FIPS = subpop)) %>%
#         dplyr::select(Update, source, FIPS, incidD)
#
#
#     # Distribute from weekly to daily
#     # -- do this mainly for seeding. it gets re-aggregated for fitting
#     # -- tbis is implemented as a spline fit to cumulative data, from which daily cum and incident are calculated.
#
#     # Limit the data to X weeks before the pulled date
#     if (!exists("config$inference$nchs_weeklag")) { config$inference$nchs_weeklag <- 2}
#     fluview_data <- fluview_data %>% filter(Update < lubridate::floor_date(Sys.Date() - config$inference$nchs_weeklag*7, "weeks"))
#
#     fluview_data <- make_daily_data(data = fluview_data, current_timescale = "week") #%>%
#     # mutate(gt_source = "nchs")
#     # fluview_data <- fluview_data %>%
#     # filter(source %in% config$subpop_setup$modeled_states)
#     # Update >= config$start_date,
#     # Update <= config$end_date_groundtruth)
#     gt_data <- append(gt_data, list(fluview_data))
#
# }














# ~ Pull HHS hospitalization  -------------------

if (any(grepl("hhs", opt$gt_data_source))){

    us_hosp <- get_covidcast_hhs_hosp(geo_level = "state",
                                      limit_date = Sys.Date())

    us_hosp <- us_hosp %>%
        # dplyr::select(-incidH_all) %>%
        # rename(incidH = incidH_confirmed) %>%
        mutate(FIPS = stringr::str_pad(FIPS, width=5, side="right", pad="0")) %>%
        filter(Update >= as_date(config$start_date) & Update <= as_date(end_date_))

    # Apply variants
    if (!is.null(config$seeding$variant_filename)){
        variant_props_file <- config$seeding$variant_filename
        adjust_for_variant <- !is.null(variant_props_file)

        if (adjust_for_variant) {
            tryCatch({
                us_hosp <- flepicommon::do_variant_adjustment(us_hosp, variant_props_file)
            }, error = function(e) {
                stop(paste0("Could not use variant file |", variant_props_file,
                            "|, with error message", e$message))
            })
        }
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
# locs <- config$subpop_setup$modeled_states
locs <- filterUSPS
us_data <- us_data %>%
    filter(source %in% locs) %>%
    filter(!is.na(source)) %>%
    rename(date = Update)




# ~ Fix non-numeric -------------------------------------------------------------
#  -- leave NAs so its not assuming an NA is a 0 and fitting to it

us_data <- us_data %>%
    # mutate(across(starts_with("incid"), ~ replace_na(.x, 0))) %>%
    mutate(across(starts_with("incid"), ~ as.numeric(.x)))


# Save
write_csv(us_data, config$inference$gt_data_path)



cat(paste0("Ground truth data saved\n",
           "  -- file:      ", config$inference$gt_data_path,".\n",
           "  -- outcomes:  ", paste(grep("incid", colnames(us_data), value = TRUE), collapse = ", ")))


# END
