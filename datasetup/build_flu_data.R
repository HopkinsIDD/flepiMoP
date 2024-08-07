

# SETUP -------------------------------------------------------------------

library(dplyr)
library(tidyr)
# library(tidycensus)
library(readr)
library(lubridate)

# #check for cdc fluview package
# httr_installed <- require(httr)
# if (!httr_installed){
#     install.packages("httr")
# }
# cdcflueview_installed <- require(cdcfluview)
# if (!cdcflueview_installed){
#     remotes::install_github("hrbrmstr/cdcfluview")
# }


option_list = list(
    optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
    optparse::make_option(c("-p", "--path"), action="store", default=Sys.getenv("FLEPI_PATH", "flepiMoP"), type='character', help="path to the flepiMoP directory"),
    optparse::make_option(c("-w", "--wide_form"), action="store",default=FALSE,type='logical',help="Whether to generate the old wide format mobility or the new long format")
)
opt = optparse::parse_args(optparse::OptionParser(option_list=option_list))

config <- flepicommon::load_config(opt$c)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}

filterUSPS <- config$subpop_setup$modeled_states

# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$subpop_setup$state_level) && config$subpop_setup$state_level, TRUE, FALSE)






# Pull Resources and Source from FluSight Github -------------------

# create needed directories
dir.create("data-locations")
dir.create("data-truth")

# get locations file
download.file("https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data/master/data-locations/locations.csv",
              "data-locations/locations.csv")

# source function and pull weekly hospitalizations from HHS
source("https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data/master/data-truth/get_truth.R")

# Pull daily hospitalizations for model run
us_data <- load_flu_hosp_data(temporal_resolution = 'daily', na.rm = TRUE)
locs <- read_csv(file.path(config$subpop_setup$geodata))

# fix string pad issue on left side
us_data <- us_data %>%
    mutate(location = stringr::str_pad(location, width = 2, side = "left", pad = "0"))

us_data <- us_data %>%
    filter(location != "US") %>%
    mutate(location = stringr::str_pad(location, width=5, side="right", pad="0")) %>%
    left_join(locs, by = c("location"="subpop")) %>%
    rename(FIPS = location,
           incidH = value,
           source = USPS) %>%
    select(-location_name, -population)

# Filter to dates we care about for speed and space
end_date_ <- config$end_date_groundtruth
if (is.null(end_date_)){
    end_date_ <- config$end_date
}
us_data <- us_data %>%
    filter(date >= lubridate::as_date(config$start_date) & date <= lubridate::as_date(end_date_)) %>%
    filter(!is.na(source))

if(!dir.exists(dirname(config$inference$gt_data_path))){
  dir.create(dirname(config$inference$gt_data_path))}
write_csv(us_data, config$inference$gt_data_path)








# PULL VARIANT DATA -------------------------------------------------------

variant_props_file <- config$seeding$variant_filename
adjust_for_variant <- !is.null(variant_props_file)

# if (adjust_for_variant){
#
#     # Variant Data (need to automate this data pull still)
#     #variant_data <- read_csv(file.path("variant/WHO_NREVSS_Clinical_Labs.csv"), skip = 1)
#     variant_data <- cdcfluview::who_nrevss(region="state", years = 2022)$clinical_labs
#
#     # location data
#     loc_data <- read_csv("data-locations/locations.csv")
#
#
#     # CLEAN DATA
#
#     variant_data <- variant_data %>%
#         select(state = region,
#                week = week,
#                year = year,
#                FluA = total_a,
#                FluB = total_b) %>%
#         # select(state = REGION,
#         #        week = WEEK,
#         #        year = YEAR,
#         #        FluA = `TOTAL A`,
#         #        FluB = `TOTAL B`) %>%
#         pivot_longer(cols = starts_with("Flu"),
#                      names_to = "variant",
#                      values_to = "n") %>%
#         # mutate(n = ifelse(n == "X", 0, n)) %>%
#         mutate(n = ifelse(is.na(n), 0, n)) %>%
#         mutate(n = as.integer(n)) %>%
#         mutate(week_end = as_date(MMWRweek::MMWRweek2Date(year, week, 7))) %>%
#         full_join(expand_grid(variant = c("FluA", "FluB"),
#                               state = unique(loc_data %>% filter(location!="US") %>% pull(location_name)),
#                               week_end = as_date(seq(lubridate::as_date(config$start_date)+6, lubridate::as_date(config$end_date_groundtruth), 7)))) %>%
#         arrange(state, week_end) %>%
#         select(-week, -year) %>%
#         group_by(state, week_end) %>%
#         mutate(prop = n / sum(n, na.rm=TRUE)) %>%
#         mutate(prop_tot = sum(prop, na.rm=TRUE)) %>%
#         ungroup() %>%
#         mutate(prop = ifelse(prop_tot==0 & variant=="FluA", 1, prop)) %>%
#         group_by(state, week_end) %>%
#         mutate(prop_tot = sum(prop, na.rm=TRUE)) %>%
#         mutate(prop = prop / sum(prop, na.rm = TRUE)) %>%
#         ungroup() %>%
#         select(-prop_tot, -n) %>%
#         mutate(prop = ifelse(is.na(prop), 0, prop)) %>%
#         filter(!is.na(week_end)) %>%
#         filter(week_end <= as_date(end_date_))
#
#     variant_data <- variant_data %>%
#         left_join(loc_data %>% select(state = location_name, source = abbreviation)) %>%
#         mutate(week = epiweek(week_end), year = epiyear(week_end))
#
#     if(end_date_ != max(variant_data$week_end)){
#         # Extend to dates of groundtruth
#         var_max_dates <- variant_data %>%
#             group_by(source, state) %>%
#             filter(week_end == max(week_end)) %>%
#             ungroup() %>%
#             mutate(max_date = as_date(end_date_)) %>%
#             mutate(weeks_missing = as.integer(max_date - week_end)/7) %>%
#             rowwise() %>%
#             mutate(weeks_missing = paste((seq(from = 1, to=weeks_missing, 1)*7 + week_end), collapse = ",")) %>%
#             # mutate(weeks_missing = list(as_date(seq(from = 1, to=weeks_missing, 1)*7 + week_end))) #%>%
#             ungroup()
#         var_max_dates <- var_max_dates %>%
#             rename(max_current = week_end) %>%
#             mutate(week_end = strsplit(as.character(weeks_missing), ",")) %>%
#             unnest(week_end) %>%
#             select(state, week, year, variant, prop, week_end, source) %>%
#             mutate(week_end = as_date(week_end))
#         variant_data <- variant_data %>%
#             bind_rows(var_max_dates)
#     }
#
#     variant_data <- variant_data %>%
#         mutate(week = epiweek(week_end), year = epiyear(week_end))
#
#     variant_data <- variant_data %>%
#         expand_grid(day = 1:7) %>%
#         mutate(date = as_date(MMWRweek::MMWRweek2Date(year, week, day))) %>%
#         select(c(variant, prop, source, date))
#
#     variant_data <- variant_data %>%
#         filter(date >= as_date(config$start_date) & date <= as_date(config$end_date_groundtruth))
#
#     write_csv(variant_data, variant_props_file)
# }
#

# APPLY VARIANTS ----------------------------------------------------------


if (adjust_for_variant) {

    us_data <- read_csv(config$inference$gt_data_path)

    tryCatch({
        us_data <- flepicommon::do_variant_adjustment(us_data, variant_props_file)
        us_data <- us_data %>%
            filter(date >= as_date(config$start_date) & date <= as_date(config$end_date_groundtruth))
        write_csv(us_data, config$inference$gt_data_path)
    }, error = function(e) {
        stop(paste0("Could not use variant file |", variant_props_file,
                    "|, with error message", e$message))
    })
}



cat(paste0("Ground truth data saved\n",
           "  -- file:      ", config$inference$gt_data_path,".\n",
           "  -- outcomes:  ", paste(grep("incid", colnames(us_data), value = TRUE), collapse = ", ")))


# END
