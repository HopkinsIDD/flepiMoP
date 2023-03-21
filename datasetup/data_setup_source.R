
# Build COVID-19 Death data from NCHS



get_covidcast_deaths <- function (scale = "US state",
                                  source = "jhu-csse", # "hhs", "jhu-csse", "nchs-mortality"
                                  #incl_unass = TRUE,
                                  fix_negatives = TRUE, adjust_for_variant = FALSE,
                                  variant_props_file = "data/variant/variant_props_long.csv",
                                  run_parallel = FALSE,
                                  n_cores = 4){

    if (scale == "US state") {

        signals <- NULL
        data_source <- NULL
        time_type <- NULL

        if ("jhu-csse" %in% source){
            signals <- c(signals, "deaths_incidence_num", "deaths_cumulative_num")
            data_source <- c(data_source, "jhu-csse", "jhu-csse")
            time_type <- c(time_type, "day", "day")
        }
        if ("nchs-mortality" %in% source){
            signals <- c(signals, "deaths_covid_incidence_num")
            data_source <- c(data_source, "nchs-mortality")
            time_type <- c(time_type, "week")
        }

        rc <- pull_covidcast_deaths(geo_level = "state",
                                    signals = signals,
                                    data_source = data_source,
                                    time_type = time_type,
                                    limit_date = Sys.Date(),
                                    fix_negatives = fix_negatives,
                                    run_parallel = FALSE,
                                    n_cores = 4) #%>%             dplyr::select(Update, FIPS, source, !!variables)
    }

    if (adjust_for_variant) {
        tryCatch({
            rc <- do_variant_adjustment(rc, variant_props_file)
        }, error = function(e) {
            stop(paste0("Could not use variant file |", variant_props_file,
                        "|, with error message", e$message))
        })
    }
    return(rc)
}







#' get_covidcast_data
#'
#' @param geo_level
#' @param signals
#' @param limit_date
#' @param fix_negatives
#' @param run_parallel
#' @param n_cores
#'
#' @return
#' @export
#'
#' @examples
pull_covidcast_deaths <- function(
        geo_level = "state",
        data_source = "jhu-csse", # "hhs", "jhu-csse", "nchs-mortality"
        signals = c("deaths_incidence_num", "deaths_cumulative_num", "confirmed_incidence_num", "confirmed_cumulative_num", "confirmed_admissions_covid_1d"),
        time_type = "day",
        limit_date = Sys.Date(),
        fix_negatives = TRUE,
        run_parallel = FALSE,
        n_cores = 4){

    # Create dictionary
    # From the GitHub: https://github.com/reichlab/covid19-forecast-hub
    loc_dictionary <- readr::read_csv("https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv")
    loc_abbr <- loc_dictionary %>% dplyr::filter(!is.na(abbreviation))
    loc_dictionary <- loc_dictionary %>%
        dplyr::mutate(state_fips = substr(location, 1, 2)) %>%
        dplyr::select(-abbreviation) %>%
        dplyr::full_join(loc_abbr %>% dplyr::select(abbreviation, state_fips=location))

    # in folder data-locations
    loc_dictionary_name <- suppressWarnings(
        setNames(c(rep(loc_dictionary$location_name, 2), "US",
                   rep(loc_dictionary$location_name[-1], 2),
                   rep(loc_dictionary$location_name, 2),
                   "New York"),
                 c(loc_dictionary$location,
                   tolower(loc_dictionary$abbreviation), "US",
                   na.omit(as.numeric(loc_dictionary$location)),
                   as.character(na.omit(
                       as.numeric(loc_dictionary$location))),
                   tolower(loc_dictionary$location_name),
                   toupper(loc_dictionary$location_name),
                   "new york state")))

    loc_dictionary_abbr <- setNames(loc_dictionary$abbreviation, loc_dictionary$location)
    loc_dictionary_pop <- setNames(loc_dictionary$population, loc_dictionary$location)

    # Set up start and end dates of data to pull
    # -- we pull the data in 6-month chunks to speed up and not overwhelm API for county-level

    start_dates <- lubridate::as_date("2020-01-01")
    end_dates <- lubridate::as_date(limit_date)

    # Call API to generate gold standard data from COVIDCast
    # Set up parallelization to speed up
    if (run_parallel & length(signals)>1){
        doParallel::registerDoParallel(cores=n_cores)
        `%do_fun%` <- foreach::`%dopar%`
    } else {
        `%do_fun%` <- foreach::`%do%`
    }


    res <- foreach::foreach(x = 1:length(signals),
                            .combine = rbind,
                            .packages = c("covidcast","dplyr","lubridate", "doParallel","foreach","vroom","purrr"),
                            .verbose = TRUE) %do_fun% {

                                df <- covidcast::covidcast_signal(data_source = data_source[x],
                                                                  signal = signals[x],
                                                                  geo_type = geo_level,
                                                                  start_day = lubridate::as_date(start_dates),
                                                                  end_day = lubridate::as_date(end_dates),
                                                                  time_type = time_type[x]) %>%
                                    as_tibble()

                                if (geo_level=="state"){
                                    df <- df %>% mutate(state_abbr = toupper(geo_value)) %>%
                                        dplyr::select(-geo_value) %>%
                                        dplyr::left_join(loc_dictionary %>%
                                                             dplyr::select(state_abbr=abbreviation, geo_value=location) %>%
                                                             dplyr::filter(stringr::str_length(geo_value)==2))
                                }
                                df <- df %>% dplyr::rename(date = time_value)

                                # Get cum hospitalizations
                                if (x == "confirmed_admissions_covid_1d"){
                                    df_cum <- df %>%
                                        dplyr::mutate(value = tidyr::replace_na(value, 0)) %>%
                                        dplyr::arrange(state_abbr, geo_value, date) %>%
                                        dplyr::group_by(data_source, signal, geo_value, state_abbr) %>%
                                        dplyr::mutate(value = cumsum(value)) %>%
                                        dplyr::ungroup() %>%
                                        dplyr::mutate(signal = "confirmed_admissions_cum")
                                    df <- rbind(df, df_cum)
                                }

                                df %>% dplyr::select(signal, Update=date, source=state_abbr, FIPS=geo_value, value)

                            }

    res <- res %>%
        dplyr::mutate(target = recode(signal,
                                      "deaths_incidence_num"="incidD",
                                      "deaths_cumulative_num"="cumD",
                                      "deaths_covid_incidence_num"="incidD",
                                      "confirmed_incidence_num"="incidC",
                                      "confirmed_cumulative_num"="cumC",
                                      "confirmed_admissions_covid_1d"="incidH",
                                      "confirmed_admissions_cum"="cumH")) %>%

        tidyr::pivot_wider(names_from = target, values_from = value) %>%
        dplyr::mutate(Update=lubridate::as_date(Update),
                      FIPS = stringr::str_replace(FIPS, stringr::fixed(".0"), ""), # clean FIPS if numeric
                      FIPS = paste0(FIPS, "000")) %>%
        dplyr::filter(as.Date(Update) <= as.Date(Sys.time())) %>%
        dplyr::distinct()

    res <- res %>% tibble::as_tibble()

    # Fix incidence counts that go negative and NA values or missing dates
    if (fix_negatives & any(c("Confirmed", "incidC", "incidD", "incidI", "Deaths", "incidDeath") %in% colnames(res))){
        res <- flepicommon::fix_negative_counts(res, "Confirmed", "incidI") %>%
            flepicommon::fix_negative_counts("Deaths", "incidDeath")
    }

    return(res)
}








make_daily_data <- function(data = nchs_data,
                            current_timescale = "week"){

    if (current_timescale != "week") stop("Only weeks implemented currently")

    data <- data %>%
        dplyr::select(-starts_with("cum")) %>%
        mutate(Update = lubridate::floor_date(Update, "weeks", week_start = 1)) %>%
        pivot_longer(cols = starts_with("incid"), names_to = "outcome", values_to = "value") %>%
        filter(!is.na(value)) %>%
        group_by(source, FIPS, outcome) %>%
        arrange(Update) %>%
        mutate(value_cum = cumsum(value)) %>%
        ungroup() %>%
        mutate(date_num = as.integer(Update))

    data %>%
        group_by(source, FIPS, outcome) %>%
        group_split() %>%
        map_dfr(~get_spline_daily(grp_dat = .)) %>%
        mutate(value = ifelse(value < 0, 0, value)) %>%
        pivot_wider(names_from = outcome, values_from = value) %>%
        dplyr::select(Update, source, FIPS, starts_with("incid"), starts_with("cum"))
}



get_spline_daily <- function(grp_dat) {

    smth <- stats::splinefun(x = grp_dat$date_num, y = grp_dat$value_cum, method="monoH.FC")
    preds <- grp_dat %>%
        dplyr::select(source, FIPS, outcome) %>%
        distinct() %>%
        expand_grid(Update = seq.Date(min(grp_dat$Update), max(grp_dat$Update), by="1 day")) %>%
        mutate(date_num = as.integer(Update))
    preds <- preds %>% mutate(value = smth(x = date_num))

    preds <- preds %>%
        mutate(outcome = gsub("incid", "cum", outcome)) %>%
        bind_rows(preds %>%
                      dplyr::arrange(Update, source, FIPS, outcome) %>%
                      mutate(value = diff(c(0, value))))
    return(preds)
}


# CHECK
# sum(data %>% filter(outcome=="incidD") %>% pull(value))
# sum(nchs_data %>% filter(!is.na(incidD)) %>% pull(incidD))


# tmp %>% ggplot(aes(x= Update, y = incidD)) + geom_line()
# # smth <- smooth.spline(x = tmp$date_num, y = tmp$cumD)
# #     plot(smth)
# smth <- stats::splinefun(x = tmp$date_num, y = tmp$cumD, method="monoH.FC")
#
# preds <- tibble(date_num = as.integer(seq.Date(min(tmp$Update), max(tmp$Update), by="1 day")))
# preds <- preds %>% mutate(pred = smth(x = date_num))
#
# # preds <- preds %>% mutate(pred = predict(smth, preds$date_num)$y)
#
#
# tmp %>% ggplot(aes(x = Update)) +
#     geom_line(aes(y=pred)) +
#     geom_point(aes(y=cumD), color = "red")
#
# tmp %>% ggplot(aes(x = Update)) +
#     geom_line(aes(y=incidD)) +
#     geom_point(aes(y=incidDweek), color = "red")
#
# sum(tmp$incidD)
# sum(tmp$incidDweek, na.rm = TRUE)
