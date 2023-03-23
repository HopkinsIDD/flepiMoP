
#' get_rawcoviddata_state_data_old
#'
#' @param fix_negatives
#'
#' @export
#'
#' @examples
get_rawcoviddata_state_data <- function(fix_negatives = TRUE){

    # install the required package if not already
    is_rawcoviddata_available <- require("rawcoviddata")
    if (!is_rawcoviddata_available){
        devtools::install_github("lmullany/rawcoviddata", force = TRUE)
    }

    # Pull CSSE data using `rawcoviddata` package from Luke Mullany
    cdp <- rawcoviddata::cssedata(return_compact=T)
    state_dat <- rawcoviddata::get_state_from_cdp(cdp=cdp, state = NULL, fix_cumul = fix_negatives, type = c("mid"))

    loc_dictionary <- readr::read_csv("https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv") %>%
        dplyr::rename(fips = location, USPS=abbreviation, Province_State=location_name, Pop2 = population) %>%
        dplyr::filter(stringr::str_length(fips)==2 & fips!="US") %>%
        data.table::as.data.table()

    state_dat <- state_dat[loc_dictionary, on = .(USPS)]

    state_dat <- state_dat %>%
        dplyr::select(Update = Date, FIPS = fips, source = USPS,
                      Confirmed = cumConfirmed, Deaths = cumDeaths,
                      incidI = Confirmed, incidDeath = Deaths)

    state_dat <- state_dat %>%
        dplyr::mutate(Update=lubridate::as_date(Update),
                      FIPS = stringr::str_replace(FIPS, stringr::fixed(".0"), ""), # clean FIPS if numeric
                      FIPS = paste0(FIPS, "000")) %>%
        dplyr::filter(as.Date(Update) <= as.Date(Sys.time())) %>%
        dplyr::distinct()

    validation_date <- Sys.getenv("VALIDATION_DATE")
    if ( validation_date != '' ) {
        print(paste("(DataUtils.R) Limiting CSSE US data to:", validation_date, sep=" "))
        state_dat <- dplyr::filter(state_dat, Update < validation_date)
    }

    # Fix incidence counts that go negative and NA values or missing dates
    if (fix_negatives){
        state_dat <- fix_negative_counts(state_dat, "cumC", "incidC") %>%
            fix_negative_counts("cumD", "incidD")
    }

    return(state_dat)
}










clean_gt_forplots <- function(gt_data){

    gt_data <- as_tibble(gt_data) %>%
        filter(source != "US")

    gt_long <- gt_data %>%
        pivot_longer(cols = -c(date, source, FIPS), names_to = "target", values_to = "incid") %>%
        group_by(source, FIPS, date, target)%>%
        summarise(incid = sum(incid)) %>%
        ungroup() %>%
        filter(grepl("incid", target, ignore.case = TRUE))

    gt_long_tmp <- gt_long %>%
        as_tibble() %>%
        mutate(incid = fix_NAs(incid)) %>%
        group_by(source, FIPS, target) %>%
        arrange(date) %>%
        mutate(incid=cumsum(incid))%>%
        ungroup() %>%
        mutate(target = gsub("incid", "cum", target))

    gt_long <- gt_long %>% full_join(gt_long_tmp)
    rm(gt_long_tmp)

    gt_long_us <- gt_long %>%
        group_by(date, target)%>%
        summarise(incid=sum(incid, na.rm = TRUE)) %>%
        mutate(source="US")
    gt_long <- gt_long %>%
        bind_rows(gt_long_us)
    rm(gt_long_us)

    gt_long <- gt_long %>% mutate(target = gsub("Death", "D", target))

    # pivot back wide now with cum
    gt_data <- gt_long %>%
        pivot_wider(names_from = target, values_from = incid)

    gt_long <- gt_long %>%
        rename(time=date, USPS=source)
    gt_long <- gt_long %>%
        rename(geoid=FIPS, outcome_name = target, outcome = incid)

    gt_data <- gt_data %>%
        rename(geoid=FIPS, time=date, USPS=source)

    return(gt_data)
}
