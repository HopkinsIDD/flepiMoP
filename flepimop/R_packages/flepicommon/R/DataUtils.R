
##' load_geodata_file
##'
##' Convenience function to load the geodata file
##'
##' @param filename filename of geodata file
##' @param subpop_len length of subpop character string
##' @param subpop_pad what to pad the subpop character string with
##' @param state_name whether to add column state with the US state name; defaults to TRUE for forecast or scenario hub runs.
##'
##' @details
##' Currently, the package only supports a geodata object with at least two columns: USPS with the state abbreviation and subpop with the geo IDs of the area. .
##'
##' @return a data frame with columns for state USPS, county subpop and population
##' @examples
##' geodata <- load_geodata_file(filename = system.file("extdata", "geodata_territories_2019_statelevel.csv", package = "flepiconfig"))
##' geodata
##'
##' @export

load_geodata_file <- function(filename,
                              subpop_len = 0,
                              subpop_pad = "0",
                              state_name = TRUE
) {

    if(!file.exists(filename)){stop(paste(filename,"does not exist in",getwd()))}
    geodata <- readr::read_csv(filename) %>%
        dplyr::mutate(subpop = as.character(subpop))

    if (!("subpop" %in% names(geodata))) {
        stop(paste(filename, "does not have a column named subpop"))
    }

    if (subpop_len > 0) {
        geodata$subpop <- stringr::str_pad(geodata$subpop, subpop_len, pad = subpop_pad)
    }

    if(state_name) {
        utils::data(fips_us_county, package = "flepicommon") # arrow::read_parquet("datasetup/usdata/fips_us_county.parquet")
        geodata <- fips_us_county %>%
            dplyr::distinct(state, state_name) %>%
            dplyr::rename(USPS = state) %>%
            dplyr::rename(state = state_name) %>%
            dplyr::mutate(state = dplyr::recode(state, "U.S. Virgin Islands" = "Virgin Islands")) %>%
            dplyr::right_join(geodata)
    }

    return(geodata)
}





#' Read parquet files with check for existence to understand errors
#'
#' @param file The file to read
#'
#' @return
#' @export
#'
#' @examples
read_parquet_with_check <- function(file){
    if(!file.exists(file)){
        stop(paste("File",file,"does not exist"))
    }
    arrow::read_parquet(file)
}












#' Depracated function that returns a function to read files of a specific type (or automatically detected type based on extension)
#' @param extension The file extension to read files of
#' @param ... Arguments to pass to the reading function
#' @return A function which will read files with that extension.
#'  - We use readr::read_csv for csv files
#'  - We use arrow::read_parquet for parquet files
#'  - We use a function that detects the extension and calls this function if the auto extension is specified.
#' @export
#'
read_file_of_type <- function(extension,...){
    if(extension == 'csv'){
        return(function(x){suppressWarnings(readr::read_csv(x,,col_types = cols(
            .default = col_double(),
            date=col_date(),
            uid=col_character(),
            comp=col_character(),
            subpop=col_character()
        )))})
    }
    if(extension == 'parquet'){
        return(function(x){
            tmp <- arrow::read_parquet(x)
            if("POSIXct" %in% class(tmp$date)){
                tmp$date <- lubridate::as_date(tz="GMT",tmp$date)
            }
            tmp
        })
    }
    if(extension == 'auto'){
        return(function(filename){
            extension <- gsub("[^.]*\\.","",filename)
            if(extension == 'auto'){stop("read_file_of_type cannot read files with file extension '.auto'")}
            read_file_of_type(extension)(filename)
        })
    }
    # if(extension == 'shp'){
    #     return(sf::st_read)
    # }
    stop(paste("read_file_of_type cannot read files of type",extension))
}



# DEFUNCT FUINCTION TO PULL DATA FROM USAfacts
#
# ##'
# ##' Download USAFacts data
# ##'
# ##' Downloads the USAFacts case and death count data
# ##'
# ##' @param filename where case data will be stored
# ##' @param url URL to CSV on USAFacts website
# ##' @param value_col_name Confirmed or Deaths
# ##' @param incl_unassigned Includes data unassigned to counties (default is FALSE)
# ##' @return data frame
# ##' @importFrom magrittr %>%
# ##' @importFrom cdlTools fips census2010FIPS stateNames
# ##'
# download_USAFacts_data <- function(filename, url, value_col_name, incl_unassigned = FALSE){
#
#   dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
#   message(paste("Downloading", url, "to", filename))
#   download.file(url, filename, "auto")
#
#   usafacts_data <- readr::read_csv(filename)
#   names(usafacts_data) <- stringr::str_to_lower(names(usafacts_data))
#   usafacts_data <- dplyr::select(usafacts_data, -statefips,-`county name`) %>% # drop statefips columns
#     dplyr::rename(FIPS=countyfips, source=state)
#   if (!incl_unassigned){
#       usafacts_data <- dplyr::filter(usafacts_data, FIPS!=0 & FIPS!=1) # Remove "Statewide Unallocated" cases
#   } else{
#       cw <- data.frame(source = sort(unique(usafacts_data$source)),
#                         FIPS = cdlTools::fips(sort(unique(usafacts_data$source)))
#                         ) %>%
#         dplyr::mutate(FIPS = as.numeric(paste0(FIPS, "000"))) %>%
#         dplyr::distinct(source, FIPS)
#       assigned <- dplyr::filter(usafacts_data, FIPS!=0 & FIPS!=1)
#       unassigned <- dplyr::filter(usafacts_data, FIPS==0 | FIPS==1) %>%
#         dplyr::select(-FIPS) %>%
#         dplyr::left_join(cw, by = c("source"))
#       usafacts_data <- dplyr::bind_rows(assigned, unassigned)
#   }
#   col_names <- names(usafacts_data)
#   date_cols <- col_names[grepl("^\\d+[-/]\\d+[-/]\\d+$", col_names)]
#   date_func <- ifelse(any(grepl("^\\d\\d\\d\\d",col_names)),lubridate::ymd, lubridate::mdy)
#   usafacts_data <- tidyr::pivot_longer(usafacts_data, tidyselect::all_of(date_cols), names_to="Update", values_to=value_col_name)
#   usafacts_data <- dplyr::mutate(usafacts_data, Update=date_func(Update), FIPS=sprintf("%05d", FIPS))
#
#   validation_date <- Sys.getenv("VALIDATION_DATE")
#   if ( validation_date != '' ) {
#     print(paste("(DataUtils.R) Limiting USAFacts data to:", validation_date, sep=" "))
#     usafacts_data <- dplyr::filter(usafacts_data, Update < validation_date )
#   }
#
#   return(usafacts_data)
# }


##'
##' Pulls island area data from NY Times
##'
##' Downloads data from NY Times and filters on island area
##'
##' Returned data preview:
##' tibble [198 × 5] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
##'  $ Update   : Date[1:198], format: "2020-03-13" "2020-03-14" "2020-03-14" ...
##'  $ source   : chr [1:198] "PR" "PR" "VI" "GU" ...
##'  $ FIPS     : chr [1:198] "72000" "72000" "78000" "66000" ...
##'  $ Confirmed: num [1:198] 3 4 1 3 5 1 3 5 2 3 ...
##'  $ Deaths   : num [1:198] 0 0 0 0 0 0 0 0 0 0 ...
##'
##' @return the case data frame
##'
get_islandareas_data <- function() {
  NYTIMES_DATA_URL <- "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
  nyt_data <- readr::read_csv(url(NYTIMES_DATA_URL))

  # Note: As of 2020-05-06, American Samoa is not in this dataset since there are no cases.
  ISLAND_AREAS = c(
    "American Samoa" = "AS",
    "Northern Mariana Islands" = "MP",
    "Guam" = "GU",
    "Puerto Rico" = "PR",
    "Virgin Islands" = "VI")

  nyt_data <- dplyr::filter(nyt_data, state %in% names(ISLAND_AREAS))
  nyt_data <- dplyr::rename(nyt_data, Update=date, source=state, FIPS=fips, Confirmed=cases, Deaths=deaths) # Rename columns
  nyt_data <- dplyr::mutate(nyt_data, FIPS=paste0(FIPS,"000"), source=dplyr::recode(source, ISLAND_AREAS))

  validation_date <- Sys.getenv("VALIDATION_DATE")
  if ( validation_date != '' ) {
    print(paste("(DataUtils.R) Limiting NYT territories data to:", validation_date, sep=" "))
    nyt_data <- dplyr::filter(nyt_data, Update < validation_date)
  }

  return(nyt_data)
}

# There are three different ways of dealing with negative incidence counts, specified by the type argument
# "high": highest estimate; this modifies the cumulative column to be the cummax of the previous cumulative counts
# "low": method: Assume all negative incidents are corrections to previous reports.
#               So, reduce previous reports to make incidents not decrease.
# "mid" method: Combination of both low and high
#               For consecutive negatives, the first negative will be modified so that the
#               cumulative is the cummax so far.
#               For the following negatives, use the low method: reduce previous reports.
# For get_USAFacts_data(), this is always the default ("mid"), but the other code is here for posterity.
#' Title
#'
#' @param .x
#' @param .y
#' @param incid_col_name
#' @param date_col_name
#' @param cum_col_name
#' @param type
#'
#' @return
#' @export
#'
#' @examples
fix_negative_counts_single_subpop <- function(.x,.y, incid_col_name, date_col_name, cum_col_name, type){
  original_names <- names(.x)

  .x <- dplyr::arrange(.x,!!rlang::sym(date_col_name))

  # Calculate new cumulative column
  calc_col_name <- paste(cum_col_name, type, sep="_")

  if (type == "high") {

    .x[[calc_col_name]] <- .x[[cum_col_name]]
    .x[[calc_col_name]][is.na(.x[[calc_col_name]]) ] <- 0 # Fixes NA values
    .x[[calc_col_name]] <- cummax(.x[[calc_col_name]])

  } else if (type == "low") {

    .x[[calc_col_name]] <- .x[[cum_col_name]]
    .x[[calc_col_name]][is.na(.x[[calc_col_name]]) ] <- Inf # Fixes NA values
    .x[[calc_col_name]] <- rev(cummin(rev(.x[[calc_col_name]])))
    .x[[calc_col_name]][!is.finite(.x[[calc_col_name]]) ] <- 0 # In case last thing in row is NA/infinity
    .x[[calc_col_name]] <- cummax(.x[[calc_col_name]]) # Replace 0's with cummax

  } else if (type == "mid") {

    .x[[calc_col_name]] <- .x[[cum_col_name]]

    ## In mid, do a local max followed by a global min (like "low")

    # Get indices where the cumulative count is bigger than the one after it
    # i.e. where incidence is negative next time
    unlagged <- .x[[calc_col_name]]
    unlagged[is.na(.x[[calc_col_name]]) ] <- -Inf # Fixes NA values
    lagged <- dplyr::lag(.x[[calc_col_name]])
    lagged[is.na(.x[[calc_col_name]]) ] <- Inf # Fixes NA values

    max_indices <- which(unlagged < lagged)
    .x[[calc_col_name]][max_indices] <- lagged[max_indices]

    # Global min
    .x[[calc_col_name]][!is.finite(.x[[calc_col_name]]) ] <- Inf
    .x[[calc_col_name]] <- rev(cummin(rev(.x[[calc_col_name]])))
    .x[[calc_col_name]][!is.finite(.x[[calc_col_name]]) ] <- 0
    .x[[calc_col_name]] <- cummax(.x[[calc_col_name]])

  } else {

    stop(paste("Invalid fix_negative_counts type. Must be ['low', 'med', 'high']. Actual: ", type))

  }

  .x[[cum_col_name]] <- .x[[calc_col_name]]

  # Recover incidence from the new cumulative
  .x[[incid_col_name]] <- c(.x[[calc_col_name]][[1]], diff(.x[[calc_col_name]]))

  .x <- .x[,original_names]
  for(col in names(.y)){
    .x[[col]] <- .y[[col]]
  }

  return(.x)
}

# Add missing dates, fix counts that go negative, and fix NA values
#
# See fix_negative_counts_single_subpop() for more details on the algorithm,
# specified by argument "type"
#' Title
#'
#' @param df
#' @param cum_col_name
#' @param incid_col_name
#' @param date_col_name
#' @param min_date
#' @param max_date
#' @param type
#'
#' @return
#' @export
#'
#' @examples
fix_negative_counts <- function(
  df,
  cum_col_name,
  incid_col_name,
  date_col_name = "Update",
  min_date = min(df[[date_col_name]]),
  max_date = max(df[[date_col_name]]),
  type="mid" # "low" or "high"
  ) {

  if(nrow(dplyr::filter(df, !!incid_col_name < 0)) > 0) {
    return(df)
  }

  df <- dplyr::group_by(df, FIPS,source)
    # Add missing dates
  df <- tidyr::complete(df, !!rlang::sym(date_col_name) := min_date + seq_len(max_date - min_date)-1)
  df <- dplyr::group_map(df, fix_negative_counts_single_subpop,
                          incid_col_name=incid_col_name,
                          date_col_name=date_col_name,
                          cum_col_name=cum_col_name,
                          type=type)
  df <- do.call(what=rbind, df)

  return(df)
}


# Add missing dates, fix counts that go negative, and fix NA values for global dataset (group by Country_Region and Province_State instead of by FIPS)
#
# See fix_negative_counts_single_subpop() for more details on the algorithm,
# specified by argument "type"
#' Title
#'
#' @param df
#' @param cum_col_name
#' @param incid_col_name
#' @param date_col_name
#' @param min_date
#' @param max_date
#' @param type
#'
#' @return
#' @export
#'
#' @examples
fix_negative_counts_global <- function(
  df,
  cum_col_name,
  incid_col_name,
  date_col_name = "Update",
  min_date = min(df[[date_col_name]]),
  max_date = max(df[[date_col_name]]),
  type="mid" # "low" or "high"
  ) {

  if(nrow(dplyr::filter(df, !!incid_col_name < 0)) > 0) {
    return(df)
  }

  df <- dplyr::group_by(df, Country_Region, Province_State, source)
    # Add missing dates
  df <- tidyr::complete(df, !!rlang::sym(date_col_name) := min_date + seq_len(max_date - min_date)-1)
  df <- dplyr::group_map(df, fix_negative_counts_single_subpop,
                          incid_col_name=incid_col_name,
                          date_col_name=date_col_name,
                          cum_col_name=cum_col_name,
                          type=type)
  df <- do.call(what=rbind, df)

  return(df)
}


# Aggregate county data to state level
#
#' Title
#'
#' @param df
#' @param state_fips
#'
#' @return
#' @export
#'
#' @examples
aggregate_counties_to_state <- function(df, state_fips){
  aggregated <- dplyr::filter(df, grepl(paste0("^", state_fips), FIPS))
  aggregated <- dplyr::group_by(aggregated, source, Update)
  aggregated <- dplyr::summarise(
    aggregated,
    Confirmed = sum(Confirmed),
    Deaths = sum(Deaths),
    incidI = sum(incidI),
    incidDeath = sum(incidDeath)
  )
  aggregated <- dplyr::mutate(aggregated, FIPS = paste0(state_fips, "000"))
  aggregated <- dplyr::ungroup(aggregated)
  nonaggregated <- dplyr::filter(df, !grepl(paste0("^", state_fips), FIPS))

  rc <- dplyr::bind_rows(nonaggregated, aggregated) %>%
    dplyr::arrange(source, FIPS, Update)

  return(rc)
}




##'
##' Download CSSE US data
##'
##' Downloads the CSSE US case and death count data
##'
##' @param filename where case data will be stored
##' @param url URL to CSV on CSSE website
##' @param value_col_name Confirmed or Deaths
##' @param incl_unassigned Includes data unassigned to county-level (default is FALSE)
##' @return data frame
##' @importFrom magrittr %>%
##'
##'
download_CSSE_US_data <- function(filename, url, value_col_name, incl_unassigned = FALSE){

  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)

  message(paste0("Downloading CSSE data from ", url, "."))

  csse_data <- readr::read_csv(url, col_types = list("FIPS" = readr::col_character())) %>%
        tibble::as_tibble()

  if (incl_unassigned){
    csse_data <- dplyr::filter(csse_data, !grepl("out of", Admin2, ignore.case = TRUE) & ## out of state records
                  !grepl("princess", Province_State, ignore.case = TRUE) & ## cruise ship cases
                  !is.na(FIPS))
  } else{
    csse_data2 <- dplyr::filter(csse_data, !grepl("out of", Admin2, ignore.case = TRUE),  ## out of state records
                    !grepl("unassigned", Admin2, ignore.case = TRUE),  ## probable cases
                    !grepl("princess", Province_State, ignore.case = TRUE),  ## cruise ship cases
                    !is.na(FIPS))
    ## include unassigned PR cases & deaths because they are being aggregated to territory level in get_CSSE_US_data
    pr_unassigned <- dplyr::filter(csse_data, grepl("unassigned", Admin2, ignore.case = TRUE),
                        Province_State == "Puerto Rico")
    csse_data <- dplyr::bind_rows(csse_data2, pr_unassigned)
  }

  csse_data <- tidyr::pivot_longer(csse_data, cols=dplyr::contains("/"), names_to="Update", values_to=value_col_name) %>%
    dplyr::mutate(Update=as.Date(lubridate::mdy(Update)),
                  FIPS = stringr::str_replace(FIPS, stringr::fixed(".0"), ""), # clean FIPS if numeric
                  FIPS = ifelse(stringr::str_length(FIPS)<=2, paste0(FIPS, "000"), stringr::str_pad(FIPS, 5, pad = "0", side = "left")),
                  FIPS = ifelse(stringr::str_sub(FIPS, 1, 3)=="900", paste0(stringr::str_sub(FIPS, 4, 5), "000"), FIPS) ## clean FIPS codes for unassigned data
                  ) %>%
    dplyr::filter(as.Date(Update) <= as.Date(Sys.time())) %>%
    dplyr::distinct()
  csse_data <- suppressWarnings(dplyr::mutate(csse_data, state_abb = state.abb[match(Province_State, state.name)]))
  csse_data <- suppressWarnings(dplyr::mutate(csse_data, source = ifelse(Province_State=="District of Columbia", "DC",
                  ifelse(is.na(state_abb) & Country_Region=="US", iso2, state_abb))))

  csse_data <- dplyr::select(csse_data, FIPS, source, Update, !!value_col_name)

  validation_date <- Sys.getenv("VALIDATION_DATE")
  if ( validation_date != '' ) {
    print(paste("(DataUtils.R) Limiting CSSE US data to:", validation_date, sep=" "))
    csse_data <- dplyr::filter(csse_data, Update < validation_date)
  }

  return(csse_data)
}



##'
##' Pull US case and death count data from JHU CSSE
##'
##' Pulls the CSSE cumulative case count and death data. Calculates incident counts.
##'
##' Returned data preview:
##' tibble [352,466 × 7] (S3: grouped_df/tbl_df/tbl/data.frame)
##'  $ FIPS       : chr [1:352466] "00001" "00001" "00001" "00001" ...
##'  $ source     : chr [1:352466] "NY" "NY" "NY" "NY" ...
##'  $ Update     : Date[1:352466], format: "2020-01-22" "2020-01-23" ...
##'  $ Confirmed  : num [1:352466] 0 0 0 0 0 0 0 0 0 0 ...
##'  $ Deaths     : num [1:352466] 0 0 0 0 0 0 0 0 0 0 ...
##'  $ incidI     : num [1:352466] 0 0 0 0 0 0 0 0 0 0 ...
##'  $ incidDeath : num [1:352466] 0 0 0 0 0 0 0 0 0 0 ...
##'
##' @param case_data_filename where case data will be stored
##' @param death_data_filename where death data will be stored
##' @param incl_unassigned Includes data unassigned to county-level (default is FALSE)
##' @return the case and deaths data frame
##'
##'
##' @export
##'
get_CSSE_US_data <- function(case_data_filename = "data/case_data/jhucsse_us_case_data_crude.csv",
                              death_data_filename = "data/case_data/jhucsse_us_death_data_crude.csv",
                             incl_unassigned = TRUE,
                             fix_negatives = TRUE){

  CSSE_US_CASE_DATA_URL <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
  CSSE_US_DEATH_DATA_URL <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
  csse_us_case <- download_CSSE_US_data(case_data_filename, CSSE_US_CASE_DATA_URL, "Confirmed", incl_unassigned)
  csse_us_death <- download_CSSE_US_data(death_data_filename, CSSE_US_DEATH_DATA_URL, "Deaths", incl_unassigned)

  csse_us_data <- dplyr::full_join(csse_us_case, csse_us_death) %>%
    dplyr::select(Update, source, FIPS, Confirmed, Deaths) %>%
    dplyr::arrange(source, FIPS, Update)

  # Create columns incidI and incidDeath
  csse_us_data <- dplyr::group_modify(
    dplyr::group_by(
      csse_us_data,
      FIPS
    ),
    function(.x,.y){
      .x$incidI = c(.x$Confirmed[1],diff(.x$Confirmed))
      .x$incidDeath = c(.x$Deaths[1],diff(.x$Deaths,))
      return(.x)
    }
  )

  # Aggregate county-level data for Puerto Rico
  csse_us_data <- tibble::as_tibble(aggregate_counties_to_state(csse_us_data, "72"))

  if(fix_negatives){
    # Fix incidence counts that go negative and NA values or missing dates
    csse_us_data <- fix_negative_counts(csse_us_data, "Confirmed", "incidI") %>%
      fix_negative_counts("Deaths", "incidDeath")
  }

  return(csse_us_data)
}



##'
##' Download CSSE global data
##'
##' Downloads the CSSE global case and death count data
##'
##' @param filename where case data will be stored
##' @param value_col_name
##' @param url URL to CSV on CSSE website
##'
##' @return data frame
##'
##' @importFrom magrittr %>%
##' @export
##'
download_CSSE_global_data <- function(filename, url, value_col_name){

  dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)
  message(paste("Downloading", url, "to", filename))
  download.file(url, filename, "auto")

  csse_data <- readr::read_csv(filename)
  csse_data <- tibble::as_tibble(csse_data)
  csse_data <- dplyr::rename(csse_data,
                  Province_State = `Province/State`,
                  Country_Region = `Country/Region`,
                  Latitude = Lat,
                  Longitude = Long)

  csse_data <- dplyr::mutate(csse_data, iso3 = suppressWarnings(suppressMessages(globaltoolboxlite::get_iso(Country_Region))))
  csse_data <- dplyr::mutate(csse_data,
      iso2 = suppressWarnings(suppressMessages(globaltoolboxlite::get_iso2_from_ISO3(iso3))),
      UID = ifelse(!is.na(Province_State), paste0(iso3, "-", Province_State), iso3))
  csse_data <- dplyr::select(csse_data, UID, Province_State:Longitude, iso2, iso3, tidyselect::everything())

  csse_data <- tidyr::pivot_longer(csse_data, cols=contains("/"), names_to="Update", values_to=value_col_name)
  csse_data <- dplyr::mutate(csse_data, Update=as.Date(lubridate::mdy(Update)))
  csse_data <- dplyr::filter(csse_data, as.Date(Update) <= as.Date(Sys.time()))
  csse_data <- dplyr::distinct(csse_data)

  # Define a single source location variable
  # - USA: States used for source
  # - China: Provinces used for source
  # - Others: Country used for source
  csse_data <- dplyr::mutate(csse_data, source = ifelse(iso3=="CHN" & !is.na(Province_State), Province_State, iso3))

  csse_data <- dplyr::select(csse_data, UID, iso2, iso3, Province_State, Country_Region, source, Latitude, Longitude, Update, !!value_col_name)

  validation_date <- Sys.getenv("VALIDATION_DATE")
  if ( validation_date != '' ) {
    print(paste("(DataUtils.R) Limiting CSSE global data to:", validation_date, sep=" "))
    csse_data <- dplyr::filter(csse_data, Update < validation_date)
  }

  return(csse_data)
}



##'
##' Pull global case and death count data from JHU CSSE
##'
##' Pulls the CSSE cumulative case count and death data. Calculates incident counts.
##'
##' Data preview:
##'  $ Update     : Date "2019-12-01" "2019-12-02" ...
##'  $ UID        : chr "AFG" "AFG" "AFG" "AFG" ...
##'  $ iso2       : chr  "AF" "AF" "AF" "AF" ...
##'  $ iso3       : chr  "AFG" "AFG" "AFG" "AFG" ...
##'  $ Latitude   : num  "AF" "AF" "AF" "AF" ...
##'  $ Longitude  : num  "AFG" "AFG" "AFG" "AFG" ...
##'  $ Confirmed  : num  0 0 0 0 0 0 0 0 0 0 ...
##'  $ Deaths     : num  0 0 0 0 0 0 0 0 0 0 ...
##'  $ incidI     : num  0 0 0 0 0 0 0 0 0 0 ...
##'  $ incidDeath : num  0 0 0 0 0 0 0 0 0 0 ...
##'  $ Country_Region  : chr  "Afghanistan" "Afghanistan" "Afghanistan" "Afghanistan" ...
##'  $ Province_State  : chr NA NA NA NA ...
##'  $ source     : chr "AFG" "AFG" "AFG" "AFG" ...
##'
##' @param case_data_filename
##' @param death_data_filename
##' @param append_wiki
##'
##' @return the case and deaths data frame
##'
##'
##' @export
##'
get_CSSE_global_data <- function(case_data_filename = "data/case_data/jhucsse_case_data_crude.csv",
                              death_data_filename = "data/case_data/jhucsse_death_data_crude.csv",
                              append_wiki = TRUE){

  CSSE_GLOBAL_CASE_DATA_URL <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
  CSSE_GLOBAL_DEATH_DATA_URL <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
  csse_global_case <- download_CSSE_global_data(case_data_filename, CSSE_GLOBAL_CASE_DATA_URL, "Confirmed")
  csse_global_death <- download_CSSE_global_data(death_data_filename, CSSE_GLOBAL_DEATH_DATA_URL, "Deaths")

  csse_global_data <- dplyr::full_join(csse_global_case, csse_global_death)
  csse_global_data <- dplyr::select(csse_global_data, UID, iso2, iso3, Province_State, Country_Region, Latitude, Longitude, Update, source, Confirmed, Deaths)
  csse_global_data <- dplyr::arrange(csse_global_data, Country_Region, Province_State, Update)

  if (append_wiki){
    data("wikipedia_cases", package = "flepicommon")
    wikipedia_cases <- dplyr::mutate(wikipedia_cases, Country_Region = ifelse(Country_Region=="Mainland China", "China", Country_Region), Update = as.Date(Update), iso2 = "CN", iso3 = "CHN", Latitude = 30.9756, Longitude = 112.2707)
    wikipedia_cases <- dplyr::mutate(wikipedia_cases, UID = ifelse(!is.na(Province_State), paste0(iso3, "-", Province_State), iso3))
    wikipedia_cases <- dplyr::select(wikipedia_cases, -Suspected)

    csse_global_data <- dplyr::bind_rows(csse_global_data, wikipedia_cases)
    csse_global_data <- dplyr::arrange(csse_global_data, Country_Region, Province_State, Update)
  }

  # Create columns incidI and incidDeath
  csse_global_data <- dplyr::group_modify(
    dplyr::group_by(
      csse_global_data,
      Country_Region,
      Province_State
    ),
    function(.x,.y){
      .x$incidI = c(.x$Confirmed[1],diff(.x$Confirmed))
      .x$incidDeath = c(.x$Deaths[1],diff(.x$Deaths,))
      return(.x)
    }
  )

  # Fix incidence counts that go negative and NA values or missing dates
  csse_global_data <- fix_negative_counts_global(csse_global_data, "Confirmed", "incidI")
  csse_global_data <- fix_negative_counts_global(csse_global_data, "Deaths", "incidDeath")

  return(csse_global_data)
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
#' @import covidcast dplyr lubridate doParallel foreach vroom purrr
#' @return
#' @export
#'
#' @examples
get_covidcast_data <- function(
    geo_level = "state",
    signals = c("deaths_incidence_num", "deaths_cumulative_num", "confirmed_incidence_num", "confirmed_cumulative_num", "confirmed_admissions_covid_1d"),
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

    if (geo_level=="county"){
        years_ <- lubridate::year("2020-01-01"):lubridate::year(limit_date)
        start_dates <- sort(c(lubridate::as_date(paste0(years_, "-01-01")),
                              lubridate::as_date(paste0(years_, "-07-01"))))
        start_dates <- start_dates[start_dates<=limit_date]

        end_dates <- sort(c(lubridate::as_date(paste0(years_, "-06-30")),
                            lubridate::as_date(paste0(years_, "-12-31")), limit_date))
        end_dates <- end_dates[end_dates<=limit_date]
    } else {
        start_dates <- lubridate::as_date("2020-01-01")
        end_dates <- lubridate::as_date(limit_date)
    }

    # Set up parallelization to speed up
    if (run_parallel){
        doParallel::registerDoParallel(cores=n_cores)
        `%do_fun%` <- foreach::`%dopar%`
    } else {
        `%do_fun%` <- foreach::`%do%`
    }

    res <- foreach::foreach(x = signals,
                            .combine = rbind,
                            .packages = c("covidcast","dplyr","lubridate", "doParallel","foreach","vroom","purrr"),
                            .verbose = TRUE) %do_fun% {

                                start_dates_ <- start_dates
                                if (x == "confirmed_admissions_covid_1d"){
                                    start_dates_[1] <- lubridate::as_date("2020-02-01")
                                }

                                # Call API to generate gold standard data from COVIDCast
                                df <- lapply(1:length(start_dates),
                                             FUN = function(y=x){
                                                 covidcast::covidcast_signal(data_source = ifelse(grepl("admissions", x), "hhs", "jhu-csse"),
                                                                             signal = x,
                                                                             geo_type = geo_level,
                                                                             start_day = lubridate::as_date(start_dates_[y]),
                                                                             end_day = lubridate::as_date(end_dates[y]))})
                                df <- data.table::rbindlist(df)

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
        dplyr::mutate(signal = recode(signal,
                                      "deaths_incidence_num"="incidD",
                                      "deaths_cumulative_num"="cumD",
                                      "confirmed_incidence_num"="incidC",
                                      "confirmed_cumulative_num"="cumC",
                                      "confirmed_admissions_covid_1d"="incidH",
                                      "confirmed_admissions_cum"="cumH")) %>%

        tidyr::pivot_wider(names_from = signal, values_from = value) %>%
        dplyr::mutate(Update=lubridate::as_date(Update),
                      FIPS = stringr::str_replace(FIPS, stringr::fixed(".0"), ""), # clean FIPS if numeric
                      FIPS = paste0(FIPS, "000")) %>%
        dplyr::filter(as.Date(Update) <= as.Date(Sys.time())) %>%
        dplyr::distinct()

    validation_date <- Sys.getenv("VALIDATION_DATE")
    if ( validation_date != '' ) {
        print(paste("(DataUtils.R) Limiting CSSE US data to:", validation_date, sep=" "))
        res <- dplyr::filter(res, Update < validation_date)
    }

    res <- res %>% tibble::as_tibble()

    # Fix incidence counts that go negative and NA values or missing dates
    if (fix_negatives & any(c("incidC", "incidH", "incidH") %in% colnames(res))){
        res <- fix_negative_counts(res, "cumC", "incidC") %>%
            fix_negative_counts("cumD", "incidD")
    }

    return(res)
}



##' Wrapper function to pull data from different sources
##'
##' Pulls a groundtruth dataset with the variables specified
##'
##' @param source name of data source: reichlab, usafacts, csse
##' @param scale geographic scale: US county, US state, country (csse only), complete (csse only)
##' @param source_file
##' @param incl_unass
##' @param fix_negatives
##' @param adjust_for_variant
##' @param variant_props_file
##' @param variables vector that may include one or more of the following variable names: Confirmed, Deaths, incidI, incidDeath
##'
##' @return data frame
##'
##' @importFrom magrittr %>%
##'
##' @export
##'
get_groundtruth_from_source <- function(
  source = "csse",
  scale = "US state",
  source_file = NULL,
  variables = c("Confirmed", "Deaths", "incidI", "incidDeath"),
  incl_unass = TRUE,
  fix_negatives = TRUE,
  adjust_for_variant = FALSE,
  variant_props_file = "data/variant/variant_props_long.csv"
) {

 if (source == "usafacts" & scale == "US county"){

    rc <- get_USAFacts_data(tempfile(), tempfile(), incl_unassigned = incl_unass) %>%
      dplyr::select(Update, FIPS, source, !!variables) %>%
      dplyr::ungroup()

  } else if(source == "usafacts" & scale == "US state"){

    rc <- get_USAFacts_data(tempfile(), tempfile(), incl_unassigned = incl_unass) %>%
      dplyr::select(Update, FIPS, source, !!variables) %>%
      dplyr::mutate(FIPS = paste0(stringr::str_sub(FIPS, 1, 2), "000")) %>%
      dplyr::group_by(Update, FIPS, source) %>%
      dplyr::summarise_if(is.numeric, sum) %>%
      dplyr::ungroup()

  } else if(source == "csse" & scale == "US county"){

    rc <- get_CSSE_US_data(tempfile(), tempfile(), incl_unassigned = incl_unass) %>%
      dplyr::select(Update, FIPS, source, !!variables)

  } else if(source == "csse" & scale == "US state"){

    rc <- get_CSSE_US_data(tempfile(), tempfile(), incl_unassigned = incl_unass) %>%
      dplyr::select(Update, FIPS, source, !!variables) %>%
      dplyr::mutate(FIPS = paste0(stringr::str_sub(FIPS, 1, 2), "000")) %>%
      dplyr::group_by(Update, FIPS, source) %>%
      dplyr::summarise_if(is.numeric, sum) %>%
      dplyr::ungroup()

  } else if(source == "csse" & scale == "country"){

    rc <- get_CSSE_global_data()
    rc <- dplyr::select(rc, UID, iso2, iso3, Province_State, Country_Region, Latitude, Longitude, Update, source, !!variables)

  } else if(source == "csse" & scale == "complete"){

    us <- get_CSSE_US_matchGlobal_data()
    us <- dplyr::select(us, Update, UID, iso2, iso3, Latitude, Longitude, source, !!variables, Country_Region, Province_State, source)
    rc <- get_CSSE_global_data()
    rc <- dplyr::select(rc, Update, UID, iso2, iso3, Latitude, Longitude, source, !!variables, Country_Region, Province_State, source)
    rc <- dplyr::bind_rows(rc, us)
    warning(print(paste("The combination of ", source, "and", scale, "is not fully working. County-level US data may be missing from this data frame.")))
    warning("Complete scale is deprecated")

    } else if(source == "covidcast" & scale == "US state"){

        # define covidcast signals

        signals <- NULL
        if (any(c("incidDeath", "Deaths", "incidD", "cumD") %in% variables)) signals <- c(signals, "deaths_incidence_num", "deaths_cumulative_num")
        if (any(c("incidI", "Confirmed", "incidC", "cumC") %in% variables)) signals <- c(signals, "confirmed_incidence_num", "confirmed_cumulative_num")
        if (any(grepl("hosp|incidH", variables))) signals <- c(signals, "confirmed_admissions_covid_1d")

        rc <- get_covidcast_data(geo_level = "state",
                                 signals = signals,
                                 limit_date = Sys.Date(),
                                 fix_negatives = fix_negatives,
                                 run_parallel = FALSE) %>%
            dplyr::select(Update, FIPS, source, !!variables)

    } else if(source == "file"){

        rc <- get_gt_file_data(source_file)

        # check that it has correct columns
        check_cols <- all(c("Update", "FIPS", "source", variables) %in% colnames(rc))
        stopifnot("Columns missing in source file. Must have [Update, FIPS, source] and desired variables." = all(check_cols))

        rc <- rc %>%
            dplyr::select(rc, Update, FIPS, source, !!variables)

  } else{
    warning(print(paste("The combination of ", source, "and", scale, "is not valid. Returning NULL object.")))
    rc <- NULL
  }


  if (adjust_for_variant) {
    tryCatch({
      rc <- do_variant_adjustment(rc, variant_props_file)
    }, error = function(e) {
      stop(paste0("Could not use variant file |", variant_props_file, "|, with error message", e$message))
    })
  }

  return(rc)

}

##' Wrapper function to apply variant adjustment
##'
##' aportions variant to reported outcomes
#'
#' @param rc
#' @param variant_props_file
#'
#' @return
#'
#' @importFrom magrittr %>%
#' @export
#'
#' @examples

do_variant_adjustment <- function(
  rc,
  variant_props_file = "data/variant/variant_props_long.csv"
) {
  non_outcome_column_names <- c("FIPS", "Update", "date", "source", "location")
  outcome_column_names <- names(rc)[!(names(rc) %in% non_outcome_column_names)]
  variant_data <- readr::read_csv(variant_props_file)
  rc <- rc %>%
    tidyr::pivot_longer(!!outcome_column_names, names_to = "outcome") %>%
    dplyr::left_join(variant_data) %>%
    dplyr::mutate(value = value * prop) %>%
    dplyr::select(-prop) %>%
    dplyr::bind_rows(
      dplyr::mutate(
        tidyr::pivot_longer(rc, !!outcome_column_names, names_to = "outcome"),
        variant = ""
      )
    )%>%
    tidyr::pivot_wider(names_from = c("outcome", "variant"), values_from = "value")
  names(rc) <- gsub("_$", "", names(rc))
  return(rc)
}


##'
##' Pull CSSE US data in format similar to that of global data
##'
##' Pulls the CSSE US confirmed cases and deaths and calculates incident cases and deaths, putting them into a verbose geographic format like that with the global dataframe
##' @return data frame
##'
##'
##'
##' @export
##'
get_CSSE_US_matchGlobal_data <- function(){

  warning("This function is still in progress. Returning NA-filled dataframe.")
  rc <- data.frame(Update = NA, UID = NA, iso2 = NA, iso3 = NA, Latitude = NA, Longitude = NA, source = NA, Confirmed = NA, Deaths = NA, incidI = NA, incidDeath = NA, Country_Region = NA, Province_State = NA, source = NA)
  return(rc)

}

