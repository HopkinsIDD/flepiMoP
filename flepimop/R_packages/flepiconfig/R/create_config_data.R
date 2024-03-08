# Process ------

#' Generate incidH intervention
#'
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param incl_subpop
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#'
#' @return data frame with columns for
#' @export
#'
#' @examples
#' dat <- set_localvar_params()
#'
#' dat
#'
set_incidH_params <- function(start_date=Sys.Date()-42,
                              sim_end_date=Sys.Date()+60,
                              incl_subpop = NULL,
                              inference = TRUE,
                              v_dist="truncnorm",
                              v_mean =  0, v_sd = 0.1, v_a = -1, v_b = 1, # TODO: add check on limits
                              p_dist="truncnorm",
                              p_mean = 0, p_sd = 0.05, p_a = -1, p_b = 1
){
    start_date <- as.Date(start_date)
    sim_end_date <- as.Date(sim_end_date)

    method = "SinglePeriodModifier"
    param_val <- "incidH::probability"

    if(is.null(incl_subpop)){
        affected_subpop = "all"
    } else{
        affected_subpop = paste0(incl_subpop, collapse='", "')
    }


    local_var <- dplyr::tibble(USPS = "",
                               subpop = affected_subpop,
                               name = "incidH_adj",
                               type = "outcome",
                               category = "incidH_adjustment",
                               parameter = param_val,
                               baseline_modifier = "",
                               start_date = start_date,
                               end_date = sim_end_date,
                               method = method,
                               param = param_val,
                               value_dist = v_dist,
                               value_mean = v_mean,
                               value_sd = v_sd,
                               value_a = v_a,
                               value_b= v_b,
                               pert_dist = p_dist,
                               pert_mean = p_mean,
                               pert_sd = p_sd,
                               pert_a = p_a,
                               pert_b = p_b) %>%
        dplyr::mutate(pert_dist = ifelse(inference, as.character(pert_dist), NA_character_),
                      dplyr::across(pert_mean:pert_b, ~ifelse(inference, as.numeric(.x), NA_real_))) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    return(local_var)
}

#' Specify parameters for NPIs
#'
#' @param intervention_file df with the location's state and ID and the intervention start and end dates, name, and method - from process_npi_shub
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param npi_cutoff_date only interventions that start before or on npi_cuttof_date are included
#' @param redux_subpop string or vector of characters indicating which subpop will have an intervention with the ModifierModifier method; it accepts "all". If any values are specified, the intervention in the subpop with the maximum start date will be selected. It defaults to NULL. .
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#'
#' @return
#'
#' @export
#'
#' @examples
#' geodata <- load_geodata_file(filename = "data/geodata_territories_2019_statelevel.csv")
#' npi_dat <- process_npi_shub(intervention_path = "data/intervention_tracking/Shelter-in-place-as-of-04302021.csv", geodata)
#'
#' npi_dat <- set_npi_params(intervention_file = npi_dat, sim_start_date = "2020-01-15", sim_end_date = "2021-07-30")
#'
set_npi_params_old <- function(intervention_file,
                               sim_start_date=as.Date("2020-01-31"),
                               sim_end_date=Sys.Date()+60,
                               npi_cutoff_date=Sys.Date()-7,
                               inference = TRUE,
                               redux_subpop = NULL,
                               v_dist = "truncnorm", v_mean=0.6, v_sd=0.05, v_a=0.0, v_b=0.9,
                               p_dist = "truncnorm", p_mean=0, p_sd=0.05, p_a=-1, p_b=1,
                               compartment = TRUE){

    param_val <- ifelse(compartment, "r0", "R0")
    sim_start_date <- lubridate::ymd(sim_start_date)
    sim_end_date <- lubridate::ymd(sim_end_date)
    npi_cuttoff_date <- lubridate::ymd(npi_cutoff_date)

    npi <- intervention_file %>%
        dplyr::filter(start_date <= npi_cutoff_date) %>%
        dplyr::filter(start_date >= sim_start_date | end_date > sim_start_date) %>% # add warning about npi period <7 days?
        dplyr::group_by(USPS, subpop) %>%
        dplyr::mutate(end_date = dplyr::case_when(is.na(end_date) | end_date == max(end_date) |
                                                      end_date > sim_end_date ~ sim_end_date,
                                                  TRUE ~ end_date),
                      value_dist = v_dist,
                      value_mean = v_mean,
                      value_sd = v_sd,
                      value_a = v_a,
                      value_b = v_b,
                      pert_dist = p_dist,
                      pert_mean = p_mean,
                      pert_sd = p_sd,
                      pert_a = p_a,
                      pert_b = p_b,
                      type = "transmission",
                      category = "NPI",
                      baseline_modifier = "",
                      parameter = dplyr::if_else(method=="MultiPeriodModifier", param_val, NA_character_)
        )

    if(any(stringr::str_detect(npi$name, "^\\d$"))) stop("Intervention names must include at least one non-numeric character.")

    npi <- npi %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference, .x, NA_real_)),
                      pert_dist = ifelse(inference, pert_dist, NA_character_)) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    if(!is.null(redux_subpop)){
        if(redux_subpop == 'all'){
            redux_subpop <- unique(npi$subpop)
        }

        npi <- npi %>%
            dplyr::filter(subpop %in% redux_subpop) %>%
            dplyr::group_by(subpop) %>%
            dplyr::filter(start_date == max(start_date)) %>%
            dplyr::mutate(category = "base_npi",
                          name = paste0(name, "_last")) %>%
            dplyr::bind_rows(
                npi %>%
                    dplyr::group_by(subpop) %>%
                    dplyr::filter(start_date != max(start_date) |! subpop %in% redux_subpop)
            ) %>%
            dplyr::ungroup()
    }

    npi <- npi %>%
        dplyr::ungroup() %>%
        dplyr::add_count(name) %>%
        dplyr::mutate(method = dplyr::if_else(n==1 & method == "MultiPeriodModifier", "SinglePeriodModifier", method),
                      parameter = dplyr::if_else(n==1 & method == "SinglePeriodModifier", param_val, parameter)) %>%
        dplyr::select(-n)

    return(npi)

}



#' Specify parameters for NPIs
#'
#' @param intervention_file df with the location's state and ID and the intervention start and end dates, name, and method - from process_npi_shub
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param npi_cutoff_date only interventions that start before or on npi_cuttof_date are included
#' @param redux_subpop string or vector of characters indicating which subpop will have an intervention with the ModifierModifier method; it accepts "all". If any values are specified, the intervention in the subpop with the maximum start date will be selected. It defaults to NULL. .
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#'
#' @return
#'
#' @export
#'
#' @examples
#' geodata <- load_geodata_file(filename = "data/geodata_territories_2019_statelevel.csv")
#' npi_dat <- process_npi_shub(intervention_path = "data/intervention_tracking/Shelter-in-place-as-of-04302021.csv", geodata)
#'
#' npi_dat <- set_npi_params(intervention_file = npi_dat, sim_start_date = "2020-01-15", sim_end_date = "2021-07-30")
#'
set_npi_params <- function (intervention_file, sim_start_date = as.Date("2020-01-31"),
                            sim_end_date = Sys.Date() + 60, npi_cutoff_date = Sys.Date() - 7,
                            inference = TRUE, redux_subpop = NULL, v_dist = "truncnorm",
                            v_mean = 0.6, v_sd = 0.05, v_a = 0, v_b = 0.9, p_dist = "truncnorm",
                            p_mean = 0, p_sd = 0.05, p_a = -1, p_b = 1, compartment = TRUE) {

    param_val <- ifelse(compartment, "r0", "R0")
    sim_start_date <- lubridate::ymd(sim_start_date)
    sim_end_date <- lubridate::ymd(sim_end_date)
    npi_cuttoff_date <- lubridate::ymd(npi_cutoff_date)
    npi <- intervention_file %>%
        dplyr::filter(start_date <= npi_cutoff_date) %>%
        dplyr::filter(start_date >= sim_start_date |  end_date > sim_start_date | is.na(end_date)) %>%
        dplyr::group_by(USPS, subpop) %>%
        dplyr::mutate(end_date = dplyr::case_when(is.na(end_date) | end_date == max(end_date) | end_date > sim_end_date ~ sim_end_date, TRUE ~ end_date),
                      value_dist = v_dist,
                      value_mean = v_mean, value_sd = v_sd, value_a = v_a,
                      value_b = v_b, pert_dist = p_dist, pert_mean = p_mean,
                      pert_sd = p_sd, pert_a = p_a, pert_b = p_b, type = "transmission",
                      category = "NPI", baseline_modifier = "", parameter = dplyr::if_else(method == "MultiPeriodModifier", param_val, NA_character_))
    if (any(stringr::str_detect(npi$name, "^\\d$")))
        stop("Intervention names must include at least one non-numeric character.")
    npi <- npi %>% dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference, .x, NA_real_)), pert_dist = ifelse(inference, pert_dist, NA_character_)) %>%
        dplyr::select(USPS, subpop,
                      start_date, end_date, name, method, type, category,
                      parameter, baseline_modifier, tidyselect::starts_with("value_"),
                      tidyselect::starts_with("pert_"))
    if (!is.null(redux_subpop)) {
        if (redux_subpop == "all") {
            redux_subpop <- unique(npi$subpop)
        }
        npi <- npi %>% dplyr::filter(subpop %in% redux_subpop) %>%
            dplyr::group_by(subpop) %>% dplyr::filter(start_date == max(start_date)) %>%
            dplyr::mutate(category = "base_npi", name = paste0(name, "_last")) %>%
            dplyr::bind_rows(npi %>% dplyr::group_by(subpop) %>% dplyr::filter(start_date != max(start_date) | !subpop %in% redux_subpop)) %>%
            dplyr::ungroup()
    }
    npi <- npi %>% dplyr::ungroup() %>%
        dplyr::add_count(name) %>%
        dplyr::mutate(method = dplyr::if_else(n == 1 & method == "MultiPeriodModifier", "SinglePeriodModifier", method),
                      parameter = dplyr::if_else(n == 1 & method == "SinglePeriodModifier", param_val, parameter)) %>%
        dplyr::select(-n)
    return(npi)
}








#' Generate seasonality file with params
#'
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#'
#' @return data frame with columns seasonal terms and set parameters.
#' @export
#'
#' @examples
#' dat <- set_seasonality_params()
#'
#' dat
#'

set_seasonality_params <- function(sim_start_date=as.Date("2020-03-31"),
                                   sim_end_date=Sys.Date()+60,
                                   inference = TRUE,
                                   method = "MultiPeriodModifier",
                                   v_dist="truncnorm",
                                   v_mean = c(-0.2, -0.133, -0.067, 0, 0.067, 0.133, 0.2, 0.133, 0.067, 0, -0.067, -0.133), # TODO function?
                                   v_sd = 0.05, v_a = -1, v_b = 1,
                                   p_dist="truncnorm",
                                   p_mean = 0, p_sd = 0.05, p_a = -1, p_b = 1,
                                   compartment = TRUE){

    sim_start_date <- as.Date(sim_start_date)
    sim_end_date <- as.Date(sim_end_date)

    param_val <- ifelse(compartment, "r0", "R0")

    years_ <- unique(lubridate::year(seq(sim_start_date, sim_end_date, 1)))

    seas <- tidyr::expand_grid(
        tidyr::tibble(month= tolower(month.abb),
                      month_num = 1:12,
                      value_dist = v_dist,
                      value_mean = v_mean,
                      value_sd = v_sd,
                      value_a = v_a,
                      value_b= v_b,
                      pert_dist = p_dist,
                      pert_mean = p_mean,
                      pert_sd = p_sd,
                      pert_a = p_a,
                      pert_b = p_b)) %>%
        tidyr::expand_grid(year = years_) %>%
        dplyr::ungroup() %>%
        dplyr::mutate(start_date = lubridate::ymd(paste0('"', year, '-', month_num, '-01"')),
                      end_date = lubridate::ceiling_date(start_date, "months")-1,
                      end_date = dplyr::if_else(end_date > sim_end_date, sim_end_date, end_date),
                      USPS = "",
                      type = "transmission",
                      parameter = param_val,
                      category = "seasonal",
                      method = method,
                      baseline_modifier = "",
                      subpop = "all",
                      name = paste0("Seas_", month),
                      pert_dist = ifelse(inference, as.character(pert_dist), NA_character_),
                      dplyr::across(pert_sd:pert_a, ~ifelse(inference, as.numeric(.x), NA_real_))
        ) %>%
        dplyr::filter(start_date <= end_date) %>%
        dplyr::filter(lubridate::ceiling_date(start_date, "months") >= lubridate::ceiling_date(sim_start_date, "months") &
                          lubridate::ceiling_date(end_date, "months") <= lubridate::ceiling_date(sim_end_date, "months")
        ) %>%
        dplyr::add_count(name) %>%
        dplyr::mutate(method = dplyr::if_else(n > 1, method, "SinglePeriodModifier"),
                      end_date = dplyr::if_else(end_date > sim_end_date, sim_end_date, end_date),
                      start_date = dplyr::if_else(start_date < sim_start_date, sim_start_date, start_date)
        ) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    return(seas)
}

#' Generate local variance
#'
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#'
#' @return data frame with columns for
#' @export
#'
#' @examples
#' dat <- set_localvar_params()
#'
#' dat
#'
set_localvar_params <- function(sim_start_date=as.Date("2020-03-31"),
                                sim_end_date=Sys.Date()+60,
                                inference = TRUE,
                                v_dist="truncnorm",
                                v_mean =  0, v_sd = 0.05, v_a = -1, v_b = 1, # TODO: add check on limits
                                p_dist="truncnorm",
                                p_mean = 0, p_sd = 0.05, p_a = -1, p_b = 1,
                                compartment = TRUE
){
    sim_start_date <- as.Date(sim_start_date)
    sim_end_date <- as.Date(sim_end_date)

    method = "SinglePeriodModifier"
    param_val <- ifelse(compartment, "r0", "R0")
    affected_subpop = "all"

    local_var <- dplyr::tibble(USPS = "",
                               subpop = "all",
                               name = "local_variance",
                               type = "transmission",
                               category = "local_variance",
                               parameter = param_val,
                               baseline_modifier = "",
                               start_date = sim_start_date,
                               end_date = sim_end_date,
                               method = method,
                               param = param_val,
                               affected_subpop = affected_subpop,
                               value_dist = v_dist,
                               value_mean = v_mean,
                               value_sd = v_sd,
                               value_a = v_a,
                               value_b= v_b,
                               pert_dist = p_dist,
                               pert_mean = p_mean,
                               pert_sd = p_sd,
                               pert_a = p_a,
                               pert_b = p_b) %>%
        dplyr::mutate(pert_dist = ifelse(inference, as.character(pert_dist), NA_character_),
                      dplyr::across(pert_mean:pert_b, ~ifelse(inference, as.numeric(.x), NA_real_))) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    return(local_var)
}

#' Generate NPI reduction interventions
#'
#' @param npi_file output from set_npi_params
#' @param incl_subpop vector of subpop to include; NULL will generate interventions for all geographies
#' @param projection_start_date first date without data to fit
#' @param redux_end_date end date for reduction interventions; default NULL uses sim_end_date in npi_file
#' @param redux_level reduction to intervention effectiveness; used to estimate mean value of reduction by month
#' @param v_dist type of distribution for reduction
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param compartment
#'
#' @return
#' @export
#'
#' @examples
#'
#'
set_redux_params <- function(npi_file,
                             projection_start_date = Sys.Date(), # baseline npi should have at least 2-3 weeks worth of data
                             redux_end_date=NULL,
                             redux_level = 0.5,
                             v_dist = "truncnorm", # TODO: change to "fixed" and add correct value, remove v_sd-v_b
                             v_mean=0.6,
                             v_sd=0.01,
                             v_a=0,
                             v_b=1,
                             compartment = TRUE
){

    projection_start_date <- as.Date(projection_start_date)
    param_val <- ifelse(compartment, "r0", "R0")

    if(!is.null(redux_end_date)){
        redux_end_date <- as.Date(redux_end_date)

        if(redux_end_date > max(npi_file$end_date)) stop("The end date for reduction interventions should be less than or equal to the sim_end_date in the npi_file.")

    }

    og <- npi_file %>%
        dplyr::filter(category == "base_npi") %>%
        dplyr::group_by(USPS, subpop) %>%
        dplyr::mutate(end_date = dplyr::if_else(is.null(redux_end_date), end_date, redux_end_date))

    if(any(projection_start_date < unique(og$start_date))){warning("Some interventions start after the projection_start_date")}

    months_start <- seq(lubridate::floor_date(projection_start_date, "month"), max(og$end_date), by="month")
    months_start[1] <- projection_start_date

    months_end <- lubridate::ceiling_date(months_start, "months")-1
    months_end[length(months_end)] <- max(og$end_date)

    month_n <- length(months_start)

    reduction <- rep(redux_level/month_n, month_n) %>% cumsum()

    redux <- dplyr::tibble(
        start_date = months_start,
        end_date = months_end,
        month = lubridate::month(months_start, label=TRUE, abbr=TRUE) %>% tolower(),
        value_mean = reduction, # TODO: reduction to value_mean
        type = rep("transmission", month_n),
        subpop = og$subpop %>% paste0(collapse = '", "')) %>%
        mutate(USPS = "",
               category = "NPI_redux",
               name = paste0(category, '_', month),
               baseline_modifier = c("base_npi", paste0("NPI_redux_", month[-length(month)])),
               method = "ModifierModifier",
               parameter = param_val,
               value_dist = v_dist,
               value_sd = v_sd,
               value_a = v_a,
               value_b = v_b,
               pert_dist = NA_character_,
               pert_mean = NA_real_,
               pert_sd = NA_real_,
               pert_a = NA_real_,
               pert_b = NA_real_) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    return(redux)
}



#' Generate vaccination rates intervention
#'
#' @param vacc_path path to vaccination rates
#' @param vacc_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param incl_subpop vector of subpop to include
#' @param scenario_num which baseline scenario will be selected from the vaccination rate file
#' @param compartment
#'
#' @return
#' @export
#'
#' @examples
#'
set_vacc_rates_params <- function (vacc_path,
                                   vacc_start_date = "2021-01-01",
                                   sim_end_date = Sys.Date() + 60,
                                   incl_subpop = NULL,
                                   scenario_num = 1,
                                   compartment = TRUE) {

    vacc_start_date <- as.Date(vacc_start_date)
    sim_end_date <- as.Date(sim_end_date)
    vacc <- readr::read_csv(vacc_path) %>% dplyr::filter(!is.na(month) &
                                                             scenario == scenario_num)
    if (!is.null(incl_subpop)) {
        vacc <- vacc %>% dplyr::filter(subpop %in% incl_subpop)
    }
    vacc <- vacc %>% dplyr::filter(start_date <= sim_end_date) %>%
        dplyr::mutate(end_date = lubridate::as_date(ifelse(end_date > sim_end_date, sim_end_date, end_date))) %>%
        dplyr::rename(value_mean = vacc_rate) %>%
        dplyr::mutate(subpop = as.character(subpop), month = lubridate::month(start_date, label = TRUE),
                      type = "transmission", category = "vaccination",
                      name = paste0("Dose1_", tolower(month), lubridate::year(start_date)),
                      method = "SinglePeriodModifier",  baseline_modifier = "",
                      value_mean = round(value_mean, 5),
                      value_dist = "fixed", value_sd = NA_real_, value_a = NA_real_,
                      value_b = NA_real_, pert_dist = NA_character_, pert_mean = NA_real_,
                      pert_sd = NA_real_, pert_a = NA_real_, pert_b = NA_real_)

    if(compartment){
        vacc <- vacc %>% mutate(parameter = rate_param)
    } else {
        vacc <- vacc %>% mutate(parameter = "transition_rate 0")
    }

    if("age_group" %in% colnames(vacc)){
        vacc <- vacc %>% mutate(name = paste0(name, "_age", age_group))
    }
    vacc <- vacc %>%
        dplyr::select(USPS, subpop, start_date, end_date, name,
                      method, type, category, parameter, baseline_modifier,
                      tidyselect::starts_with("value_"), tidyselect::starts_with("pert_")) %>%
        dplyr::filter(start_date >= vacc_start_date & value_mean > 0)
    return(vacc)
}



#' Generate vaccination rates intervention
#'
#' @param vacc_path path to vaccination rates
#' @param vacc_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param incl_subpop vector of subpop to include
#' @param scenario_num which baseline scenario will be selected from the vaccination rate file
#' @param compartment
#' @param rate_param
#'
#' @return
#' @export
#'
#' @examples
#'
set_vacc_rates_params_dose3 <- function (vacc_path,
                                         vacc_start_date = "2021-01-01", sim_end_date = Sys.Date() + 60,
                                         incl_subpop = NULL,
                                         rate_groups = c("nu_3y","nu_3o"),
                                         scenario_num = 1,
                                         compartment = TRUE,
                                         rate_param=NA) {

    vacc_start_date <- as.Date(vacc_start_date)
    sim_end_date <- as.Date(sim_end_date)
    vacc <- readr::read_csv(vacc_path) %>% dplyr::filter(!is.na(month) &
                                                             scenario == scenario_num)
    if (!is.null(incl_subpop)) {
        vacc <- vacc %>% dplyr::filter(subpop %in% incl_subpop)
    }

    if(compartment){
        vacc <- vacc %>% mutate(parameter=rate_param)
    } else {
        vacc <- vacc %>% mutate(parameter="transition_rate 0")
    }

    vacc <- vacc %>% dplyr::filter(start_date <= sim_end_date) %>%
        dplyr::mutate(end_date = lubridate::as_date(ifelse(end_date >
                                                               sim_end_date, sim_end_date, end_date))) %>% dplyr::rename(value_mean = vacc_rate) %>%
        dplyr::mutate(subpop = as.character(subpop), month = lubridate::month(start_date,
                                                                            label = TRUE), type = "transmission", category = "vaccination",
                      name = paste0("Dose3_", tolower(month), lubridate::year(start_date), "_",age_group),
                      method = "SinglePeriodModifier",
                      baseline_modifier = "",
                      value_dist = "fixed", value_sd = NA_real_, value_a = NA_real_,
                      value_b = NA_real_, pert_dist = NA_character_, pert_mean = NA_real_,
                      pert_sd = NA_real_, pert_a = NA_real_, pert_b = NA_real_) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name,
                      method, type, category, parameter, baseline_modifier,
                      tidyselect::starts_with("value_"), tidyselect::starts_with("pert_")) %>%
        dplyr::filter(start_date >= vacc_start_date & value_mean > 0)

    return(vacc)
}







#' Generate variant interventions
#'
#' @param b117_only whether to generate estimates for B117 variant only or both B117 and B1617
#' @param variant_path_1 path to B117 variant
#' @param variant_path_2 path to B1617 variant
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param inference_cutoff_date no inference is applied for interventions that start on or after this day
#' @param variant_lb
#' @param varian_effect change in transmission for variant default is 50% from Davies et al 2021
#' @param month_shift
#' @param geodata file with columns for state/county abbreviation (USPS) and admin code (subpop); only required if state_level is TRUE
#' @param state_level whether there is state-level data on the variant; requires a geodata file
#' @param transmission_increase transmission increase in B1617 relative to B117
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param v_dist type of distribution for reduction
#' @param v_mean reduction mean
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#'
#'
#' @return
#' @export
#'
#' @examples
#'
set_variant_params <- function(b117_only = FALSE, variant_path, variant_path_2 = NULL,
                               sim_start_date, sim_end_date, inference_cutoff_date = Sys.Date() - 7,
                               variant_lb = 1.4, variant_effect = 1.5, month_shift = NULL,
                               state_level = TRUE, geodata = NULL,
                               transmission_increase = c(1, 1.45, (1.6 * 1.6)),
                               variant_compartments = c("WILD", "ALPHA", "DELTA"),
                               compartment = TRUE, inference = TRUE,
                               v_dist = "truncnorm", v_sd = 0.01, v_a = -1.5, v_b = 0,
                               p_dist = "truncnorm", p_mean = 0, p_sd = 0.01, p_a = -1, p_b = 1){

    inference_cutoff_date <- as.Date(inference_cutoff_date)
    if (compartment) {
        variant_data <- generate_compartment_variant2(variant_path = variant_path,
                                                      variant_compartments = variant_compartments, transmission_increase = transmission_increase,
                                                      geodata = geodata, sim_start_date = sim_start_date,
                                                      sim_end_date = sim_end_date)
    } else {
        # we can get rid of this B117 part eventually
        if (b117_only) {
            variant_data <- config.writer::generate_variant_b117(variant_path = variant_path,
                                                                 sim_start_date = sim_start_date, sim_end_date = sim_end_date,
                                                                 variant_lb = variant_lb, variant_effect = variant_effect,
                                                                 month_shift = month_shift) %>% dplyr::mutate(subpop = "all",
                                                                                                              USPS = "")
        } else if (state_level) {
            if (is.null(variant_path_2)) {
                stop("You must specify a path for the second variant.")
            }
            if (is.null(geodata)) {
                stop("You must specify a geodata file")
            }
            variant_data <- generate_multiple_variants_state(variant_path_1 = variant_path,
                                                             variant_path_2 = variant_path_2, sim_start_date = sim_start_date,
                                                             sim_end_date = sim_end_date, variant_lb = variant_lb,
                                                             variant_effect = variant_effect, transmission_increase = transmission_increase,
                                                             geodata = geodata)
        } else {
            if (is.null(variant_path_2)) {
                stop("You must specify a path for the second variant.")
            }
            variant_data <- generate_multiple_variants(variant_path_1 = variant_path,
                                                       variant_path_2 = variant_path_2, sim_start_date = sim_start_date,
                                                       sim_end_date = sim_end_date, variant_lb = variant_lb,
                                                       variant_effect = variant_effect, transmission_increase = transmission_increase) %>%
                dplyr::mutate(subpop = "all", USPS = "")
        }
    }
    variant_data <- variant_data %>% dplyr::mutate(type = "transmission",
                                                   category = "variant",
                                                   name = paste(USPS, "variantR0adj", paste0("Week", lubridate::week(start_date)), sep = "_"),
                                                   name = stringr::str_remove(name, "^\\_"),
                                                   method = "SinglePeriodModifier",
                                                   parameter = "R0",
                                                   value_dist = v_dist, value_mean = 1 - R_ratio, value_sd = v_sd, value_a = v_a, value_b = v_b,
                                                   pert_dist = p_dist, pert_mean = p_mean, pert_sd = p_sd,
                                                   pert_a = p_a, pert_b = p_b, baseline_modifier = "") %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference & start_date < inference_cutoff_date, .x, NA_real_)),
                      pert_dist = ifelse(inference & start_date < inference_cutoff_date,
                                         pert_dist, NA_character_)) %>%
        dplyr::select(USPS,
                      subpop, start_date, end_date, name, method, type, category,
                      parameter, baseline_modifier, tidyselect::starts_with("value_"),
                      tidyselect::starts_with("pert_"))

    return(variant_data)
}



#' Generate outcome interventions based on vaccination rates
#'
#' @param outcome_path path to vaccination adjusted outcome interventions
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param incl_subpop vector of subpop to include
#' @param scenario which scenario will be selected from the outcome intervention file
#' @param v_dist type of distribution for reduction
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#' @param variant_compartments
#'
#' @return
#' @export
#'
#' @examples
#'
set_vacc_outcome_params <- function(age_strat = "under65",
                                    variant_compartments = c("WILD","ALPHA","DELTA"),
                                    vaccine_compartments = c("unvaccinated"),
                                    national_level = TRUE, # whether to do national interventions to reduce number
                                    redux_round = 0.1,
                                    outcome_path,
                                    sim_start_date = as.Date("2020-03-31"),
                                    sim_end_date = Sys.Date() + 60,
                                    inference = FALSE,
                                    incl_subpop = NULL,
                                    scenario_num = 1,
                                    v_dist = "truncnorm", v_sd = 0.01, v_a = 0, v_b = 1,
                                    p_dist = "truncnorm", p_mean = 0, p_sd = 0.05,
                                    p_a = -1, p_b = 1){

    sim_start_date <- as.Date(sim_start_date)
    sim_end_date <- as.Date(sim_end_date)
    outcome <- readr::read_csv(outcome_path) %>%
        dplyr::filter(!is.na(month) & month != "baseline") %>%
        dplyr::filter(scenario == scenario_num) %>%
        dplyr::filter(prob_redux!=1)

    if (!is.null(incl_subpop)){
        outcome <- outcome %>% dplyr::filter(subpop %in% incl_subpop)
    }
    if(!is.null(outcome$age_strata)){
        if(!is.null(age_strat)){
            outcome <- outcome %>% filter(age_strata %in% age_strat)
        }
    }

    if(national_level){
        outcome <- outcome %>%
            group_by(age_strata, start_date, end_date, month, year, var) %>%
            summarise(prob_redux = mean(prob_redux, na.rm=TRUE)) %>%
            mutate(USPS="US", subpop='all')
    }

    outcome <- outcome %>%
        mutate(prob_redux = round(prob_redux / redux_round)*redux_round) %>%
        filter(prob_redux!=1)

    outcome <- outcome %>%
        dplyr::mutate(month = tolower(month)) %>%
        dplyr::mutate(prob_redux = 1 - prob_redux) %>%
        dplyr::filter(start_date <= sim_end_date) %>%
        dplyr::mutate(end_date = lubridate::as_date(ifelse(end_date > sim_end_date, sim_end_date, end_date)),
                      start_date = lubridate::as_date(ifelse(end_date > start_date & start_date < sim_start_date, sim_start_date, start_date))) %>%
        dplyr::filter(start_date >= sim_start_date) %>%
        dplyr::rename(value_mean = prob_redux) %>%
        dplyr::mutate(subpop = as.character(subpop),
                      type = "outcome",
                      category = "vacc_outcome",baseline_modifier = "",
                      value_dist = v_dist, value_sd = v_sd, value_a = v_a,
                      value_b = v_b, pert_dist = p_dist, pert_mean = p_mean,
                      pert_sd = p_sd, pert_a = p_a, pert_b = p_b)

    outcome <- outcome %>%
        dplyr::full_join(
            expand_grid(var = c("rr_death_inf", "rr_hosp_inf"),
                        variant=variant_compartments,
                        vacc=vaccine_compartments,
                        age_strata=unique(outcome$age_strata)) %>%
                dplyr::mutate(param = dplyr::case_when(var == "rr_death_inf" ~ "incidD", var == "rr_hosp_inf" ~ "incidH",
                                                       TRUE ~ NA_character_),
                              param = paste(param, vacc, variant, age_strat, sep="_")) %>%
                dplyr::filter(!is.na(param))) %>%
        dplyr::mutate(
            #    name = paste(param, "vaccadj", month, sep = "_"), method = "SinglePeriodModifier",
            #    name = paste(param, "vaccadj", USPS, (1-value_mean), sep = "_"), method = "SinglePeriodModifier",
            name = paste(param, "vaccadj", (1-value_mean), sep = "_"), method = "SinglePeriodModifier",
            parameter = paste0(param, "::probability")) %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b,
                                    ~ifelse(inference, .x, NA_real_)),
                      pert_dist = ifelse(inference,
                                         pert_dist, NA_character_)) %>%
        dplyr::select(USPS, subpop,
                      start_date, end_date, name, method, type, category,
                      parameter, baseline_modifier, tidyselect::starts_with("value_"),
                      tidyselect::starts_with("pert_"))
    return(outcome)
}




#' Generate incidC shift interventions
#'
#' @param periods vector of dates that include a shift in incidC
#' @param geodata df with USPS and subpop column for subpop with a shift in incidC
#' @param baseline_ifr assumed true infection fatality rate
#' @param cfr_data optional file with estimates of cfr by state
#' @param epochs character vector with the selection of epochs from the cfr_data file, any of "NoSplit", "MarJun", "JulOct", "NovJan". Required if cfr_data is specified.
#' @param outcomes_parquet_file path to file with subpop-specific adjustments to IFR; required if cfr_data is specified
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param v_dist type of distribution for reduction
#' @param v_mean state-specific initial value. will be taken from empirical CFR estimates if it exists, otherwise this used. If a vector is specified, then each value is added to the corresponding period
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @return
#' @export
#'
#' @examples
set_incidC_shift <- function(periods,
                             geodata,
                             baseline_ifr = 0.005,
                             cfr_data = NULL,
                             epochs = NULL,
                             outcomes_parquet_file = NULL,
                             inference = TRUE,
                             v_dist="truncnorm",
                             v_mean=0.25, v_sd = 0.05, v_a = 0, v_b = 1,
                             p_dist="truncnorm",
                             p_mean = 0, p_sd = 0.01, p_a = -1, p_b = 1
){
    periods <- as.Date(periods)

    if(is.null(cfr_data)){
        epochs <- 1:(length(periods)-1)

        cfr_data <- geodata %>%
            dplyr::select(USPS, subpop) %>%
            tidyr::expand_grid(value_mean = v_mean,
                               epoch=epochs)
    } else{
        if(is.null(epochs) | length(epochs) != (length(periods)-1)){stop("The number of epochs selected should be equal to the number of periods with a shift in incidC")}
        if(any(!epochs %in% c("NoSplit", "MarJun", "JulOct", "NovJan"))){stop('Unknown epoch selected, choose from: "NoSplit", "MarJun", "JulOct", "NovJan"')}
        if(is.null(outcomes_parquet_file)){stop("Must specify a file with the age-adjustments to IFR by state")}

        relative_outcomes <- arrow::read_parquet(outcomes_parquet_file)

        relative_ifr <- relative_outcomes %>%
            dplyr::filter(source == 'incidI' & outcome == "incidD") %>%
            dplyr::filter(subpop %in% geodata$subpop) %>%
            dplyr::select(USPS,subpop,value) %>%
            dplyr::rename(rel_ifr=value) %>%
            dplyr::mutate(ifr=baseline_ifr*rel_ifr)

        cfr_data <- readr::read_csv(cfr_data) %>%
            dplyr::rename(USPS=state, delay=lag) %>%
            dplyr::select(USPS, epoch, delay, cfr) %>%
            dplyr::filter(epoch %in% epochs) %>%
            dplyr::left_join(relative_ifr) %>%
            dplyr::filter(subpop %in% geodata$subpop) %>%
            dplyr::mutate(incidC = pmin(0.99,ifr/cfr),  # get effective case detection rate based in assumed IFR.
                          value_mean = pmax(0,1-incidC),
                          value_mean = signif(value_mean, digits = 2)) %>% # get effective reduction in incidC assuming baseline incidC
            dplyr::select(USPS,subpop, epoch, value_mean)


        no_cfr_data <- relative_ifr %>%
            tidyr::expand_grid(value_mean = v_mean,
                               epoch = epochs) %>%
            dplyr::filter(!subpop %in% cfr_data$subpop) %>%
            dplyr::select(USPS, subpop, epoch, value_mean)

        cfr_data <- dplyr::bind_rows(cfr_data,
                                     no_cfr_data)
    }

    outcome <- list()
    for(i in 1:(length(periods)-1)){
        outcome[[i]] <- cfr_data %>%
            dplyr::filter(epoch == epochs[i]) %>%
            dplyr::select(-epoch) %>%
            dplyr::mutate(
                method = "SinglePeriodModifier",
                name = paste0("incidCshift_", i),
                type = "outcome",
                category = "incidCshift",
                parameter = "incidC::probability",
                baseline_modifier = "",
                start_date = periods[i],
                end_date = periods[i+1]-1,
                value_dist = v_dist,
                value_mean = value_mean,
                value_sd = v_sd,
                value_a = v_a,
                value_b = v_b,
                pert_dist = p_dist,
                pert_mean = p_mean,
                pert_sd = p_sd,
                pert_a = p_a,
                pert_b = p_b
            )

    }

    outcome <- dplyr::bind_rows(outcome) %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference, .x, NA_real_)),
                      pert_dist = ifelse(inference, pert_dist, NA_character_)) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category, parameter, baseline_modifier, tidyselect::starts_with("value_"), tidyselect::starts_with("pert_"))

    return(outcome)

}

#' Generate interventions to adjust hospitalizations
#'
#' @param outcome_path path to vaccination adjusted outcome interventions
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param geodata df with USPS and subpop column for subpop with an incidH adjustment
#' @param v_dist type of distribution for reduction
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @param compartment
#' @param variant_compartments
#'
#' @return
#' @export
#'
#' @examples

set_incidH_adj_params <- function(outcome_path,
                                  sim_start_date=as.Date("2020-03-31"),
                                  sim_end_date=Sys.Date()+60,
                                  geodata,
                                  inference = FALSE,
                                  v_dist = "fixed", v_sd = 0.01, v_a = -10, v_b = 2,
                                  p_dist = "truncnorm", p_mean = 0, p_sd = 0.05, p_a = -1, p_b = 1,
                                  compartment = TRUE,
                                  variant_compartments = c("WILD", "ALPHA", "DELTA")
)
{
    variant_compartments <- stringr::str_to_upper(variant_compartments)

    sim_start_date <- lubridate::as_date(sim_start_date)
    sim_end_date <- lubridate::as_date(sim_end_date)
    outcome <- readr::read_csv(outcome_path) %>%
        dplyr::filter(!is.na(ratio) & USPS != "US")

    outcome <- outcome %>%
        dplyr::left_join(geodata %>% dplyr::select(USPS, subpop))

    outcome <- outcome %>% dplyr::mutate(param = "incidH") %>%
        # dplyr::mutate(month = tolower(month)) %>%
        dplyr::mutate(prob_redux = 1 - (1/ratio)) %>%
        #dplyr::mutate(prob_redux = 1 / ratio) %>%
        dplyr::mutate(end_date = sim_end_date,
                      start_date = sim_start_date) %>%
        dplyr::rename(value_mean = prob_redux) %>%
        dplyr::mutate(subpop = as.character(subpop),
                      type = "outcome",
                      category = "outcome_adj",
                      name = paste(param, "adj",USPS, sep = "_"),
                      method = "SinglePeriodModifier",
                      parameter = paste0(param, "::probability"),
                      baseline_modifier = "",
                      value_dist = v_dist,
                      value_sd = v_sd,
                      value_a = v_a,
                      value_b = v_b,
                      pert_dist = p_dist,
                      pert_mean = p_mean,
                      pert_sd = p_sd,
                      pert_a = p_a,
                      pert_b = p_b) %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference, .x, NA_real_)),
                      pert_dist = ifelse(inference, pert_dist, NA_character_)) %>%
        dplyr::select(USPS, subpop, start_date, end_date, name, method, type, category,
                      parameter, baseline_modifier, tidyselect::starts_with("value_"),
                      tidyselect::starts_with("pert_"))

    if(compartment){
        temp <- list()
        for(i in 1:length(variant_compartments)){
            temp[[i]] <- outcome %>%
                dplyr::mutate(parameter = stringr::str_replace(parameter, "::probability", paste0("_", variant_compartments[i],"::probability")),
                              name = paste0(name, "_", variant_compartments[i]))
        }

        outcome <- dplyr::bind_rows(temp)

    }

    return(outcome)
}


#' Generate interventions to adjust vaccine effectiveness on transmission
#'
#' @param variant_path path to variant data
#' @param VE vaccine effectiveness against wild strain for the first and second doses, respectively
#' @param VE_delta vaccine effectivenes against variant or the first and second doses, respectively
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param geodata df with USPS and subpop column for subpop with an incidH adjustment
#' @param v_dist type of distribution for reduction
#' @param v_sd reduction sd
#' @param v_a reduction a
#' @param v_b reduction b
#' @param inference logical indicating whether inference will be performed on intervention (default is TRUE); perturbation values are replaced with NA if set to FALSE.
#' @param p_dist type of distribution for perturbation
#' @param p_mean perturbation mean
#' @param p_sd perturbation sd
#' @param p_a perturbation a
#' @param p_b perturbation b
#' @return
#' @export
#'
#' @examples
#'

set_ve_shift_params <- function(variant_path,
                                VE = c(0.5, 0.9),
                                VE_delta = c(0.35, 0.8),
                                sim_start_date=as.Date("2020-03-31"),
                                sim_end_date=Sys.Date()+60,
                                geodata,
                                inference = FALSE,
                                v_dist = "fixed", v_sd = 0.01, v_a = -1, v_b = 2,
                                p_dist = "truncnorm", p_mean = 0, p_sd = 0.01, p_a = -1, p_b = 1,
                                compartment = TRUE){

    par_val_1 <- ifelse(compartment, "theta_1A", "susceptibility_reduction 1")
    par_val_2 <- ifelse(compartment, "theta_2A", "susceptibility_reduction 2")
    sim_start_date <- lubridate::as_date(sim_start_date)
    sim_end_date <- lubridate::as_date(sim_end_date)

    outcome <- readr::read_csv(variant_path) %>%
        dplyr::filter(location == "US", date >= "2021-04-01") %>%
        dplyr::mutate(month = lubridate::month(date, label=TRUE), year = lubridate::year(date),
                      monthyr = paste0(month,substr(year,3,4))) %>%
        dplyr::group_by(month, monthyr, year) %>%
        dplyr::summarize(start_date = min(date),
                         end_date = as.Date(max(date)),
                         variant_prop = median(fit)) %>%
        dplyr::mutate(ve_1 = VE[1]*(1-variant_prop)+VE_delta[1]*variant_prop,
                      ve_2 = VE[2]*(1-variant_prop)+VE_delta[2]*variant_prop) %>%
        dplyr::mutate(ve1_redux = 1 - (ve_1/VE[1]),
                      ve2_redux = 1 - (ve_2/VE[2])) %>%
        dplyr::select(-ve_1, -ve_2) %>%
        tidyr::pivot_longer(cols = ends_with("redux"), names_to = "dose", values_to = "value_mean") %>%
        dplyr::mutate(value_mean = round(value_mean, 2)) %>%
        dplyr::group_by(value_mean, dose) %>%
        dplyr::summarise(month = ifelse(length(month)==1, as.character(monthyr), paste0(monthyr[which.min(start_date)], "-", monthyr[which.max(start_date)])),
                         start_date = min(start_date),
                         end_date = max(end_date)) %>%
        dplyr::filter(value_mean != 0)


    outcome <- outcome %>%
        dplyr::mutate(name = paste0("VEshift_", tolower(month), "_dose", stringr::str_sub(dose, 3, 3))) %>%
        dplyr::select(-dose) %>%
        dplyr::filter(start_date <= sim_end_date & end_date > sim_start_date) %>%
        dplyr::mutate(end_date = lubridate::as_date(ifelse(end_date > sim_end_date, sim_end_date, end_date))) %>%
        dplyr::mutate(USPS = "",
                      subpop = "all",
                      type = "transmission",
                      parameter = dplyr::if_else(stringr::str_detect(name, "ose1"), par_val_1, par_val_2),
                      category = "ve_shift",
                      method = "SinglePeriodModifier",
                      baseline_modifier = "",
                      value_dist = v_dist,
                      value_sd = v_sd,
                      value_a = v_a,
                      value_b = v_b,
                      pert_dist = p_dist,
                      pert_mean = p_mean,
                      pert_sd = p_sd, # dont want much perturbation on this if it gets perturbed
                      pert_a = p_a,
                      pert_b = p_b)  %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(inference, .x, NA_real_)),
                      pert_dist = ifelse(inference, pert_dist, NA_character_))
    return(outcome)
}


#' Bind interventions and prevents inference on interventions with no data
#'
#' @param ... intervention dfs with config params
#' @param inference_cutoff_date no inference is applied for interventions that start on or after this day
#' @param sim_start_date simulation start date
#' @param sim_end_date simulation end date
#' @param save_name directory to save dataframe; NULL if no safe
#' @param filter_dates whether to filter interventions by sim_start_date and sim_end_date
#'
#' @return
#' @export
#'
#' @examples
#'
bind_interventions <- function(...,
                               inference_cutoff_date = Sys.Date() - 7,
                               sim_start_date,
                               sim_end_date,
                               save_name,
                               filter_dates=FALSE) {

    inference_cutoff_date <- as.Date(inference_cutoff_date)
    sim_end_date <- as.Date(sim_end_date)
    sim_start_date <- as.Date(sim_start_date)
    dat <- dplyr::bind_rows(...)
    if (filter_dates){
        dat <- dat %>%
            filter(start_date < sim_end) %>%
            filter(end_date > sim_start) %>%
            mutate(start_date = as_date(ifelse(start_date<sim_start, sim_start, start_date)))
    } else {
        if (min(dat$start_date) < sim_start_date)
            stop("At least one intervention has a start date before the sim_start_date.")
        if (max(dat$end_date) > sim_end_date)
            stop("At least one intervention has an end date after the sim_end_date.")
    }
    check <- dat %>% dplyr::filter(category == "NPI") %>%
        dplyr::group_by(USPS, subpop, type, category) %>% dplyr::arrange(USPS, subpop, start_date) %>%
        dplyr::mutate(note = dplyr::case_when(end_date >= dplyr::lead(start_date) ~ "Overlap", dplyr::lead(start_date) - end_date > 1 ~ "Gap", TRUE ~ NA_character_)) %>%
        dplyr::mutate(dplyr::across(pert_mean:pert_b, ~ifelse(start_date < inference_cutoff_date, .x, NA_real_)), pert_dist = ifelse(start_date < inference_cutoff_date, pert_dist, NA_character_)) %>%
        dplyr::filter(!is.na(note))
    if (nrow(check) > 0) {
        if (any(check$note == "Overlap"))
            warning(paste0("There are ", nrow(check[check$note == "Overlap", ]), " NPIs of the same category/subpop that overlap in time"))
        if (any(check$note == "Gap"))
            warning(paste0("There are ", nrow(check[check$note == "Gap", ]), " NPIs of the same category/subpop that are discontinuous."))
    }
    if (!is.null(save_name)) {
        readr::write_csv(dat, file = save_name)
    }
    return(dat)
}


#' Estimate average reduction in transmission per day per subpop
#'
#' @param dat
#' @param plot
#'
#' @return
#' @export
#'
#' @examples
#'

daily_mean_reduction <- function(dat,
                                 plot = FALSE){

    dat <- dat %>%
        dplyr::filter(type == "transmission") %>%
        dplyr::mutate(mean = dplyr::case_when(value_dist == "truncnorm" ~
                                                  truncnorm::etruncnorm(a=value_a, b=value_b, mean=value_mean, sd=value_sd),
                                              value_dist == "fixed" ~
                                                  value_mean,
                                              value_dist == "uniform" ~
                                                  (value_a+value_b)/2)
        ) %>%
        dplyr::select(USPS, subpop, start_date, end_date, mean)

    timeline <- tidyr::crossing(time = seq(from=min(dat$start_date), to=max(dat$end_date), by = 1),
                                subpop = unique(dat$subpop))

    if(any(stringr::str_detect(dat$subpop, '", "'))){
        mtr_subpop <- dat %>%
            dplyr::filter(stringr::str_detect(subpop, '", "'))

        temp <- list()
        for(i in 1:nrow(mtr_subpop)){
            temp[[i]] <- tidyr::expand_grid(subpop = mtr_subpop$subpop[i] %>% stringr::str_split('", "') %>% unlist(),
                                            mtr_subpop[i,] %>% dplyr::ungroup() %>% dplyr::select(-subpop)) %>%
                dplyr::select(colnames(mtr_subpop))
        }

        dat <- dat %>%
            dplyr::filter(stringr::str_detect(subpop, '", "', negate = TRUE)) %>%
            dplyr::bind_rows(
                dplyr::bind_rows(temp)
            )
    }

    dat <- dat %>%
        dplyr::filter(subpop=="all") %>%
        dplyr::ungroup() %>%
        dplyr::select(-subpop) %>%
        tidyr::crossing(subpop=unique(dat$subpop[dat$subpop!="all"])) %>%
        dplyr::select(subpop, start_date, end_date, mean) %>%
        dplyr::bind_rows(dat %>% dplyr::filter(subpop!="all") %>% dplyr::ungroup() %>% dplyr::select(-USPS)) %>%
        dplyr::left_join(timeline) %>%
        dplyr::filter(time >= start_date & time <= end_date) %>%
        dplyr::group_by(subpop, time) %>%
        dplyr::summarize(mean = prod(1-mean))

    if(plot){
        dat<- ggplot2::ggplot(data= dat, ggplot2::aes(x=time, y=mean))+
            ggplot2::geom_line()+
            ggplot2::facet_wrap(~subpop)+
            ggplot2::theme_bw()+
            ggplot2::ylab("Average reduction")+
            ggplot2::scale_x_date(date_breaks = "3 months", date_labels = "%b\n%y")+
            ggplot2::scale_y_continuous(breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0))

    }

    return(dat)
}
