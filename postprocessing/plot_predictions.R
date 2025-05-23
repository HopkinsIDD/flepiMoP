#..............................................................................................................

# This script gets run by: run_sim_processing.R`
# Do not modify unless changing the plotting or saving structures

#..............................................................................................................

library(tidyverse)


center_line <- ifelse(point_est==0.5, "median", "mean") ## mean or median model line
center_line_var <- ifelse(point_est==0.5, "point", "point-mean")
proj_data <- data_comb


#### Which valid locations are missing from our submission?

# locs <- read_csv("https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv")
# mismatched <- unique(proj_data$location)[which(!(unique(proj_data$location) %in% locs$location))]
# missing_from_fc <- unique(locs$location)[which(!(locs$location %in% unique(proj_data$location)))]
#
# locs %>% filter(location %in% missing_from_fc)


# STATE DATA --------------------------------------------------------------
utils::data(fips_us_county, package = "flepicommon")

# State Data #
state_cw <- fips_us_county %>% 
  dplyr::distinct(state, state_code) %>%
  dplyr::select(USPS = state, location = state_code) %>%
  dplyr::mutate(location = str_pad(location, 2, side = "left", pad = "0")) %>%
  distinct(location, USPS) %>%
  dplyr::mutate(location = as.character(location), USPS = as.character(USPS)) %>%
  bind_rows(tibble(USPS = "US", location = "US"))



# GROUND TRUTH ------------------------------------------------------------

gt_data <- gt_data %>% 
  mutate(date = lubridate::as_date(date))
colnames(gt_data) <- gsub("incidI", "incidC", colnames(gt_data))
gt_outcomes <- outcomes_[outcomes_ != "I" & sapply(X = paste0("incid", outcomes_), FUN = function(x=X, y) any(grepl(pattern = x, x = y)), y = colnames(gt_data)) ]
outcomes_gt_ <- outcomes_[outcomes_ %in% gt_outcomes]
outcomes_time_gt_ <- outcomes_time_[outcomes_ %in% gt_outcomes]
outcomes_cum_gt_ <- outcomes_cum_[outcomes_ %in% gt_outcomes]
outcomes_cumfromgt_gt_ <- outcomes_cumfromgt[outcomes_ %in% gt_outcomes]

use_obs_data_forcum <- ifelse(any(outcomes_cumfromgt_gt_),TRUE, FALSE)
gt_data_2 <- gt_data
gt_data_2 <- gt_data_2 %>% mutate(cumH = 0) # incidH is only cumulative from start of simulation


# ~ Weekly Outcomes -----------------------------------------------------------
gt_cl <- NULL
if (any(outcomes_time_=="weekly")) {
  # Incident
  gt_data_st_week <- get_weekly_incid(gt_data %>% dplyr::select(date, subpop, USPS, paste0("incid", outcomes_gt_[outcomes_time_gt_=="weekly"])) %>% mutate(sim_num = 0),
  # gt_data_st_week <- get_weekly_incid(gt_data %>% dplyr::select(date, subpop, USPS, paste0("incid", outcomes_gt_[outcomes_time_gt_=="weekly"])) %>% mutate(sim_num = 0),
                                      outcomes = outcomes_gt_[outcomes_time_gt_=="weekly"]) 
  
  # Cumulative
  weekly_cum_outcomes_ <- outcomes_gt_[outcomes_cum_gt_ & outcomes_time_gt_=="weekly"]
  if (length(weekly_cum_outcomes_)>0) {
    gt_data_st_weekcum <- get_cum_sims(sim_data = gt_data_st_week %>%
                                         mutate(agestrat="age0to130") %>%
                                         rename(outcome = outcome_name, value = outcome) %>%
                                         filter(outcome %in% paste0("incid", weekly_cum_outcomes_)),
                                       obs_data = gt_data_2,
                                       gt_cum_vars = paste0("cum", outcomes_gt_[outcomes_cumfromgt_gt_]), # variables to get cum from GT
                                       forecast_date = lubridate::as_date(forecast_date),
                                       aggregation="week",
                                       loc_column = "USPS",
                                       use_obs_data = use_obs_data_forcum) %>%
      rename(outcome_name = outcome, outcome = value) %>%
      select(-agestrat)

    gt_data_st_week <- gt_data_st_week %>%
      bind_rows(gt_data_st_weekcum)
  }
  gt_cl <- gt_cl %>% bind_rows(gt_data_st_week %>% mutate(time_aggr = "weekly"))
}
if (any(outcomes_time_=="daily")) {
  # Incident
  gt_data_st_day <- get_daily_incid(gt_data %>% dplyr::select(date, subpop, USPS, paste0("incid", outcomes_gt_[outcomes_time_gt_=="daily"])) %>% mutate(sim_num = 0),
                                    outcomes = outcomes_gt_[outcomes_time_gt_=="daily"]) 
  
  # Cumulative
  daily_cum_outcomes_ <- outcomes_gt_[outcomes_cum_gt_ & outcomes_time_gt_=="daily"]
  if (length(daily_cum_outcomes_)>0){
    gt_data_st_daycum <- get_cum_sims(sim_data = gt_data_st_day  %>%
                                        mutate(agestrat="age0to130") %>%
                                        rename(outcome = outcome_name, value = outcome) %>%
                                        filter(outcome %in% paste0("incid", daily_cum_outcomes_)),
                                      obs_data = gt_data_2,
                                      gt_cum_vars = paste0("cum", outcomes_gt_[outcomes_cumfromgt_gt_]), # variables to get cum from GT
                                      forecast_date = lubridate::as_date(forecast_date),
                                      aggregation="day",
                                      loc_column = "USPS",
                                      use_obs_data = use_obs_data_forcum) %>%
      rename(outcome_name = outcome, outcome = value) %>%
      select(-agestrat)
    gt_data_st_day <- gt_data_st_day %>% bind_rows(gt_data_st_daycum)
  }
  gt_cl <- gt_cl %>% bind_rows(gt_data_st_day %>% mutate(time_aggr = "daily"))
}



# Remove incomplete weeks from ground truth #
gt_cl <- gt_cl
# if(!((max(gt_cl$date)-lubridate::days(7)) %in% unique(gt_cl$date))){
#   dat_st_cl <- dat_st_cl %>% filter(date != max(date))
# }
# if(!((max(inc_dat_st_vars$date)-lubridate::days(7)) %in% unique(inc_dat_st_vars$date))){
#   inc_dat_st_vars <- inc_dat_st_vars %>% filter(date != max(date))
# }

dat_st_cl2 <- gt_cl %>%
  select(date, USPS, target = outcome_name, time_aggr, value = outcome) %>%
  mutate(incid_cum = ifelse(grepl("inc", target), "inc", "cum")) %>%
  mutate(aggr_target = !grepl('_', target)) %>%
  mutate(outcome = substr(gsub("cum|incid", "", target), 1,1)) %>%
  mutate(pre_gt_end = date<=validation_date)





# <> OVERALL --------------------------------------------------------------

# PRIMARY FORECAST DATA ----------------------------------------------------

forecast_st <- proj_data %>%
  filter(nchar(location)==2 & (quantile %in% sort(unique(c(quant_values, 0.5))) | is.na(quantile))) %>%
  left_join(state_cw, by = c("location"))

# filter out incid or cum
if (!plot_incid) {  forecast_st <- forecast_st %>% filter(!grepl(" inc ", target)) }
if (!plot_cum) {  forecast_st <- forecast_st %>% filter(!grepl(" cum ", target)) }

# filter to keep only outcomes of interest
outcomes_name <- recode(outcomes_, "I"="inf", "C"="case", "H"="hosp", "D"="death")
if(any(outcomes_cum_)){
  cum_outcomes_name <- paste0("cum ", recode(cum_wk_outcomes_, "I"="inf", "C"="case", "H"="hosp", "D"="death"))
}else{
  cum_outcomes_name <- NULL
}
forecast_st <- forecast_st %>% filter(grepl(paste0(c(paste0("inc ", outcomes_name), cum_outcomes_name), collapse = "|"), target))

# create cat variables
forecast_st_plt <- forecast_st %>%
  mutate(incid_cum = ifelse(grepl("inc ", target), "inc", "cum")) %>%
  mutate(outcome = stringr::word(target, 5)) %>%
  mutate(outcome = recode(outcome, "inf"="I", "case"="C", "hosp"="H", "death"="D")) %>%
  dplyr::mutate(quantile_cln = ifelse(!is.na(quantile), paste0("q", paste0(as.character(quantile*100), "%")),
                                      ifelse(type=="point-mean", paste0("mean"),
                                             ifelse(type=="point", paste0("median"), NA)))) %>%
  mutate(target_type = paste0(incid_cum, outcome))

pltdat_truth <- dat_st_cl2 %>% 
  # filter(aggr_target) %>% 
  rename(gt = value) %>%
  mutate(target = gsub("incid", "inc", target)) %>%
  rename(target_type = target) %>%
  filter(USPS %in% unique(forecast_st_plt$USPS)) %>%
  filter(target_type %in% unique(forecast_st_plt$target_type))

if(center_line == "mean"){
  forecast_st_plt <- forecast_st_plt %>% mutate(quantile_cln = gsub("mean", "ctr", quantile_cln))
} else{
  forecast_st_plt <- forecast_st_plt %>% mutate(quantile_cln = gsub("q50%", "ctr", quantile_cln))
}

forecast_st_plt <- forecast_st_plt %>%
  select(scenario_name, scenario_id, target = target_type, incid_cum, outcome, date = target_end_date, USPS, quantile_cln, value) %>%
  pivot_wider(names_from = quantile_cln, values_from = value) %>%
  mutate(type = "projection") %>%
  full_join(pltdat_truth %>%
              mutate(type = "gt", scenario_name = ifelse(pre_gt_end, "gt-pre-projection", "gt-post-projection")) %>%
              select(date, USPS, target = target_type, incid_cum, type, scenario_name, ctr=gt)) %>%
  filter(date >= trunc_date & date <= sim_end_date)




# PRODUCE PDF OF ALL LOCATIONS --------------------------------------------


# set up colors
scenarios_plot <- unique(forecast_st_plt$scenario_name)
scenarios_plot <- scenarios_plot[!grepl('gt', scenarios_plot)]
cols <- c("black", "orange", c("green", "coral", "blue", "purple")[1:length(scenarios_plot)])
names(cols) <- c("gt-pre-projection", "gt-post-projection", scenarios_plot)

options(scipen = 999)
scale_y_funct <- scale_y_continuous

stplot_fname_nosqrt <- paste0(stplot_fname, ".pdf")

pdf(stplot_fname_nosqrt, width=7, height=11)
for(usps in unique(forecast_st_plt$USPS)){

  print(paste0("Plotting: ", usps))
  cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_name)]

  target_labs <- paste0(str_to_title(outcomes_time_[match(gsub("inc","",unique(forecast_st_plt$target)),outcomes_)]), " incident ", gsub("inc","",unique(forecast_st_plt$target)))
  names(target_labs) <- unique(forecast_st_plt$target)

  inc_st_plt <- forecast_st_plt %>%
    filter(USPS == usps) %>%
    filter(incid_cum=="inc") %>%
    mutate(scenario_name = factor(scenario_name)) %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = projection_date, color="red", alpha =0.5) +
    scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
    scale_y_funct(glue::glue("Incident, {usps}")) +
    # scale_y_funct(glue::glue("Weekly Incident, {usps}")) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
    theme_bw() + xlab(NULL) +
    guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
    coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
    facet_wrap(~target, ncol = 1, scales = "free_y",
               labeller = as_labeller(target_labs)) +
    theme(legend.position = "bottom", legend.text = element_text(size=10),
          axis.text.x = element_text(size=6, angle = 45))
  plot(inc_st_plt)


  if (plot_cum) {

    target_labs <- paste0(str_to_title(outcomes_time_[match(gsub("cum","",unique(forecast_st_plt$target)),outcomes_)]), " cumulative ", gsub("cum","",unique(forecast_st_plt$target)))
    names(target_labs) <- unique(forecast_st_plt$target)

    cum_st_plt <- forecast_st_plt %>%
      filter(USPS == usps) %>%
      filter(incid_cum=="cum") %>%
      mutate(scenario_name = factor(scenario_name)) %>%
      ggplot(aes(x = date)) +
      list(
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = projection_date, color="red", alpha =0.5) +
      scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
      scale_y_funct(glue::glue("Cumulative, {usps}"))  +
      scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
      theme_bw() + xlab(NULL) +
      guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
      coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
      facet_wrap(~target, ncol = 1, scales = "free_y",
                 labeller = as_labeller(target_labs)) +
      theme(legend.position = "bottom", legend.text = element_text(size=10),
            axis.text.x = element_text(size=6, angle = 45))

    plot(cum_st_plt)
  }
}
dev.off()


stplot_fname_sqrt <- paste0(stplot_fname, "_sqrt.pdf")
scale_y_funct <- scale_y_sqrt

pdf(stplot_fname_sqrt, width=7, height=11)
for(usps in unique(forecast_st_plt$USPS)){

  print(paste0("Plotting: ", usps))
  cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_name)]

  target_labs <- paste0(str_to_title(outcomes_time_[match(gsub("inc","",unique(forecast_st_plt$target)),outcomes_)]), " incident ", gsub("inc","",unique(forecast_st_plt$target)))
  names(target_labs) <- unique(forecast_st_plt$target)

  inc_st_plt <- forecast_st_plt %>%
    filter(USPS == usps) %>%
    filter(incid_cum=="inc") %>%
    mutate(scenario_name = factor(scenario_name)) %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = projection_date, color="red", alpha =0.5) +
    scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
    scale_y_funct(glue::glue("Incident, {usps}")) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
    theme_bw() + xlab(NULL) +
    guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
    coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
    facet_wrap(~target, ncol = 1, scales = "free_y",
               labeller = as_labeller(target_labs)) +
    theme(legend.position = "bottom", legend.text = element_text(size=10),
          axis.text.x = element_text(size=6, angle = 45))
  plot(inc_st_plt)

  if (plot_cum) {

    target_labs <- paste0(str_to_title(outcomes_time_[match(gsub("cum","",unique(forecast_st_plt$target)),outcomes_)]), " cumulative ", gsub("cum","",unique(forecast_st_plt$target)))
    names(target_labs) <- unique(forecast_st_plt$target)

    cum_st_plt <- forecast_st_plt %>%
      filter(USPS == usps) %>%
      filter(incid_cum=="cum") %>%
      mutate(scenario_name = factor(scenario_name)) %>%
      ggplot(aes(x = date)) +
      list(
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = projection_date, color="red", alpha =0.5) +
      scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
      scale_y_funct(glue::glue("Cumulative, {usps}")) +
      scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
      theme_bw() + xlab(NULL) +
      guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
      coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
      facet_wrap(~target, ncol = 1, scales = "free_y",
                 labeller = as_labeller(target_labs)) +
      theme(legend.position = "bottom", legend.text = element_text(size=10),
            axis.text.x = element_text(size=6, angle = 45))

    plot(cum_st_plt)
  }
}
dev.off()



# WHERE SAVED
print(paste0("Plots created in: \n\n", stplot_fname_nosqrt, " & \n", stplot_fname_sqrt, "\n\n"))
