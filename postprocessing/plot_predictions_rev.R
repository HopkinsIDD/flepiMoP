#..............................................................................................................

# This script gets run by: run_sim_processing.R`
# Do not modify unless changing the plotting or saving structures

#..............................................................................................................

library(tidyverse)


#### Which valid locations are missing from our submission?

# locs <- read_csv("https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv")
# mismatched <- unique(data_submission$location)[which(!(unique(data_submission$location) %in% locs$location))]
# missing_from_fc <- unique(locs$location)[which(!(locs$location %in% unique(data_submission$location)))]
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

gt_data <- gt_data_weekly %>%
  mutate(date = lubridate::as_date(date)) %>%
  add_cum_sims(res = .,  # add cum
               obs_data = NULL,
               origin_date = origin_date)

dat_st_cl2 <- gt_data %>%
  .[, location := substr(subpop, 1, 2)] %>%
  collapse::join(state_cw, how = "left") %>%
  select(date, USPS, target, value) %>%
  mutate(incid_cum = ifelse(grepl("inc", target), "inc", "cum")) %>%
  mutate(aggr_target = !grepl('_', target)) %>%
  mutate(pre_gt_end = date<=origin_date)





# <> OVERALL --------------------------------------------------------------

# PRIMARY FORECAST DATA ----------------------------------------------------

forecast_st <- data_submission %>%
  .[nchar(location)==2 & output_type == "quantile" & output_type_id %in% sort(unique(c(quant_values, 0.5))),] %>%
  collapse::join(state_cw, how = "left")

# filter out incid or cum
if (!plot_incid) {  forecast_st <- forecast_st %>% filter(!grepl(" inc ", target)) }
if (!plot_cum) {  forecast_st <- forecast_st %>% filter(!grepl(" cum ", target)) }

# filter to keep only outcomes of interest
forecast_st <- forecast_st %>% filter(target %in% c(cum_targets_plot, inc_targets_plot))

# create cat variables
forecast_st_plt <- forecast_st %>%
  .[, incid_cum := ifelse(grepl("inc ", target), "inc", "cum")] %>%
  .[, quantile_cln := paste0("q", paste0(as.character(output_type_id*100), "%"))] %>%
  .[, quantile_cln := gsub("q50%", "ctr", quantile_cln)]

pltdat_truth <- dat_st_cl2 %>%
  rename(gt = value) %>%
  mutate(age_group = "0-130") %>%
  as.data.table()


forecast_st_plt <- forecast_st_plt %>%
  # as.data.table() %>%
  .[, date := origin_date + (horizon*7)-1] %>%
  # dplyr::select(date, scenario_name, scenario_id, target, incid_cum, USPS, location, age_group, quantile_cln, value) %>%
  dplyr::select(date, scenario_id, target, incid_cum, USPS, location, age_group, quantile_cln, value) %>%
  pivot_wider(names_from = quantile_cln, values_from = value) %>%
  mutate(type = "projection") %>%
  as.data.table() %>%
  bind_rows(pltdat_truth %>%
              # mutate(type = "gt", scenario_name = ifelse(pre_gt_end, "gt-pre-projection", "gt-post-projection")) %>%
              mutate(type = "gt", scenario_id = ifelse(pre_gt_end, "gt-pre-projection", "gt-post-projection")) %>%
              # select(date, USPS, target, incid_cum, type, age_group, scenario_name, ctr=gt)) %>%
              select(date, USPS, target, incid_cum, type, age_group, scenario_id, ctr=gt)) %>%
  .[date >= trunc_date & date <= sim_end_date]




# PRODUCE PDF OF ALL LOCATIONS --------------------------------------------

usps_to_plot <- sort(unique(forecast_st_plt$USPS))
usps_to_plot <- c("US", usps_to_plot[usps_to_plot!="US"])

# set up colors
# scenarios_plot <- unique(forecast_st_plt$scenario_name)
scenarios_plot <- unique(forecast_st_plt$scenario_id)
scenarios_plot <- scenarios_plot[!grepl('gt', scenarios_plot)]
cols <- c("black", "orange", c("green", "coral", "blue", "purple")[1:length(scenarios_plot)])
names(cols) <- c("gt-pre-projection", "gt-post-projection", scenarios_plot)

options(scipen = 999)
scale_y_funct <- scale_y_continuous

stplot_fname_nosqrt <- paste0(stplot_fname, ".pdf")

pdf(stplot_fname_nosqrt, width=7, height=11)
for(usps in unique(forecast_st_plt$USPS)){
  
  print(paste0("Plotting: ", usps))
  # cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_name)]
  cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_id)]
  
  target_labs <- grep("inc", unique(forecast_st_plt$target), value = TRUE)
  names(target_labs) <- target_labs #unique(forecast_st_plt$target)
  
  inc_st_plt <- forecast_st_plt %>%
    .[USPS == usps & incid_cum=="inc" & age_group == "0-130"] %>%
    # .[, scenario_name := factor(scenario_name)] %>%
    .[, scenario_id := factor(scenario_id)] %>%
    as_tibble() %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_id)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_id)), alpha = 0.25)}
      # if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      # if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_id)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_id)), size = 1.5, pch=21, fill=NA) +
    # geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    # geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
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
    
    target_labs <- grep("cum", unique(forecast_st_plt$target), value = TRUE)
    names(target_labs) <- target_labs #unique(forecast_st_plt$target)
    
    cum_st_plt <- forecast_st_plt %>%
      .[USPS == usps & incid_cum=="cum" & age_group == "0-130",] %>%
      # .[, scenario_name := factor(scenario_name)] %>%
      .[, scenario_id := factor(scenario_id)] %>%
      as_tibble() %>%
      ggplot(aes(x = date)) +
      list(
        # if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        # if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_id)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_id)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_id)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_id)), size = 1.5, pch=21, fill=NA) +
      # geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      # geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
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
for(usps in usps_to_plot){
  
  print(paste0("Plotting: ", usps))
  cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_name)]
  
  target_labs <- grep("inc", unique(forecast_st_plt$target), value = TRUE)
  names(target_labs) <- target_labs #unique(forecast_st_plt$target)
  
  inc_st_plt <- forecast_st_plt %>%
    .[USPS == usps & incid_cum=="inc" & age_group == "0-130",] %>%
    .[, scenario_name := factor(scenario_name)] %>%
    as_tibble() %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
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
    
    target_labs <- grep("cum", unique(forecast_st_plt$target), value = TRUE)
    names(target_labs) <- target_labs #unique(forecast_st_plt$target)
    
    cum_st_plt <- forecast_st_plt %>%
      .[USPS == usps & incid_cum=="cum" & age_group == "0-130",] %>%
      .[, scenario_name := factor(scenario_name)] %>%
      as_tibble() %>%
      ggplot(aes(x = date)) +
      list(
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
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





# AGE SPECIFIC ------------------------------------------------------------

stplot_fname_nosqrt_age <- gsub(".pdf", "_ages.pdf", stplot_fname_nosqrt)
pdf(stplot_fname_nosqrt_age, width=7, height=11)
for(usps in unique(forecast_st_plt$USPS)){
  
  print(paste0("Plotting: ", usps, ", age-specific"))
  cols_tmp <- cols[names(cols) %in% unique(forecast_st_plt$scenario_name)]
  
  target_labs <- grep("inc", unique(forecast_st_plt$target), value = TRUE)
  names(target_labs) <- target_labs #unique(forecast_st_plt$target)
  
  inc_st_plt_hosp <- forecast_st_plt %>%
    .[USPS == usps & incid_cum=="inc" & grepl("hosp", target),] %>%
    .[, scenario_name := factor(scenario_name)] %>%
    as_tibble() %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
    scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
    scale_y_funct(glue::glue("Incident, {usps}")) +
    # scale_y_funct(glue::glue("Weekly Incident, {usps}")) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
    theme_bw() + xlab(NULL) +
    guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
    coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
    facet_wrap(~age_group, ncol = 1, scales = "free_y") +
    theme(legend.position = "bottom", legend.text = element_text(size=10),
          axis.text.x = element_text(size=6, angle = 45))+
    ggtitle("Hospitalizations")
  plot(inc_st_plt_hosp)
  
  
  inc_st_plt_death <- forecast_st_plt %>%
    .[USPS == usps & incid_cum=="inc" & grepl("death", target),] %>%
    .[, scenario_name := factor(scenario_name)] %>%
    as_tibble() %>%
    ggplot(aes(x = date)) +
    list(
      if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
      if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
    ) +
    geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
    geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
    geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
    scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
    scale_y_funct(glue::glue("Incident, {usps}")) +
    # scale_y_funct(glue::glue("Weekly Incident, {usps}")) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
    theme_bw() + xlab(NULL) +
    guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
    coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
    facet_wrap(~age_group, ncol = 1, scales = "free_y") +
    theme(legend.position = "bottom", legend.text = element_text(size=10),
          axis.text.x = element_text(size=6, angle = 45)) +
    ggtitle("Deaths")
  plot(inc_st_plt_death)
  
  
  if (plot_cum) {
    
    target_labs <- grep("cum", unique(forecast_st_plt$target), value = TRUE)
    names(target_labs) <- target_labs #unique(forecast_st_plt$target)
    
    cum_st_plt_hosp <- forecast_st_plt %>%
      .[USPS == usps & incid_cum=="cum" & grepl("hosp", target),] %>%
      .[, scenario_name := factor(scenario_name)] %>%
      as_tibble() %>%
      ggplot(aes(x = date)) +
      list(
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
      scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
      scale_y_funct(glue::glue("Cumulative, {usps}")) +
      # scale_y_funct(glue::glue("Weekly Cumulative, {usps}")) +
      scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
      theme_bw() + xlab(NULL) +
      guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
      coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
      facet_wrap(~age_group, ncol = 1, scales = "free_y") +
      theme(legend.position = "bottom", legend.text = element_text(size=10),
            axis.text.x = element_text(size=6, angle = 45))+
      ggtitle("Hospitalizations")
    plot(cum_st_plt_hosp)
    
    
    cum_st_plt_death <- forecast_st_plt %>%
      .[USPS == usps & incid_cum=="cum" & grepl("death", target),] %>%
      .[, scenario_name := factor(scenario_name)] %>%
      as_tibble() %>%
      ggplot(aes(x = date)) +
      list(
        if("q2.5%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q2.5%`, ymax = `q97.5%`, fill = factor(scenario_name)), alpha = 0.2)},
        if("q25%" %in% colnames(forecast_st_plt)){ geom_ribbon(data = . %>% filter(type=="projection"), aes(ymin = `q25%`, ymax = `q75%`, fill = factor(scenario_name)), alpha = 0.25)}
      ) +
      geom_line(data = . %>% filter(type=="projection"), aes(y = ctr, color = factor(scenario_name)), linewidth = 1.25) +
      geom_point(data = . %>% filter(type=="gt"), aes(y = ctr, color = factor(scenario_name)), size = 1.5, pch=21, fill=NA) +
      geom_vline(xintercept = origin_date, color="red", alpha =0.5) +
      scale_color_manual(values = cols_tmp, aesthetics = c("color", "fill")) +
      scale_y_funct(glue::glue("Cumulative, {usps}")) +
      # scale_y_funct(glue::glue("Weekly Cumulative, {usps}")) +
      scale_x_date(date_breaks = "1 month", date_labels = "%b %y") +
      theme_bw() + xlab(NULL) +
      guides(color=guide_legend(title = NULL, nrow=2,byrow=TRUE), fill = "none") +
      coord_cartesian(xlim=lubridate::as_date(c(trunc_date, sim_end_date))) +
      facet_wrap(~age_group, ncol = 1, scales = "free_y") +
      theme(legend.position = "bottom", legend.text = element_text(size=10),
            axis.text.x = element_text(size=6, angle = 45)) +
      ggtitle("Deaths")
    plot(cum_st_plt_death)
  }
}
dev.off()






# WHERE SAVED
message(paste0("\nPlots created in: \n\n  ", stplot_fname_nosqrt, " \n  ", stplot_fname_sqrt, " \n  ", stplot_fname_nosqrt_age, "\n\n"))
