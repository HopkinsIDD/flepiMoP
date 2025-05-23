
start_time <- Sys.time()

# LIBRARIES AND DIRECTORIES -----------------------------------------------

library(ellipse)
library(progress)
library(cowplot)
library(arrow)
library(lubridate)
library(zoo)

# s3 name
s3_name <- "idd-inference-runs"

# PULL GEODATA ------------------------------------------------------------

# Pull in subpop data
geodata_states <- read.csv(paste0("./data/",
                           config$subpop_setup$geodata)) %>%
  mutate(subpop = stringr::str_pad(subpop, width = 5, side = "left", pad = "0"))

# PULL OUTCOMES FROM S3 ---------------------------------------------------

# List of outcomes to pull
outcomes_list <-
  c("hnpi", "hpar", "llik", "seed", "seir", "snpi", "spar")

# Download all final outcomes from AWS buckets
for (i in 1:length(outcomes_list)) {
  sys_call_s3 <-
    paste0(
      'aws s3 cp --recursive s3://',
      s3_name, "/",
      config$name,
      "-",
      scenario_s3_buckets,
      '/',
      "model_output",
      '/',
      outcomes_list[i],
      ' ',
      scenario_dir,
      '/',
      outcomes_list[i],
      ' --exclude="*" --include="*/final/*"'
    )
  system(sys_call_s3)
}

# Download intermediate likelihoods from AWS buckets
for (i in 1:length(outcomes_list)) {
  sys_call_s3 <-
    paste0(
      'aws s3 cp --recursive s3://',
      s3_name, "/",
      config$name,
      "-",
      scenario_s3_buckets,
      '/',
      "model_output",
      '/',
      "llik",
      ' ',
      scenario_dir,
      '/',
      "llik",
      ' --exclude="*" --include="*/intermediate/*"'
    )
  system(sys_call_s3)
}

# FUNCTIONS ---------------------------------------------------------------

import_s3_outcome <- function(scn_dir, outcome, global_opt, final_opt){
  dir_ <- paste0(scn_dir, "/",
                 outcome, "/",
                 config$name, "/",
                 config$interventions$scenarios, "/",
                 config$outcome_modifiers$scenarios)
  subdir_ <- paste0(dir_, "/", list.files(dir_),
                    "/",
                    global_opt,
                    "/",
                    final_opt)
  subdir_list <- list.files(subdir_)

  out_ <- NULL
  total <- length(subdir_list)
  pb <- txtProgressBar(min=0, max=total, style = 3)

  print(paste0("Importing ", outcome, " files (n = ", total, "):"))

  for (i in 1:length(subdir_list)) {
    if(any(grepl("parquet", subdir_list))){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/"))
    }
    if(outcome == "hosp"){
      dat <- arrow::read_parquet(paste(subdir_, subdir_list[i], sep = "/")) %>%
        select(date, subpop, incidI, incidC, incidH, incidD)
    }
    if(any(grepl("csv", subdir_list))){
      dat <- read.csv(paste(subdir_, subdir_list[i], sep = "/"))
    }
    if(final_opt == "final"){
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
    }
    if(final_opt == "intermediate"){
      dat$slot <- as.numeric(str_sub(subdir_list[i], start = 1, end = 9))
      dat$block <- as.numeric(str_sub(subdir_list[i], start = 11, end = 19))
    }
    out_ <- rbind(out_, dat)

    # Increase the amount the progress bar is filled by setting the value to i.
    setTxtProgressBar(pb, value = i)
  }
  close(pb)
  return(out_)
}

# IMPORT OUTCOMES ---------------------------------------------------------

# Current workaround for issues with `shared` directory
# work_dir <- scenario_dir
work_dir <- paste0(getwd(), "/", scenario_dir)

hnpi <- import_s3_outcome(work_dir, "hnpi", "global", "final") %>%
  full_join(geodata_states, by = "subpop")
hosp <- import_s3_outcome(work_dir, "hosp", "global", "final") %>%
  full_join(geodata_states, by = "subpop")
hpar <- import_s3_outcome(work_dir, "hpar", "global", "final") %>%
  full_join(geodata_states, by = "subpop")
llik <- import_s3_outcome(work_dir, "llik", "global", "final") %>%
  full_join(geodata_states, by = "subpop")
global_int_llik <- import_s3_outcome(work_dir, "llik", "global", "intermediate") %>%
  full_join(geodata_states, by = "subpop")
chimeric_int_llik <- import_s3_outcome(work_dir, "llik", "chimeric", "intermediate") %>%
  full_join(geodata_states, by = "subpop")
seed <- import_s3_outcome(work_dir, "seed", "global", "final") %>%
  mutate(subpop = stringr::str_pad(subpop, width = 5, side = "left", pad = "0")) %>%
  full_join(geodata_states, by = "subpop")
snpi <- import_s3_outcome(work_dir, "snpi", "global", "final") %>%
  full_join(geodata_states, by = "subpop")
spar <- import_s3_outcome(work_dir, "spar", "global", "final")

# DERIVED OBJECTS ---------------------------------------------------------

bind_hnpi_llik <- full_join(x = hnpi, y = llik)
if(all(!is.na(hnpi$modifier_name))){
  var_bind_hnpi_llik <- bind_hnpi_llik %>%
    group_by(modifier_name) %>%
    summarize(var = var(value)) %>%
    filter(var > 0.0001)
  pivot_bind_hnpi_llik <- bind_hnpi_llik %>%
    dplyr::filter(modifier_name %in% var_bind_hnpi_llik$modifier_name) %>%
    pivot_wider(names_from = modifier_name, values_from = value)
}

bind_hosp_llik <- full_join(x = hosp, y = llik) %>%
  pivot_longer(incidI:incidD, names_to = "outcome", values_to = "value")

bind_hpar_llik <- full_join(x = hpar, y = llik) %>%
  dplyr::filter(quantity == "probability")
var_bind_hpar_llik <- bind_hpar_llik %>%
  group_by(outcome) %>%
  summarize(var = var(value)) %>%
  filter(var > 0.0001)
pivot_bind_hpar_llik <- bind_hpar_llik %>%
  dplyr::filter(outcome %in% var_bind_hpar_llik$outcome) %>%
  pivot_wider(names_from = outcome, values_from = value)

int_llik <- rbind(global_int_llik %>%
                    mutate(type = "global"),
                  chimeric_int_llik %>%
                    mutate(type = "chimeric"))

bind_snpi_llik <- full_join(x = snpi, y = llik)
var_bind_snpi_llik <- bind_snpi_llik %>%
  group_by(modifier_name) %>%
  summarize(var = var(value)) %>%
  filter(var > 0.0001)
pivot_bind_snpi_llik <- bind_snpi_llik %>%
  dplyr::filter(modifier_name %in% var_bind_snpi_llik$modifier_name) %>%
  pivot_wider(names_from = modifier_name, values_from = value)

bind_spar_llik <- full_join(x = spar, y = llik)
var_bind_spar_llik <- bind_spar_llik %>%
  group_by(parameter) %>%
  summarize(var = var(value)) %>%
  filter(var > 0.0001)
pivot_bind_spar_llik <- bind_spar_llik %>%
  dplyr::filter(parameter %in% var_bind_spar_llik$parameter) %>%
  pivot_wider(names_from = parameter, values_from = value)

# ANALYSES ----------------------------------------------------------------

USPS <- geodata_states %>%
  arrange(USPS) %>%
  pull(USPS)

# llik
all_ll_plot <- llik %>%
  ggplot(aes(x = USPS, y = ll)) +
  geom_violin(scale = "width") +
  geom_jitter(shape = ".", height = 0, width = 0.2) +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(y = "log likelihood", label = "Convergence of likelihood across slots")

accept_plot <- llik %>%
  ggplot(aes(x = slot, y = accept_avg)) +
  geom_point() +
  theme_bw(base_size = 10) +
  labs(y = "average acceptance rate", label = "Acceptance rate across slots")

hnpi_plot <- list()
hnpi_llik_plot <- list()
hosp_llik_plot <- list()
hpar_plot <- list()
hpar_llik_plot <- list()
int_llik_plot <- list()
seed_plot <- list()
snpi_plot <- list()
snpi_llik_plot <- list()
spar_plot <- list()
spar_llik_plot <- list()
state_plot1 <- list()
state_plot2 <- list()
state_plot3 <- list()
state_plot4 <- list()
state_plot5 <- list()

pb2 <- txtProgressBar(min=0, max=length(USPS), style = 3)

for(i in 1:length(USPS)){

  state <- USPS[i]

  print(paste0("Preparing plots for ", state))

  # hnpi
  if(all(is.na(hnpi$modifier_name))){
    print("hnpi files are empty")
  } else {
    hnpi_plot[[i]] <- pivot_bind_hnpi_llik %>%
      filter(USPS == state) %>%
      dplyr::select(ll, all_of(var_bind_hnpi_llik$modifier_name)) %>%
      pivot_longer(cols = all_of(var_bind_hnpi_llik$modifier_name)) %>%
      drop_na(value) %>%
      ggplot(aes(x = name, y = value)) +
      geom_violin(scale = "width") +
      geom_jitter(aes(group = name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.8) +
      theme_bw(base_size = 10) +
      theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
      scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
      labs(x = "parameter", title = paste0(state, " hnpi values"))

    hnpi_llik_plot[[i]] <- pivot_bind_hnpi_llik %>%
      filter(USPS == state) %>%
      dplyr::select(ll, all_of(var_bind_hnpi_llik$modifier_name)) %>%
      pivot_longer(cols = all_of(var_bind_hnpi_llik$modifier_name)) %>%
      drop_na(value) %>%
      ggplot(aes(x = value, y = ll)) +
      geom_point(size = 0.5, alpha = 0.8) +
      facet_wrap(~name, scales = "free_x") +
      geom_smooth(formula = y ~ x, method = "lm", se = FALSE) +
      theme_bw(base_size = 10) +
      labs(y = "log likelihood", title = paste0(state, " hnpi correlation with likelihood"))
    }

  # hosp
  state_llik_rank <- llik %>%
    dplyr::filter(USPS == state) %>%
    mutate(rank = rank(desc(ll))) %>%
    arrange(rank)

  filter_state_hosp <- bind_hosp_llik %>%
    filter(USPS == state,
           slot %in% c(head(state_llik_rank, 5)$slot, tail(state_llik_rank, 5)$slot)) %>%
    mutate(llik_bin = case_when(slot %in% head(state_llik_rank, 5)$slot ~ "top",
                                slot %in% tail(state_llik_rank, 5)$slot ~ "bottom"))

  filter_gt_data <- gt_data %>%
    filter(USPS == state) %>%
    select(USPS, subpop, date, dplyr::contains("incid") & !dplyr::contains("_")) %>%
    pivot_longer(dplyr::contains('incid'), names_to = "outcome", values_to = "value") %>%
    mutate(week = lubridate::week(date)) %>%
    group_by(outcome, week) %>%
    mutate(rollmean = zoo::rollmean(x = value, k = 7, fill = NA))

  hosp_llik_plot[[i]] <- ggplot() +
    geom_line(data = filter_state_hosp,
              aes(x = as.POSIXct(date), y = value, group = slot, color = ll, linetype = llik_bin)) +
    scale_x_datetime(date_breaks = "1 month", date_labels = "%b %Y", name = "date") +
    scale_y_sqrt(name = "count") +
    scale_linetype_manual(values = c(2, 1), name = "likelihood\nbin") +
    scale_color_viridis_c(option = "D", name = "log\nlikelihood") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
    ggtitle(paste0(state, " incidence")) +
    facet_grid(outcome~., scales = "free_y") +
    geom_point(data = filter_gt_data,
               aes(x = as.POSIXct(date), y = value),
               pch = 1,
               size = 0.25, color = "grey50") +
    geom_line(data = filter_gt_data %>% drop_na(rollmean),
              aes(x = as.POSIXct(date), y = rollmean), color = "red")

  # hpar
  if(all(is_empty(var_bind_hpar_llik$name))){
    print("no varying hpar outcomes to plot")
  } else {
    hpar_plot[[i]] <- pivot_bind_hpar_llik %>%
      filter(USPS == state) %>%
      dplyr::select(ll, all_of(var_bind_hpar_llik$outcome)) %>%
      pivot_longer(cols = all_of(var_bind_hpar_llik$outcome)) %>%
      drop_na(value) %>%
      ggplot(aes(x = name, y = value)) +
      geom_violin(scale = "width") +
      geom_jitter(aes(group = name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.8) +
      theme_bw(base_size = 10) +
      theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
      scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
      labs(x = "parameter", title = paste0(state, " hpar probability values"))

    hpar_llik_plot[[i]] <- pivot_bind_hpar_llik %>%
      filter(USPS == state) %>%
      dplyr::select(ll, all_of(var_bind_hpar_llik$outcome)) %>%
      pivot_longer(cols = all_of(var_bind_hpar_llik$outcome)) %>%
      drop_na(value) %>%
      ggplot(aes(x = value, y = ll)) +
      geom_point(size = 0.5, alpha = 0.8) +
      facet_wrap(~name, scales = "free_x") +
      geom_smooth(formula = y ~ x, method = "lm", se = FALSE) +
      theme_bw(base_size = 10) +
      theme(strip.text = element_text(size = 4)) +
      labs(y = "log likelihood", title = paste0(state, " hpar correlation with likelihood"))
  }

  # llik
  int_llik_plot[[i]] <- int_llik %>%
    filter(USPS == state,
           between(slot, 1, 8)) %>%
    ggplot() +
    geom_point(aes(x = block, y = ll, linetype = type, color = factor(slot))) +
    geom_step(aes(x = block, y = ll, linetype = type, color = factor(slot))) +
    scale_color_brewer(palette = "Dark2", name = "slot") +
    theme_bw(base_size = 10) +
    labs(y = "log likelihood", title = paste0(state, " global and chimeric intermediate likelihoods"))

  # seed
  seed_plot[[i]] <- seed %>%
    filter(destination_infection_stage == "E",
           USPS == state) %>%
    ggplot(aes(x = as.Date(date), y = amount, color = destination_variant_type)) +
    geom_count(alpha = 0.8) +
    scale_x_date(date_breaks = "1 week", date_labels = "%Y-%m-%d") +
    scale_color_brewer(palette = "Dark2", name = "variant") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    labs(x = "seeding date", title = paste0(state, " seeding across all slots"))

  # snpi
  snpi_plot[[i]] <- pivot_bind_snpi_llik %>%
    filter(USPS == state) %>%
    dplyr::select(ll, all_of(var_bind_snpi_llik$modifier_name)) %>%
    pivot_longer(cols = all_of(var_bind_snpi_llik$modifier_name)) %>%
    drop_na(value) %>%
    ggplot(aes(x = name, y = value)) +
    geom_violin(scale = "width") +
    geom_jitter(aes(group = name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.8) +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
    labs(x = "value", title = paste0(state, " snpi values"))

  snpi_llik_plot[[i]] <- pivot_bind_snpi_llik %>%
    filter(USPS == state) %>%
    dplyr::select(ll, all_of(var_bind_snpi_llik$modifier_name)) %>%
    pivot_longer(cols = all_of(var_bind_snpi_llik$modifier_name)) %>%
    drop_na(value) %>%
    ggplot(aes(x = value, y = ll)) +
    geom_point(size = 0.5, alpha = 0.8) +
    facet_wrap(~name, scales = "free_x") +
    geom_smooth(formula = y ~ x, method = "lm", se = FALSE) +
    theme_bw(base_size = 10) +
    labs(y = "log likelihood", title = paste0(state, " snpi correlation with likelihood"))

  # spar
  spar_plot[[i]] <- pivot_bind_spar_llik %>%
    filter(USPS == state) %>%
    dplyr::select(ll, all_of(var_bind_spar_llik$parameter)) %>%
    pivot_longer(cols = all_of(var_bind_spar_llik$parameter)) %>%
    drop_na(value) %>%
    ggplot(aes(x = name, y = value)) +
    geom_violin(scale = "width") +
    geom_jitter(aes(group = name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.8) +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
    labs(x = "parameter", title = paste0(state, " spar parameter values"))

  spar_llik_plot[[i]] <- pivot_bind_spar_llik %>%
    filter(USPS == state) %>%
    dplyr::select(ll, all_of(var_bind_spar_llik$parameter)) %>%
    pivot_longer(cols = all_of(var_bind_spar_llik$parameter)) %>%
    drop_na(value) %>%
    ggplot(aes(x = value, y = ll)) +
    geom_point(size = 0.5, alpha = 0.8) +
    facet_wrap(~name, scales = "free_x") +
    geom_smooth(formula = y ~ x, method = "lm", se = FALSE) +
    theme_bw(base_size = 10) +
    labs(y = "log likelihood", title = paste0(state, " spar correlation with likelihood"))

  # Combined output plots
  state_plot1[[i]] <- plot_grid(int_llik_plot[[i]],
                                seed_plot[[i]],
                                nrow = 2, ncol = 1)
  if(all(is.na(hnpi$modifier_name))){
    state_plot2[[i]] <- NA
  } else {
    state_plot2[[i]] <- plot_grid(hnpi_plot[[i]],
                                  hnpi_llik_plot[[i]],
                                  nrow = 2, ncol = 1)
  }
  if(all(is_empty(var_bind_hpar_llik$name))){
    state_plot3[[i]] <- NA
  } else {
    state_plot3[[i]] <- plot_grid(hpar_plot[[i]],
                                  hpar_llik_plot[[i]],
                                  nrow = 2, ncol = 1)
  }
  state_plot4[[i]] <- plot_grid(snpi_plot[[i]],
                                snpi_llik_plot[[i]],
                                nrow = 2, ncol = 1)
  state_plot5[[i]] <- plot_grid(spar_plot[[i]],
                                spar_llik_plot[[i]],
                                nrow = 2, ncol = 1)

  # Increase the amount the progress bar is filled by setting the value to i.
  setTxtProgressBar(pb2, value = i)
}

# OUTPUT FILES ------------------------------------------------------------
pdf(file = paste0(round_directory, "/", fch_date, "_", disease, "_", smh_or_fch, "_R", round_num, "_", scenarios, "_", lubridate::ymd(today()), ".pdf"),
    height = 12,
    width = 9)
plot(all_ll_plot)
plot(accept_plot)
for(i in 1:length(USPS)){
  state <- USPS[i]
  print(paste0("Outputting plots for ", state))
  plot(state_plot1[[i]])
  if(all(!is.na(state_plot2[[i]]))){
    plot(state_plot2[[i]])
  }
  if(all(!is.na(state_plot3[[i]]))){
    plot(state_plot3[[i]])
  }
  plot(state_plot4[[i]])
  plot(state_plot5[[i]])
  plot(hosp_llik_plot[[i]])
}
dev.off()

end_time <- Sys.time()
print(end_time - start_time)
