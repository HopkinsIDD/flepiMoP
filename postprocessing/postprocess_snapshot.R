## Script runs automatically after batch run: for an inference run, compares outcomes to data

suppressMessages(library(parallel))
suppressMessages(library(foreach))
suppressMessages(library(inference))
suppressMessages(library(tidyverse))
suppressMessages(library(tidyr))
suppressMessages(library(doParallel))
suppressMessages(library(dplyr))
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(ggforce))
suppressMessages(library(ggforce))
suppressMessages(library(gridExtra))


options(readr.num_columns = 0)

option_list = list(
  optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH", Sys.getenv("CONFIG_PATH")), type='character', help="path to the config file"),
  optparse::make_option(c("-u","--run-id"), action="store", dest = "run_id", type='character', help="Unique identifier for this run", default = Sys.getenv("FLEPI_RUN_INDEX",flepicommon::run_id())),
  optparse::make_option(c("-R", "--results-path"), action="store", dest = "results_path",  type='character', help="Path for model output", default = Sys.getenv("FS_RESULTS_PATH", Sys.getenv("FS_RESULTS_PATH"))),
  # optparse::make_option(c("-p", "--flepimop-repo"), action="store", dest = "flepimop_repo", default=Sys.getenv("FLEPI_PATH", Sys.getenv("FLEPI_PATH")), type='character', help="path to the flepimop repo"),
  optparse::make_option(c("-o", "--select-outputs"), action="store", dest = "select_outputs", default=Sys.getenv("OUTPUTS","hosp, hnpi, snpi, llik"), type='character', help="A list of outputs to plot.")
)

parser=optparse::OptionParser(option_list=option_list)
opt = optparse::parse_args(parser, convert_hyphens_to_underscores = TRUE)

print("Generating plots")
print(paste("Config:", opt$config, 'for', opt$run_id, 'saved in', opt$results_path))

if(opt$config == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a config YAML file with either -c option or CONFIG_PATH environment variable."
  ))
}

if(opt$results_path == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a results path with either -P option or FS_RESULTS_PATH environment variable."
  ))
}

# if(opt$flepimop_repo == ""){
#   optparse::print_help(parser)
#   stop(paste(
#     "Please specify a flepiMoP path with -p option or FLEPI_PATH environment variable."
#   ))
# }

print(paste('Processing run ',opt$results_path))

opt$select_outputs <- strsplit(opt$select_outputs, ', ')[[1]]
print(opt$select_outputs)
## SETUP -----------------------------------------------------------------------

config <- flepicommon::load_config(opt$config)

# Pull in subpop data
geodata <- setDT(read.csv(file.path(config$subpop_setup$geodata))) %>%
  .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")]

subpops <- unique(geodata$subpop)

## gt_data MUST exist directly after a run
gt_data <- data.table::fread(config$inference$gt_data_path) %>%
  .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")]

# store list of files to save
files_ <- c()
dir.create("pplot")

pdf.options(useDingbats = TRUE)

# FUNCTIONS ---------------------------------------------------------------

import_model_outputs <- function(scn_dir, outcome, global_opt, final_opt, run_id = opt$run_id,
                                 lim_hosp = c("date", 
                                              sapply(1:length(names(config$inference$statistics)), function(i) purrr::flatten(config$inference$statistics[i])$sim_var),
                                              "subpop")){
  # model_output/USA_inference_fake/20231016_204739CEST/hnpi/global/intermediate/000000001.000000001.000000030.20231016_204739CEST.hnpi.parquet
  dir_ <- file.path(scn_dir, 
                 paste0(config$name, "_", config$seir_modifiers$scenarios[scenario_num], "_", config$outcome_modifiers$scenarios[scenario_num]),
                 run_id, 
                 outcome)
  subdir_ <- paste0(dir_, "/",
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
        select(all_of(lim_hosp))
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
## TO DO: SYNTHESISE WITH WHAT ALISON DID
scenario_num <- length(config$seir_modifiers$scenarios)
setup_prefix <- paste0(config$name,
                       ifelse(is.null(config$seir_modifiers$scenarios),"",paste0("_",config$seir_modifiers$scenarios[scenario_num])),
                       ifelse(is.null(config$outcome_modifiers$scenarios),"",paste0("_",config$outcome_modifiers$scenarios[scenario_num])))

res_dir <- file.path(opt$results_path, ifelse(is.null(config$model_output_dirname),"model_output", config$model_output_dirname))
print(res_dir)

results_filelist <- file.path(res_dir, 
                                 paste0(config$name, "_", config$seir_modifiers$scenarios[scenario_num], "_", config$outcome_modifiers$scenarios[scenario_num]),
                                 opt$run_id)

model_outputs <- list.files(results_filelist)[match(opt$select_outputs,
                                           list.files(results_filelist))]
if("llik" %in% model_outputs){
  # opts <- c("global", "chimeric").  ## NOT DOING ANYTHING WITH INT YET, TO DO
  # int_llik <- lapply(opts, function(i) setDT(import_model_outputs(res_dir, "llik", i, "intermediate")))
}else{
  model_outputs <- c(model_outputs, "llik")
}
start_time <- Sys.time()

outputs_global <- lapply(model_outputs, function(i) setDT(import_model_outputs(res_dir, outcome = i, 'global', 'final')))
names(outputs_global) <- model_outputs

end_time <- Sys.time()
print(end_time - start_time)


## HOSP --------------------------------------------------------------------
# Compare inference statistics sim_var to data_var
if("hosp" %in% model_outputs){
  
  gg_cols <- 2
  num_nodes <- length(unique(outputs_global$hosp %>% .[,subpop]))
  pdf_dims <- data.frame(width = gg_cols*2, length = num_nodes/gg_cols * 2)
  
  fname <- paste0("pplot/hosp_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)
  # pdf(fname, width = 20, height = 18)
  # pdf(fname)
  fit_stats <- names(config$inference$statistics)
  subpop_names <- unique(outputs_global$hosp %>% .[,subpop])
  
  for(i in 1:length(fit_stats)){
    statistics <- purrr::flatten(config$inference$statistics[i])
    cols_sim <- c("date", statistics$sim_var, "subpop","slot")
    cols_data <- c("date", "subpop", statistics$data_var)
    hosp_outputs_global_tmp <- copy(outputs_global$hosp)[,..cols_sim]
    
    # aggregate based on what is in the config
    df_sim <- lapply(subpop_names, function(y) {
      lapply(unique(outputs_global$hosp$slot), function(x)
        purrr::flatten_df(inference::getStats(
          hosp_outputs_global_tmp %>% .[subpop == y & slot == x] ,
          "date",
          "sim_var",
          stat_list = config$inference$statistics[i],
          start_date = config$start_date_groundtruth,
          end_date = config$end_date_groundtruth
        )) %>% dplyr::mutate(subpop = y, slot = x)) %>% dplyr::bind_rows() 
    }) %>% dplyr::bind_rows() 
    
    df_data <- lapply(subpop_names, function(x) {
      purrr::flatten_df(
        inference::getStats(
          gt_data %>% .[subpop == x,..cols_data],
          "date",
          "data_var",
          stat_list = config$inference$statistics[i],
          start_date = config$start_date_groundtruth,
          end_date = config$end_date_groundtruth
        )) %>% dplyr::mutate(subpop = x) %>% 
        mutate(data_var = as.numeric(data_var)) }) %>% dplyr::bind_rows()
    
    
    ## summarize slots 
    # print(outputs_global$hosp %>%
    #   .[, ..cols_sim] %>%
    #   .[, date := lubridate::as_date(date)] %>%
    #   .[, as.list(quantile(get(statistics$sim_var), c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", "subpop")] %>%
    #   ggplot() + 
    #   geom_ribbon(aes(x = date, ymin = V1, ymax = V5), alpha = 0.1) +
    #   geom_ribbon(aes(x = date, ymin = V2, ymax = V4), alpha = 0.1) +
    #   geom_line(aes(x = date, y = V3)) + 
    #   geom_point(data = gt_data %>%
    #                .[, ..cols_data],
    #              aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
    #   facet_wrap(~subpop, scales = 'free', ncol = gg_cols) +
    #   labs(x = 'date', y = fit_stats[i], title = statistics$sim_var) +
    #   theme_classic()
    # )
    print(
      df_sim %>%
      setDT() %>%
      .[, date := lubridate::as_date(date)] %>%
      .[, as.list(quantile(sim_var, c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", "subpop")] %>%
      setnames(., paste0("V", 1:5), paste0("q", c(.05,.25,.5,.75,.95))) %>%
      ggplot() + 
      geom_ribbon(aes(x = date, ymin = q0.05, ymax = q0.95), alpha = 0.1) +
      geom_ribbon(aes(x = date, ymin = q0.25, ymax = q0.75), alpha = 0.1) +
      geom_line(aes(x = date, y = q0.5)) + 
      # if inference, plot gt along side
      geom_point(data = df_data,
                 aes(lubridate::as_date(date), data_var), color = 'firebrick', alpha = 0.2, size=1) +
      facet_wrap(~subpop, scales = 'free') +
      labs(x = 'date', y = fit_stats[i]) +
      theme_classic()
    )
    
    # ## plot all slots
    # print(outputs_global$hosp %>% 
    #   ggplot() +
    #   geom_line(aes(lubridate::as_date(date), get(sim_var), group = as.factor(slot)), alpha = 0.1) + 
    #   facet_wrap(~get(config$subpop_setup$subpop), scales = 'free') +
    #   geom_point(data = gt_data %>%
    #                .[, ..cols_data],
    #              aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
    #   theme_classic() + 
    #   labs(x = 'date', y = fit_stats[i]) +
    #   theme(legend.position="none")
    # )
    
    ## plot cumulatives
    print(
      df_sim %>%
        setDT() %>%
        .[, date := lubridate::as_date(date)] %>%
        .[, .(date, subpop, sim_var, slot)] %>%
        data.table::melt(., id.vars = c("date", "slot", "subpop")) %>%
        # dplyr::arrange(subpop, slot, date) %>% 
        .[, csum := cumsum(value), by = .(slot, subpop, variable)] %>%
        .[, as.list(quantile(csum, c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", config$subpop_setup$subpop)] %>%
        setnames(., paste0("V", 1:5), paste0("q", c(.05,.25,.5,.75,.95))) %>%
        ggplot() + 
        geom_ribbon(aes(x = date, ymin = q0.05, ymax = q0.95), alpha = 0.1) +
        geom_ribbon(aes(x = date, ymin = q0.25, ymax = q0.75), alpha = 0.1) +
        geom_line(aes(x = date, y = q0.5)) + 
        geom_point(data = df_data %>% setDT() %>%
                     .[, csum := cumsum(data_var) , by = .(subpop)],
                   aes(lubridate::as_date(date), csum), color = 'firebrick', alpha = 0.2, size=1) +
        facet_wrap(~subpop, scales = 'free') +
        # facet_wrap(~get(subpop), scales = 'free') +      
        labs(x = 'date', y = paste0("cumulative ", fit_stats[i])) +
        theme_classic() 
      # outputs_global$hosp %>%
      #       .[, ..cols_sim] %>%
      #       .[, date := lubridate::as_date(date)] %>%
      #       .[, csum := cumsum(get(statistics$sim_var)), by = .(subpop, slot)] %>%
      #       .[, as.list(quantile(csum, c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", "subpop")] %>%
      #       ggplot() + 
      #       geom_ribbon(aes(x = date, ymin = V1, ymax = V5), alpha = 0.1) +
      #       geom_ribbon(aes(x = date, ymin = V2, ymax = V4), alpha = 0.1) +
      #       geom_line(aes(x = date, y = V3)) + 
      #       geom_point(data = gt_data %>%
      #                    .[, ..cols_data] %>%
      #                    .[, csum := cumsum(replace_na(get(statistics$data_var), 0)) , by = .(subpop)]
      #                    ,
      #                  aes(lubridate::as_date(date), csum), color = 'firebrick', alpha = 0.1) + 
      #       facet_wrap(~subpop, scales = 'free', ncol = gg_cols) +
      #       labs(x = 'date', y = fit_stats[i], title = paste0("cumulative ", statistics$sim_var)) +
      #       theme_classic()
    )
    
  }
  dev.off()
  files_ <- c(files_, fname)
  
  ## hosp by highest and lowest llik
  
  if("llik" %in% model_outputs){
    llik_rank <- copy(outputs_global$llik) %>% 
      .[, .SD[order(ll)], subpop] 
    high_low_llik <- rbindlist(list(data.table(llik_rank, key = "subpop") %>%
                                      .[, head(.SD,5), by = subpop] %>% 
                                      .[, llik_bin := "top"], 
                                    data.table(llik_rank, key = "subpop") %>%
                                      .[, tail(.SD,5), by = subpop]%>% 
                                      .[, llik_bin := "bottom"])
    )
  }
  
  fname <- paste0("pplot/hosp_by_llik_mod_outputs_", opt$run_id,".pdf")
  # pdf_dims <- data.frame(width = gg_cols*2, length = num_nodes/gg_cols * 2)
  # pdf(fname, width = pdf_dims$width, height = pdf_dims$length)
  pdf(fname, width = 20, height = 10)

  for(i in 1:length(fit_stats)){
    statistics <- purrr::flatten(config$inference$statistics[i])
    cols_sim <- c("date", statistics$sim_var, "subpop","slot")
    cols_data <- c("date", "subpop", statistics$data_var)
    hosp_outputs_global_tmp <- copy(outputs_global$hosp)[,..cols_sim]
    
    if("llik" %in% model_outputs){
      # high_low_hosp_llik <- copy(outputs_global$hosp) %>% 
      #   .[high_low_llik, on = c("slot", "subpop"), allow.cartesian = TRUE]
      
      # aggregate simulation output and data by time based on what is in the config
      df_sim <- lapply(subpop_names, function(y) {
        lapply(unique(outputs_global$hosp$slot), function(x)
          purrr::flatten_df(inference::getStats(
            hosp_outputs_global_tmp %>% .[subpop == y & slot == x] ,
            "date",
            "sim_var",
            stat_list = config$inference$statistics[i],
            start_date = config$start_date_groundtruth,
            end_date = config$end_date_groundtruth
          )) %>% dplyr::mutate(subpop = y, slot = x)) %>% dplyr::bind_rows() 
      }) %>% dplyr::bind_rows() %>% setDT()
      
      df_data <- lapply(subpop_names, function(x) {
        purrr::flatten_df(
          inference::getStats(
            gt_data %>% .[subpop == x,..cols_data],
            "date",
            "data_var",
            stat_list = config$inference$statistics[i],
            start_date = config$start_date_groundtruth,
            end_date = config$end_date_groundtruth
          )) %>% dplyr::mutate(subpop = x) %>% 
          dplyr::mutate(data_var = as.numeric(data_var)) %>%
          dplyr::mutate(date = lubridate::as_date(date)) }) %>%
        dplyr::bind_rows() %>% setDT()
      
      # add likelihood ranking to simulation output
      high_low_hosp_llik <- df_sim %>% 
        .[high_low_llik, on = c("slot", "subpop"), allow.cartesian=TRUE] %>% # right join by "on" variables
        .[subpop != "Total"]
      
      hosp_llik_plots <- lapply(unique(high_low_hosp_llik %>% .[, subpop]),
                           function(e){
                             high_low_hosp_llik %>%
                               .[subpop == e] %>%
                               .[, date := lubridate::as_date(date)] %>%
                               ggplot() + 
                               geom_line(aes(x = date, y = sim_var, group = slot, color = ll)) + 
                               scale_linetype_manual(values = c(1, 2), name = "likelihood\nbin") +
                               scale_color_viridis_c(option = "D", name = "log\nlikelihood") +
                               geom_point(data = df_data %>% .[subpop == e],
                                          aes(lubridate::as_date(date), data_var), color = 'firebrick', alpha = 0.2, size=1) +
                               facet_wrap(~subpop, scales = 'free') +
                               labs(x = 'date', y = fit_stats[i]) +
                               theme_classic() + 
                               theme(legend.key.size = unit(0.2, "cm"))
                             # high_low_hosp_llik %>%
                             #   .[, date := lubridate::as_date(date)] %>%
                             #   .[subpop == e] %>%
                             #   ggplot() +
                             #   geom_line(aes(lubridate::as_date(date), get(statistics$data_var), 
                             #                 group = slot, color = ll))+#, linetype = llik_bin)) +
                             #   # scale_linetype_manual(values = c(1, 2), name = "likelihood\nbin") +
                             #   scale_color_viridis_c(option = "D", name = "log\nlikelihood") +
                             #   geom_point(data = gt_data %>%
                             #                .[, ..cols_data] %>%
                             #                .[subpop == e] ,
                             #              aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
                             #   facet_wrap(~subpop, scales = 'free', ncol = gg_cols) +
                             #   labs(x = 'date', y = fit_stats[i]) + #, title = paste0("top 5, bottom 5 lliks, ", statistics$sim_var)) +
                             #   theme_classic() +
                             #   guides(linetype = 'none')
                           }
      )
      
      print(do.call("grid.arrange", c(hosp_llik_plots, ncol=gg_cols)))

    }#end if
  }#end loop inference statistic
  dev.off()
  files_ <- c(files_, fname)
  
}


## HNPI --------------------------------------------------------------------
if("hnpi" %in% model_outputs){
  
  gg_cols <- 4
  num_nodes <- length(unique(outputs_global$hnpi %>% .[,subpop]))
  # pdf_dims <- data.frame(width = gg_cols*3, length = num_nodes/gg_cols * 2)
  pdf_dims <- data.frame(width = 20, length = 10)

  fname <- paste0("pplot/hnpi_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)
  
  
  hnpi_plots <- lapply(sort(unique(outputs_global$hnpi %>% .[, subpop])),
         function(i){
           outputs_global$hnpi %>%
             .[outputs_global$llik, on = c("subpop", "slot")] %>%
             .[subpop == i] %>%
             ggplot(aes(modifier_name,value)) + 
             geom_violin() +
             geom_jitter(aes(group = modifier_name, color = ll), size = 0.6, height = 0, width = 0.2, alpha = 1) +
             facet_wrap(~subpop, scales = 'free') +
             scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
	     theme_classic() +
	     theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6))
         }
  )
  
  print(do.call("grid.arrange", c(hnpi_plots, ncol=gg_cols)))
  dev.off()
  
  files_ <- c(files_,fname)
}

## HPAR --------------------------------------------------------------------
if("hpar" %in% model_outputs){

  
}

## LLIK --------------------------------------------------------------------
if("llik" %in% model_outputs){
  
}

## SEED --------------------------------------------------------------------
## THIS IS BROKEN FOR > ONE COMPARTMENT. TO FIX
if("seed" %in% model_outputs){ ## TO DO: MODIFIED FOR WHEN LOTS MORE SEEDING COMPARTMENTS
  
  fname <- paste0("pplot/seed_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = 15, height = 45)
  
  dest <- unname(config$seeding$seeding_compartments)
  source_comp <- lapply(dest, function(i) i$source_compartment)
  dest_comp <- lapply(dest, function(i) i$destination_compartment)
  compartments <- names(config$compartments)
  destination_columns <- paste0("destination_", compartments)

  tmp_ <- paste("+", destination_columns, collapse = "")
  facet_formula <- paste("~", substr(tmp_, 2, nchar(tmp_)))
  
  seed_plots <- lapply(sort(unique(setDT(geodata) %>% .[, subpop])),
                       function(i){
                         outputs_global$seed %>%
                           .[subpop == i] %>%
                           ggplot(aes(x = as.Date(date), y = amount)) +
                           facet_wrap(as.formula(facet_formula), scales = 'free', ncol=1,
                                      labeller = label_wrap_gen(multi_line=FALSE)) +
                           geom_count(alpha = 0.8) +
                           labs(x = 'date', title = i) +
                           theme_classic()
                       }
  )
  
  print(do.call("grid.arrange", c(seed_plots, ncol=4)))
  
  # 
  # for(i in unique(outputs_global$seed$subpop)){
  #   print(outputs_global$seed %>%
  #     .[subpop == i] %>%
  #       ggplot(aes(x = as.Date(date), y = amount)) +
  #       facet_wrap(as.formula(facet_formula), scales = 'free', ncol=1, 
  #                  labeller = label_wrap_gen(multi_line=FALSE)) +
  #       geom_count(alpha = 0.8) +
  #       labs(x = 'date', title = i) +
  #       theme_classic()
  #   )
  # }
  
  dev.off()
  files_ <- c(files_, fname)
}

## SEIR --------------------------------------------------------------------
if("seir" %in% model_outputs){
  print("TO DO")
  
}

## SNPI --------------------------------------------------------------------
if("snpi" %in% model_outputs){
  
  gg_cols <- 4
  num_nodes <- length(unique(outputs_global$snpi %>% .[,subpop]))
  pdf_dims <- data.frame(width = gg_cols*4, length = num_nodes/gg_cols * 3)
  
  fname <- paste0("pplot/snpi_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)

  node_names <- unique(sort(outputs_global$snpi %>% .[ , subpop]))
  node_names <- c(node_names[str_detect(node_names,",")], node_names[!str_detect(node_names,",")])
  
  snpi_plots <- lapply(node_names,
                       function(i){
                         if(!grepl(',', i)){
                           outputs_global$snpi %>%
                             .[outputs_global$llik, on = c("subpop", "slot")] %>%
                             .[subpop == i] %>%
                             ggplot(aes(modifier_name,value)) + 
                             geom_violin() + 
                             geom_jitter(aes(group = modifier_name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.5) +
                             theme_bw(base_size = 10) +
                             theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
                             scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
                             labs(x = "parameter", title = i)
                         }else{
                           nodes_ <- unlist(strsplit(i,","))
                           ll_across_nodes <- 
                             outputs_global$llik %>% 
                             .[subpop %in% nodes_] %>%
                             .[, .(ll_sum = sum(ll)), by = .(slot)]
                           
                           outputs_global$snpi %>%
                             .[subpop == i] %>%
                             .[ll_across_nodes, on = c("slot")] %>%
                             ggplot(aes(modifier_name,value)) + 
                             geom_violin() + 
                             geom_jitter(aes(group = modifier_name, color = ll_sum), size = 0.5, height = 0, width = 0.2, alpha = 0.5) +
                             theme_bw(base_size = 10) +
                             theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
                             scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
                             labs(x = "parameter")
                           }
                       }
                       )
  
  print(do.call("grid.arrange", c(snpi_plots, ncol=gg_cols)))
  
  dev.off()
  
  files_ <- c(files_, fname)
}

## SPAR --------------------------------------------------------------------
if("spar" %in% model_outputs){
  print("TO DO")
  
}

# file.copy(from = files_,
#           to = file.path(gt_data_path, "pplot",basename(files_)),
#           overwrite = TRUE)

