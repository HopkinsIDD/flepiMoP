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
  optparse::make_option(c("-p", "--flepimop-repo"), action="store", dest = "flepimop_repo", default=Sys.getenv("FLEPI_PATH", Sys.getenv("FLEPI_PATH")), type='character', help="path to the flepimop repo"),
  optparse::make_option(c("-o", "--select-outputs"), action="store", dest = "select_outputs", default=Sys.getenv("OUTPUTS","hosp, hpar, snpi, hnpi, llik"), type='character', help="path to the flepimop repo")
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

if(opt$flepimop_repo == ""){
  optparse::print_help(parser)
  stop(paste(
    "Please specify a flepiMoP path with -p option or FLEPI_PATH environment variable."
  ))
}

print(paste('Processing run ',opt$results_path))

opt$select_outputs <- strsplit(opt$select_outputs, ', ')[[1]]
print(opt$select_outputs)
## SETUP -----------------------------------------------------------------------

config <- flepicommon::load_config(opt$config)

# Pull in subpop data
geodata <- setDT(read.csv(file.path(config$data_path, config$spatial_setup$geodata)))

## gt_data MUST exist directly after a run
gt_data <- data.table::fread(config$inference$gt_data_path) %>%
  .[, subpop := stringr::str_pad(FIPS, width = 5, side = "left", pad = "0")]

# store list of files to save
files_ <- c()
dir.create("pplot")

pdf.options(useDingbats = TRUE)

# FUNCTIONS ---------------------------------------------------------------

import_model_outputs <- function(scn_dir, outcome, global_opt, final_opt,
                                 lim_hosp = c("date", 
                                              sapply(1:length(names(config$inference$statistics)), function(i) purrr::flatten(config$inference$statistics[i])$sim_var),
                                              config$spatial_setup$subpop)){
  dir_ <- paste0(scn_dir, "/",
                 outcome, "/",
                 config$name, "/",
                 config$interventions$scenarios, "/",
                 config$outcomes$scenarios)
  subdir_ <- paste0(dir_, "/", list.files(dir_),
                    "/",
                    global_opt,
                    "/",
                    final_opt)
  subdir_list <- list.files(subdir_)
  
  out_ <- NULL
  total <- length(subdir_list)
  
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
    
  }
  return(out_)
}

# IMPORT OUTCOMES ---------------------------------------------------------

res_dir <- file.path(opt$results_path, config$model_output_dirname)
print(res_dir)

model_outputs <- list.files(res_dir)[match(opt$select_outputs,list.files(res_dir))]
if("llik" %in% model_outputs){
  # opts <- c("global", "chimeric").  ## NOT DOING ANYTHING WITH INT YET, TO DO
  # int_llik <- lapply(opts, function(i) setDT(import_model_outputs(res_dir, "llik", i, "intermediate")))
}else{
  model_outputs <- c(model_outputs, "llik")
}
start_time <- Sys.time()

outputs_global <- lapply(model_outputs, function(i) setDT(import_model_outputs(res_dir, i, 'global', 'final')))
names(outputs_global) <- model_outputs

end_time <- Sys.time()
print(end_time - start_time)


## HOSP --------------------------------------------------------------------
# Compare inference statistics sim_var to data_var
if("hosp" %in% model_outputs){
  
  gg_cols <- 8
  num_nodes <- length(unique(outputs_global$hosp %>% .[,get(config$spatial_setup$subpop)]))
  pdf_dims <- data.frame(width = gg_cols*2, length = num_nodes/gg_cols * 2)
  
  fname <- paste0("pplot/hosp_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)
  fit_stats <- names(config$inference$statistics)
  
  for(i in 1:length(fit_stats)){
    statistics <- purrr::flatten(config$inference$statistics[i])
    cols_sim <- c("date", statistics$sim_var, config$spatial_setup$subpop,"slot")
    cols_data <- c("date", config$spatial_setup$subpop, statistics$data_var)
    ## summarize slots 
    print(outputs_global$hosp %>%
      .[, ..cols_sim] %>%
      .[, date := lubridate::as_date(date)] %>%
      { if(config$spatial_setup$subpop == 'subpop'){
        .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
      } %>% 
      { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
      } %>%
      .[, as.list(quantile(get(statistics$sim_var), c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", config$spatial_setup$subpop)] %>%
      ggplot() + 
      geom_ribbon(aes(x = date, ymin = V1, ymax = V5), alpha = 0.1) +
      geom_ribbon(aes(x = date, ymin = V2, ymax = V4), alpha = 0.1) +
      geom_line(aes(x = date, y = V3)) + 
      geom_point(data = gt_data %>%
                   .[, ..cols_data] %>%
                   { if(config$spatial_setup$subpop == 'subpop'){
                     .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
                   } %>% 
                   { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
                   } ,
                 aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
      facet_wrap(~get(config$spatial_setup$subpop), scales = 'free', ncol = gg_cols) +
      labs(x = 'date', y = fit_stats[i], title = statistics$sim_var) +
      theme_classic()
    )
    
    # ## plot all slots
    # print(outputs_global$hosp %>% 
    #   ggplot() +
    #   geom_line(aes(lubridate::as_date(date), get(sim_var), group = as.factor(slot)), alpha = 0.1) + 
    #   facet_wrap(~get(config$spatial_setup$subpop), scales = 'free') +
    #   geom_point(data = gt_data %>%
    #                .[, ..cols_data],
    #              aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
    #   theme_classic() + 
    #   labs(x = 'date', y = fit_stats[i]) +
    #   theme(legend.position="none")
    # )
    
    ## plot cumulatives
    print(outputs_global$hosp %>%
            .[, ..cols_sim] %>%
            .[, date := lubridate::as_date(date)] %>%
            { if(config$spatial_setup$subpop == 'subpop'){
              .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
            } %>% 
            { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
            } %>%
            .[, csum := cumsum(get(statistics$sim_var)), by = .(get(config$spatial_setup$subpop), slot)] %>%
            .[, as.list(quantile(csum, c(.05, .25, .5, .75, .95), na.rm = TRUE, names = FALSE)), by = c("date", config$spatial_setup$subpop)] %>%
            ggplot() + 
            geom_ribbon(aes(x = date, ymin = V1, ymax = V5), alpha = 0.1) +
            geom_ribbon(aes(x = date, ymin = V2, ymax = V4), alpha = 0.1) +
            geom_line(aes(x = date, y = V3)) + 
            geom_point(data = gt_data %>%
                         .[, ..cols_data] %>%
                         { if(config$spatial_setup$subpop == 'subpop'){
                           .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
                         } %>% 
                         { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
                         } %>%
                         .[, csum := cumsum(replace_na(get(statistics$data_var), 0)) , by = .(get(config$spatial_setup$subpop))]
                         ,
                       aes(lubridate::as_date(date), csum), color = 'firebrick', alpha = 0.1) + 
            facet_wrap(~get(config$spatial_setup$subpop), scales = 'free', ncol = gg_cols) +
            labs(x = 'date', y = fit_stats[i], title = paste0("cumulative ", statistics$sim_var)) +
            theme_classic()
    )
    
  }
  dev.off()
  files_ <- c(files_, fname)
  
  ## hosp by highest and lowest llik
  
  fname <- paste0("pplot/hosp_by_llik_mod_outputs_", opt$run_id,".pdf")
  pdf_dims <- data.frame(width = gg_cols*4, length = num_nodes/gg_cols * 3)
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)

  for(i in 1:length(fit_stats)){
    statistics <- purrr::flatten(config$inference$statistics[i])
    cols_sim <- c("date", statistics$sim_var, config$spatial_setup$subpop,"slot")
    cols_data <- c("date", config$spatial_setup$subpop, statistics$data_var)
    if("llik" %in% model_outputs){
      llik_rank <- copy(outputs_global$llik) %>% 
        .[, .SD[order(ll)], eval(config$spatial_setup$subpop)] 
      high_low_llik <- rbindlist(list(data.table(llik_rank, key = eval(config$spatial_setup$subpop)) %>%
                                        .[, head(.SD,5), by = eval(config$spatial_setup$subpop)] %>% 
                                        .[, llik_bin := "top"], 
                                      data.table(llik_rank, key = eval(config$spatial_setup$subpop)) %>%
                                        .[, tail(.SD,5), by = eval(config$spatial_setup$subpop)]%>% 
                                        .[, llik_bin := "bottom"])
      )
      
      high_low_hosp_llik <- copy(outputs_global$hosp) %>% 
        .[high_low_llik, on = c("slot", eval(config$spatial_setup$subpop))]
      
      hosp_llik_plots <- lapply(unique(high_low_hosp_llik %>% .[, get(config$spatial_setup$subpop)]),
                           function(e){
                             high_low_hosp_llik %>%
                               .[, date := lubridate::as_date(date)] %>%
                               { if(config$spatial_setup$subpop == 'subpop'){
                                 .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
                               } %>% 
                               .[get(config$spatial_setup$subpop) == e] %>%
                               { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
                               } %>%
                               ggplot() +
                               geom_line(aes(lubridate::as_date(date), get(statistics$data_var), 
                                             group = slot, color = ll, linetype = llik_bin)) +
                               scale_linetype_manual(values = c(1, 2), name = "likelihood\nbin") +
                               scale_color_viridis_c(option = "D", name = "log\nlikelihood") +
                               geom_point(data = gt_data %>%
                                            .[, ..cols_data] %>%
                                            { if(config$spatial_setup$subpop == 'subpop'){
                                              .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
                                            } %>% 
                                            .[get(config$spatial_setup$subpop) == e] %>%
                                            { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
                                            } ,
                                          aes(lubridate::as_date(date), get(statistics$data_var)), color = 'firebrick', alpha = 0.1) + 
                               facet_wrap(~get(config$spatial_setup$subpop), scales = 'free', ncol = gg_cols) +
                               labs(x = 'date', y = fit_stats[i]) + #, title = paste0("top 5, bottom 5 lliks, ", statistics$sim_var)) +
                               theme_classic() +
                               guides(linetype = 'none')
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
  num_nodes <- length(unique(outputs_global$hosp %>% .[,get(config$spatial_setup$subpop)]))
  pdf_dims <- data.frame(width = gg_cols*3, length = num_nodes/gg_cols * 2)
  
  fname <- paste0("pplot/hnpi_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)
  
  
  hnpi_plots <- lapply(sort(unique(outputs_global$hnpi %>% .[, get(config$spatial_setup$subpop)])),
         function(i){
           outputs_global$hnpi %>%
             .[outputs_global$llik, on = c(config$spatial_setup$subpop, "slot")] %>%
             { if(config$spatial_setup$subpop == 'subpop'){
               .[geodata %>% .[, subpop := stringr::str_pad(subpop, width = 5, side = "left", pad = "0")], on = .(subpop)]} 
             } %>% 
             .[get(config$spatial_setup$subpop) == i] %>%
             { if(config$spatial_setup$subpop == 'subpop'){ .[, subpop := USPS]} 
             } %>%
             ggplot(aes(npi_name,reduction)) + 
             geom_violin() +
             geom_jitter(aes(group = npi_name, color = ll), size = 0.6, height = 0, width = 0.2, alpha = 1) +
             facet_wrap(~get(config$spatial_setup$subpop), scales = 'free') +
             scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
             theme_classic()
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
  
  seed_plots <- lapply(sort(unique(setDT(geodata) %>% .[, get(config$spatial_setup$subpop)])),
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
  num_nodes <- length(unique(outputs_global$hosp %>% .[,get(config$spatial_setup$subpop)]))
  pdf_dims <- data.frame(width = gg_cols*4, length = num_nodes/gg_cols * 3)
  
  fname <- paste0("pplot/snpi_mod_outputs_", opt$run_id,".pdf")
  pdf(fname, width = pdf_dims$width, height = pdf_dims$length)

  node_names <- unique(sort(outputs_global$snpi %>% .[ , get(config$spatial_setup$subpop)]))
  node_names <- c(node_names[str_detect(node_names,",")], node_names[!str_detect(node_names,",")])
  
  snpi_plots <- lapply(node_names,
                       function(i){
                         if(!grepl(',', i)){
                           
                           i_lab <- ifelse(config$spatial_setup$subpop == 'subpop', geodata[subpop == i, USPS], i)
                             
                           outputs_global$snpi %>%
                             .[outputs_global$llik, on = c(config$spatial_setup$subpop, "slot")] %>%
                             .[get(config$spatial_setup$subpop) == i] %>%
                             ggplot(aes(npi_name,reduction)) + 
                             geom_violin() + 
                             geom_jitter(aes(group = npi_name, color = ll), size = 0.5, height = 0, width = 0.2, alpha = 0.5) +
                             theme_bw(base_size = 10) +
                             theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 6)) +
                             scale_color_viridis_c(option = "B", name = "log\nlikelihood") +
                             labs(x = "parameter", title = i_lab)
                         }else{
                           nodes_ <- unlist(strsplit(i,","))
                           ll_across_nodes <- 
                             outputs_global$llik %>% 
                             .[get(config$spatial_setup$subpop) %in% nodes_] %>%
                             .[, .(ll_sum = sum(ll)), by = .(slot)]
                           
                           outputs_global$snpi %>%
                             .[get(config$spatial_setup$subpop) == i] %>%
                             .[ll_across_nodes, on = c("slot")] %>%
                             ggplot(aes(npi_name,reduction)) + 
                             geom_violin() + 
                             geom_jitter(aes(group = npi_name, color = ll_sum), size = 0.5, height = 0, width = 0.2, alpha = 0.5) +
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


## MOVE FILES TO /pplot --------------------------------------------------------------------

# file.copy(from = files_,
#           to = file.path(data_path, "pplot",basename(files_)),
#           overwrite = TRUE)

