##'
##' Function that does a rapid testing of the inference procedures on a single
##' date series.
##'
##' @param seeding the initial seeding
##' @param config the config file with inference info.
##' @param date_bounds acceptable bounds for the seeding dates
##' @param n_slots the number of slots to run, MCMC iterations per slot are in the config file
##' @param ncores the number of cores to run
##' @param npi_file file to write NPI samples to
##' @param seeding_file file to write seeding samples to
##' @param loglik_file file to write neg log-loglik samples to
##' @param epi_dir file to write simulated trajectory samples to
##' 
##' @return inference run results on this particular area
##'
#' @export
single_loc_inference_test <- function(to_fit,
                                      S0, # TODO change to geodata.csv
                                      seeding,
                                      config,
                                      date_bounds,
                                      n_slots,
                                      ncores, 
                                      npi_file,
                                      seeding_file,
                                      loglik_file,
                                      epi_dir) {
    
    cl <- parallel::makeCluster(ncores)
    registerDoSNOW(cl)
    
    # Column name that stores subpop unique id
    obs_subpop <- config$subpop_setup$subpop
    
    # Set number of simulations
    iterations_per_slot <- config$inference$iterations_per_slot
    
    # SEIR parameters for simulations
    R0 <- flepicommon::as_evaled_expression(config$seir$parameters$R0s$value)
    gamma <- flepicommon::as_evaled_expression(config$seir$parameters$gamma$value)
    sigma <- flepicommon::as_evaled_expression(config$seir$parameters$sigma)
    
    # Data to fit
    obs <- to_fit
    
    # dates based on config
    sim_dates <- seq.Date(as.Date(config$start_date), as.Date(config$end_date), by = "1 days")
    
    # Get unique geonames
    geonames <- unique(obs[[obs_subpop]])
    
    # Compute statistics of observations
    data_stats <- lapply(
        geonames,
        function(x) {
            df <- obs[obs[[obs_subpop]] == x, ]
            getStats(
                df,
                "date",
                "data_var",
                stat_list = config$inference$statistics)
        }) %>%
        set_names(geonames)
    
    all_locations <- unique(obs[[obs_subpop]])
    
    # Inference loops
    required_packages <- c("dplyr", "magrittr", "xts", "zoo", "purrr", "stringr", "truncnorm",
                           "readr", "flepicommon", "hospitalization", "data.table",
                           "inference")  # packages required for dopar
    
    
    # Loop over number of slots
    res <- foreach(s = 1:n_slots, 
                   .combine = rbind,
                   .packages = required_packages,
                   .inorder = F,
                   .export = c("epi_dir")
    ) %dopar% {
        
        npis_init <- npis_dataframe(config, random = T)
        
        seeding_init <- seeding
        for (i in 1:nrow(seeding_init)) {
            seeding_init$date[i] <- sample(sim_dates[1:20], 1)
            # TODO change amount based on data
            seeding_init$amount[i] <- rpois(1, 10)
        }
        
        initial_seeding <- perturb_seeding(seeding_init, config$seeding$perturbation_sd, date_bounds)
        initial_npis <- perturb_expand_npis(npis_init, config$interventions$settings)
        
        # Write to file
        initial_seeding %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(seeding_file, append = file.exists(seeding_file))
        
        initial_npis %>% 
            distinct(value, modifier_name, subpop) %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(npi_file, append = file.exists(npi_file))
        
        # Simulate initial hospitalizatoins
        initial_sim_hosp <- simulate_single_epi(dates = sim_dates, 
                                                seeding = initial_seeding,
                                                R0 = R0, 
                                                S0 = S0, 
                                                gamma = gamma,
                                                sigma = sigma,
                                                beta_mults = 1-initial_npis$value) %>% 
            single_hosp_run(config) %>% 
            dplyr::filter(date %in% obs$date)
        
        write_csv(initial_sim_hosp, glue::glue("{epi_dir}sim_slot_{s}_index_0.csv"))
        
        initial_sim_stats <- getStats(
            initial_sim_hosp,
            "date",
            "sim_var",
            end_date = max(obs$date),
            config$inference$statistics
        )
        
        # Get observation statistics
        log_likelihood <- list()
        for(var in names(data_stats[[1]])) {
            log_likelihood[[var]] <- logLikStat(
                obs = data_stats[[1]][[var]]$data_var,
                sim = initial_sim_stats[[var]]$sim_var,
                dist = config$inference$statistics[[var]]$likelihood$dist,
                param = config$inference$statistics[[var]]$likelihood$param,
                add_one = config$inference$statistics[[var]]$add_one
            )
        }
        # Compute log-likelihoods
        initial_log_likelihood_data <- dplyr::tibble(
            ll = sum(unlist(log_likelihood)),
            subpop = 1
        )
        
        # Compute total loglik for each sim
        likelihood <- initial_log_likelihood_data
        
        likelihood %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(loglik_file, append = file.exists(loglik_file))
        
        ## For logging
        current_likelihood <- likelihood
        current_index <- 0
        previous_likelihood_data <- initial_log_likelihood_data
        
        for (index in seq_len(iterations_per_slot)) {
            current_seeding <- perturb_seeding(initial_seeding, config$seeding$perturbation_sd, date_bounds)
            current_npis <- perturb_expand_npis(initial_npis, config$interventions$settings)
            
            # Simulate  hospitalizatoins
            sim_hosp <- simulate_single_epi(dates = sim_dates,
                                            seeding = current_seeding,
                                            R0 = R0,
                                            S0 = S0,
                                            gamma = gamma,
                                            sigma = sigma,
                                            beta_mults = 1-current_npis$reduction) %>% 
                single_hosp_run(config) %>% 
                dplyr::filter(date %in% obs$date)
            
            sim_stats <- getStats(
                sim_hosp,
                "date",
                "sim_var",
                end_date = max(obs$date),
                config$inference$statistics
            )
            
            # Get observation statistics
            log_likelihood <- list()
            for(var in names(data_stats[[1]])) {
                log_likelihood[[var]] <- logLikStat(
                    obs = data_stats[[1]][[var]]$data_var,
                    sim = sim_stats[[var]]$sim_var,
                    dist = config$inference$statistics[[var]]$likelihood$dist,
                    param = config$inference$statistics[[var]]$likelihood$param,
                    add_one = config$inference$statistics[[var]]$add_one
                )
            }
            # Compute log-likelihoods
            log_likelihood_data <- dplyr::tibble(
                ll = sum(unlist(log_likelihood)),
                subpop = 1
            )
            
            # Compute total loglik for each sim
            likelihood <- log_likelihood_data
            
            ## For logging
            print(paste("Current likelihood",current_likelihood$ll,"Proposed likelihood",likelihood$ll))
            
            # Update states
            if(iterateAccept(current_likelihood, likelihood, 'll')){
                current_index <- index
                current_likelihood <- likelihood
                
                write_csv(sim_hosp, glue::glue("{epi_dir}sim_slot_{s}_index_{index}.csv"))
            }
            
            # Upate seeding and NPIs by location
            seeding_npis_list <- accept_reject_proposals(
                seeding_orig = initial_seeding,
                seeding_prop = current_seeding,
                npis_orig = distinct(initial_npis, value, modifier_name, subpop),
                npis_prop = distinct(current_npis, value, modifier_name, subpop),
                orig_lls = previous_likelihood_data,
                prop_lls = log_likelihood_data
            )
            initial_seeding <- seeding_npis_list$seeding
            initial_npis <- inner_join(seeding_npis_list$npis, select(current_npis, -value), by = c("subpop", "modifier_name"))
            previous_likelihood_data <- seeding_npis_list$ll
            
            # Write to file
            initial_seeding %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(seeding_file, append = file.exists(seeding_file))
            
            seeding_npis_list$npis %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(npi_file, append = file.exists(npi_file))
            
            seeding_npis_list$ll %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(loglik_file, append = file.exists(loglik_file))
            
        }
        
    }
    parallel::stopCluster(cl)
}

##'
##' Function that does a rapid testing of the inference procedures on a single
##' date series.
##'
##' @param seeding the initial seeding
##' @param config the config file with inference info.
##' @param date_bounds acceptable bounds for the seeding dates
##' @param n_slots the number of slots to run, MCMC iterations per slot are in the config file
##' @param ncores the number of cores to run
##' @param npi_file file to write NPI samples to
##' @param seeding_file file to write seeding samples to
##' @param loglik_file file to write neg log-loglik samples to
##' @param epi_dir file to write simulated trajectory samples to
##' 
##' @return inference run results on this particular area
##'
#' @export
multi_loc_inference_test <- function(to_fit,
                                     S0s, # TODO change to geodata.csv
                                     seedings,
                                     mob,
                                     offsets,
                                     config,
                                     date_bounds,
                                     n_slots,
                                     ncores, 
                                     npi_file,
                                     seeding_file,
                                     loglik_file,
                                     location_loglik_file,
                                     epi_dir) {
    library(tidyverse)
    cl <- parallel::makeCluster(ncores)
    registerDoSNOW(cl)
    
    N <- length(S0s)
    # Column name that stores subpop unique id
    obs_subpop <- config$subpop_setup$subpop
    
    # Set number of simulations
    iterations_per_slot <- config$inference$iterations_per_slot
    
    # SEIR parameters for simulations
    R0 <- flepicommon::as_evaled_expression(config$seir$parameters$R0s$value)
    gamma <- flepicommon::as_evaled_expression(config$seir$parameters$gamma$value)
    sigma <- flepicommon::as_evaled_expression(config$seir$parameters$sigma)
    
    # Data to fit
    obs <- to_fit
    
    # dates based on config
    sim_dates <- seq.Date(as.Date(config$start_date), as.Date(config$end_date), by = "1 days")
    
    # Get unique geonames
    geonames <- unique(obs[[obs_subpop]])
    
    # Compute statistics of observations
    data_stats <- lapply(
        geonames,
        function(x) {
            df <- obs[obs[[obs_subpop]] == x, ]
            getStats(
                df,
                "date",
                "data_var",
                stat_list = config$inference$statistics)
        }) %>%
        set_names(geonames)
    
    all_locations <- unique(obs[[obs_subpop]])
    
    # Inference loops
    required_packages <- c("dplyr", "magrittr", "xts", "zoo", "purrr", "stringr", "truncnorm",
                           "readr", "flepicommon", "hospitalization", "data.table",
                           "inference", "purrr", "tidyr")  # packages required for dopar
    
    
    # Loop over number of slots
    res <- foreach(s = 1:n_slots, 
                   .combine = rbind,
                   .packages = required_packages,
                   .inorder = F,
                   .export = c("epi_dir")
    ) %dopar% {
        
        npis_init <- pmap(list(x = 1:N, y = offsets),
                     function(x,y) 
                         npis_dataframe(config, 
                                        subpop = x,
                                        offset = y,
                                        random = T)) %>% 
            bind_rows()
        
        seeding_init <- seedings
        for (i in 1:nrow(seeding_init)) {
            seeding_init$date[i] <- sample(sim_dates[1:20], 1)
            # TODO change amount based on data
            seeding_init$amount[i] <- rpois(1, 10)
        }
        
        initial_seeding <- perturb_seeding(seeding_init, config$seeding$perturbation_sd, date_bounds)
        initial_npis <- perturb_expand_npis(npis_init, config$interventions$settings, multi = T)
        
        # Write to file
        initial_seeding %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(seeding_file, append = file.exists(seeding_file))
        
        initial_npis %>% 
            distinct(value, modifier_name, subpop) %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(npi_file, append = file.exists(npi_file))
        
        npi_mat <- select(initial_npis, date, subpop, value) %>% 
            pivot_wider(values_from = "value", names_from = "subpop", id_cols = "date")
        
        # Simulate epi
        initial_sim_hosp <- simulate_multi_epi(dates = sim_dates,
                                               seedings = initial_seeding,
                                               R0 = R0,
                                               S0s = S0s,
                                               N = length(S0s),
                                               gamma = gamma,
                                               sigma = sigma,
                                               mob = mob,
                                               beta_mults = 1-as.matrix(npi_mat[,-1])) %>% 
            multi_hosp_run(N, config) %>% 
            dplyr::filter(date %in% obs$date)
        
        write_csv(initial_sim_hosp, glue::glue("{epi_dir}sim_slot_{s}_index_0_multi.csv"))
        
        initial_likelihood_data <- list()
        for(location in all_locations) {
            
            local_sim_hosp <- dplyr::filter(initial_sim_hosp, !!rlang::sym(obs_subpop) == location) %>%
                dplyr::filter(date %in% unique(obs$date[obs$subpop == location]))
            initial_sim_stats <- inference::getStats(
                local_sim_hosp,
                "date",
                "sim_var",
                #end_date = max(obs$date[obs[[obs_subpop]] == location]),
                stat_list = config$inference$statistics
            )
            
            
            # Get observation statistics
            log_likelihood <- list()
            for(var in names(data_stats[[location]])) {
                log_likelihood[[var]] <- inference::logLikStat(
                    obs = data_stats[[location]][[var]]$data_var,
                    sim = initial_sim_stats[[var]]$sim_var,
                    dist = config$inference$statistics[[var]]$likelihood$dist,
                    param = config$inference$statistics[[var]]$likelihood$param,
                    add_one = config$inference$statistics[[var]]$add_one
                )
            }
            # Compute log-likelihoods
            initial_likelihood_data[[location]] <- dplyr::tibble(
                ll = sum(unlist(log_likelihood)),
                subpop = location
            )
        }
        
        initial_likelihood_data <- initial_likelihood_data %>% do.call(what=rbind)
        
        initial_likelihood_data %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(location_loglik_file, append = file.exists(location_loglik_file))
        
        # Compute total loglik for each sim
        likelihood <- initial_likelihood_data %>%
            summarise(ll = sum(ll, na.rm = T))
        
        likelihood %>% 
            mutate(slot = s, index = 0) %>% 
            write_csv(loglik_file, append = file.exists(loglik_file))
        
        ## For logging
        current_likelihood <- likelihood
        current_index <- 0
        previous_likelihood_data <- initial_likelihood_data
        
        for (index in seq_len(iterations_per_slot)) {
            current_seeding <- perturb_seeding(initial_seeding, config$seeding$perturbation_sd, date_bounds)
            current_npis <- perturb_expand_npis(initial_npis, config$interventions$settings, multi = T)
            
            npi_mat <- select(current_npis, date, subpop, reduction) %>% 
                pivot_wider(values_from = "reduction", names_from = "subpop", id_cols = "date")
            
            # Simulate  hospitalizatoins
            sim_hosp <- simulate_multi_epi(dates = sim_dates,
                                           seedings = current_seeding,
                                           R0 = R0,
                                           S0s = S0s,
                                           N = length(S0s),
                                           gamma = gamma,
                                           sigma = sigma,
                                           mob = mob,
                                           beta_mults = 1-as.matrix(npi_mat[,-1])) %>% 
                multi_hosp_run(N, config) %>% 
                dplyr::filter(date %in% obs$date)
            
            current_likelihood_data <- list()
            
            for(location in all_locations) {
                local_sim_hosp <- dplyr::filter(sim_hosp, !!rlang::sym(obs_subpop) == location) %>%
                    dplyr::filter(date %in% unique(obs$date[obs$subpop == location]))
                sim_stats <- inference::getStats(
                    local_sim_hosp,
                    "date",
                    "sim_var",
                    #end_date = max(obs$date[obs[[obs_subpop]] == location]),
                    stat_list = config$inference$statistics
                )
                
                
                # Get observation statistics
                log_likelihood <- list()
                for(var in names(data_stats[[location]])) {
                    log_likelihood[[var]] <- inference::logLikStat(
                        obs = data_stats[[location]][[var]]$data_var,
                        sim = sim_stats[[var]]$sim_var,
                        dist = config$inference$statistics[[var]]$likelihood$dist,
                        param = config$inference$statistics[[var]]$likelihood$param,
                        add_one = config$inference$statistics[[var]]$add_one
                    )
                }
                # Compute log-likelihoods
                current_likelihood_data[[location]] <- dplyr::tibble(
                    ll = sum(unlist(log_likelihood)),
                    subpop = location
                )
            }
            
            current_likelihood_data <- current_likelihood_data %>% do.call(what=rbind)
            
            # Compute total loglik for each sim
            likelihood <- current_likelihood_data %>%
                summarise(ll = sum(ll, na.rm = T)) 
            
            ## For logging
            print(paste("Current likelihood",current_likelihood$ll,"Proposed likelihood",likelihood$ll))
            
            # Update states
            if(iterateAccept(current_likelihood, likelihood, 'll')){
                current_index <- index
                current_likelihood <- likelihood
                
                likelihood %>% 
                    mutate(slot = s, index = index) %>% 
                    write_csv(loglik_file, append = file.exists(loglik_file))
                
                    write_csv(sim_hosp, glue::glue("{epi_dir}sim_slot_{s}_index_{index}_multi.csv"))
            }
            
            # Upate seeding and NPIs by location
            seeding_npis_list <- accept_reject_proposals(
                seeding_orig = initial_seeding,
                seeding_prop = current_seeding,
                npis_orig = distinct(initial_npis, value, modifier_name, subpop),
                npis_prop = distinct(current_npis, value, modifier_name, subpop),
                orig_lls = previous_likelihood_data,
                prop_lls = current_likelihood_data
            )
            initial_seeding <- seeding_npis_list$seeding
            initial_npis <- inner_join(seeding_npis_list$npis, select(current_npis, -value), by = c("subpop", "modifier_name"))
            previous_likelihood_data <- seeding_npis_list$ll
            
            # Write to file
            initial_seeding %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(seeding_file, append = file.exists(seeding_file))
            
            seeding_npis_list$npis %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(npi_file, append = file.exists(npi_file))
            
            seeding_npis_list$ll %>% 
                mutate(slot = s, index = index) %>% 
                write_csv(location_loglik_file, append = file.exists(location_loglik_file))
            
        }
        
    }
    parallel::stopCluster(cl)
}




##'
##'
##' Fill in the epi curve from a baseline state given current parameters
##'
##' TODO: Modify to just call python code
##'
##' @param dates the dates to run the epidemic on
##' @param seeding the seeding to use
##' @param R0 the reproductive number
##' @param S0 the initial number susceptible
##' @param gamma the date between compartments
##' @param sigma the incubation period/latent period
##' @param beta_mults multipliers to capture the impact of interventions
##' @param alpha defaults to 1
##'
##' @return the epimic states
##'
##' @export
simulate_single_epi <- function(dates,
                                seeding,
                                S0,
                                R0,
                                gamma,
                                sigma,
                                beta_mults = rep(1, length(dates)),
                                alpha = 1) {
    
    require(tidyverse)
    
    ##Scale R0 to beta
    beta <- R0 * gamma/3
    
    ##set up return matrix
    epi <- matrix(0, nrow=length(dates), ncol=7)
    colnames(epi) <- c("S","E","I1","I2","I3","R","incidI")
    
    
    ##column indices for convenience
    S <- 1
    E <- 2
    I1 <- 3
    I2 <- 4
    I3 <- 5
    R  <- 6
    incidI <- 7
    
    ##seed the first case
    epi[1,S] <- S0
    
    ##get the indices where seeding occurs
    seed_dates <- which(dates%in%seeding$date)
    
    for (i in 1:(length(dates)-1)) {
        ##Seed if possible
        if(i%in% seed_dates) {
            tmp <- seeding$amount[which(i==seed_dates)]
            epi[i,S] <- epi[i,S] - tmp
            epi[i,E] <- epi[i,E] + tmp
        }
        
        ##Draw transitions
        #print(beta)
        dSE <- rbinom(1,epi[i,S],beta*beta_mults[i]*sum(epi[i,I1:I3])^alpha/S0)
        dEI <- rbinom(1,epi[i,E],sigma)
        dI12 <- rbinom(1, epi[i,I1], gamma)
        dI23 <- rbinom(1, epi[i,I2], gamma)
        dIR <- rbinom(1, epi[i,I3], gamma)
        
        ##Make transitions
        epi[i+1,S] <- epi[i,S] - dSE
        epi[i+1,E] <- epi[i,E] + dSE - dEI
        epi[i+1,I1] <- epi[i,I1] + dEI - dI12
        epi[i+1,I2] <- epi[i,I2] + dI12 - dI23
        epi[i+1,I3] <- epi[i,I3] + dI23 - dIR
        epi[i+1,R] <- epi[i,R] + dIR
        epi[i+1,incidI] <- dEI
        
    }
    
    epi <- as.data.frame(epi) %>%
        mutate(date=dates)%>%
        pivot_longer(-date,values_to="N", names_to="comp")
    
    return(epi)
}


##'
##'
##' Fill in the epi curve from a baseline state given current parameters
##'
##' TODO: Modify to just call python code
##'
##' @param dates the dates to run the epidemic on
##' @param seedings the seeding to use
##' @param R0s the reproductive number
##' @param S0s the initial number susceptibles
##' @param gamma the date between compartments
##' @param sigma the incubation period/latent period
##' @param N the number of nodes
##' @param mob mobility matrix N x N
##' @param pa proportion of date away
##' @param beta_mults multipliers to capture the impact of interventions
##' @param alpha defaults to 1
##'
##' @return the epimic states
##'
##' @export
simulate_multi_epi <- function(dates,
                               seedings,
                               S0s,
                               R0,
                               gamma,
                               sigma,
                               N,
                               mob,
                               pa = .5,
                               beta_mults = matrix(rep(1, N*length(dates)), ncol = N),
                               alpha = 1) {
    
    require(tidyverse)
    
    # proportion date away
    paoverh <- pa/S0s
    oneminusp <- 1-paoverh*rowSums(mob)
    
    ##Scale R0 to beta
    beta <- R0 * gamma/3
    
    ##set up return matrix
    epi <- array(0, dim = c(length(dates), 7, N))
    colnames(epi) <- c("S","E","I1","I2","I3","R","incidI")
    
    ##column indices for convenience
    S <- 1
    E <- 2
    I1 <- 3
    I2 <- 4
    I3 <- 5
    R  <- 6
    incidI <- 7
    
    ##seed the first case
    epi[1,S,] <- S0s
    
    ##get the indices where seeding occurs
    seed_dates <- which(dates%in%seedings$date)
    
    for (i in 1:(length(dates)-1)) {
        
        betaIh <- matrix(beta*beta_mults[i,]*(colSums(epi[i,I1:I3,])^alpha)/S0s, ncol = 1)
        mobbetaIh <- mob %*% betaIh
        
        for (j in 1:N) {
            ##Seed if possible
            if(i%in% seed_dates) {
                ind <- which(i==seed_dates)
                if (ind == j) {
                    tmp <- seedings$amount[ind]
                    epi[i,S,j] <- epi[i,S,j] - tmp
                    epi[i,E,j] <- epi[i,E,j] + tmp
                }
            }
            
            ##Draw transitions
            #print(beta)
            
            foi <- oneminusp[j]*betaIh[j] + paoverh[j]*mobbetaIh[j]
            dSE <- rbinom(1,epi[i,S,j],foi)
            dEI <- rbinom(1,epi[i,E,j],sigma)
            dI12 <- rbinom(1, epi[i,I1,j], gamma)
            dI23 <- rbinom(1, epi[i,I2,j], gamma)
            dIR <- rbinom(1, epi[i,I3,j], gamma)
            
            ##Make transitions
            epi[i+1,S,j] <- epi[i,S,j] - dSE
            epi[i+1,E,j] <- epi[i,E,j] + dSE - dEI
            epi[i+1,I1,j] <- epi[i,I1,j] + dEI - dI12
            epi[i+1,I2,j] <- epi[i,I2,j] + dI12 - dI23
            epi[i+1,I3,j] <- epi[i,I3,j] + dI23 - dIR
            epi[i+1,R,j] <- epi[i,R,j] + dIR
            epi[i+1,incidI,j] <- dEI
        }
    }
    
    epi <- lapply(1:N, function(x) as.data.frame(epi[,,x]) %>% mutate(subpop = x)) %>%
        bind_rows() %>% 
        mutate(date=rep(dates, N)) %>%
        pivot_longer(cols = c(-date, -subpop), values_to="N", names_to="comp")
    
    return(epi)
}

##'
##' Run hospitalizations and deaths for single epi curve
##' Functions taken from the hospitalization package
##'
##' @param epi SEIR output
##' @param config configuration file
##'
##' @return the epimic states
##'
##'
##' @export
single_hosp_run <- function(epi, config) {
    
    p_death <- config$hospitalization$parameters$p_death
    p_death_rate <- config$hospitalization$parameters$p_death_rate
    p_hosp <- p_death/p_death_rate
    date_hosp_pars <- as_evaled_expression(config$hospitalization$parameters$date_hosp)
    date_disch_pars <- as_evaled_expression(config$hospitalization$parameters$date_disch)
    date_hosp_death_pars <- as_evaled_expression(config$hospitalization$parameters$date_hosp_death)
    dat_ <- dplyr::filter(epi, comp == "incidI") %>% 
        select(-comp) %>% 
        rename(incidI = N) %>%
        mutate(uid = epi$subpop[1]) %>% 
        as.data.table()
    
    if ("subpop" %in% colnames(dat_)) {
        dat_ <- select(dat_, -subpop)
    }
    
    dat_H <- hosp_create_delay_frame('incidI',p_hosp,dat_,date_hosp_pars,"H")
    data_D <- hosp_create_delay_frame('incidH',p_death_rate,dat_H,date_hosp_death_pars,"D")
    R_delay_ <- round(exp(date_disch_pars[1]))
    res <- Reduce(function(x, y, ...) merge(x, y, all = TRUE, ...),
                  list(dat_, dat_H, data_D)) %>%
        tidyr::replace_na(
            list(incidI = 0,
                 incidH = 0,
                 incidD = 0,
                 hosp_curr = 0)) %>%
        dplyr::mutate(date_inds = as.integer(date - min(date) + 1)) %>%
        dplyr::arrange(date_inds) %>%
        split(.$uid) %>%
        purrr::map_dfr(function(.x){
            .x$hosp_curr <- cumsum(.x$incidH) - lag(cumsum(.x$incidH),
                                                    n=R_delay_,default=0)
            return(.x)
        }) %>%
        replace_na(
            list(hosp_curr = 0)) %>%
        arrange(date_inds) %>% 
        select(-date_inds) %>% 
        mutate(subpop = uid) %>% 
        select(-uid)
    
    return(res)
}

##' @export
multi_hosp_run <- function(epi, N, config) {
    map_df(1:N, 
           ~ single_hosp_run(dplyr::filter(epi, subpop == .), config)) %>%
        dplyr::filter(date >= config$start_date,
               date <= config$end_date)
}

##'
##' Create NPIs dataframe for a single location for input to SEIR code
##'
##' @param dates vector of dates to use
##' @param config configuration file
##'
##' @return a dataframe with npi reduction values by date
##'
##'
##' @export
npis_dataframe <- function(config, random = F, subpop = 1, offset = 0, intervention_multi = 1) {
    
    dates <- seq.Date(as.Date(config$start_date), as.Date(config$end_date), by = "1 days")
    npis <- tibble(date = dates, value = 0, modifier_name = "local_variation", subpop = subpop)
    interventions <- config$interventions$settings
    date_changes <- map_chr(interventions[1:2], 
                            ~ifelse(is.null(.$period_start_date),
                                    config$start_date, 
                                    .$period_start_date)) %>% as.Date()
    # Apply offset if specified
    date_changes[2:length(date_changes)] <- date_changes[2:length(date_changes)] + offset
    
    # Apply interventions
    for (d in 1:length(date_changes)) {
        npis$value[dates >= date_changes[d]] <- interventions[[d]]$value$mean * intervention_multi
        npis$modifier_name[dates >= date_changes[d]] <- names(interventions)[d]
    }
    
    if(random) {
        # Randomly assign interventions
        for (d in 1:length(date_changes)) {
            if (names(interventions)[d] == "local_variation") {
                npis$value[dates >= date_changes[d]] <- runif(1, -.5, .5)
            } else {
                npis$value[dates >= date_changes[d]] <- runif(1, 0, 1)
            }
        }
    }
    return(npis)
}

##'
##' Create synthetic data to run inference on
##' Runs an SEIR simulation and applies hospitalization and death on it
##'
##' @param S0 population size
##' @param config configuration file
##'
##' @return the synthetic data
##'
##'
##' @export
synthetic_data <- function(S0, seeding, config) {
    
    # Simulate single epidemic 
    dates <- seq.Date(as.Date(config$start_date), as.Date(config$end_date), by = "1 days")
    npis <- npis_dataframe(config)
    R0 <- flepicommon::as_evaled_expression(config$seir$parameters$R0s$value)
    gamma <- flepicommon::as_evaled_expression(config$seir$parameters$gamma$value)
    sigma <- flepicommon::as_evaled_expression(config$seir$parameters$sigma)
    
    # Simulate epi
    epi <- simulate_single_epi(dates = dates,
                               seeding = seeding,
                               R0 = R0,
                               S0 = S0,
                               gamma = gamma,
                               sigma = sigma,
                               beta_mults = 1-npis$value)
    
    # - - - -
    # Setup fake data
    fake_data <- single_hosp_run(epi, config) %>%
        rename(date = date) %>% 
        dplyr::filter(date >= config$start_date,
               date <= config$end_date)
    
    return(fake_data)
}

##'
##' Create synthetic data to run inference on
##' Runs an SEIR simulation and applies hospitalization and death on it
##'
##' @param S0s population sizes
##' @param seedings seeding dataframe
##' @param config configuration file
##'
##' @return the synthetic data
##'
##'
##' @export
synthetic_data_multi <- function(S0s, seedings, mob, config, offsets, interventions_multi) {
    
    N <- length(S0s)  # number of nodes
    
    # Simulate single epidemic 
    dates <- seq.Date(as.Date(config$start_date), as.Date(config$end_date), by = "1 days")
    npis <- pmap(list(x = 1:N, y = offsets, z = interventions_multi),
                 function(x,y,z) 
                     npis_dataframe(config, 
                                    subpop = x,
                                    offset = y,
                                    intervention_multi = z)) %>% 
        bind_rows()
    
    R0 <- flepicommon::as_evaled_expression(config$seir$parameters$R0s$value)
    gamma <- flepicommon::as_evaled_expression(config$seir$parameters$gamma$value)
    sigma <- flepicommon::as_evaled_expression(config$seir$parameters$sigma)
    
    npi_mat <- select(npis, date, subpop, value) %>% 
        pivot_wider(values_from = "value", names_from = "subpop", id_cols = "date")
    
    # Simulate epi
    epi <- simulate_multi_epi(dates = dates,
                              seedings = seedings,
                              R0 = R0,
                              S0s = S0s,
                              N = length(S0s),
                              gamma = gamma,
                              sigma = sigma,
                              mob = mob,
                              beta_mults = 1-as.matrix(npi_mat[,-1]))
    
    # - - - -
    # Setup fake data
    fake_data <- map_df(1:N, 
                        ~ single_hosp_run(dplyr::filter(epi, subpop == .), config)) %>%
        rename(date = date) %>% 
        dplyr::filter(date >= config$start_date,
               date <= config$end_date)
    
    return(fake_data)
}

##'
##' Expands the perturb npi into a dataframe with date
##'
##' @param npis the npis in "long" version (one value by npi, date and place)
##' @param intervention_settings intervention_settings from the config file
##'
##' @return the perturbed npi dataframe
##'
##'
##' @export
perturb_expand_npis <- function(npis, intervention_settings, multi = F) {
    if(multi) {
        npis %>% 
            distinct(value, modifier_name, subpop) %>%
            group_by(subpop) %>% 
            group_map(~perturb_npis(.x, intervention_settings) %>% 
                          mutate(subpop = .y$subpop[1])) %>% 
            bind_rows() %>% 
            inner_join(select(npis, -value), by = c("modifier_name", "subpop"))
    } else {
        npis %>% 
            distinct(value, modifier_name) %>% 
            perturb_npis(intervention_settings) %>% 
            inner_join(select(npis, -value), by = c("modifier_name"))
    }
}
