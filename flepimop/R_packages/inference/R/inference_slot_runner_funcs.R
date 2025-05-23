###Functions that help with the running of Filter MC. Can take in full confit.


##'Function that performs aggregation and calculates likelihood data across all given locations.
##'
##' @param all_locations all of the locations to calculate likelihood for
##' @param modeled_outcome  the hospital data for the simulations
##' @param obs_subpop the name of the column containing locations.
##' @param config the full configuration setup
##' @param obs the full observed data
##' @param ground_truth_data the data we are going to compare to aggregated to the right statistic
##' @param hosp_file the filename of the hosp file being used (unclear if needed in scope)
##' @param hierarchical_stats the hierarchical stats to use
##' @param defined_priors information on defined priors.
##' @param geodata the geographics data to help with hierarchies
##' @param snpi the file with the npi information for seir, only used for heirarchical likelihoods
##' @param hnpi the file with the npi information for outcomes, only used for heirarchical likelihoods
##' @param hpar data frame of hospitalization parameters, only used for heirarchical likelihoods
##'
##' @return a data frame of likelihood data.
##'
##' @export
##'
aggregate_and_calc_loc_likelihoods <- function(
        all_locations,
        modeled_outcome,
        obs_subpop,
        targets_config,
        obs, # should remove this, it's not used
        ground_truth_data,
        hosp_file, # should remove this, it's not used
        hierarchical_stats,
        defined_priors,
        geodata,
        snpi=NULL,
        hnpi=NULL,
        hpar=NULL,
        start_date = NULL,
        end_date = NULL
) {

    ##Holds the likelihoods for all locations
    likelihood_data <- list()



    ##iterate over locations
    for (location in all_locations) {
        ##Pull out the local sim from the complete sim
        this_location_modeled_outcome <-
            ## Filter to this location
            dplyr::filter(
                modeled_outcome,
                !!rlang::sym(obs_subpop) == location
            ) %>%
            ## Reformat into form the algorithm is looking for
            inference::getStats(
                "date",
                "sim_var",
                stat_list = targets_config,
                start_date = start_date,
                end_date = end_date
            )


        ## Get observation statistics
        this_location_log_likelihood <- 0
        for (var in names(ground_truth_data[[location]])) {

          obs_tmp1 <- ground_truth_data[[location]][[var]]
          obs_tmp <- obs_tmp1[!is.na(obs_tmp1$data_var) & !is.na(obs_tmp1$date),]
          sim_tmp1 <- this_location_modeled_outcome[[var]]
          sim_tmp <- sim_tmp1[match(lubridate::as_date(sim_tmp1$date),
                                    lubridate::as_date(obs_tmp$date)),] %>% na.omit()


            this_location_log_likelihood <- this_location_log_likelihood +
                ## Actually compute likelihood for this location and statistic here:
                sum(inference::logLikStat(
                    obs = as.numeric(obs_tmp$data_var),
                    sim = as.numeric(sim_tmp$sim_var),
                    dist = targets_config[[var]]$likelihood$dist,
                    param = targets_config[[var]]$likelihood$param,
                    add_one = targets_config[[var]]$add_one
                ))
        }

        ## Compute log-likelihoods
        ## We use a data frame for debugging, only ll is used
        likelihood_data[[location]] <- dplyr::tibble(
            ll = this_location_log_likelihood,
            subpop = location,
            accept = 0, # acceptance decision (0/1) . Will be updated later when accept/reject decisions made
            accept_avg = 0, # running average acceptance decision
            accept_prob = 0 # probability of acceptance of proposal
        )
        names(likelihood_data)[names(likelihood_data) == 'subpop'] <- obs_subpop
    }

    #' @importFrom magrittr %>%
    likelihood_data <- likelihood_data %>% do.call(what = rbind)

    ##Update  likelihood data based on hierarchical_stats (NOT SUPPORTED FOR INIT FILES)
    for (stat in names(hierarchical_stats)) {

        if (hierarchical_stats[[stat]]$module %in% c("seir_interventions", "seir")) {
            ll_adjs <- inference::calc_hierarchical_likadj(
                stat = hierarchical_stats[[stat]]$name,
                infer_frame = snpi,
                geodata = geodata,
                geo_group_column = hierarchical_stats[[stat]]$geo_group_col,
                transform = hierarchical_stats[[stat]]$transform
            )

        } else if (hierarchical_stats[[stat]]$module == "outcomes_interventions") {
            ll_adjs <- inference::calc_hierarchical_likadj(
                stat = hierarchical_stats[[stat]]$name,
                infer_frame = hnpi,
                geodata = geodata,
                geo_group_column = hierarchical_stats[[stat]]$geo_group_col,
                transform = hierarchical_stats[[stat]]$transform
            )

        } else if (hierarchical_stats[[stat]]$module %in% c("hospitalization", "outcomes_parameters")) {

            ll_adjs <- inference::calc_hierarchical_likadj(
                stat = hierarchical_stats[[stat]]$name,
                infer_frame = hpar,
                geodata = geodata,
                geo_group_column = hierarchical_stats[[stat]]$geo_group_col,
                transform = hierarchical_stats[[stat]]$transform,
                stat_col = "value",
                stat_name_col = "parameter"
            )

        } else if (hierarchical_stats[[stat]]$module == "seir_parameters") {
            stop("We currently do not support hierarchies on seir parameters, since we don't do inference on them except via npis.")
        } else {
            stop("unsupported hierarchical stat module")
        }

        ##probably a more efficient what to do this, but unclear...
        likelihood_data <- dplyr::left_join(likelihood_data, ll_adjs, by = obs_subpop) %>%
            tidyr::replace_na(list(likadj = 0)) %>% ##avoid unmatched location problems
            dplyr::mutate(ll = ll + likadj) %>%
            dplyr::select(-likadj)
    }


    ##Update likelihoods based on priors
    for (prior in names(defined_priors)) {
        if (defined_priors[[prior]]$module %in% c("seir_interventions", "seir")) {
            #' @importFrom magrittr %>%
            ll_adjs <- snpi %>%
                dplyr::filter(modifier_name == defined_priors[[prior]]$name) %>%
                dplyr::mutate(likadj = calc_prior_likadj(value,
                                                         defined_priors[[prior]]$likelihood$dist,
                                                         defined_priors[[prior]]$likelihood$param
                )) %>%
                dplyr::select(subpop, likadj)

        } else if (defined_priors[[prior]]$module == "outcomes_interventions") {
            #' @importFrom magrittr %>%
            ll_adjs <- hnpi %>%
                dplyr::filter(modifier_name == defined_priors[[prior]]$name) %>%
                dplyr::mutate(likadj = calc_prior_likadj(value,
                                                         defined_priors[[prior]]$likelihood$dist,
                                                         defined_priors[[prior]]$likelihood$param
                )) %>%
                dplyr::select(subpop, likadj)

        }  else if (defined_priors[[prior]]$module %in% c("outcomes_parameters", "hospitalization")) {

            ll_adjs <- hpar %>%
                dplyr::filter(parameter == defined_priors[[prior]]$name) %>%
                dplyr::mutate(likadj = calc_prior_likadj(value,
                                                         defined_priors[[prior]]$likelihood$dist,
                                                         defined_priors[[prior]]$likelihood$param
                )) %>%
                dplyr::select(subpop, likadj)

        } else if (hierarchical_stats[[stat]]$module == "seir_parameters") {
            stop("We currently do not support priors on seir parameters, since we don't do inference on them except via npis.")
        } else {
            stop("unsupported prior module")
        }

        ##probably a more efficient what to do this, but unclear...
        likelihood_data <- dplyr::left_join(likelihood_data, ll_adjs, by = obs_subpop) %>%
            dplyr::mutate(ll = ll + likadj) %>%
            dplyr::select(-likadj)
    }

    if(any(is.na(likelihood_data$ll))) {
        print("Full Likelihood")
        print(likelihood_data)
        print("NA only Likelihoods")
        print(likelihood_data[is.na(likelihood_data$ll), ])
        stop("The likelihood was NA")
    }

    return(likelihood_data)
}



##'
##' Function that performs the necessary file copies the end of an MCMC iteration of
##' inference_slot.
##'
##'@param current_index the current index in the run
##'@param slot what is the current slot number
##'@param block what is the current block
##'@param run_id what is the id of this run
##'@param global_local_prefix the prefix to be put on both global and local runs.
##'@param gf_prefix the prefix for the directory containing the current globally accepted files.
##'@param global_block_prefix prefix that describes this block.
##'
##'@return TRUE if this succeeded.
##'
##'@export
##'
perform_MCMC_step_copies_global <- function(current_index,
                                            slot,
                                            block,
                                            run_id,
                                            global_intermediate_filepath_suffix,
                                            slotblock_filename_prefix,
                                            slot_filename_prefix
) {

    rc_file_types <- c("seed", "init", "seir", "hosp", "llik", "snpi", "hnpi", "spar", "hpar")
    rc_file_ext <- c("csv", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet")

    rc <- list()

    if(current_index != 0){

        #move files from global/intermediate/slot.block.run to global/final/slot

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_gf")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = global_intermediate_filepath_suffix,
                                              filename_prefix = slotblock_filename_prefix,
                                              index = current_index,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = "global/final",
                                              filename_prefix = "",
                                              index=slot,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }


        #move files from global/intermediate/slot.block.run to global/intermediate/slot.block

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_block")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = global_intermediate_filepath_suffix,
                                              filename_prefix = slotblock_filename_prefix,
                                              index = current_index,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = global_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index=block,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }

    } else {

        #move files from global/intermediate/slot.(block-1) to global/intermediate/slot.block

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_prevblk")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = global_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index = block - 1,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = global_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index=block,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }
    }

    return(rc)
}


##'
##' Function that performs the necessary file copies the end of an MCMC iteration of
##' inference_slot.
##'
##'@param current_index the current index in the run
##'@param slot what is the current slot number
##'@param block what is the current block
##'@param run_id what is the id of this run
##'@param chimeric_local_prefix the prefix to be put on both chimeric and local runs.
##'@param cf_prefix the prefix for the directory containing the current chimericly accepted files.
##'@param chimeric_block_prefix prefix that describes this block.
##'
##'@return TRUE if this succeeded.
##'
##'@export
##'
perform_MCMC_step_copies_chimeric <- function(current_index,
                                              slot,
                                              block,
                                              run_id,
                                              chimeric_intermediate_filepath_suffix,
                                              slotblock_filename_prefix,
                                              slot_filename_prefix) {


    rc_file_types <- c("seed", "init", "llik", "snpi", "hnpi", "spar", "hpar")
    rc_file_ext <- c("csv", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet")

    rc <- list()

    if(current_index != 0){

        #move files from chimeric/intermediate/slot.block.run to chimeric/final/slot

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_gf")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = chimeric_intermediate_filepath_suffix,
                                              filename_prefix = slotblock_filename_prefix,
                                              index = current_index,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = "chimeric/final",
                                              filename_prefix = "",
                                              index=slot,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }


        #move files from chimeric/intermediate/slot.block.run to chimeric/intermediate/slot.block

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_block")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = chimeric_intermediate_filepath_suffix,
                                              filename_prefix = slotblock_filename_prefix,
                                              index = current_index,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = chimeric_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index=block,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }

    } else {

        #move files from chimeric/intermediate/slot.(block-1) to chimeric/intermediate/slot.block

        for (i in 1:length(rc_file_types)){
            rc[[paste0(rc_file_types[i], "_prevblk")]] <- file.copy(
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = chimeric_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index = block - 1,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                flepicommon::create_file_name(run_id = run_id,
                                              prefix = setup_prefix,
                                              filepath_suffix = chimeric_intermediate_filepath_suffix,
                                              filename_prefix = slot_filename_prefix,
                                              index=block,
                                              type = rc_file_types[i],
                                              extension = rc_file_ext[i]),
                overwrite = TRUE
            )
        }
    }

    return(rc)

}

## Create a list with a filename of each type/extension.  A convenience function for consistency in file names
#' @export
create_filename_list <- function(
        run_id,
        prefix,
        filepath_suffix,
        filename_prefix,
        index,
        types = c("seed", "init", "seir", "snpi", "hnpi", "spar", "hosp", "hpar", "llik"),
        extensions = c("csv", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet")) {

    if(length(types) != length(extensions)){
        stop("Please specify the same number of types and extensions.  Given",length(types),"and",length(extensions))
    }
    rc <- mapply(
        x=types,
        y=extensions,
        function(x,y){
            flepicommon::create_file_name(run_id = run_id,
                                          prefix = prefix,
                                          filepath_suffix = filepath_suffix,
                                          filename_prefix = filename_prefix,
                                          index = index,
                                          type = x,
                                          extension = y,
                                          create_directory = TRUE)
        }
    )
    names(rc) <- paste(names(rc),"filename",sep='_')
    return(rc)
}

##'@name initialize_mcmc_first_block
##'@title initialize_mcmc_first_block
##'@param slot what is the current slot number
##'@param block what is the current block
##'@param run_id what is the id of this run
##'@param global_intermediate_filepath_suffix the suffix to use for global files
##'@param chimeric_intermediate_filepath_suffix the suffix to use for chimeric files
##'@param filename_prefix the prefix to use for all files
##'@param gempyor_inference_runner An already initialized copy of python inference runner
##'@export
initialize_mcmc_first_block <- function(
        run_id,
        block,
        setup_prefix,
        global_intermediate_filepath_suffix,
        chimeric_intermediate_filepath_suffix,
        filename_prefix,
        gempyor_inference_runner,
        likelihood_calculation_function,
        is_resume = FALSE) {

    global_types <- c("seed", "init", "seir", "snpi", "hnpi", "spar", "hosp", "hpar", "llik")
    global_extensions <- c("csv", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet")
    chimeric_types <- c("seed", "init", "snpi", "hnpi", "spar", "hpar", "llik")
    chimeric_extensions <- c("csv", "parquet", "parquet", "parquet", "parquet", "parquet", "parquet")
    non_llik_types <- paste(c("seed", "init", "seir", "snpi", "hnpi", "spar", "hosp", "hpar"), "filename", sep = "_")

    # Get names of files saved at end of previous block, to initiate this block from
    # makes file names of the form {setup_prefix}/{run_id}/{global_type}/global/intermediate/{filename_prefix}.(block-1).{run_id}.{global_type}.{ext}
    global_files <- create_filename_list(run_id=run_id,  prefix=setup_prefix, filepath_suffix=global_intermediate_filepath_suffix, filename_prefix = filename_prefix, index=block - 1, types=global_types, extensions=global_extensions)
    # makes file names of the form {setup_prefix}/{run_id}/{chimeric_type}/chimeric/intermediate/{filename_prefix}.(block-1).{run_id}.{chimeric_type}.{ext}
    chimeric_files <- create_filename_list(run_id=run_id, prefix=setup_prefix, filepath_suffix=chimeric_intermediate_filepath_suffix, filename_prefix = filename_prefix, index=block - 1, types=chimeric_types, extensions=chimeric_extensions)

    ## If this isn't the first block, all of the files should definitely exist

    global_check <- sapply(global_files, file.exists)
    chimeric_check <- sapply(chimeric_files, file.exists)

    if (block > 1) {

        if (any(!global_check)) {
            stop(paste(
                "Could not find file",
                names(global_files)[!global_check],
                ":",
                global_files[!global_check],
                "needed to resume",
                collapse = "\n"
            ))
        }
        if (any(!chimeric_check)) {
            stop(paste(
                "Could not find file",
                names(chimeric_files)[!chimeric_check],
                ":",
                chimeric_files[!chimeric_check],
                "needed to resume",
                collapse = "\n"
            ))
        }
        return(TRUE)
    }
    # TODO: no check has been added for init files below
    if (is_resume) {
        print(global_check)
        important_global_check <- global_check[
            !(names(global_check) %in% c("llik_filename", "hosp_filename", "seir_filename", "seed_filename", "init_filename"))
        ]
        if (!all(important_global_check)) {
            all_file_types <- names(important_global_check)
            missing_file_types <- names(important_global_check)[!important_global_check]
            missing_files <- global_files[missing_file_types]
            stop(paste(
                "For a resume, all global files must be present.",
                "Could not find the following file types:",
                paste(missing_file_types, collapse = ", "),
                "\nWas expecting the following files:",
                paste(all_file_types, collapse = ", "),
                "\nLooking for them in these files",
                paste(missing_files, collapse = ", ")
            ))
        }
    }

    if (any(global_check)) {
        warning(paste(
            "Found file",
            names(global_files)[global_check],
            "when creating first block, using that",
            collapse = "\n"
        ))
    }

    if (any(chimeric_check)) {
        warning(paste(
            "Found file",
            names(global_files)[chimeric_check],
            "when creating first block, ignoring that file and replacing with global",
            collapse = "\n"
        ))
    }

    global_file_names <- names(global_files[!global_check]) # names are of the form "variable_filename", only files that DONT already exist will be in this list


    ## seed
    if (!is.null(config$seeding)){
        if ("seed_filename" %in% global_file_names) {
            print("need to create seeding directory")
            if(!file.exists(config$seeding$lambda_file)) {
                print("Will create seeding lambda file using flepimop/main_scripts/create_seeding.R") #Need to document this
                err <- system(paste(
                    opt$rpath,
                    paste(opt$flepi_path, "flepimop", "main_scripts", "create_seeding.R", sep = "/"),
                    "-c", opt$config
                ))
                if (err != 0) {
                    stop("Could not run seeding")
                }
            }
            print("Will copy seeding lambda file to the seeding directory")
            err <- !(file.copy(config$seeding$lambda_file, global_files[["seed_filename"]]))
            if (err != 0) {
                stop("Could not copy seeding")
            }
        }

        # additional seeding for new variants or introductions to add to fitted seeding (for resumes)
        #   need to document!!
        if (!is.null(config$seeding$added_seeding) & is_resume & block <= 1){
            if(!file.exists(config$seeding$added_seeding$added_lambda_file)) {
                err <- system(paste(
                    opt$rpath,
                    paste(opt$flepi_path, "flepimop", "main_scripts", "create_seeding_added.R", sep = "/"),
                    "-c", opt$config
                ))
                if (err != 0) {
                    stop("Could not run added seeding")
                }
            }

            # load and add to original seeding
            seed_new <-  readr::read_csv(global_files[["seed_filename"]],show_col_types = FALSE)
            added_seeding <- readr::read_csv(config$seeding$added_seeding$added_lambda_file,show_col_types = FALSE)

            if (!is.null(config$seeding$added_seeding$fix_original_seeding) &&
                config$seeding$added_seeding$fix_original_seeding){
                seed_new$no_perturb <- TRUE
            }
            if (!is.null(config$seeding$added_seeding$fix_added_seeding)  &&
                config$seeding$added_seeding$fix_added_seeding){
                added_seeding$no_perturb <- TRUE
            }

            if (!is.null(config$seeding$added_seeding$filter_previous_seedingdates) &&
                config$seeding$added_seeding$filter_previous_seedingdates){
                seed_new <- seed_new %>%
                    dplyr::filter(date < lubridate::as_date(config$seeding$added_seeding$start_date) &
                                      date > lubridate::as_date(config$seeding$added_seeding$end_date))
            }
            seed_new <- seed_new %>% dplyr::bind_rows(added_seeding)

            readr::write_csv(seed_new, global_files[["seed_filename"]])
        }
    }


    ## initial conditions (init)

    if (!is.null(config$initial_conditions)){
      if(config$initial_conditions$method  != "plugin"){

        if ("init_filename" %in% global_file_names) {

          if (config$initial_conditions$method %in% c("FromFile", "SetInitialConditions")){

            if (is.null(config$initial_conditions$initial_conditions_file)) {
              stop("ERROR: Initial conditions file needs to be specified in the config under `initial_conditions:initial_conditions_file`")
            }
            initial_init_file <- config$initial_conditions$initial_conditions_file

          } else if (config$initial_conditions$method %in% c("InitialConditionsFolderDraw", "SetInitialConditionsFolderDraw")) {
            print("Initial conditions in inference has not been fully implemented yet for the 'folder draw' methods,
                      and no copying to global or chimeric files is being done.")

            if (is.null(config$initial_conditions$initial_file_type)) {
              stop("ERROR: Initial conditions file needs to be specified in the config under `initial_conditions:initial_conditions_file`")
            }
            initial_init_file <- global_files[[paste0(config$initial_conditions$initial_file_type, "_filename")]]
          }


          if (!file.exists(initial_init_file)) {
            stop("ERROR: Initial conditions file specified but does not exist.")
          }

          if (grepl(".csv", initial_init_file)){
            initial_init <- readr::read_csv(initial_init_file,show_col_types = FALSE)
          }else{
            initial_init <- arrow::read_parquet(initial_init_file)
          }

          # if the initial conditions file contains a 'date' column, filter for config$start_date

          if("date" %in% colnames(initial_init)){

            initial_init <- initial_init %>%
              dplyr::mutate(date = as.POSIXct(date, tz="UTC")) %>%
              dplyr::filter(date == as.POSIXct(paste0(config$start_date, " 00:00:00"), tz="UTC"))

            if (nrow(initial_init) == 0) {
              stop("ERROR: Initial conditions file specified but does not contain the start date.")
            }

          }

          arrow::write_parquet(initial_init, global_files[["init_filename"]])
        }

        # if the initial conditions file contains a 'date' column, filter for config$start_date
        if (grepl(".csv", global_files[["init_filename"]])){
          initial_init <- readr::read_csv(global_files[["init_filename"]],show_col_types = FALSE)
        }else{
          initial_init <- arrow::read_parquet(global_files[["init_filename"]])
        }

        if("date" %in% colnames(initial_init)){

          initial_init <- initial_init %>%
            dplyr::mutate(date = as.POSIXct(date, tz="UTC")) %>%
            dplyr::filter(date == as.POSIXct(paste0(config$start_date, " 00:00:00"), tz="UTC"))

          if (nrow(initial_init) == 0) {
            stop("ERROR: Initial conditions file specified but does not contain the start date.")
          }

        }
        arrow::write_parquet(initial_init, global_files[["init_filename"]])
      }else if(config$initial_conditions$method  == "plugin"){
        print("Initial conditions files generated by gempyor using plugin method.")
      }
    }


    ## seir, snpi, spar

    checked_par_files <- c("snpi_filename", "spar_filename", "hnpi_filename", "hpar_filename")
    checked_sim_files <- c("seir_filename", "hosp_filename")
    # These functions save variables to files of the form variable/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.(block-1),runID.variable.ext
    if (any(checked_par_files %in% global_file_names)) {
        if (!all(checked_par_files %in% global_file_names)) {
            stop("Provided some GempyorInference input, but not all")
        }
        if (any(checked_sim_files %in% global_file_names)) {
            if (!all(checked_sim_files %in% global_file_names)) {
                stop("Provided only one of hosp or seir input file, with some output files. Not supported anymore")
            }
            tryCatch({
                gempyor_inference_runner$one_simulation(sim_id2write = block - 1)
            }, error = function(e) {
                print("GempyorInference failed to run (call on l. 687 of inference_slot_runner_funcs.R).")
                print("Here is all the debug information I could find:")
                for(m in reticulate::py_last_error()) print(m)
                stop("GempyorInference failed to run... stopping")
            })
            #gempyor_inference_runner$one_simulation(sim_id2write = block - 1)
        } else {
            stop("Provided some GempyorInference output(seir, hosp), but not GempyorInference input")
        }
    } else {
        if (any(checked_sim_files %in% global_file_names)) {
            if (!all(checked_sim_files %in% global_file_names)) {
                stop("Provided only one of hosp or seir input file, not supported anymore")
            }
            warning("SEIR and Hosp input provided, but output not found. This is unstable for stochastic runs")
            tryCatch({
                gempyor_inference_runner$one_simulation(sim_id2write = block - 1, load_ID = TRUE, sim_id2load = block - 1)
            }, error = function(e) {
                print("GempyorInference failed to run (call on l. 687 of inference_slot_runner_funcs.R).")
                print("Here is all the debug information I could find:")
                for(m in reticulate::py_last_error()) print(m)
                stop("GempyorInference failed to run... stopping")
            })
            #gempyor_inference_runner$one_simulation(sim_id2write=block - 1, load_ID=TRUE, sim_id2load=block - 1)
        }
    }

    ## llik
    if (!("llik_filename" %in% global_file_names)) {
        stop("Please do not provide a likelihood file")
    }

    extension <- gsub(".*[.]", "", global_files[["hosp_filename"]])
    hosp_data <- flepicommon::read_file_of_type(extension)(global_files[["hosp_filename"]])

    ## Refactor me later:
    global_likelihood_data <- likelihood_calculation_function(hosp_data)
    arrow::write_parquet(global_likelihood_data, global_files[["llik_filename"]]) # save global likelihood data to file of the form llik/name/seir_modifiers_scenario/outcome_modifiers_scenario/run_id/global/intermediate/slot.(block-1).run_ID.llik.ext

    #print("from inside initialize_mcmc_first_block: column names of likelihood dataframe")
    #print(colnames(global_likelihood_data))

    for (type in names(chimeric_files)) {
        file.copy(global_files[[type]], chimeric_files[[type]], overwrite = TRUE) # copy files that were in global directory into chimeric directory
    }
    return()
}
