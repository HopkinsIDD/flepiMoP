context("perform_MCMC_step_copies")


##THESE TESTS CAN BE MADE MORE DETAILED...JUST MAKING PLACE HOLDERS
test_that("MCMC step copies (global) are correctly performed when we are not at the start of a block", {
    
    
    
    skip("These tests need to be revised to work with new file structures.")
    ## ** NEED TO REVISE TO WORK!!! ***
    
    ##some information on our phantom runs
    current_index <- 2
    slot <- 2
    block <- 5
    run_id <- "TEST_RUN"
    slot_prefix <- flepicommon::create_prefix("config","seir_modifiers_scenario","outcome_modifiers_scenario",run_id,sep='/',trailing_separator='/')
    gf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','final',sep='/',trailing_separator='/')
    gi_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','intermediate',sep='/',trailing_separator='/')
    global_block_prefix <- flepicommon::create_prefix(prefix=gi_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')
    global_local_prefix <- flepicommon::create_prefix(prefix=global_block_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')
    slotblock_filename_prefix <- flepicommon::create_prefix(slot=list(slot,"%09d"), block=list(block,"%09d"), sep='.', trailing_separator='.')

    
    ##To be save make a directory
    dir.create("MCMC_step_copy_test")
    setwd("MCMC_step_copy_test")
    ##get file names
    seed_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'seed','csv')
    init_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'init','parquet')
    seir_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'seir','parquet')
    hosp_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'hosp','parquet')
    llik_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'llik','parquet')
    snpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'snpi','parquet')
    spar_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'spar','parquet')
    hnpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'hnpi','parquet')
    hpar_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_local_prefix, filepath_suffix=gi_prefix, filename_prefix=slotblock_filename_prefix, index=current_index,'hpar','parquet')



    ##create the copy from  files
    readr::write_csv(data.frame(file="seed"), seed_src)
    arrow::write_parquet(data.frame(file="init"), init_src)
    arrow::write_parquet(data.frame(file="seir"), seir_src)
    arrow::write_parquet(data.frame(file="hosp"), hosp_src)
    arrow::write_parquet(data.frame(file="llik"), llik_src)
    arrow::write_parquet(data.frame(file="snpi"), snpi_src)
    arrow::write_parquet(data.frame(file="spar"), spar_src)
    arrow::write_parquet(data.frame(file="hnpi"), hnpi_src)
    arrow::write_parquet(data.frame(file="hpar"), hpar_src)

    ##print(hosp_src)
    ##print(flepicommon::create_file_name(run_id=run_id, prefix=gf_prefix,slot,'hosp','parquet'))

    res <- perform_MCMC_step_copies_global(current_index,
                                    slot,
                                    block,
                                    run_id,
                                    global_local_prefix,
                                    gf_prefix,
                                    global_block_prefix)


    expect_equal(prod(unlist(res)),1)

    ##clean up
    setwd("..")
    unlink("MCMC_step_copy_test", recursive=TRUE)

})


test_that("MCMC step copies (global) are correctly performed when we are at the start of a block", {
    ##some information on our phantom runs
    current_index <- 0
    slot <- 2
    block <- 5
    run_id <- "TEST_RUN"
    slot_prefix <- flepicommon::create_prefix("config","seir_modifiers_scenario","outcome_modifiers_scenario",run_id,sep='/',trailing_separator='/')
    gf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','final',sep='/',trailing_separator='/')
    gi_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'global','intermediate',sep='/',trailing_separator='/')
    global_block_prefix <- flepicommon::create_prefix(prefix=gi_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')
    global_local_prefix <- flepicommon::create_prefix(prefix=global_block_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')

    ##To be save make a direectory
    dir.create("MCMC_step_copy_test")
    setwd("MCMC_step_copy_test")
    ##get file names
    seed_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'seed','csv')
    init_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'init','parquet')
    seir_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'seir','parquet')
    hosp_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'hosp','parquet')
    llik_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'llik','parquet')
    snpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'snpi','parquet')
    spar_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'spar','parquet')
    hnpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'hnpi','parquet')
    hpar_src <- flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block-1,'hpar','parquet')

    ##create the copy from  files
    readr::write_csv(data.frame(file="seed"), seed_src)
    arrow::write_parquet(data.frame(file="init"), init_src)
    arrow::write_parquet(data.frame(file="seir"), seir_src)
    arrow::write_parquet(data.frame(file="hosp"), hosp_src)
    arrow::write_parquet(data.frame(file="llik"), llik_src)
    arrow::write_parquet(data.frame(file="snpi"), snpi_src)
    arrow::write_parquet(data.frame(file="spar"), spar_src)
    arrow::write_parquet(data.frame(file="hnpi"), hnpi_src)
    arrow::write_parquet(data.frame(file="hpar"), hpar_src)

    print(hosp_src)
    print(flepicommon::create_file_name(run_id=run_id, prefix=global_block_prefix,block,'hosp','parquet'))

    res <- perform_MCMC_step_copies_global(current_index,
                                    slot,
                                    block,
                                    run_id,
                                    global_local_prefix,
                                    gf_prefix,
                                    global_block_prefix)


    expect_equal(prod(unlist(res)),1)

    ##clean up
    setwd("..")
    unlink("MCMC_step_copy_test", recursive=TRUE)

})


test_that("MCMC step copies (chimeric) are correctly performed when we are not at the start of a block", {
    ##some information on our phantom runs
    current_index <- 2
    slot <- 2
    block <- 5
    run_id <- "TEST_RUN"
    slot_prefix <- flepicommon::create_prefix("config","seir_modifiers_scenario","outcome_modifiers_scenario",run_id,sep='/',trailing_separator='/')
    cf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','final',sep='/',trailing_separator='/')
    ci_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','intermediate',sep='/',trailing_separator='/')
    chimeric_block_prefix <- flepicommon::create_prefix(prefix=ci_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')
    chimeric_local_prefix <- flepicommon::create_prefix(prefix=chimeric_block_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')

    ##To be save make a directory
    dir.create("MCMC_step_copy_test")
    setwd("MCMC_step_copy_test")
    ##get file names
    seed_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'seed','csv')
    seir_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'seir','parquet')
    hosp_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'hosp','parquet')
    llik_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'llik','parquet')
    snpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'snpi','parquet')
    spar_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'spar','parquet')
    hnpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'hnpi','parquet')
    hpar_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_local_prefix,current_index,'hpar','parquet')



    ##create the copy from  files
    arrow::write_parquet(data.frame(file="seed"), seed_src)
    arrow::write_parquet(data.frame(file="seir"), seir_src)
    arrow::write_parquet(data.frame(file="hosp"), hosp_src)
    arrow::write_parquet(data.frame(file="llik"), llik_src)
    arrow::write_parquet(data.frame(file="snpi"), snpi_src)
    arrow::write_parquet(data.frame(file="spar"), spar_src)
    arrow::write_parquet(data.frame(file="hnpi"), hnpi_src)
    arrow::write_parquet(data.frame(file="hpar"), hpar_src)

    ##print(hosp_src)
    ##print(flepicommon::create_file_name(run_id=run_id, prefix=cf_prefix,slot,'hosp','parquet'))

    res <- perform_MCMC_step_copies_chimeric(current_index,
                                           slot,
                                           block,
                                           run_id,
                                           chimeric_local_prefix,
                                           cf_prefix,
                                           chimeric_block_prefix)


    expect_equal(prod(unlist(res)),1)

    ##clean up
    setwd("..")
    unlink("MCMC_step_copy_test", recursive=TRUE)


})


test_that("MCMC step copies (chimeric) are correctly performed when we are at the start of a block", {
    ##some information on our phantom runs
    current_index <- 0
    slot <- 2
    block <- 5
    run_id <- "TEST_RUN"
    slot_prefix <- flepicommon::create_prefix("config","seir_modifiers_scenario","outcome_modifiers_scenario",run_id,sep='/',trailing_separator='/')
    cf_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','final',sep='/',trailing_separator='/')
    ci_prefix <- flepicommon::create_prefix(prefix=slot_prefix,'chimeric','intermediate',sep='/',trailing_separator='/')
    chimeric_block_prefix <- flepicommon::create_prefix(prefix=ci_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')
    chimeric_local_prefix <- flepicommon::create_prefix(prefix=chimeric_block_prefix, slot=list(slot,"%09d"), sep='.',
                                                      trailing_separator='.')

    ##To be save make a direectory
    dir.create("MCMC_step_copy_test")
    setwd("MCMC_step_copy_test")
    ##get file names
    seed_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'seed','csv')
    seir_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'seir','parquet')
    hosp_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'hosp','parquet')
    llik_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'llik','parquet')
    snpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'snpi','parquet')
    spar_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'spar','parquet')
    hnpi_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'hnpi','parquet')
    hpar_src <- flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block-1,'hpar','parquet')



    ##create the copy from  files
    arrow::write_parquet(data.frame(file="seed"), seed_src)
    arrow::write_parquet(data.frame(file="seir"), seir_src)
    arrow::write_parquet(data.frame(file="hosp"), hosp_src)
    arrow::write_parquet(data.frame(file="llik"), llik_src)
    arrow::write_parquet(data.frame(file="snpi"), snpi_src)
    arrow::write_parquet(data.frame(file="spar"), spar_src)
    arrow::write_parquet(data.frame(file="hnpi"), hnpi_src)
    arrow::write_parquet(data.frame(file="hpar"), hpar_src)

    print(hosp_src)
    print(flepicommon::create_file_name(run_id=run_id, prefix=chimeric_block_prefix,block,'hosp','parquet'))

    res <- perform_MCMC_step_copies_chimeric(current_index,
                                           slot,
                                           block,
                                           run_id,
                                           chimeric_local_prefix,
                                           cf_prefix,
                                           chimeric_block_prefix)


    expect_equal(prod(unlist(res)),1)

    ##clean up
    setwd("..")
    unlink("MCMC_step_copy_test", recursive=TRUE)


})

