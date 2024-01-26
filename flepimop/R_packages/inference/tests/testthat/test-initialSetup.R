context("initial MCMC setup")

test_that("initialize_mcmc_first_block works for block > 1",{
#     
#     filenames <- c(
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "global", 
#             filename_prefix = "filename_prefix",
#             index = 1,
#             types = c("seed","init", "seir", "snpi", "spar", "hosp", "hnpi", "hpar","llik"),
#             extensions = c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         ),
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "chimeric", 
#             filename_prefix = "filename_prefix",
#             index = 1,
#             c("seed","init", "seir", "snpi", "spar", "hosp", "hnpi", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         )
#     )
#     
#     expect_false({
#         suppressWarnings(unlink("model_output",recursive=TRUE))
#         # suppressWarnings(lapply(filenames,file.remove))
#         any(file.exists(filenames))
#     })
#     
#     expect_error({
#         initialize_mcmc_first_block(
#             run_id = "test_run",
#             block = 2,
#             setup_prefix = "tests",
#             global_intermediate_filepath_suffix = "global",
#             chimeric_intermediate_filepath_suffix = "chimeric",
#             filename_prefix = "filename_prefix",
#             gempyor_inference_runner = NULL,
#             likelihood_calculation_function = NULL,
#             is_resume = FALSE
#         )
#     })
#     
#     filenames <- c(
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "global", 
#             filename_prefix = "filename_prefix",
#             index = 1,
#             c("seed", "init", "seir", "snpi", "spar", "hosp", "hnpi", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         ),
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "chimeric", 
#             filename_prefix = "filename_prefix",
#             index = 1,
#             c("seed","init", "seir", "snpi", "spar", "hosp", "hnpi", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         )
#     )
#     
#     expect_true({
#         lapply(filenames,function(x){write.csv(file=x,data.frame(missing=TRUE))})
#         all(file.exists(filenames))
#     })
#     
#     expect_error({
#         initialize_mcmc_first_block(
#             run_id = "test_run",
#             block = 2,
#             setup_prefix = "prefix",
#             global_intermediate_filepath_suffix = "global",
#             chimeric_intermediate_filepath_suffix = "chimeric",
#             filename_prefix = "filename_prefix",
#             gempyor_inference_runner = NULL,
#             likelihood_calculation_function = NULL,
#             is_resume = FALSE
#         )
#     }, NA)
#     
#     expect_false({
#         suppressWarnings(unlink("model_output",recursive=TRUE))
#         any(file.exists(filenames))
#     })
#     
# })
# 
# 
# 
# test_that("initialize_mcmc_first_block works for block < 1",{
#     
#     filenames <- c(
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "global", 
#             filename_prefix = "filename_prefix",
#             index = -1,
#             c("seed","init",  "seir", "snpi", "spar", "hosp", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         ),
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "chimeric", 
#             filename_prefix = "filename_prefix",
#             index = -1,
#             c("seed","init",  "seir", "snpi", "spar", "hosp", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet", "parquet")
#         )
#     )
#     
#     expect_false({
#         suppressWarnings(unlink("model_output",recursive=TRUE))
#         # suppressWarnings(lapply(filenames,file.remove))
#         all(file.exists(filenames))
#     })
#     
#     expect_error({
#         initialize_mcmc_first_block(
#             run_id = "test_run",
#             block = 0,
#             setup_prefix = "tests",
#             global_intermediate_filepath_suffix = "global",
#             chimeric_intermediate_filepath_suffix = "chimeric",
#             filename_prefix = "filename_prefix",
#             gempyor_inference_runner = NULL,
#             likelihood_calculation_function = NULL,
#             is_resume = FALSE
#         )
#     })
#     
#     filenames <- c(
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "global", 
#             filename_prefix = "filename_prefix",
#             index = -1,
#             c("seed","init", "seir", "snpi", "spar", "hosp", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         ),
#         create_filename_list(
#             run_id = "test_run",
#             prefix = "tests",
#             filepath_suffix = "chimeric", 
#             filename_prefix = "filename_prefix",
#             index = -1,
#             c("seed", "init", "seir", "snpi", "spar", "hosp", "hpar","llik"),
#             c("csv","parquet","parquet","parquet","parquet","parquet","parquet","parquet")
#         )
#     )
#     
#     expect_true({
#         lapply(filenames,function(x){write.csv(file=x,data.frame(missing=TRUE))})
#         all(file.exists(filenames))
#     })
#     
#     expect_error({
#         initialize_mcmc_first_block(
#             run_id = "test_run",
#             block = 0,
#             setup_prefix = "tests",
#             global_intermediate_filepath_suffix = "global",
#             chimeric_intermediate_filepath_suffix = "chimeric",
#             filename_prefix = "filename_prefix",
#             gempyor_inference_runner = NULL,
#             likelihood_calculation_function = NULL,
#             is_resume = FALSE
#         )
#     })
#     
#     expect_false({
#         suppressWarnings(unlink("model_output",recursive=TRUE))
#         any(file.exists(filenames))
#     })
    
})
