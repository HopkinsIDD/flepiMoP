

# SETUP -------------------------------------------------------------------

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(flepicommon)


option_list = list(
    optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
    optparse::make_option(c("-r", "--rpath"), action="store", default=Sys.getenv("RSCRIPT_PATH","Rscript"), type = 'character', help = "path to R executable"),
    optparse::make_option(c("-p", "--flepi_path"), action="store", type='character', default = Sys.getenv("FLEPI_PATH", "flepiMoP/"), help="path to the flepiMoP directory")
    # optparse::make_option(c("-R", "--resume_location"), action="store", default=Sys.getenv("RESUME_LOCATION", NA), type = 'character', help = "Is this run a resume")
)

opt = optparse::parse_args(optparse::OptionParser(option_list=option_list))
# opt$is_resume <- !is.na(opt$resume_location)

config <- flepicommon::load_config(opt$c)
if (length(config) == 0) {
    stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}





# RUN INITIAL SEEDING -----------------------------------------------------

## Run initial seeding
if(!file.exists(config$seeding$lambda_file)) {
    err <- system(paste(
        opt$rpath,
        paste(opt$flepi_path, "flepimop", "main_scripts", "create_seeding.R", sep = "/"),
        "-c", opt$config
    ))
    if (err != 0) {
        stop("Could not run seeding")
    }
}



# RUN ADDED SEEDING -----------------------------------------------------


# Run additional seeding for new variants or introductions to add to fitted seeding (for resumes)
if (!is.null(config$seeding$added_seeding)){
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
}
