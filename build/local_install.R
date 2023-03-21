# set-up the repository
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org"
       options(repos=r)
})

# Installs the custom-made packages in this repository

library(devtools)

install.packages(c("covidcast","data.table","vroom","dplyr","RSocrata"), force=TRUE)
# devtools::install_github("hrbrmstr/cdcfluview")

# To run if operating in the container
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)]) # get the name of this file, by looking for the option "--file" in the arguments that were used to start this R instance and getting the term that comes after
pkg.dir <- paste0(dirname(script.name), "../R_packages/") # find the directory that this file is within

#list of local packages (reorder so flepicommon is installed first)
loc_pkgs <- list.files(pkg.dir,full.names=TRUE)
loc_pkgs <- loc_pkgs[c(which(grepl("flepicommon", loc_pkgs)), which(!grepl("flepicommon", loc_pkgs)))]

# Install them
install.packages(loc_pkgs,type='source',repos=NULL)

# to run within a local instance of R studio

#install.packages(list.files("../flepimop/R_packeages/",full.names=TRUE),type='source',repos=NULL) #install from files. Run from flepiMoP folder. Might need to run twice since packages are interdependent and might not be installed in correct order
# devtools::install_github("HopkinsIDD/globaltoolboxlite") #install the covidimportation package from a separate Github repo
# devtools::install_github("HopkinsIDD/covidImportation")
#
