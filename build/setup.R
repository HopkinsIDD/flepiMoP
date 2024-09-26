#!/usr/bin/env Rscript

.args <- commandArgs(trailingOnly = TRUE)

if (length(.args) != 1) {
  stop("Usage: setup.R <flepimop-path>")
}

# TODO sniff for upgrade mode

if (!require(remotes)) {
  install.packages("remotes", repos = c(getOption("repos"), "http://cran.r-project.org"));
  stopifnot("Could not load `remotes` package." = require(remotes))
}

rpkgs <- list.files(file.path(.args[1], "flepimop", "R_packages"), full.names = TRUE)

for (pkg in rpkgs) {
  install.packages(pkg, repos = NULL, type = "source")
}

# other dependencies for analysis scripts
for (pkg in c("ggfortify", "flextable", "optparse", "cowplot")) {
  if (!requireNamespace(pkg, quietly = TRUE)) { 
    install.packages(pkg, repos = c(getOption("repos"), "http://cran.r-project.org"));
    if (!requireNamespace(pkg)) {
       stop(sprintf("Could not install and/or load `%s` package.", pkg))
    }
  }
}

# install the R scripts as executables
inference::install_cli()