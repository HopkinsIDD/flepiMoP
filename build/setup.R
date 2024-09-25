#!/usr/bin/env Rscript

# TODO sniff for upgrade mode

if (!require(remotes)) {
  install.packages("remotes", repos = c(getOption("repos"), "http://cran.r-project.org"));
  stopifnot("Could not load `remotes` package." = require(remotes))
}

for (pkg in c("flepicommon", "flepiconfig", "inference")) {
  if (!requireNamespace(pkg, quietly = TRUE)) remotes::install_github("HopkinsIDD/flepiMoP", subdir = sprintf("flepimop/R_packages/%s", pkg))
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
