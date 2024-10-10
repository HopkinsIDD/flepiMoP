#!/usr/bin/env Rscript

# Helper functions
split_pkgs <- \(x) unique(unlist(strsplit(gsub("\\s+", "", x), ",")))

# Light argument parsing
args <- commandArgs(trailingOnly = TRUE)
flepi_path <- if (length(args)) args[1L] else getwd()

# Get R package dependencies
rpkgs <- list.files(
  file.path(flepi_path, "flepimop", "R_packages"),
  full.names = TRUE
)
dependencies <- sapply(rpkgs, function(rpkg) {
  description <- read.dcf(file.path(rpkg, "DESCRIPTION"))
  sections <- c("Depends", "Imports")
  contained_sections <- sections %in% colnames(description)
  if (sum(contained_sections) >= 1L) {
    return(split_pkgs(description[, sections[contained_sections]]))
  }
  character()
}, USE.NAMES = FALSE)
dependencies <- sort(unique(unlist(dependencies)))
dependencies <- setdiff(
  dependencies,
  c("arrow", "covidcast", "methods", basename(rpkgs))
)
dependencies <- dependencies[!grepl("^R(\\(.*\\))?$", dependencies)]

# Construct environment.yml file
environment_yml <- file.path(flepi_path, "environment.yml")
new_environment_yml <- c(
  "channels:",
  "- conda-forge",
  "- defaults",
  "- r",
  "- dnachun",
  "dependencies:",
  "- python=3.10",
  "- pip",
  "- r-base>=4.3",
  "- pyarrow=17.0.0",
  "- r-arrow=17.0.0",
  "- r-sf",
  paste0("- r-", dependencies)
)
if (file.exists(environment_yml)) {
  old_environment_yml <- readLines(environment_yml)
} else {
  old_environment_yml <- character()
}
old_environment_yml <- old_environment_yml[!grepl("^#", old_environment_yml)]
if (!identical(new_environment_yml, old_environment_yml)) {
  new_environment_yml <- c(
    paste0("# ", format(Sys.time(), "%a %b %d %X %Y %Z")),
    new_environment_yml
  )
  writeLines(new_environment_yml, environment_yml)
}
