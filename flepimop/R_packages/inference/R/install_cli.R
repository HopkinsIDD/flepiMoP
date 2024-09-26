
#' @title Install Inference Scripts
#' 
#' @description
#' Installs the scripts for R-based FlepiMoP inference.
#' 
#' @param path The path to install the scripts to. Default is `usr/local/bin` (unix-like).
#' 
#' @param overwrite Whether to overwrite existing scripts. Default is `TRUE`. see [base::file.copy()].
#' 
#' @export
install_cli <- function(
  path = if (.Platform$OS.type == "unix") file.path("usr", "local", "bin") else stop("Unsupported OS"),
  overwrite = TRUE
) {
  scriptfiles <- list.files(
    system.file("scripts", package = utils::packageName()), pattern = "flepimop-.*", full.names = TRUE
  )
  from <- scriptfiles
  to <- file.path(path, gsub("\.R$", "", basename(scriptfiles)))
  file.copy(from, to, overwrite = overwrite)
}