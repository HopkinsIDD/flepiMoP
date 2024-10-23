
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
  path = {
    condapth <- Sys.getenv("CONDA_PREFIX")
    if (condapth != "") { 
      file.path(condapth, "bin")
    } else {
      stop("only support default path installation when conda is running.")
    }
  }
) {
  scriptfiles <- list.files(
    system.file("scripts", package = utils::packageName()), pattern = "flepimop-.*", full.names = TRUE
  )
  from <- scriptfiles
  to <- file.path(path, gsub("\\.R$", "", basename(scriptfiles)))
  to_remove <- file.exists(to)
  if (any(to_remove)) {
    file.remove(to[to_remove])
  }
  file.symlink(from, to)
}
