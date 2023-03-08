## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(comment = "#>", collapse = TRUE)

## ----echo=FALSE---------------------------------------------------------------
simple_inline <- "#' @title Title `r 1 + 1`
#' @description Description `r 2 + 2`
#' @md
foo <- function() NULL
"

## ----code=simple_inline-------------------------------------------------------
#' @title Title `r 1 + 1`
#' @description Description `r 2 + 2`
#' @md
foo <- function() NULL


## ----code = roxygen2:::markdown(simple_inline)--------------------------------
#' @title Title 2
#' @description Description 4
#' @md
foo <- function() NULL

## ----echo=FALSE---------------------------------------------------------------
backtick <- "#' Title
#' Description `r paste0('\\x60', 'bar', '\\x60')`
#' @md
foo <- function() NULL
"

## ----code = backtick----------------------------------------------------------
#' Title
#' Description `r paste0('\x60', 'bar', '\x60')`
#' @md
foo <- function() NULL


## ----code = roxygen2:::markdown_pass1(backtick)-------------------------------
#' Title
#' Description `bar`
#' @md
foo <- function() NULL


## ----code = roxygen2:::markdown(backtick)-------------------------------------
#' Title
#' Description \code{bar}
#' @md
foo <- function() NULL

## ---- echo=FALSE, results="asis"----------------------------------------------
cat(
  "\x60\x60\x60rd\n",
  format(roxygen2:::roc_proc_text(roxygen2::rd_roclet(), backtick)[[1]]),
  "\n\x60\x60\x60"
)

## ----echo=FALSE---------------------------------------------------------------
simple_fenced <- "#' @title Title
#' @details Details
#' ```{r lorem}
#' 1+1
#' ```
#' @md
foo <- function() NULL
"

## ----code=simple_fenced-------------------------------------------------------
#' @title Title
#' @details Details
#' ```{r lorem}
#' 1+1
#' ```
#' @md
foo <- function() NULL


## ----lorem--------------------------------------------------------------------
1+1

## ---- echo=FALSE, results="asis"----------------------------------------------
cat(
  "\x60\x60\x60rd\n",
  format(roxygen2:::roc_proc_text(roxygen2::rd_roclet(), simple_fenced)[[1]]),
  "\n\x60\x60\x60"
)

## ----include=FALSE------------------------------------------------------------
code_envs <- "#' Title `r baz <- 420` `r baz`
#'
#' Description `r exists('baz', inherits = FALSE)`
#' @md
bar <- function() NULL

#' Title
#' 
#' Description `r exists('baz', inherits = FALSE)`
#' @md
zap <- function() NULL
"

## ----code=code_envs-----------------------------------------------------------
#' Title `r baz <- 420` `r baz`
#'
#' Description `r exists('baz', inherits = FALSE)`
#' @md
bar <- function() NULL

#' Title
#' 
#' Description `r exists('baz', inherits = FALSE)`
#' @md
zap <- function() NULL


## ---- echo=FALSE, results="asis"----------------------------------------------
cat(
  "\x60\x60\x60rd\n",
  format(roxygen2:::roc_proc_text(roxygen2::rd_roclet(), code_envs)[[1]]),
  format(roxygen2:::roc_proc_text(roxygen2::rd_roclet(), code_envs)[[2]]),
  "\n\x60\x60\x60"
)

## -----------------------------------------------------------------------------
#' \enumerate{
#'   \item First item
#'   \item Second item
#' }

## -----------------------------------------------------------------------------
#' \itemize{
#'   \item First item
#'   \item Second item
#' }

## -----------------------------------------------------------------------------
#' \describe{
#'   \item{One}{First item}
#'   \item{Two}{Second item}
#' }

## -----------------------------------------------------------------------------
tabular <- function(df, ...) {
  stopifnot(is.data.frame(df))

  align <- function(x) if (is.numeric(x)) "r" else "l"
  col_align <- purrr::map_chr(df, align)

  cols <- lapply(df, format, ...)
  contents <- do.call("paste",
    c(cols, list(sep = " \\tab ", collapse = "\\cr\n#'   ")))

  paste("#' \\tabular{", paste(col_align, collapse = ""), "}{\n#'   ",
    paste0("\\strong{", names(df), "}", sep = "", collapse = " \\tab "), " \\cr\n#'   ",
    contents, "\n#' }\n", sep = "")
}

cat(tabular(mtcars[1:5, 1:5]))

