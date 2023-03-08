## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(comment = "#>", collapse = TRUE)

## ---- eval = FALSE------------------------------------------------------------
#  my_function <- function(x, y) {
#    pkg::fun(x) * y
#  }

## -----------------------------------------------------------------------------
#' @importFrom pkg fun 
my_function <- function(x, y) {
  fun(x) * y
}

## -----------------------------------------------------------------------------
#' @importFrom pkg fun1 fun2
#' @importFrom pkg2 fun3
#' @importFrom pkg3 fun4
NULL

## -----------------------------------------------------------------------------
#' @import zoo

#' Different name for calling zoo.
#'
#' @params ... passed to zoo.
#' @return zoo object.
#'
#' @export 
zoo2 <- function(...) zoo(...)

## -----------------------------------------------------------------------------
#' @import zoo
NULL

#' Different name for calling zoo.
#'
#' @params ... passed to zoo.
#' @return zoo object.
#'
#' @export 
zoo2 <- function(...) zoo(...)

