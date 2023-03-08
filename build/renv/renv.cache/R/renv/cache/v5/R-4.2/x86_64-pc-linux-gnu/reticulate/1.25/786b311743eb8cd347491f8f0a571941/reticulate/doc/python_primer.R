## ----include=FALSE------------------------------------------------------------
library(reticulate)

# this vignette requires python 3.8 or newer
eval <- tryCatch({
  config <- py_config()
  numeric_version(config$version) >= "3.8" && py_numpy_available()
}, error = function(e) FALSE)

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = eval
)

## ----setup--------------------------------------------------------------------
library(reticulate)

## -----------------------------------------------------------------------------
if (TRUE) {
  cat("This is one expression. \n")
  cat("This is another expression. \n")
}

## -----------------------------------------------------------------------------
library(reticulate)
l <- r_to_py(list(1, 2, 3))
it <- as_iterator(l)

iter_next(it)
iter_next(it)
iter_next(it)
iter_next(it, completed = "StopIteration")

## -----------------------------------------------------------------------------
my_function <- function(name = "World") {
  cat("Hello", name, "\n")
}

my_function()
my_function("Friend")

## ---- eval = FALSE------------------------------------------------------------
#  dplyr <- loadNamespace("dplyr")

## ---- error = TRUE------------------------------------------------------------
library(reticulate)
py$a_strict_Python_function(3)             # error
py$a_strict_Python_function(3L)            # success
py$a_strict_Python_function(as.integer(3)) # success

