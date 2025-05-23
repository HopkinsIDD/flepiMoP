# File naming ------------------------------------------------------------------

##' Create a unique identifier for a run via time stamp
##' @export
run_id <- function(){
  rc <- "test"
  try({
    rc <- format(lubridate::now(),"%Y%m%d_%H%M%S%Z")
  }, silent=TRUE)
  return(rc)
}


##' @name create_slotprefix
##' @title create_slotprefix
##' @description Function for creating scenario tags from components; Intended for use in filename construction
##' @param .dots A set of strings, or lists of value,format (see sprintf).
##' @param trailing_separator How to end the prefix
##' @export
create_setup_prefix <- function(..., trailing_separator=""){
  args <- list(...)
  args <- args[which(!sapply(args, is.null))]
  formats <- sapply(args, function(x){x[2]})
  formats[is.na(formats)] <- '%s'
  values <- c(lapply(args,function(x){x[[1]]}))
  prefix <- paste0(file.path(do.call(purrr::partial(sprintf, fmt=paste(formats, collapse = '_')), values)), trailing_separator)
  return(prefix)
}


##' @name create_prefix
##' @title create_prefix
##' @description Function for creating scenario tags from components; Intended for use in filename construction
##' @param .dots A set of strings, or lists of value,format (see sprintf).
##' @param sep A character to use to separate different components of the scenario in the tag.  This argument cannot appear in any of the  .dots arguments.
##' @export
create_prefix <- function(..., prefix='',sep='-',trailing_separator=""){
  args <- list(...)
  formats <- sapply(args, function(x){x[2]})
  formats[is.na(formats)] <- '%s'
  values <- c(lapply(args,function(x){x[[1]]}))
  if(any(grepl(sep,values,fixed=TRUE))){
    stop("scenario elements cannot contain the seperator")
  }
  prefix <- paste0(prefix,do.call(purrr::partial(sprintf,fmt=paste(formats,collapse = sep)),values),trailing_separator)

  return(prefix)
}

## Function for creating file names from their components
##' @export
create_file_name <- function(run_id, 
                             prefix, 
                             filepath_suffix, 
                             filename_prefix, 
                             index, 
                             type, 
                             extension='parquet', 
                             create_directory = TRUE){
  rc <- sprintf("model_output/%s/%s/%s/%s/%s%09d.%s.%s.%s", 
                prefix, run_id, type, 
                filepath_suffix, filename_prefix, 
                index, run_id, type, extension)
  
  if(create_directory){ # Add filename prefix here.
    if(!dir.exists(dirname(rc))){
      dir.create(dirname(rc), recursive = TRUE)
    }
  }
  return(rc)
}
