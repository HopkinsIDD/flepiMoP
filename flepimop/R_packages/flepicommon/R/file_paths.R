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


##' @name create_prefix
##' @title create_prefix
##' @description Function for creating scenario tags from components; Intended for use in filename construction
##' @param .dots A set of strings, or lists of value,format (see sprintf).
##' @param sep A character to use to separate different components of the scenario in the tag.  This argument cannot appear in any of the  .dots arguments.
##' @export
create_prefix <- function(...,prefix='',sep='-',trailing_separator=""){
  args <- list(...)
  formats <- sapply(args,function(x){x[2]})
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
create_file_name <- function(run_id,prefix,index,type,extension='parquet',create_directory = TRUE){
  rc <- sprintf("model_output/%s/%s%09d.%s.%s.%s",type,prefix,index,run_id,type,extension)
  if(create_directory){
    if(!dir.exists(dirname(rc))){
      dir.create(dirname(rc), recursive = TRUE)
    }
  }
  return(rc)
}

