##
# @file
# @brief Creates mobility and geodata for US
#
# @details
#
# ## Configuration Items
#
# ```yaml
# data_path: <path to directory>
# subpop_setup:
#   modeled_states: <list of state postal codes> e.g. MD, CA, NY
#   mobility: <path to file relative to data_path> optional; default is 'mobility.csv'
#   geodata: <path to file relative to data_path> optional; default is 'geodata.csv'
#
# importation:
#   census_api_key: <string, optional> default is environment variable CENSUS_API_KEY. Environment variable is preferred so you don't accidentally commit your key.
# ```
#
# ## Input Data
#
# None
#
# ## Output Data
#
# * {data_path}/{subpop_setup::mobility}
# * {data_path}/{subpop_setup::geodata}
#

## @cond



# SETUP -------------------------------------------------------------------


library(dplyr)
library(tidyr)
# library(tidycensus)


option_list = list(
  optparse::make_option(c("-c", "--config"), action="store", default=Sys.getenv("CONFIG_PATH"), type='character', help="path to the config file"),
  optparse::make_option(c("-p", "--path"), action="store", default=Sys.getenv("FLEPI_PATH", "flepiMoP"), type='character', help="path to the flepiMoP directory"),
  optparse::make_option(c("-w", "--wide_form"), action="store",default=FALSE,type='logical',help="Whether to generate the old wide format mobility or the new long format")
)
opt = optparse::parse_args(optparse::OptionParser(option_list=option_list))

config <- flepicommon::load_config(opt$c)
if (length(config) == 0) {
  stop("no configuration found -- please set CONFIG_PATH environment variable or use the -c command flag")
}


# Aggregation to state level if in config
state_level <- ifelse(!is.null(config$subpop_setup$state_level) && config$subpop_setup$state_level, TRUE, FALSE)

# commute_data <- arrow::read_parquet(file.path(opt$p,"datasetup", "usdata","united-states-commutes","commute_data.gz.parquet"))
# census_data <- arrow::read_parquet(file.path(opt$p,"datasetup", "usdata","united-states-commutes","census_tracts_2010.gz.parquet"))


# # Get census key
# census_key = Sys.getenv("CENSUS_API_KEY")
# if(length(config$importation$census_api_key) != 0){
#   census_key = config$importation$census_api_key
# }
# if(census_key == ""){
#   stop("no census key found -- please set CENSUS_API_KEY environment variable or specify importation::census_api_key in config file")
# }
# tidycensus::census_api_key(key = census_key)



filterUSPS <- c("WY","VT","DC","AK","ND","SD","DE","MT","RI","ME","NH","HI","ID","WV","NE","NM",
                "KS","NV","MS","AR","UT","IA","CT","OK","OR","KY","LA","AL","SC","MN","CO","WI",
                "MD","MO","IN","TN","MA","AZ","WA","VA","NJ","MI","NC","GA","OH","IL","PA","NY","FL","TX","CA")

# GEODATA (CENSUS DATA) -------------------------------------------------------------



# # Retrieved from:
# census_data <- tidycensus::get_acs(geography="county", state=filterUSPS,
#                                    variables="B01003_001", year=config$subpop_setup$census_year,
#                                    keep_geo_vars=TRUE, geometry=FALSE, show_call=TRUE)
census_data <- arrow::read_parquet(paste0(opt$p,"/datasetup/usdata/us_county_census_2019.parquet")) %>%
  dplyr::rename(population=estimate, subpop=GEOID) %>%
  dplyr::select(subpop, population) %>%
  dplyr::mutate(subpop = substr(subpop,1,5))

# Add USPS column
#data(fips_codes)
fips_codes <- arrow::read_parquet(paste0(opt$p,"/datasetup/usdata/fips_us_county.parquet"))
fips_subpop_codes <- dplyr::mutate(fips_codes, subpop=paste0(state_code,county_code)) %>%
  dplyr::group_by(subpop) %>%
  dplyr::summarize(USPS=unique(state))

census_data <- dplyr::left_join(census_data, fips_subpop_codes, by="subpop") 


# Make each territory one county.
# Puerto Rico is the only one in the 2018 ACS estimates right now. Aggregate it.
# Keeping the other territories in the aggregation just in case they're there in the future.
name_changer <- setNames(
  unique(census_data$subpop),
  unique(census_data$subpop)
)
name_changer[grepl("^60",name_changer)] <- "60000" # American Samoa
name_changer[grepl("^66",name_changer)] <- "66000" # Guam
name_changer[grepl("^69",name_changer)] <- "69000" # Northern Mariana Islands
name_changer[grepl("^72",name_changer)] <- "72000" # Puerto Rico
name_changer[grepl("^78",name_changer)] <- "78000" # Virgin Islands

census_data <- census_data %>%
  dplyr::mutate(subpop = name_changer[subpop]) %>%
  dplyr::group_by(subpop) %>%
  dplyr::summarize(USPS = unique(USPS), population = sum(population))


# Territory populations (except Puerto Rico) taken from from https://www.census.gov/prod/cen2010/cph-2-1.pdf
terr_census_data <- arrow::read_parquet(file.path(opt$p,"datasetup", "usdata","united-states-commutes","census_tracts_island_areas_2010.gz.parquet"))

census_data <- terr_census_data %>%
  dplyr::rename(subpop = geoid) %>%
  dplyr::filter(length(filterUSPS) == 0 | ((USPS %in% filterUSPS) & !(USPS %in% census_data)))%>%
  rbind(census_data)


# State-level aggregation if desired
if (state_level){
  census_data <- census_data %>%
    dplyr::mutate(subpop = as.character(paste0(substr(subpop,1,2), "000"))) %>%
    dplyr::group_by(USPS, subpop) %>%
    dplyr::summarise(population=sum(population, na.rm=TRUE)) %>%
    tibble::as_tibble()
}


# Sort by population
census_data <- census_data %>%
  dplyr::arrange(population)

if (!is.null(config$subpop_setup$popnodes)) {
  names(census_data)[names(census_data) == "population"] <- config$subpop_setup$popnodes
}

if (length(config$subpop_setup$geodata) > 0) {
  geodata_file <- config$subpop_setup$geodata
} else {
  geodata_file <- 'geodata.csv'
}

# manually remove PR
census_data <- census_data %>% filter(USPS != "PR")


if(!dir.exists(dirname(config$subpop_setup$geodata))){
  dir.create(dirname(config$subpop_setup$geodata))}
write.csv(file = file.path(geodata_file), census_data, row.names=FALSE)
print(paste("Wrote geodata file:", file.path(geodata_file)))




# MOBILITY DATA (COMMUTER DATA) ------------------------------------------------------------


if(state_level & !file.exists(paste0("/", config$subpop_setup$mobility))){

  warning(paste("State-level mobility files must be created manually because `build_US_setup.R` does not generate a state-level mobility file automatically. No valid mobility file named", paste0(config$data_path, "/", config$subpop_setup$mobility), "(specified in the config) currently exists. Please check again."))

} else if(state_level & file.exists(paste0(config$data_path, "/", config$subpop_setup$mobility))){

  warning(paste("Using existing state-level mobility file named", paste0(config$data_path, "/", config$subpop_setup$mobility)))

} else{

  commute_data <- arrow::read_parquet(file.path(opt$p,"datasetup","usdata","united-states-commutes","commute_data.gz.parquet"))
  commute_data <- commute_data %>%
    dplyr::mutate(OFIPS = substr(OFIPS,1,5), DFIPS = substr(DFIPS,1,5)) %>%
    dplyr::mutate(OFIPS = name_changer[OFIPS], DFIPS = name_changer[DFIPS]) %>%
    dplyr::filter(OFIPS %in% census_data$subpop, DFIPS %in% census_data$subpop) %>%
    dplyr::group_by(OFIPS,DFIPS) %>%
    dplyr::summarize(FLOW = sum(FLOW)) %>%
    dplyr::filter(OFIPS != DFIPS)

  if(opt$w){
    mobility_file <- 'mobility.txt'
  } else if (length(config$subpop_setup$mobility) > 0) {
    mobility_file <- config$subpop_setup$mobility
  } else {
    mobility_file <- 'mobility.csv'
  }

  if(endsWith(mobility_file, '.txt')) {

    # Pads 0's for every subpop and itself, so that nothing gets dropped on the pivot
    padding_table <- tibble::tibble(
      OFIPS = census_data$subpop,
      DFIPS = census_data$subpop,
      FLOW = 0
    )

    rc <- dplyr::bind_rows(padding_table, commute_data) %>%
      dplyr::arrange(match(OFIPS, census_data$subpop), match(DFIPS, census_data$subpop)) %>%
      tidyr::pivot_wider(OFIPS,names_from=DFIPS,values_from=FLOW, values_fill=c("FLOW"=0),values_fn = list(FLOW=sum))
    if(!isTRUE(all(rc$OFIPS == census_data$subpop))){
      print(rc$OFIPS)
      print(census_data$subpop)
      stop("There was a problem generating the mobility matrix")
    }
    write.table(file = file.path(mobility_file), as.matrix(rc[,-1]), row.names=FALSE, col.names = FALSE, sep = " ")

    } else if(endsWith(mobility_file, '.csv')) {

      rc <- commute_data
      names(rc) <- c("ori","dest","amount")

      rc <- rc[rc$ori != rc$dest,]
      write.csv(file = file.path(mobility_file), rc, row.names=FALSE)

    } else {
      stop("Only .txt and .csv extensions supported for mobility matrix. Please check config's subpop_setup::mobility.")
    }

    print(paste("Wrote mobility file:", file.path(mobility_file)))
}



## @endcond

