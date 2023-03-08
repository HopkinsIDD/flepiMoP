## ---- message=FALSE-----------------------------------------------------------
library(dplyr)

data <- read.csv(system.file("extdata", "covid-tracking-project-oct-2020.csv",
                             package = "covidcast", mustWork = TRUE))

data %>%
    select(date, state, death, deathIncrease, hospitalizedCurrently,
           hospitalizedIncrease) %>%
    head() %>%
    knitr::kable()

## ---- message=FALSE-----------------------------------------------------------
library(covidcast)

hospitalized <- data %>%
    select(time_value = date,
           geo_value = state,
           value = hospitalizedIncrease) %>%
    mutate(geo_value = tolower(geo_value),
           time_value = as.Date(time_value)) %>%
    as.covidcast_signal(geo_type = "state",
                        data_source = "covid-tracking",
                        signal = "hospitalized_increase")

head(hospitalized) %>%
    knitr::kable()

## -----------------------------------------------------------------------------
plot(hospitalized, plot_type = "choro")

## ---- message=FALSE-----------------------------------------------------------
deaths <- covidcast_signal("indicator-combination", "deaths_incidence_prop",
                           start_day = "2020-10-01",
                           end_day = "2020-10-31",
                           geo_type = "state")

covidcast_cor(deaths, hospitalized, by = "time_value")

## -----------------------------------------------------------------------------
death_hosp <- aggregate_signals(list(deaths, hospitalized),
                                format = "wide")

head(death_hosp) %>%
    knitr::kable()

