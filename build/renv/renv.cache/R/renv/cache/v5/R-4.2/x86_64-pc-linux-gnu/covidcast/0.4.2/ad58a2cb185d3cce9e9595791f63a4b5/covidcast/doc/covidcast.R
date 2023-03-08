## ---- message=FALSE-----------------------------------------------------------
library(covidcast)
library(dplyr)

cli <- suppressMessages(
  covidcast_signal(data_source = "fb-survey", signal = "smoothed_cli",
                   start_day = "2020-05-01", end_day = "2020-05-07",
                   geo_type = "county")
)
knitr::kable(head(cli))

## -----------------------------------------------------------------------------
summary(cli)

## -----------------------------------------------------------------------------
cli <- suppressMessages(
  covidcast_signal(data_source = "fb-survey", signal = "smoothed_cli",
                   start_day = "2020-05-01", end_day = "2020-05-07",
                   geo_type = "state")
)
knitr::kable(head(cli))

## -----------------------------------------------------------------------------
cli <- suppressMessages(
  covidcast_signal(data_source = "fb-survey", signal = "smoothed_cli",
                   start_day = "2020-05-01", end_day = "2020-05-07",
                   geo_type = "county", geo_value = "42003")
)
knitr::kable(head(cli))

## -----------------------------------------------------------------------------
plot(cli, plot_type = "line",
     title = "Survey results in Allegheny County, PA")

## -----------------------------------------------------------------------------
name_to_fips("Allegheny")
name_to_cbsa("Pittsburgh")

## -----------------------------------------------------------------------------
county_fips_to_name("42003")
cbsa_to_name("38300")

## -----------------------------------------------------------------------------
meta <- covidcast_meta()
knitr::kable(head(meta))

## ---- eval = FALSE------------------------------------------------------------
#  summary(meta)

## ---- message = FALSE---------------------------------------------------------
covidcast_signal(data_source = "doctor-visits", signal = "smoothed_cli",
                 start_day = "2020-05-01", end_day = "2020-05-01",
                 geo_type = "state", geo_values = "pa", as_of = "2020-05-07")

## ---- message = FALSE---------------------------------------------------------
covidcast_signal(data_source = "doctor-visits", signal = "smoothed_cli",
                 start_day = "2020-05-01", end_day = "2020-05-01",
                 geo_type = "state", geo_values = "pa")

## ---- message = FALSE---------------------------------------------------------
covidcast_signal(data_source = "doctor-visits", signal = "smoothed_cli",
                 start_day = "2020-05-01", end_day = "2020-05-01",
                 geo_type = "state", geo_values = "pa",
                 issues = c("2020-05-01", "2020-05-15")) %>%
  knitr::kable()

## ---- message = FALSE---------------------------------------------------------
covidcast_signal(data_source = "doctor-visits", signal = "smoothed_cli",
                 start_day = "2020-05-01", end_day = "2020-05-07",
                 geo_type = "state", geo_values = "pa", lag = 7) %>%
  knitr::kable()

## ---- message = FALSE---------------------------------------------------------
covidcast_signal(data_source = "doctor-visits", signal = "smoothed_cli",
                 start_day = "2020-05-03", end_day = "2020-05-03",
                 geo_type = "state", geo_values = "pa",
                 issues = c("2020-05-09", "2020-05-15")) %>%
  knitr::kable()

