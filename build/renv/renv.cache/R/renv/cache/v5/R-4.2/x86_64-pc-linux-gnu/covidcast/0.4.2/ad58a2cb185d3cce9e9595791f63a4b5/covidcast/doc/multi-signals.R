## ---- message = FALSE---------------------------------------------------------
library(covidcast)

start_day <- "2020-06-01"
end_day <- "2020-10-01"

signals <- suppressMessages(
  covidcast_signals(data_source = "usa-facts",
                    signal = c("confirmed_incidence_num",
                               "deaths_incidence_num"),
                    start_day = start_day, end_day = end_day,
                    geo_type = "state")
)

summary(signals[[1]])
summary(signals[[2]])

## ---- message=FALSE-----------------------------------------------------------
library(dplyr)

aggregate_signals(signals) %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals, dt = c(-1, 0)) %>%
  filter(geo_value == "tx") %>% head()
aggregate_signals(signals, dt = list(0, c(-1, 0, 1))) %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals[[1]], dt = c(-1, 0, 1)) %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals, format = "long") %>%
  filter(geo_value == "tx") %>% head()

aggregate_signals(signals, dt = c(-1, 0), format = "long") %>%
  filter(geo_value == "tx") %>% head()

aggregate_signals(signals, dt = list(-1, 0), format = "long") %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals[[1]], dt = c(-1, 0), format = "long") %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals, dt = list(-1, 0)) %>%
  covidcast_longer() %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
aggregate_signals(signals, dt = list(-1, 0), format = "long") %>%
  covidcast_wider() %>%
  filter(geo_value == "tx") %>% head()

## -----------------------------------------------------------------------------
df_cor1 <- covidcast_cor(x = aggregate_signals(signals[[1]], dt = -7,
                                              format = "long"),
                        y = signals[[2]])

df_cor2 <- covidcast_cor(x = signals[[1]], y = signals[[2]], dt_x = -7)
identical(df_cor1, df_cor2)

