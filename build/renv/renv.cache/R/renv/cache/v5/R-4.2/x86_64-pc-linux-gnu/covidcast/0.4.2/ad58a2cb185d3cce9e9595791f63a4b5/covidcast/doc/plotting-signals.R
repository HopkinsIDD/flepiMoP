## ---- message=FALSE-----------------------------------------------------------
library(covidcast)

comb <- suppressMessages(
  covidcast_signal(data_source = "indicator-combination",
                   signal = "nmf_day_doc_fbc_fbs_ght",
                   start_day = "2020-07-01", end_day = "2020-07-14")
)
summary(comb)

inum <- suppressMessages(
  covidcast_signal(data_source = "usa-facts",
                   signal = "confirmed_7dav_incidence_num",
                   start_day = "2020-07-01", end_day = "2020-07-14")
)
summary(inum)

## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(fig.width = 10, fig.height = 8)

## -----------------------------------------------------------------------------
plot(comb)

## -----------------------------------------------------------------------------
plot(comb, time_value = "2020-07-04", choro_col = cm.colors(10), alpha = 0.4,
     title = "Combination of COVID-19 indicators on 2020-07-04")

## -----------------------------------------------------------------------------
breaks <- c(0, 1, 2, 5, 10, 20, 50, 100, 200)
colors <- c("#D3D3D3", "#FFFFCC", "#FEDDA2", "#FDBB79", "#FD9950", "#EB7538",
            "#C74E32", "#A3272C", "#800026")

# Note that length(breaks) == length(colors) by design. This works as follows:
# we assign colors[i] iff the value satisfies breaks[i] <= value < breaks[i+1],
# where we take breaks[0] = -Inf and breaks[N+1] = Inf, for N = length(breaks)

plot(inum, choro_col = colors, choro_params = list(breaks = breaks),
     title = "New COVID cases (7-day trailing average) on 2020-07-14")

## -----------------------------------------------------------------------------
cprop <- suppressMessages(
  covidcast_signal(data_source = "usa-facts",
                   signal = "confirmed_cumulative_prop",
                   start_day = "2020-07-01", end_day = "2020-07-14")
)

breaks <- c(0, 1000)
colors <- c("#D3D3D3", "#FFC0CB")

plot(cprop, choro_col = colors,
     choro_params = list(breaks = breaks, legend_width = 3),
     title = "Cumulative COVID cases per 100k people on 2020-07-14")

## -----------------------------------------------------------------------------
plot(inum, plot_type = "bubble")

## -----------------------------------------------------------------------------
plot(inum, plot_type = "bubble",
     bubble_params = list(breaks = seq(20, 200, len = 6)))

## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(fig.width = 10, fig.height = 6)

## -----------------------------------------------------------------------------
iprop <- suppressMessages(
  covidcast_signal(data_source = "usa-facts",
                   signal = "confirmed_7dav_incidence_prop",
                   start_day = "2020-07-01", end_day = "2020-07-14")
)

## -----------------------------------------------------------------------------
library(gridExtra)

breaks1 <- c(1, 10, 100, 1000)
breaks2 <- c(10, 50, 100, 500)

p1 <- plot(inum, plot_type = "bubble",
           bubble_params = list(breaks = breaks1, max_size = 6),
           include = "TX", bubble_col = "red",
           title = paste("Incidence number on", max(inum$time_value)))
p2 <- plot(iprop, plot_type = "bubble",
           bubble_params = list(breaks = breaks2, max_size = 6),
           include = "TX", bubble_col = "red",
           title = paste("Incidence rate on", max(iprop$time_value)))

grid.arrange(p1, p2, nrow = 1)

## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(fig.width = 8, fig.height = 6)

## -----------------------------------------------------------------------------
suppressMessages({
comb_st <- covidcast_signal(data_source = "indicator-combination",
                               signal = "nmf_day_doc_fbc_fbs_ght",
                               start_day = "2020-04-15", end_day = "2020-07-01",
                               geo_type = "state")
inum_st <- covidcast_signal(data_source = "usa-facts",
                               signal = "confirmed_7dav_incidence_num",
                               start_day = "2020-04-15", end_day = "2020-07-01",
                               geo_type = "state")
})

## ---- message = FALSE---------------------------------------------------------
library(dplyr)

states <- c("ca", "pa", "tx", "ny")
plot(comb_st %>% filter(geo_value %in% states), plot_type = "line")
plot(inum_st %>% filter(geo_value %in% states), plot_type = "line")

## ---- warning = FALSE---------------------------------------------------------
library(ggplot2)

suppressMessages({
comb_md <- covidcast_signal(data_source = "indicator-combination",
                               signal = "nmf_day_doc_fbc_fbs_ght",
                               start_day = "2020-06-01", end_day = "2020-07-15",
                               geo_values = name_to_fips("Miami-Dade"))
inum_md <- covidcast_signal(data_source = "usa-facts",
                               signal = "confirmed_7dav_incidence_num",
                               start_day = "2020-06-01", end_day = "2020-07-15",
                               geo_values = name_to_fips("Miami-Dade"))
})

# Compute the ranges of the two signals
range1 <- inum_md %>% select("value") %>% range
range2 <- comb_md %>% select("value") %>% range

# Function to transform from one range to another
trans <- function(x, from_range, to_range) {
  (x - from_range[1]) / (from_range[2] - from_range[1]) *
    (to_range[2] - to_range[1]) + to_range[1]
}

# Convenience functions for our two signal ranges
trans12 <- function(x) trans(x, range1, range2)
trans21 <- function(x) trans(x, range2, range1)

# Transform the combined signal to the incidence range, then stack
# these rowwise into one data frame
df <- select(rbind(comb_md %>% mutate_at("value", trans21),
                   inum_md), c("time_value", "value"))
df$signal <- c(rep("Combined indicator", nrow(comb_md)),
               rep("New COVID-19 cases", nrow(inum_md)))

# Finally, plot both signals
ggplot(df, aes(x = time_value, y = value)) +
  labs(x = "Date", title = "Miami-Dade County") +
  geom_line(aes(color = signal)) +
  scale_y_continuous(
    name = "New COVID-19 cases (7-day trailing average)",
    sec.axis = sec_axis(trans12, name = "Combination of COVID-19 indicators")
  ) +
  theme(legend.position = "bottom",
        legend.title = ggplot2::element_blank())

