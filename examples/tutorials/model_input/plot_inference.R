
library(yaml)
library(arrow)
library(data.table)
library(ggplot2)

.args <- # if (interactive()) c(
  "config_sample_2pop_inference.yml"
#) else commandArgs(trailingOnly = TRUE)

config <- read_yaml(.args[1])

gt_data <- fread(config$inference$gt_data_path)

outputdir <- sprintf(
  "model_output/%s_%s_%s",
  config$name, config$seir_modifiers$scenarios[1], config$outcome_modifiers$scenarios[1]
) |>
dir("EDT", full.names = TRUE) |>
dir("hosp", full.names = TRUE) |>
dir("global", full.names = TRUE) |>
dir("final", full.names = TRUE)

gt <- list.files(outputdir, full.names = TRUE) |> lapply(read_parquet) |> rbindlist(idcol = "sample")

plot_dt <- gt[, .(
  sample, date = as.Date(date), subpop, incidH = incidHosp
)]

p <- ggplot(plot_dt) + aes(date, incidH) +
  facet_grid(subpop ~ ., scale = "free_y") +
  geom_line(aes(group = sample), alpha = 0.5, color = "dodgerblue") +
  geom_point(data = gt_data) +
  theme_minimal() +
  scale_x_date(NULL) +
  scale_y_continuous("Incidence")

ggsave("example.png", p, bg = "white", width = 10, height = 10)
