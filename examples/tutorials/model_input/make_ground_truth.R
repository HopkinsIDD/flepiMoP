
library(yaml)
library(arrow)
library(data.table)

.args <- # if (interactive()) c(
  "config_sample_2pop_modifiers.yml"
#) else commandArgs(trailingOnly = TRUE)

config <- read_yaml(.args[1])
outputdir <- sprintf(
  "model_output/%s_%s_%s/",
  config$name, config$seir_modifiers$scenarios[1], config$outcome_modifiers$scenarios[1]
)

gt <- list.files(outputdir, "hosp", recursive = TRUE, full.names = TRUE) |>
  read_parquet() |> as.data.table()

gt[, .(date = as.Date(date), subpop, incidH = incidHosp)] |> fwrite("data/sample_2pop_cases.csv")
