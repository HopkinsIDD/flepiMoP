## -----------------------------------------------------------------------------
arrow::arrow_with_s3()

## ---- eval = FALSE------------------------------------------------------------
#  arrow::copy_files("s3://ursa-labs-taxi-data", "nyc-taxi")

## ---- eval = FALSE------------------------------------------------------------
#  bucket <- "https://ursa-labs-taxi-data.s3.us-east-2.amazonaws.com"
#  for (year in 2009:2019) {
#    if (year == 2019) {
#      # We only have through June 2019 there
#      months <- 1:6
#    } else {
#      months <- 1:12
#    }
#    for (month in sprintf("%02d", months)) {
#      dir.create(file.path("nyc-taxi", year, month), recursive = TRUE)
#      try(download.file(
#        paste(bucket, year, month, "data.parquet", sep = "/"),
#        file.path("nyc-taxi", year, month, "data.parquet"),
#        mode = "wb"
#      ), silent = TRUE)
#    }
#  }

## -----------------------------------------------------------------------------
dir.exists("nyc-taxi")

## -----------------------------------------------------------------------------
library(arrow, warn.conflicts = FALSE)
library(dplyr, warn.conflicts = FALSE)

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  ds <- open_dataset("nyc-taxi", partitioning = c("year", "month"))

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  ds

## ---- echo = FALSE, eval = !file.exists("nyc-taxi")---------------------------
cat("
FileSystemDataset with 125 Parquet files
vendor_id: string
pickup_at: timestamp[us]
dropoff_at: timestamp[us]
passenger_count: int8
trip_distance: float
pickup_longitude: float
pickup_latitude: float
rate_code_id: null
store_and_fwd_flag: string
dropoff_longitude: float
dropoff_latitude: float
payment_type: string
fare_amount: float
extra: float
mta_tax: float
tip_amount: float
tolls_amount: float
total_amount: float
year: int32
month: int32

See $metadata for additional Schema metadata
")

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  system.time(ds %>%
#    filter(total_amount > 100, year == 2015) %>%
#    select(tip_amount, total_amount, passenger_count) %>%
#    mutate(tip_pct = 100 * tip_amount / total_amount) %>%
#    group_by(passenger_count) %>%
#    collect() %>%
#    summarise(
#      median_tip_pct = median(tip_pct),
#      n = n()
#    ) %>%
#    print())

## ---- echo = FALSE, eval = !file.exists("nyc-taxi")---------------------------
cat("
# A tibble: 10 x 3
   passenger_count median_tip_pct      n
             <int>          <dbl>  <int>
 1               0           9.84    380
 2               1          16.7  143087
 3               2          16.6   34418
 4               3          14.4    8922
 5               4          11.4    4771
 6               5          16.7    5806
 7               6          16.7    3338
 8               7          16.7      11
 9               8          16.7      32
10               9          16.7      42

   user  system elapsed
  4.436   1.012   1.402
")

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  ds %>%
#    filter(total_amount > 100, year == 2015) %>%
#    select(tip_amount, total_amount, passenger_count) %>%
#    mutate(tip_pct = 100 * tip_amount / total_amount) %>%
#    group_by(passenger_count)

## ---- echo = FALSE, eval = !file.exists("nyc-taxi")---------------------------
cat("
FileSystemDataset (query)
tip_amount: float
total_amount: float
passenger_count: int8
tip_pct: expr

* Filter: ((total_amount > 100) and (year == 2015))
* Grouped by passenger_count
See $.data for the source Arrow object
")

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  sampled_data <- ds %>%
#    filter(year == 2015) %>%
#    select(tip_amount, total_amount, passenger_count) %>%
#    map_batches(~ sample_frac(as.data.frame(.), 1e-4)) %>%
#    mutate(tip_pct = tip_amount / total_amount)
#  
#  str(sampled_data)

## ---- echo = FALSE, eval = !file.exists("nyc-taxi")---------------------------
cat("
'data.frame':	15603 obs. of  4 variables:
 $ tip_amount     : num  0 0 1.55 1.45 5.2 ...
 $ total_amount   : num  5.8 16.3 7.85 8.75 26 ...
 $ passenger_count: int  1 1 1 1 1 6 5 1 2 1 ...
 $ tip_pct        : num  0 0 0.197 0.166 0.2 ...
")

## ---- eval = file.exists("nyc-taxi")------------------------------------------
#  model <- lm(tip_pct ~ total_amount + passenger_count, data = sampled_data)
#  
#  ds %>%
#    filter(year == 2015) %>%
#    select(tip_amount, total_amount, passenger_count) %>%
#    mutate(tip_pct = tip_amount / total_amount) %>%
#    map_batches(function(batch) {
#      batch %>%
#        as.data.frame() %>%
#        mutate(pred_tip_pct = predict(model, newdata = .)) %>%
#        filter(!is.nan(tip_pct)) %>%
#        summarize(sse_partial = sum((pred_tip_pct - tip_pct)^2), n_partial = n())
#    }) %>%
#    summarize(mse = sum(sse_partial) / sum(n_partial)) %>%
#    pull(mse)

## ---- echo = FALSE, eval = !file.exists("nyc-taxi")---------------------------
cat("
[1] 0.1304284
")

