## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  eval = TRUE,
  warning = FALSE,
  comment = "#>"
)

## ---- echo = FALSE, results = 'asis'------------------------------------------
# we do NOT want the vignette to build on CRAN...it's taking too long
if (!identical(Sys.getenv("IN_PKGDOWN"), "true") && 
    !tolower(as.list(Sys.info())$user) %in% c("sjobergd", "currym", "whitingk", "whiting")) {
  msg <- 
    paste("View this vignette on the",
          "[package website](https://www.danieldsjoberg.com/gtsummary/articles/gallery.html).")
  cat(msg)
  knitr::knit_exit()
}

## ----setup, message = FALSE, warning=FALSE------------------------------------
library(gtsummary); library(gt); library(survival)
library(dplyr); library(stringr); library(purrr); library(forcats); library(tidyr)

## -----------------------------------------------------------------------------
trial %>%
  select(trt, age, grade) %>%
  tbl_summary(
    by = trt, 
    missing = "no",
    statistic = all_continuous() ~ "{median} ({p25}, {p75})"
  ) %>%
  modify_header(all_stat_cols() ~ "**{level}**<br>N = {n} ({style_percent(p)}%)") %>%
  add_n() %>%
  bold_labels() %>%
  modify_spanning_header(all_stat_cols() ~ "**Chemotherapy Treatment**")

## -----------------------------------------------------------------------------
trial %>%
  select(trt, age, marker) %>%
  tbl_summary(
    by = trt,
    type = all_continuous() ~ "continuous2",
    statistic = all_continuous() ~ c("{N_nonmiss}",
                                     "{mean} ({sd})", 
                                     "{median} ({p25}, {p75})", 
                                     "{min}, {max}"),
    missing = "no"
  ) %>%
  italicize_levels()

## ---- message = FALSE---------------------------------------------------------
trial %>%
  select(response, age, grade) %>%
  mutate(response = factor(response, labels = c("No Tumor Response", "Tumor Responded"))) %>%
  tbl_summary(
    by = response, 
    missing = "no",
    label = list(age ~ "Patient Age", grade ~ "Tumor Grade")
  ) %>%
  add_p(pvalue_fun = ~style_pvalue(.x, digits = 2)) %>%
  add_q()

## -----------------------------------------------------------------------------
trial %>%
  select(response, age, grade) %>%
  mutate(
    response = factor(response, labels = c("No Tumor Response", "Tumor Responded")) %>% 
      fct_explicit_na(na_level = "Missing Response Status")
  ) %>%
  tbl_summary(
    by = response, 
    label = list(age ~ "Patient Age", grade ~ "Tumor Grade")
  ) 

## -----------------------------------------------------------------------------
trial %>%
  select(response, marker, trt) %>%
  tbl_summary(
    by = trt,
    statistic = list(all_continuous() ~ "{mean} ({sd})",
                     all_categorical() ~ "{p}%"),
    missing = "no"
  ) %>%
  add_difference() %>%
  add_n() %>%
  modify_header(all_stat_cols() ~ "**{level}**") %>%
  modify_footnote(all_stat_cols() ~ NA)

## -----------------------------------------------------------------------------
# imagine that each patient received Drug A and Drug B (adding ID showing their paired measurements)
trial_paired <-
  trial %>%
  select(trt, marker, response) %>%
  group_by(trt) %>%
  mutate(id = row_number()) %>%
  ungroup()

# you must first delete incomplete pairs from the data, then you can build the table
trial_paired %>%
  # delete missing values
  filter(complete.cases(.)) %>%
  # keep IDs with both measurements
  group_by(id) %>%
  filter(n() == 2) %>%
  ungroup() %>%
  # summarize data
  tbl_summary(by = trt, include = -id) %>%
  add_p(test = list(marker ~ "paired.t.test",
                    response ~ "mcnemar.test"), 
        group = id)

## -----------------------------------------------------------------------------
# table summarizing data with no p-values
small_trial <- trial %>% select(grade, age, response)
t0 <- small_trial %>%
  tbl_summary(by = grade, missing = "no") %>%
  modify_header(all_stat_cols() ~ "**{level}**")

# table comparing grade I and II
t1 <- small_trial %>%
  filter(grade %in% c("I", "II")) %>%
  tbl_summary(by = grade, missing = "no") %>%
  add_p() %>%
  modify_header(p.value ~ md("**I vs. II**")) %>%
  # hide summary stat columns
  modify_column_hide(all_stat_cols())

# table comparing grade I and II
t2 <- small_trial %>%
  filter(grade %in% c("I", "III")) %>%
  tbl_summary(by = grade, missing = "no") %>%
  add_p()  %>%
  modify_header(p.value ~ md("**I vs. III**")) %>%
  # hide summary stat columns
  modify_column_hide(all_stat_cols())

# merging the 3 tables together, and adding additional gt formatting
tbl_merge(list(t0, t1, t2)) %>%
  modify_spanning_header(
    list(
      all_stat_cols() ~ "**Tumor Grade**",
      starts_with("p.value") ~ "**p-values**"
    )
  )

## -----------------------------------------------------------------------------

trial %>%
  select(age, marker) %>%
  tbl_summary(statistic = all_continuous() ~ "{mean} ({sd})", missing = "no") %>%
  modify_header(stat_0 ~ "**Mean (SD)**") %>%
  add_ci()


## -----------------------------------------------------------------------------
trial %>%
  select(trt, grade, marker) %>%
  tbl_continuous(variable = marker, by = trt) %>%
  modify_spanning_header(all_stat_cols() ~ "**Treatment Assignment**")

## -----------------------------------------------------------------------------
trial %>%
  select(trt, grade, age, stage) %>%
  mutate(grade = paste("Grade", grade)) %>%
  tbl_strata(
    strata = grade, 
    ~.x %>%
      tbl_summary(by = trt, missing = "no") %>%
      modify_header(all_stat_cols() ~ "**{level}**")
  )

## -----------------------------------------------------------------------------
trial %>%
  select(response, age, grade) %>%
  tbl_uvregression(
    method = glm,
    y = response, 
    method.args = list(family = binomial),
    exponentiate = TRUE
  ) %>%
  add_nevent()

## -----------------------------------------------------------------------------
gt_r1 <- glm(response ~ trt + grade, trial, family = binomial) %>%
  tbl_regression(exponentiate = TRUE)
gt_r2 <- coxph(Surv(ttdeath, death) ~ trt + grade, trial) %>%
  tbl_regression(exponentiate = TRUE)
gt_t1 <- trial[c("trt", "grade")] %>% 
  tbl_summary(missing = "no") %>% 
  add_n() %>%
  modify_header(stat_0 ~ "**n (%)**") %>%
  modify_footnote(stat_0 ~ NA_character_)

theme_gtsummary_compact()
tbl_merge(
  list(gt_t1, gt_r1, gt_r2),
  tab_spanner = c(NA_character_, "**Tumor Response**", "**Time to Death**")
)

## ---- echo=FALSE--------------------------------------------------------------
reset_gtsummary_theme()

## -----------------------------------------------------------------------------
trial %>%
  select(ttdeath, death, stage, grade) %>%
  tbl_uvregression(
    method = coxph,
    y = Surv(ttdeath, death), 
    exponentiate = TRUE,
    hide_n = TRUE
  ) %>%
  add_nevent(location = "level")

## -----------------------------------------------------------------------------
trial %>%
  select(age, marker, trt) %>%
  tbl_uvregression(
    method = lm,
    x = trt,
    show_single_row = "trt",
    hide_n = TRUE
  ) %>%
  modify_header(list(
    label ~"**Model Outcome**",
    estimate ~ "**Treatment Coef.**"
  )) %>%
  modify_footnote(estimate ~ "Values larger than 0 indicate larger values in the Drug B group.")

## -----------------------------------------------------------------------------
my_tidy <- function(x, exponentiate =  FALSE, conf.level = 0.95, ...) {
  dplyr::bind_cols(
    broom::tidy(x, exponentiate = exponentiate, conf.int = FALSE),
    # calculate the confidence intervals, and save them in a tibble
    stats::confint.default(x) %>%
      tibble::as_tibble() %>%
      rlang::set_names(c("conf.low", "conf.high"))  )
}

lm(age ~ grade + marker, trial) %>%
  tbl_regression(tidy_fun = my_tidy)

## -----------------------------------------------------------------------------
trial %>%
  select(ttdeath, death, stage, grade) %>%
  tbl_uvregression(
    method = coxph,
    y = Surv(ttdeath, death), 
    exponentiate = TRUE,
  ) %>%
  add_significance_stars()

