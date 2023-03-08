## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  warning = FALSE,
  comment = "#>"
)

## ----setup, message=FALSE-----------------------------------------------------
library(gtsummary)

tbl_regression_ex <-
  lm(age ~ grade + marker, trial) %>%
  tbl_regression() %>%
  bold_p(t = 0.5) 

tbl_summary_ex <-
  trial %>%
  select(trt, age, grade, response) %>%
  tbl_summary(by = trt)

## -----------------------------------------------------------------------------
tbl_summary_ex$table_body

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "hide", "Logical indicating whether the column is hidden in the output. This column is also scoped in `modify_header()` (and friends) to be used in a selecting environment",
  "align", "Specifies the alignment/justification of the column, e.g. 'center' or 'left'",
  "label", "Label that will be displayed (if column is displayed in output)",
  "interpret_label", "the {gt} function that is used to interpret the column label, `gt::md()` or `gt::html()`",
  "spanning_header", "Includes text printed above columns as spanning headers.",
  "interpret_spanning_header", "the {gt} function that is used to interpret the column spanning headers, `gt::md()` or `gt::html()`",
  "modify_stat_{*}", "any column beginning with `modify_stat_` is a statistic available to report in `modify_header()` (and others)",
  "modify_selector_{*}", "any column beginning with `modify_selector_` is a column that is scoped in `modify_header()` (and friends) to be used in a selecting environment"
) %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "rows", "expression selecting rows in `.$table_body`, `NA` indicates to add footnote to header",
  "footnote", "string containing footnote to add to column/row"
) %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "rows", "expression selecting rows in `.$table_body`",
  "fmt_fun", "list of formatting/styling functions"
) %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "rows", "expression selecting rows in `.$table_body`",
  "format_type", "one of `c('bold', 'italic', 'indent')`",
  "undo_text_format", "logical indicating where the formatting indicated should be undone/removed."
)%>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "rows", "expression selecting rows in `.$table_body`",
  "symbol", "string to replace missing values with, e.g. an em-dash"
) %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## ---- echo=FALSE--------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "column", "Column name from `.$table_body`",
  "rows", "expression selecting rows in `.$table_body`",
  "pattern", "glue pattern directing how to combine/merge columns. The merged columns will replace the column indicated in 'column'."
) %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

## -----------------------------------------------------------------------------
tbl_regression_ex$table_styling

## -----------------------------------------------------------------------------
tbl_regression_ex %>%
  purrr::pluck("table_body") %>%
  select(variable, row_type, label)

## ---- eval = FALSE------------------------------------------------------------
#  print.gtsummary <- function(x) {
#    get_theme_element("pkgwide-str:print_engine") %>%
#      switch(
#        "gt" = as_gt(x),
#        "flextable" = as_flex_table(x),
#        "huxtable" = as_hux_table(x),
#        "kable_extra" = as_kable_extra(x),
#        "kable" = as_kable(x)
#      ) %>%
#      print()
#  }

## ---- echo = FALSE------------------------------------------------------------
tibble::tribble(
  ~Column, ~Description,
  "`variable`", "String of the variable name",
  "`label`", "String matching the variable's values in `.$table_body$label`",
  "`col_name`", "The column name the statistics appear under in `.$table_body`, e.g. `'stat_0'`, `'stat_1'`",
  "`variable_levels`", "This column appears if and only if the variable being summarized has multiple levels. The column is equal to the variable's levels.",
  "`<statistics>`", "Primarily, the tibble stores the summary statistics for each variable. For example, when the mean is requested in `tbl_summary()`, there will be a column called `'mean'`."
)%>%
  gt::gt() %>%
  gt::fmt_markdown(columns = everything()) %>%
  gt::tab_options(
            table.font.size = "small",
            data_row.padding = gt::px(1),
            summary_row.padding = gt::px(1),
            grand_summary_row.padding = gt::px(1),
            footnotes.padding = gt::px(1),
            source_notes.padding = gt::px(1),
            row_group.padding = gt::px(1)
          )

