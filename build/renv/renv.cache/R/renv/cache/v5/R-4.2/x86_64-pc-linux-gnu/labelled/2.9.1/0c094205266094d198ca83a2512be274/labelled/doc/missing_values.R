## -----------------------------------------------------------------------------
library(labelled)

## -----------------------------------------------------------------------------
x <- c(1:5, tagged_na("a"), tagged_na("z"), NA)

## -----------------------------------------------------------------------------
x
is.na(x)

## -----------------------------------------------------------------------------
na_tag(x)
print_tagged_na(x)
format_tagged_na(x)

## -----------------------------------------------------------------------------
is.na(x)
is_tagged_na(x)
# You can test for specific tagged NAs with the second argument
is_tagged_na(x, "a")
is_regular_na(x)

## ---- error=TRUE--------------------------------------------------------------
y <- c("a", "b", tagged_na("z"))
y
is_tagged_na(y)
format_tagged_na(y)

z <- c(1L, 2L, tagged_na("a"))
typeof(z)
format_tagged_na(z)

## -----------------------------------------------------------------------------
x <- c(1, 2, tagged_na("a"), 1, tagged_na("z"), 2, tagged_na("a"), NA)
x %>% print_tagged_na()

unique(x) %>% print_tagged_na()
unique_tagged_na(x) %>% print_tagged_na()

duplicated(x)
duplicated_tagged_na(x)

sort(x, na.last = TRUE) %>% print_tagged_na()
sort_tagged_na(x) %>% print_tagged_na()

## -----------------------------------------------------------------------------
x <- c(1, 0, 1, tagged_na("r"), 0, tagged_na("d"), tagged_na("z"), NA)
val_labels(x) <- c(
  no = 0, yes = 1,
  "don't know" = tagged_na("d"),
  refusal = tagged_na("r")
)
x

## -----------------------------------------------------------------------------
to_factor(x)

## -----------------------------------------------------------------------------
to_factor(x, explicit_tagged_na = TRUE)
to_factor(x, levels = "prefixed", explicit_tagged_na = TRUE)

## -----------------------------------------------------------------------------
tagged_na_to_user_na(x)
tagged_na_to_user_na(x, user_na_start = 10)

## -----------------------------------------------------------------------------
tagged_na_to_regular_na(x)
tagged_na_to_regular_na(x) %>% is_tagged_na()

## -----------------------------------------------------------------------------
v <- labelled(c(1,2,3,9,1,3,2,NA), c(yes = 1, no = 3, "don't know" = 9))
v
na_values(v) <- 9
v

na_values(v) <- NULL
v

na_range(v) <- c(5, Inf)
na_range(v)
v

## -----------------------------------------------------------------------------
library(dplyr)
# setting value labels and user NAs
df <- tibble(s1 = c("M", "M", "F", "F"), s2 = c(1, 1, 2, 9)) %>%
  set_value_labels(s2 = c(yes = 1, no = 2)) %>%
  set_na_values(s2 = 9)
df$s2

# removing user NAs
df <- df %>% set_na_values(s2 = NULL)
df$s2

## -----------------------------------------------------------------------------
v
is.na(v)
is_user_na(v)
is_regular_na(v)

## -----------------------------------------------------------------------------
x <- c(1:5, 11:15)
na_range(x) <- c(10, Inf)
val_labels(x) <- c("dk" = 11, "refused" = 15)
x
mean(x)

## -----------------------------------------------------------------------------
user_na_to_na(x)
mean(user_na_to_na(x), na.rm = TRUE)

## -----------------------------------------------------------------------------
user_na_to_tagged_na(x)
mean(user_na_to_tagged_na(x), na.rm = TRUE)

## -----------------------------------------------------------------------------
remove_user_na(x)
mean(remove_user_na(x))

