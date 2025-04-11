context("iterateAccept")

test_that("iterateAccept always accepts better likelihoods",{
  ll_ref <- -1
  ll_new <-  -.99
  N <- 10000
  expect_equal({
    all(sapply(seq_len(N),function(x){iterateAccept(ll_ref,ll_new)$accept}))
  },TRUE)
})

test_that("iterateAccept sometimes accepts worse likelihoods",{
 
  ll_ref <-  -1
  ll_new <- -7
  N <- 10000
  expect_gt(
    mean(sapply(seq_len(N),function(x){iterateAccept(ll_ref,ll_new)$accept})),
    0
  )   
  
  # ll_ref <-  -1
  # ll_new <- -.99
  # N <- 10000
  # expect_lt(
  #   abs(mean(sapply(seq_len(N),function(x){iterateAccept(ll_new,ll_ref)})) - exp(ll_ref - ll_new)),
  #   4.5e-3
  # )
  # 
  # ll_ref <- -1
  # ll_new <-  -.9
  # N <- 10000
  # expect_lt(
  #   abs(mean(sapply(seq_len(N),function(x){iterateAccept(ll_new,ll_ref)})) - exp(ll_ref - ll_new)),
  #   1.15e-2
  # )
})

test_that("iterateAccept is less likely to accept worse likelihoods",{
  
  ll_ref <-  -1
  ll_new1 <- -6
  ll_new2 <- -7
  N <- 10000
  expect_gt(
    mean(sapply(seq_len(N),function(x){iterateAccept(ll_ref,ll_new1)$accept})),
    mean(sapply(seq_len(N),function(x){iterateAccept(ll_ref,ll_new2)$accept}))
  )  
})
