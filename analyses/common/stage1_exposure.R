# Reusable Stage-1 hierarchical brms exposure model specifications and helpers
# for both Grambank (Track 1) and LDL/Wug (Track 2) two-stage admixture models.
#
# Stage 1 maps individual-level or population-level genetic admixture observations to
# language-level posterior means (eta_mean) and standard deviations (eta_sd):
#   - individual-level measures:  y ~ 1 + (1 | lang) + (1 | lang:population)
#   - population-level w/ SE:     y | se(se_y, sigma = TRUE) ~ 1 + (1 | lang)
#   - population-level plain:     y ~ 1 + (1 | lang)   (gaussian or bernoulli for graff)

get_stage1_measure_specs <- function() {
  list(
    ent         = list(level = "ind", col = "ent"),
    nonmax      = list(level = "ind", col = "nonmax"),
    ent_K12     = list(level = "ind", col = "ent_K12"),
    ent_K30     = list(level = "ind", col = "ent_K30"),
    pop_ent     = list(level = "pop", col = "pop_ent", se = "pop_ent_se"),
    het         = list(level = "pop", col = "het", se = "het_se"),
    support     = list(level = "pop", col = "support", se = "support_se"),
    amount      = list(level = "pop", col = "amount", se = "amount_se"),
    graff       = list(level = "pop", col = "graff_target", bern = TRUE),
    neigh       = list(level = "pop", col = "neigh_flow"),
    f3_z        = list(level = "pop", col = "f3_z"),
    f3_est      = list(level = "pop", col = "f3_est"),
    f3k_z       = list(level = "pop", col = "f3k_z"),
    f3k_est     = list(level = "pop", col = "f3k_est"),
    f3b3_z      = list(level = "pop", col = "f3b3_z"),
    f3b3_est    = list(level = "pop", col = "f3b3_est"),
    sqrt_het    = list(level = "pop", col = "sqrt_het", se = "sqrt_het_se"),
    f3k_est_pos = list(level = "pop", col = "f3k_est_pos")
  )
}

subset_of_measure <- function(m) {
  switch(m,
    neigh = "neigh",
    het = , sqrt_het = "het",
    f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
    f3b3_z = , f3b3_est = "f3b3",
    "all"
  )
}

augment_population_f3_columns <- function(pop_df, k5_file = NULL) {
  pop_df$f3b3_z <- ifelse(pop_df$n_B >= 3, pop_df$f3_z, NA)
  pop_df$f3b3_est <- ifelse(pop_df$n_B >= 3, pop_df$f3_est, NA)
  if (!is.null(k5_file) && file.exists(k5_file)) {
    k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
    i <- match(pop_df$population, k5$population)
    pop_df$f3k_z <- -k5$min_z[i]
    pop_df$f3k_est <- -k5$min_est[i]
  } else if (!("f3k_est" %in% names(pop_df))) {
    pop_df$f3k_z <- pop_df$f3k_est <- NA_real_
  }
  pop_df$f3k_est_pos <- pmax(0, pop_df$f3k_est)
  pop_df
}

fit_stage1_exposure <- function(
  measure,
  ind_df,
  pop_df,
  set_name = "main",
  fits_dir = tempdir(),
  fit_prefix = "",
  dry = FALSE,
  chains = 4L,
  iter = 3000L,
  warmup = 1500L,
  threads = 1L,
  seed = 20261021L,
  adapt_delta = 0.99
) {
  suppressPackageStartupMessages({
    library(brms)
    library(posterior)
  })
  specs <- get_stage1_measure_specs()
  stopifnot(measure %in% names(specs))
  s <- specs[[measure]]

  dat <- if (s$level == "ind") ind_df[ind_df$population %in% pop_df$population, ] else pop_df
  dat <- dat[!is.na(dat[[s$col]]), ]
  dat$y <- dat[[s$col]]
  dat$lang <- dat$glottocode
  bern <- isTRUE(s$bern)
  ymu <- NA; sdy <- NA
  if (!bern) {
    ymu <- mean(dat$y)
    sdy <- sd(dat$y)
    dat$y <- (dat$y - ymu) / sdy
  }
  pr <- c(prior(normal(0, 1.5), class = Intercept), prior(exponential(1), class = sd))
  if (!bern) pr <- c(pr, prior(exponential(1), class = sigma))
  if (s$level == "ind") {
    f <- bf(y ~ 1 + (1 | lang) + (1 | lang:population))
  } else if (!is.null(s$se)) {
    dat$se_y <- dat[[s$se]] / sdy
    f <- bf(y | se(se_y, sigma = TRUE) ~ 1 + (1 | lang))
  } else {
    f <- bf(y ~ 1 + (1 | lang))
  }
  fam <- if (bern) bernoulli() else gaussian()
  fit_name <- paste0("exposure_", fit_prefix, set_name, "__", measure)
  langs <- sort(unique(dat$lang))

  if (dry) {
    make_stancode(f, data = dat, family = fam, prior = pr, threads = threading(threads))
    make_standata(f, data = dat, family = fam, prior = pr)
    return(list(
      e = data.frame(set = set_name, glottocode = langs, eta_mean = rnorm(length(langs)),
                     eta_sd = 0.3, measure = measure, y_mean = ymu, y_sd = sdy),
      diag = data.frame(set = set_name, measure = measure, n_obs = nrow(dat), n_lang = length(langs))
    ))
  }

  fit <- brm(
    f, data = dat, family = fam, prior = pr,
    chains = chains, iter = iter, warmup = warmup, cores = chains, threads = threading(threads),
    seed = seed, control = list(adapt_delta = adapt_delta, max_treedepth = 12),
    file = file.path(fits_dir, fit_name), file_refit = "on_change", refresh = 0, silent = 2
  )
  dr <- as_draws_df(fit)
  eta <- sapply(langs, function(l) dr$b_Intercept + dr[[sprintf("r_lang[%s,Intercept]", l)]])
  sm <- summarise_draws(dr)
  list(
    e = data.frame(set = set_name, glottocode = langs, eta_mean = colMeans(eta),
                   eta_sd = apply(eta, 2, sd), measure = measure, y_mean = ymu, y_sd = sdy),
    diag = data.frame(
      set = set_name, measure = measure, level = s$level, n_obs = nrow(dat), n_lang = length(langs),
      max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
      divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
      sd_lang = mean(dr$sd_lang__Intercept),
      sd_pop = if (s$level == "ind") mean(dr$`sd_lang:population__Intercept`) else NA,
      sd_resid = if (bern) NA else mean(dr$sigma)
    )
  )
}
