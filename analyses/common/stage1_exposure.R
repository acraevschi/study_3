# Reusable Stage-1 hierarchical brms exposure model specifications and helpers
# for both Grambank (Track 1) and Paradigm (Track 2) two-stage admixture models.
#
# Stage 1 maps individual-level or population-level genetic admixture observations to
# language-level posterior means (eta_mean) and standard deviations (eta_sd):
#   - individual-level measures:  y ~ 1 + (1 | lang) + (1 | lang:population)
#   - population-level w/ SE:     y | se(se_y, sigma = TRUE) ~ 1 + (1 | lang)
#   - population-level plain:     y ~ 1 + (1 | lang)   (gaussian or bernoulli for graff)
#   - language-level max_p:       max_{p in l} y_p across populations with n_individuals >= min_n_ind

get_stage1_measure_specs <- function() {
  list(
    # Fixed clade-stratified non-conspecific f3 & SI benchmarks
    f3_fixed       = list(level = "pop", col = "f3_fixed"),
    f3_fixed_iso   = list(level = "pop", col = "f3_fixed_iso"),
    f3_fixed_maxp  = list(level = "lang_max", col = "f3_fixed"),
    f3_fixed_raw   = list(level = "pop", col = "f3_fixed_raw", se = "f3_fixed_se"),
    f3_all_fixed   = list(level = "pop", col = "f3_all_fixed"),
    f4_est         = list(level = "pop", col = "f4_est", se = "f4_se"),
    # FST neighbor & drift controls (558-pop FST.csv)
    neigh_flow_out = list(level = "pop", col = "neigh_flow_out"),
    median_fst_all = list(level = "pop", col = "median_fst_all"),
    # Discrete-K ADMIXTURE SI negative-control diagnostics (K = 12, 20, 30)
    ent_K12        = list(level = "ind", col = "ent_K12"),
    ent_K20        = list(level = "ind", col = "ent_K20"),
    ent_K30        = list(level = "ind", col = "ent_K30"),
    rmsd_K12       = list(level = "pop", col = "rmsd_K12", min_n_ind = 5L),
    rmsd_K20       = list(level = "pop", col = "rmsd_K20", min_n_ind = 5L),
    rmsd_K30       = list(level = "pop", col = "rmsd_K30", min_n_ind = 5L),
    # Legacy measures (preserved for backward compatibility)
    ent            = list(level = "ind", col = "ent"),
    nonmax         = list(level = "ind", col = "nonmax"),
    pop_ent        = list(level = "pop", col = "pop_ent", se = "pop_ent_se"),
    het            = list(level = "pop", col = "het", se = "het_se"),
    support        = list(level = "pop", col = "support", se = "support_se"),
    amount         = list(level = "pop", col = "amount", se = "amount_se"),
    graff          = list(level = "pop", col = "graff_target", bern = TRUE),
    neigh          = list(level = "pop", col = "neigh_flow"),
    f3_z           = list(level = "pop", col = "f3_z"),
    f3_est         = list(level = "pop", col = "f3_est"),
    f3k_z          = list(level = "pop", col = "f3k_z"),
    f3k_est        = list(level = "pop", col = "f3k_est"),
    f3b3_z         = list(level = "pop", col = "f3b3_z"),
    f3b3_est       = list(level = "pop", col = "f3b3_est"),
    sqrt_het       = list(level = "pop", col = "sqrt_het", se = "sqrt_het_se"),
    f3k_est_pos    = list(level = "pop", col = "f3k_est_pos")
  )
}

subset_of_measure <- function(m) {
  switch(m,
    f3_fixed = , f3_fixed_maxp = , f3_fixed_raw = , f3_all_fixed = "f3_fixed",
    f3_fixed_iso = "f3_fixed_iso",
    f4_est = "f4",
    neigh_flow_out = "neigh_out",
    median_fst_all = "all",
    rmsd_K12 = , rmsd_K20 = , rmsd_K30 = "rmsd5",
    neigh = "neigh",
    het = , sqrt_het = "het",
    f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
    f3b3_z = , f3b3_est = "f3b3",
    "all"
  )
}

augment_population_f3_columns <- function(
  pop_df,
  k5_file = NULL,
  f3_fixed_file = NULL,
  fst_regimes_file = NULL,
  discrete_k_file = NULL
) {
  if (all(c("n_B", "f3_z", "f3_est") %in% names(pop_df))) {
    pop_df$f3b3_z <- ifelse(pop_df$n_B >= 3, pop_df$f3_z, NA)
    pop_df$f3b3_est <- ifelse(pop_df$n_B >= 3, pop_df$f3_est, NA)
  }
  if (!is.null(k5_file) && file.exists(k5_file)) {
    k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
    i <- match(pop_df$population, k5$population)
    pop_df$f3k_z <- -k5$min_z[i]
    pop_df$f3k_est <- -k5$min_est[i]
  } else if (!("f3k_est" %in% names(pop_df))) {
    pop_df$f3k_z <- pop_df$f3k_est <- NA_real_
  }
  pop_df$f3k_est_pos <- pmax(0, pop_df$f3k_est)

  if (is.null(f3_fixed_file) && file.exists("analyses/common/data/f3_fixed_scores.csv")) {
    f3_fixed_file <- "analyses/common/data/f3_fixed_scores.csv"
  }
  if (!is.null(f3_fixed_file) && file.exists(f3_fixed_file)) {
    ff <- read.csv(f3_fixed_file, stringsAsFactors = FALSE)
    idx <- match(pop_df$population, ff$population)
    pop_df$f3_fixed          <- as.numeric(ff$f3_fixed[idx])
    pop_df$f3_fixed_iso      <- as.numeric(ff$f3_fixed_iso[idx])
    pop_df$f3_fixed_raw      <- as.numeric(ff$f3_est_neg[idx])
    pop_df$f3_fixed_se       <- as.numeric(ff$best_se[idx])
    pop_df$f3_all_fixed      <- as.numeric(ff$f3_all_fixed[idx])
    pop_df$iso_fallback_used <- as.integer(ff$iso_fallback_used[idx])
    pop_df$f4_est            <- as.numeric(ff$f4_est[idx])
    pop_df$f4_se             <- as.numeric(ff$f4_se[idx])
    pop_df$f4_z              <- as.numeric(ff$f4_z[idx])
  }

  if (is.null(fst_regimes_file) && file.exists("analyses/common/data/fst_neighbor_regimes.csv")) {
    fst_regimes_file <- "analyses/common/data/fst_neighbor_regimes.csv"
  }
  if (!is.null(fst_regimes_file) && file.exists(fst_regimes_file)) {
    fr <- read.csv(fst_regimes_file, stringsAsFactors = FALSE)
    idx <- match(pop_df$population, fr$population)
    pop_df$neigh_flow_out     <- as.numeric(fr$neigh_flow_out[idx])
    pop_df$median_fst_all     <- as.numeric(fr$median_fst_all[idx])
    pop_df$fst_mismatch_ratio <- as.numeric(fr$fst_mismatch_ratio[idx])
    pop_df$barbieri_mismatch  <- as.integer(fr$barbieri_mismatch[idx])
    pop_df$regime_a_enclave   <- as.integer(fr$regime_a_enclave[idx])
    pop_df$regime_b_shift     <- as.integer(fr$regime_b_shift[idx])
  }

  if (is.null(discrete_k_file) && file.exists("analyses/common/data/discrete_k_diagnostics.csv")) {
    discrete_k_file <- "analyses/common/data/discrete_k_diagnostics.csv"
  }
  if (!is.null(discrete_k_file) && file.exists(discrete_k_file)) {
    dk <- read.csv(discrete_k_file, stringsAsFactors = FALSE)
    idx <- match(pop_df$population, dk$population)
    for (col in c("rmsd_K12", "rmsd_K20", "rmsd_K30", "pop_ent_K12", "pop_ent_K20", "pop_ent_K30",
                  "max_q_K12", "max_q_K20", "max_q_K30")) {
      if (col %in% names(dk)) pop_df[[col]] <- as.numeric(dk[[col]][idx])
    }
  }

  pop_df
}

fit_stage1_exposure <- function(
  measure,
  ind_df,
  pop_df,
  set_name = "main",
  fits_dir = tempdir(),
  fit_prefix = "",
  min_n_ind = NULL,
  dry = FALSE,
  chains = 4L,
  iter = 3000L,
  warmup = 1500L,
  cores = getOption("mc.cores", chains),
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

  eff_min_n <- if (!is.null(min_n_ind)) as.integer(min_n_ind) else if (!is.null(s$min_n_ind)) as.integer(s$min_n_ind) else NULL
  pop_use <- pop_df
  if (!is.null(eff_min_n) && "n_individuals" %in% names(pop_use)) {
    pop_use <- pop_use[!is.na(pop_use$n_individuals) & pop_use$n_individuals >= eff_min_n, , drop = FALSE]
  }

  if (identical(s$level, "lang_max")) {
    if (!(s$col %in% names(pop_use))) {
      stop("Column '", s$col, "' required for measure '", measure, "' is missing from pop_df.")
    }
    dat <- pop_use[!is.na(pop_use[[s$col]]), , drop = FALSE]
    if (nrow(dat) == 0L) {
      stop("No non-missing observations available for measure '", measure, "' (col='", s$col, "').")
    }
    langs <- sort(unique(dat$glottocode))
    raw_max <- vapply(langs, function(l) max(dat[[s$col]][dat$glottocode == l], na.rm = TRUE), numeric(1))
    ymu <- mean(raw_max)
    sdy <- sd(raw_max)
    eta_std <- if (!is.na(sdy) && sdy > 1e-12) (raw_max - ymu) / sdy else rep(0.0, length(langs))
    return(list(
      e = data.frame(
        set = set_name, glottocode = langs, eta_mean = as.numeric(eta_std),
        eta_sd = rep(0.0, length(langs)), measure = measure, y_mean = ymu, y_sd = sdy,
        stringsAsFactors = FALSE
      ),
      diag = data.frame(
        set = set_name, measure = measure, level = "lang_max",
        n_obs = nrow(dat), n_lang = length(langs),
        max_rhat = 1.0, min_ess_bulk = NA_real_, divergences = 0L,
        sd_lang = NA_real_, sd_pop = NA_real_, sd_resid = NA_real_,
        stringsAsFactors = FALSE
      )
    ))
  }

  dat <- if (s$level == "ind") ind_df[ind_df$population %in% pop_use$population, , drop = FALSE] else pop_use
  if (!(s$col %in% names(dat))) {
    stop("Column '", s$col, "' required for measure '", measure, "' is missing from ",
         if (s$level == "ind") "ind_df" else "pop_df", ".")
  }
  dat <- dat[!is.na(dat[[s$col]]), , drop = FALSE]
  if (!is.null(s$se)) {
    if (!(s$se %in% names(dat))) {
      stop("SE column '", s$se, "' required for measure '", measure, "' is missing from data.")
    }
    dat <- dat[!is.na(dat[[s$se]]) & is.finite(dat[[s$se]]) & dat[[s$se]] > 0, , drop = FALSE]
  }
  if (nrow(dat) == 0L) {
    stop("No non-missing observations available for measure '", measure, "' (col='", s$col, "').")
  }
  dat$y <- dat[[s$col]]
  dat$lang <- dat$glottocode
  bern <- isTRUE(s$bern)
  ymu <- NA_real_; sdy <- NA_real_
  if (!bern) {
    ymu <- mean(dat$y)
    sdy <- sd(dat$y)
    if (is.na(sdy) || sdy <= 1e-12) {
      stop("Exposure measure '", measure, "' has zero or undefined variance (n=", nrow(dat), ").")
    }
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
  n_sfx <- if (!is.null(eff_min_n)) paste0("_n", eff_min_n) else ""
  fit_name <- paste0("exposure_", fit_prefix, set_name, n_sfx, "__", measure)
  langs <- sort(unique(dat$lang))

  if (dry) {
    make_stancode(f, data = dat, family = fam, prior = pr, threads = threading(threads))
    make_standata(f, data = dat, family = fam, prior = pr)
    return(list(
      e = data.frame(
        set = set_name, glottocode = langs, eta_mean = rnorm(length(langs)),
        eta_sd = 0.3, measure = measure, y_mean = ymu, y_sd = sdy,
        stringsAsFactors = FALSE
      ),
      diag = data.frame(
        set = set_name, measure = measure, level = s$level,
        n_obs = nrow(dat), n_lang = length(langs),
        max_rhat = NA_real_, min_ess_bulk = NA_real_, divergences = NA_integer_,
        sd_lang = NA_real_, sd_pop = NA_real_, sd_resid = NA_real_,
        stringsAsFactors = FALSE
      )
    ))
  }

  fit <- brm(
    f, data = dat, family = fam, prior = pr,
    chains = chains, iter = iter, warmup = warmup, cores = cores, threads = threading(threads),
    seed = seed, control = list(adapt_delta = adapt_delta, max_treedepth = 12),
    file = file.path(fits_dir, fit_name), file_refit = "on_change", refresh = 0, silent = 2
  )
  dr <- as_draws_df(fit)
  eta <- sapply(langs, function(l) dr$b_Intercept + dr[[sprintf("r_lang[%s,Intercept]", l)]])
  sm <- summarise_draws(dr)
  list(
    e = data.frame(
      set = set_name, glottocode = langs, eta_mean = colMeans(eta),
      eta_sd = apply(eta, 2, sd), measure = measure, y_mean = ymu, y_sd = sdy,
      stringsAsFactors = FALSE
    ),
    diag = data.frame(
      set = set_name, measure = measure, level = s$level, n_obs = nrow(dat), n_lang = length(langs),
      max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
      divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
      sd_lang = mean(dr$sd_lang__Intercept),
      sd_pop = if (s$level == "ind") mean(dr$`sd_lang:population__Intercept`) else NA_real_,
      sd_resid = if (bern) NA_real_ else mean(dr$sigma),
      stringsAsFactors = FALSE
    )
  )
}
