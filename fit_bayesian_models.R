#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Bayesian phylogenetic / spatial models of inflectional predictability
# ---------------------------------------------------------------------------
# Response : correct | trials(total)  -- MGN cell-to-cell prediction accuracy
# Question : does L1 speaker population predict inflectional predictability?
#
# CHANGES FROM THE PREVIOUS VERSION, AND WHY
#
# 1. beta_binomial() instead of binomial().
#    MGN's per-pair accuracy is the BEST of three edge-weighting configurations
#    under cross-validation (Guzman Naranjo 2024, p.445), not an i.i.d. Bernoulli
#    sequence. A plain binomial is therefore badly overdispersed, which makes the
#    credible interval on the population effect far too narrow -- exactly the
#    coefficient the study is about.
#
# 2. (1 | iso_sanitized) added alongside (1 | gr(phylo, cov = A)), where `phylo`
#    is a duplicate of the language column (brms refuses two group-level terms
#    that name the same factor).
#    With only the phylogenetic term, ALL residual between-language variation is
#    forced through the phylogenetic covariance, inflating apparent phylogenetic
#    signal and leaving language-level overdispersion unmodelled. The standard
#    phylogenetic mixed model estimates both components.
#
# 3. log10_nvar_z added as a control.
#    Raw accuracy is not comparable across cell pairs without accounting for how
#    many inflection classes the classifier chooses among (the chance baseline).
#    nvar correlates -0.32 with accuracy.
#
# 4. mo(distance) replaces the linear distance_rel term, matching the original
#    design. Morphosyntactic distance is an ordinal count (1..14) with no reason
#    to act linearly on the logit scale.
#
# 5. altitude_range_scaled is DROPPED from Model 2 (user decision, 2026-09-03).
#    NOTE: this does not remove imputation exposure from Model 2. The same 11
#    languages (arb, eng, fra, hbs, hin, lav, por, rus, spa, urd, ydd) have
#    imputed contact_richness and roughness too, and 4 of them have imputed area.
#    Bromham has no polygon for them, and all four covariates derive from that
#    polygon. Use MGN_DROP_ECO_IMPUTED=1 to refit Model 2 on the 53 languages
#    with measured values. Model 1 contains no imputed covariate at all.
#
# 6. The Gaussian process runs on 3-D unit-sphere coordinates rather than raw
#    lat/lon degrees. A degree of longitude is 111 km at the equator and 55 km at
#    60N, and raw degrees do not wrap at the antimeridian. `gr = TRUE` groups by
#    the 64 unique locations instead of fitting over all rows.
#
# 7. Fits the full dataset by default. The previous version silently kept
#    slice_sample(n = 100) per language x POS -- 6.7% of the data -- described only
#    as "fast, stable". Set MGN_SUBSAMPLE to opt in, and it is reported loudly.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(brms)
  library(cmdstanr)
  library(ggplot2)
  library(bayesplot)
})

FIT_DIR     <- Sys.getenv("MGN_FIT_DIR", "fits")
RESULT_DIR  <- Sys.getenv("MGN_RESULT_DIR", "results")
PLOT_DIR    <- Sys.getenv("MGN_PLOT_DIR", "plots")
dir.create(RESULT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIT_DIR,    showWarnings = FALSE, recursive = TRUE)
dir.create(PLOT_DIR,   showWarnings = FALSE, recursive = TRUE)

CHAINS      <- as.integer(Sys.getenv("MGN_CHAINS",  "4"))
THREADS     <- as.integer(Sys.getenv("MGN_THREADS", "1"))  # cores WITHIN each chain
ITER        <- as.integer(Sys.getenv("MGN_ITER",    "2000"))
WARMUP      <- as.integer(Sys.getenv("MGN_WARMUP",  "1000"))
SUBSAMPLE   <- Sys.getenv("MGN_SUBSAMPLE", "")   # rows per language x POS; "" = full data
DROP_ECO_IMPUTED <- Sys.getenv("MGN_DROP_ECO_IMPUTED", "0") == "1"
ADAPT_DELTA <- as.numeric(Sys.getenv("MGN_ADAPT_DELTA", "0.99"))
MAX_TREEDEPTH <- as.integer(Sys.getenv("MGN_MAX_TREEDEPTH", "12"))
REFRESH     <- as.integer(Sys.getenv("MGN_REFRESH", "100"))

# Observation unit -- see the aggregation block below.
UNIT <- match.arg(tolower(Sys.getenv("MGN_UNIT", "pair")), c("pair", "source_cell"))

# Which models to fit: "minimal", "comprehensive", or "both".
MODELS <- match.arg(tolower(Sys.getenv("MGN_MODELS", "both")),
                    c("both", "minimal", "comprehensive"))
FIT_MIN  <- MODELS %in% c("both", "minimal")
FIT_COMP <- MODELS %in% c("both", "comprehensive")

cat(sprintf(paste0("Sampler: %d chains x %d iter (warmup %d) | %d thread(s) per chain",
                   " = %d cores\n         adapt_delta %.3f | max_treedepth %d | models: %s\n"),
            CHAINS, ITER, WARMUP, max(THREADS, 1), CHAINS * max(THREADS, 1),
            ADAPT_DELTA, MAX_TREEDEPTH, MODELS))

cat("Loading mgn_modeling_dataset.csv and phylo_cov_matrix.rds...\n")
data <- read.csv("mgn_modeling_dataset.csv", stringsAsFactors = FALSE)
A    <- readRDS("phylo_cov_matrix.rds")

data$iso_sanitized <- as.factor(data$iso_sanitized)
data$pos           <- as.factor(data$pos)

# Align dataset and covariance matrix -------------------------------------
valid <- intersect(levels(data$iso_sanitized), rownames(A))
dropped <- setdiff(levels(data$iso_sanitized), rownames(A))
if (length(dropped)) {
  warning("Languages in the dataset but absent from A (dropped): ",
          paste(dropped, collapse = ", "))
}
data <- data %>% filter(iso_sanitized %in% valid)
data$iso_sanitized <- droplevels(data$iso_sanitized)
A_sub <- A[levels(data$iso_sanitized), levels(data$iso_sanitized)]

# Sensitivity option: drop languages whose macro-ecological covariates were
# imputed rather than measured (11 of 64: eng, spa, rus, fra, por, hin, urd,
# arb, hbs, lav, ydd).
if (DROP_ECO_IMPUTED) {
  n0 <- nrow(data)
  data <- data %>% filter(!eco_imputed)
  data$iso_sanitized <- droplevels(data$iso_sanitized)
  A_sub <- A[levels(data$iso_sanitized), levels(data$iso_sanitized)]
  cat(sprintf("MGN_DROP_ECO_IMPUTED=1: kept %d of %d rows, %d languages\n",
              nrow(data), n0, nlevels(data$iso_sanitized)))
}

# ---------------------------------------------------------------------------
# Observation unit
# ---------------------------------------------------------------------------
# "pair"        one row per directed cell pair (cell_1 -> cell_2). 111,315 rows.
#
# "source_cell" one row per (language, POS, SOURCE cell): successes and trials
#               summed over every target cell, pair-level morphological measures
#               averaged with trial weights. 3,175 rows.
#
#   This is the aggregation from the earlier prelim_analysis_mgn_cells_binomial_
#   median.R ("aggregate to average the results across multiple runs for the same
#   cell_1"), ported to the beta-binomial response. That script took the MEDIAN of
#   the per-pair accuracy rate and modelled it with zero_one_inflated_beta, which
#   discards the trial counts entirely; summing `correct` and `total` keeps the
#   binomial structure, so a source cell tested on 12,000 trials still outweighs
#   one tested on 60.
#
#   `distance` cannot survive the collapse: a source cell reaches targets a median
#   of 6 distinct distances away (max 13), so every aggregated row mixes all of
#   them. `mo(distance)` is therefore dropped from both formulas under this unit --
#   it is the one term the source-cell model cannot carry.
if (UNIT == "source_cell") {
  n0 <- nrow(data)

  # Constant within (iso_sanitized, pos, cell_1); verified against the dataset.
  # Everything else is either a target-side label (cell_2, unimorph_2), a pair
  # measure aggregated below, or `distance`/`distance_rel`, which are dropped.
  CONSTANT_COLS <- intersect(
    c("lang", "glottocode", "language_name", "family", "macro_area", "num_used",
      "unimorph_1", "n_pairs", "population_l1", "population_source",
      "l2_proportion", "vehicularity", "log10_pop", "log10_pop_z",
      "paradigm_size", "log10_paradigm_size", "eco_imputed",
      "contact_richness_scaled", "log10_area_scaled", "altitude_range_scaled",
      "roughness_scaled", "lat", "lon"),
    names(data))

  data <- data %>%
    group_by(iso_sanitized, pos, cell_1) %>%
    summarise(
      # Trial-weighted, to match the mixture that `correct / total` now is.
      # These MUST come first: summarise() evaluates in order, so reassigning
      # `total` below would leave the weights a length-1 sum.
      nvar      = weighted.mean(nvar, total),
      nph       = weighted.mean(nph, total),
      nmarkers  = weighted.mean(nmarkers, total),
      # Reference only; NOT a model term (see the note above).
      distance_mean = weighted.mean(distance, total),
      n_targets = dplyr::n(),
      across(all_of(CONSTANT_COLS), dplyr::first),
      correct   = sum(correct),
      total     = sum(total),
      .groups = "drop"
    ) %>%
    as.data.frame()

  # The two trial-level z-scores were standardised over the 111,315 pairs. Their
  # moments no longer describe the sample being fitted, and `nvar` is now a
  # weighted mean rather than a per-pair count, so both are recomputed here.
  # log10_pop_z, contact_richness_scaled, log10_area_scaled and roughness_scaled
  # are standardised over the 64 LANGUAGES upstream, so aggregation cannot move
  # them and they are carried through untouched.
  data$log10_nvar            <- log10(pmax(data$nvar, 1))
  data$log10_nvar_z          <- as.numeric(scale(data$log10_nvar))
  data$log10_paradigm_size_z <- as.numeric(scale(data$log10_paradigm_size))

  cat(sprintf(paste0("MGN_UNIT=source_cell: aggregated %d cell pairs into %d source",
                     " cells (%.1f%%),\n                      %d languages, median %d",
                     " targets collapsed per row.\n",
                     "                      mo(distance) dropped; log10_nvar_z and",
                     " log10_paradigm_size_z\n                      restandardised on",
                     " the aggregated rows.\n"),
              n0, nrow(data), 100 * nrow(data) / n0,
              nlevels(data$iso_sanitized), median(data$n_targets)))
}

# brms rejects two group-level terms that name the same factor, so the
# phylogenetic term needs its own column pointing at the same languages. This is
# the idiom from the brms phylogenetic vignette (`phylo` + `species`): one term
# carries the phylogenetically structured variance, the other the independent
# language-specific variance.
data$phylo <- data$iso_sanitized

# Spherical coordinates for the spatial GP --------------------------------
lat_r <- data$lat * pi / 180
lon_r <- data$lon * pi / 180
data$geo_x <- cos(lat_r) * cos(lon_r)
data$geo_y <- cos(lat_r) * sin(lon_r)
data$geo_z <- sin(lat_r)

# Optional subsample -------------------------------------------------------
if (nzchar(SUBSAMPLE)) {
  n_per <- as.integer(SUBSAMPLE)
  set.seed(42)
  n0 <- nrow(data)
  data <- data %>%
    group_by(iso_sanitized, pos) %>%
    slice_sample(n = n_per) %>%
    ungroup()
  cat(sprintf(paste0("!! SUBSAMPLING ACTIVE: %d of %d rows (%.1f%%), max %d per ",
                     "language x POS.\n!! Results are NOT based on the full dataset.\n"),
              nrow(data), n0, 100 * nrow(data) / n0, n_per))
} else {
  cat("Using the full dataset (no subsampling).\n")
}

cat(sprintf("Fitting on %d observations across %d languages.\n",
            nrow(data), nlevels(data$iso_sanitized)))
cat(sprintf("Trials per language: min %d, median %.0f, max %d\n",
            min(table(data$iso_sanitized)),
            median(table(data$iso_sanitized)),
            max(table(data$iso_sanitized))))

priors <- c(
  prior(normal(0, 2),   class = "b"),
  prior(normal(0, 3),   class = "Intercept"),
  prior(exponential(1), class = "sd")
)

fit_args <- list(
  data    = data,
  data2   = list(A = A_sub),
  family  = beta_binomial(link = "logit", link_phi = "log"),
  prior   = priors,
  chains  = CHAINS,
  cores   = CHAINS,
  threads = if (THREADS > 1) threading(THREADS) else NULL,
  backend = "cmdstanr",
  stan_model_args = list(stanc_options = list("O1")),
  iter    = ITER,
  warmup  = WARMUP,
  refresh = REFRESH,
  control = list(adapt_delta = ADAPT_DELTA, max_treedepth = MAX_TREEDEPTH),
  file_refit = "always"
)

# ---------------------------------------------------------------------------
# Model 1: minimal sociodemographic
# ---------------------------------------------------------------------------
cat("\n=== Model 1: Minimal Sociodemographic ===\n")
# `mo(distance)` is a cell-PAIR term. Under MGN_UNIT=source_cell every row mixes
# all the distances that source cell reaches, so the term is dropped rather than
# fed a meaningless average.
DIST_TERM <- if (UNIT == "pair") "mo(distance) +" else ""

formula_min <- bf(as.formula(paste(
  "correct | trials(total) ~",
  "log10_pop_z * pos +",
  "log10_paradigm_size_z * pos +",
  "log10_nvar_z +",
  DIST_TERM,
  "(1 | iso_sanitized) +",
  "(1 | gr(phylo, cov = A)) +",
  "gp(geo_x, geo_y, geo_z, k = 15, c = 5 / 4, gr = TRUE)"
)))
cat("Formula 1: ", deparse1(formula_min$formula), "\n")

fit_minimal <- if (FIT_MIN) {
  do.call(brm, c(list(formula = formula_min,
                      file = file.path(FIT_DIR, "model_minimal")), fit_args))
} else NULL
if (FIT_MIN) cat("Model 1 fitted.\n") else cat("Model 1 skipped (MGN_MODELS).\n")

# ---------------------------------------------------------------------------
# Model 2: macro-ecological & contact
# ---------------------------------------------------------------------------
cat("\n=== Model 2: Macro-Ecological & Contact ===\n")
formula_comp <- bf(as.formula(paste(
  "correct | trials(total) ~",
  "log10_pop_z * pos +",
  "contact_richness_scaled +",
  "log10_area_scaled +",
  "roughness_scaled +",
  "log10_paradigm_size_z * pos +",
  "log10_nvar_z +",
  DIST_TERM,
  "(1 | iso_sanitized) +",
  "(1 | gr(phylo, cov = A)) +",
  "gp(geo_x, geo_y, geo_z, k = 15, c = 5 / 4, gr = TRUE)"
)))
cat("Formula 2: ", deparse1(formula_comp$formula), "\n")

fit_comp <- if (FIT_COMP) {
  do.call(brm, c(list(formula = formula_comp,
                      file = file.path(FIT_DIR, "model_comprehensive")), fit_args))
} else NULL
if (FIT_COMP) cat("Model 2 fitted.\n") else cat("Model 2 skipped (MGN_MODELS).\n")

# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------
cat("\n=== Convergence Diagnostics ===\n")
diagnose <- function(fit, label) {
  s  <- posterior::summarise_draws(as_draws_df(fit))
  np <- nuts_params(fit)
  div <- sum(subset(np, Parameter == "divergent__")$Value)
  out <- data.frame(
    model         = label,
    max_rhat      = max(s$rhat, na.rm = TRUE),
    min_ess_bulk  = min(s$ess_bulk, na.rm = TRUE),
    min_ess_tail  = min(s$ess_tail, na.rm = TRUE),
    divergences   = div
  )
  cat(sprintf("%-14s max Rhat %.4f | min bulk ESS %.0f | min tail ESS %.0f | divergences %d\n",
              label, out$max_rhat, out$min_ess_bulk, out$min_ess_tail, div))
  if (out$max_rhat >= 1.01)     warning(label, ": Rhat >= 1.01, chains have not mixed.")
  if (out$min_ess_bulk < 400)   warning(label, ": bulk ESS < 400.")
  if (div > 0)                  warning(label, ": ", div, " divergent transitions.")
  out
}
# Only the models actually fitted this run.
fits <- list(Minimal = fit_minimal, Comprehensive = fit_comp)
fits <- fits[!vapply(fits, is.null, logical(1))]

diag_df <- bind_rows(lapply(names(fits), function(k) diagnose(fits[[k]], k)))
write.csv(diag_df, file.path(RESULT_DIR, "convergence_diagnostics.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
# On the full dataset the pointwise log-likelihood is draws x observations
# (8000 x 111,315 ~= 7 GB in doubles), so this is the memory peak of the whole
# script. It runs after each fit is already written to FIT_DIR, so a failure here
# costs the LOO table, never the sampling.
cat("\n=== LOO-CV ===\n")
loos <- lapply(fits, function(f) try(loo(f), silent = TRUE))
ok   <- !vapply(loos, inherits, logical(1), "try-error")
if (any(!ok)) {
  warning("loo() failed for: ", paste(names(loos)[!ok], collapse = ", "),
          " -- the fits themselves are unaffected and saved in ", FIT_DIR)
  for (k in names(loos)[!ok]) cat(k, ": ", as.character(loos[[k]]), "\n", sep = "")
}
loos <- loos[ok]

if (length(loos) >= 2) {
  comp_df <- as.data.frame(loo_compare(loos[[1]], loos[[2]]))
  comp_df$model <- rownames(comp_df)
  write.csv(comp_df, file.path(RESULT_DIR, "loo_model_comparison.csv"), row.names = FALSE)
  print(comp_df)
} else if (length(loos) == 1) {
  cat("Only one model fitted -- no comparison. Reporting its own LOO.\n")
}

if (length(loos)) {
  elpd_df <- bind_rows(lapply(names(loos), function(k) {
    e <- loos[[k]]$estimates
    data.frame(model = k, statistic = rownames(e),
               Estimate = e[, "Estimate"], SE = e[, "SE"],
               bad_pareto_k = sum(loo::pareto_k_values(loos[[k]]) > 0.7))
  }))
  write.csv(elpd_df, file.path(RESULT_DIR, "loo_estimates.csv"), row.names = FALSE)
  print(elpd_df)
}

# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
cat("\n=== Posterior Summaries ===\n")
extract_fixed <- function(fit, label) {
  fx <- as.data.frame(summary(fit)$fixed)
  fx$parameter <- rownames(fx)
  fx$model <- label
  fx
}
all_params <- bind_rows(lapply(names(fits), function(k) extract_fixed(fits[[k]], k)))
write.csv(all_params, file.path(RESULT_DIR, "posterior_parameter_summaries.csv"),
          row.names = FALSE)
print(all_params[, c("model", "parameter", "Estimate", "l-95% CI", "u-95% CI",
                     "Rhat", "Bulk_ESS")])

# Random-effect SDs: how much variance is phylogenetic vs language-specific?
# Both grouping terms must be reported. Reading only $iso_sanitized silently
# discards the phylogenetic component, which is the reason the two terms exist.
extract_re <- function(fit, label) {
  re <- summary(fit)$random
  bind_rows(lapply(names(re), function(g) {
    df <- as.data.frame(re[[g]])
    df$group <- g
    df$term  <- rownames(df)
    df$model <- label
    df
  }))
}
re_sd <- bind_rows(lapply(names(fits), function(k) extract_re(fits[[k]], k)))
write.csv(re_sd, file.path(RESULT_DIR, "random_effect_sds.csv"), row.names = FALSE)
cat("Random-effect SDs (phylo = phylogenetically structured, iso_sanitized = independent):\n")
print(re_sd[, c("model", "group", "Estimate", "l-95% CI", "u-95% CI")])

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
cat("\n=== Plots ===\n")
# Plot the most complex model available this run.
plot_fit   <- if (!is.null(fit_comp)) fit_comp else fit_minimal
plot_label <- if (!is.null(fit_comp)) "Macro-ecological" else "Minimal sociodemographic"

p1 <- mcmc_plot(plot_fit, type = "intervals", variable = "^b_", regex = TRUE) +
  theme_minimal(base_size = 12) +
  labs(title = paste(plot_label, "regression coefficients"),
       subtitle = "Beta-binomial model of MGN cell-to-cell prediction accuracy",
       x = "Posterior estimate (log-odds)")
ggsave(file.path(PLOT_DIR, "posterior_coefficients.png"), p1,
       width = 8, height = 6, dpi = 300)

p2 <- plot(conditional_effects(plot_fit, effects = "log10_pop_z:pos"), plot = FALSE)[[1]] +
  theme_minimal(base_size = 12) +
  labs(title = "Speaker population and inflectional predictability",
       x = "Standardised log10 L1 population", y = "Predicted accuracy")
ggsave(file.path(PLOT_DIR, "marginal_effects_population.png"), p2,
       width = 7, height = 5, dpi = 300)

cat("\nAll models, diagnostics, comparisons and plots complete.\n")
