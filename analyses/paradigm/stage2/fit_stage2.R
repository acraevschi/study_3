# Stage-2 Bayesian Berkson, Equivalence (BF_01 / ROPE), and Cross-Track Validation
# for Study 3 Track 2: Morphological Paradigm Predictability (`V` and `N` separate).
#
# Mirrors `analyses/grambank/baseline/fit.R` and `analyses/common/stage1_exposure.R`:
#   - Strictly separate 1-row-per-`glottocode` Verb (`V`) and Noun (`N`) pipelines.
#   - Stage-1 hierarchical shrinkage via `fit_stage1_exposure()`, standardized on the
#     exact Stage-2 language set in Berkson `plain` mode (`x = (eta_mean - mu) / sd`,
#     never `me(x, x_sd)` double-shrinkage), plus a 20-draw posterior-imputation
#     Berkson variance propagation check.
#   - Precision-weighted Stage-2 Gaussian regressions `y | se(se_y, sigma = TRUE)`
#     (`se_y = se_raw / sd_raw`) with `Normal(0, 0.5)` slope priors matching Track 1.
#   - Savage-Dickey `BF_01` at `beta = 0` (both Gaussian and nonparametric KDE),
#     posterior mass in ROPE `[-0.15, +0.15]` SD, and `P(beta < 0.44)` / `P(|beta| < 0.44)`
#     against Track 1's `+0.44` SD inflectional-breadth effect.
#
# Environment variables:
#   PARA_DRY=1:          build Stan code & data for every Stage-1 and Stage-2 model; sample nothing
#   PARA_SMOKE=1:        fast 400-iteration 2-chain smoke fit
#   PARA_CORES=4:        total CPU core cap across chains/workers (default 4, hard cap 16)
#   PARA_POS=V,N:        POS pipelines to run (default: V,N)
#   PARA_SETS=main,...:  link sets to run (default: main,no_doubtful_language,no_other_language_stand_ins,strict)
#   PARA_MEASURES=...:   Stage-1 exposures (default: ent,sqrt_het,f3k_est_pos)
#   PARA_DATA_DIR=...:   directory containing prepared Stage-2 CSV/NWK files
#   PARA_OUT_DIR=...:    output directory for results (default: analyses/paradigm/results/stage2)
#   PARA_N_IMP=20:       number of Stage-1 posterior draws for Berkson propagation check

Sys.setenv(
  OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))

suppressPackageStartupMessages({
  library(brms)
  library(ape)
  library(loo)
  library(posterior)
  library(parallel)
  library(nlme)
})
options(brms.backend = "cmdstanr")

cmd_args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_all, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else getwd()
if (!dir.exists("analyses/paradigm/stage2") &&
    dir.exists(file.path(script_dir, "..", "..", "..", "analyses", "paradigm", "stage2"))) {
  setwd(normalizePath(file.path(script_dir, "..", "..", "..")))
}

common_dir <- normalizePath("analyses/common", mustWork = TRUE)
source(file.path(common_dir, "phylogeny.R"))
source(file.path(common_dir, "stage1_exposure.R"))

dry <- nzchar(Sys.getenv("PARA_DRY"))
smoke <- nzchar(Sys.getenv("PARA_SMOKE"))

CORES <- min(16L, max(1L, as.integer(Sys.getenv("PARA_CORES", "4"))))
if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- 2L
} else {
  ITER <- 3000L; WARMUP <- 1500L; CHAINS <- min(4L, CORES)
}
THREADS <- 1L
PAR_FITS <- max(1L, CORES %/% CHAINS)
stopifnot(PAR_FITS * CHAINS * THREADS <= 16L)

SEED <- 20261112L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
TRACK1_BREADTH_REF <- 0.44
ROPE_HALF_WIDTH <- 0.15

N_IMP_DEFAULT <- if (dry) 2L else if (smoke) 4L else 20L
N_IMP <- max(2L, as.integer(Sys.getenv("PARA_N_IMP", as.character(N_IMP_DEFAULT))))

data_dir <- Sys.getenv("PARA_DATA_DIR", "analyses/paradigm/results/stage2/prepared")
base_out_dir <- Sys.getenv("PARA_OUT_DIR", "analyses/paradigm/results/stage2")
fits_dir <- if (dry) tempdir() else file.path(base_out_dir, if (smoke) "fits_smoke" else "fits")
res_dir <- if (dry) base_out_dir else file.path(base_out_dir, if (smoke) "results_smoke" else ".")
ALL_SETS <- c("main", "no_doubtful_language", "no_other_language_stand_ins", "strict")
pos_list <- strsplit(Sys.getenv("PARA_POS", "V,N"), ",")[[1]]
sets_active <- strsplit(Sys.getenv("PARA_SETS", paste(ALL_SETS, collapse = ",")), ",")[[1]]
measures_active <- strsplit(Sys.getenv("PARA_MEASURES", "ent,sqrt_het,f3k_est_pos"), ",")[[1]]
stopifnot(all(pos_list %in% c("V", "N")), all(sets_active %in% ALL_SETS))
stopifnot(all(measures_active %in% names(get_stage1_measure_specs())))

stan_cache_dir <- file.path(base_out_dir, paste0("stan_cache_", paste(pos_list, collapse = "_")))
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(stan_cache_dir, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = stan_cache_dir)
}

k5_file <- file.path(common_dir, "data/f3_scores_k5.csv")

# --- Helper Functions -------------------------------------------------------------------------

kde_density_at_zero <- function(draws_vec) {
  s <- sd(draws_vec)
  if (is.na(s) || s < 1e-12) return(NA_real_)
  if (requireNamespace("logspline", quietly = TRUE)) {
    ls_fit <- tryCatch(logspline::logspline(draws_vec), error = function(e) NULL)
    if (!is.null(ls_fit)) {
      val <- tryCatch(logspline::dlogspline(0.0, ls_fit), error = function(e) NA_real_)
      if (is.finite(val) && val >= 0) return(val)
    }
  }
  bw_val <- tryCatch( stats::bw.SJ(draws_vec), error = function(e) stats::bw.nrd0(draws_vec) )
  dens <- stats::density(
    draws_vec, bw = bw_val,
    from = min(draws_vec, 0.0) - 3.0 * s,
    to = max(draws_vec, 0.0) + 3.0 * s,
    n = 2048L
  )
  as.numeric(stats::approx(dens$x, dens$y, xout = 0.0)$y)
}

summarise_slope_equivalence <- function(b_draws, pfx = "b", b_rhat_val = NA_real_, b_ess_val = NA_real_) {
  mu <- mean(b_draws)
  s <- sd(b_draws)
  d0_norm <- dnorm(0.0, mean = mu, sd = s)
  d0_kde <- kde_density_at_zero(b_draws)
  vals <- list(
    mean = mu,
    sd = s,
    q025 = unname(quantile(b_draws, 0.025)),
    q05  = unname(quantile(b_draws, 0.05)),
    q95  = unname(quantile(b_draws, 0.95)),
    q975 = unname(quantile(b_draws, 0.975)),
    rhat = b_rhat_val,
    ess_bulk = b_ess_val,
    p_pos = mean(b_draws > 0.0),
    p_neg = mean(b_draws < 0.0),
    bf01_norm = d0_norm / PRIOR_DENS_ZERO,
    bf01_kde  = if (is.finite(d0_kde)) d0_kde / PRIOR_DENS_ZERO else NA_real_,
    p_rope_015 = mean(abs(b_draws) <= ROPE_HALF_WIDTH),
    p_lt_044 = mean(b_draws < TRACK1_BREADTH_REF),
    p_abs_lt_044 = mean(abs(b_draws) < TRACK1_BREADTH_REF)
  )
  setNames(vals, paste0(pfx, "_", names(vals)))
}

prepare_bouckaert_for_sample <- function(glottocodes) {
  tr <- tryCatch(load_bouckaert_tree(apply_split_map = FALSE), error = function(e) NULL)
  if (is.null(tr)) return(NULL)
  # Apply pre-split / sister mappings ONLY when target tip is needed and source tip is not needed
  extra_map <- c(
    BOUCKAERT_SPLIT_MAP,
    ladi1250 = "ladi1251",
    cent2322 = "sout3253"
  )
  for (old_c in names(extra_map)) {
    new_c <- unname(extra_map[old_c])
    if ((new_c %in% glottocodes) && !(old_c %in% glottocodes)) {
      idx <- which(tr$tip.label == old_c)
      if (length(idx) == 1L && !(new_c %in% tr$tip.label)) {
        tr$tip.label[idx] <- new_c
      }
    }
  }
  tr
}

make_stage2_prior <- function(has_b, has_sd, has_gp) {
  pr <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(exponential(1), class = sigma)
  )
  if (isTRUE(has_b))  pr <- c(pr, prior(normal(0, 0.5), class = b))
  if (isTRUE(has_sd)) pr <- c(pr, prior(exponential(1), class = sd))
  if (isTRUE(has_gp)) pr <- c(pr, prior(exponential(1), class = sdgp))
  pr
}

fit_stage2_brm <- function(name, formula_obj, data_df, prior_obj, data2_list = NULL, adapt = 0.99) {
  if (dry) {
    make_stancode(formula_obj, data = data_df, data2 = data2_list, family = gaussian(),
                  prior = prior_obj, threads = threading(THREADS))
    make_standata(formula_obj, data = data_df, data2 = data2_list, family = gaussian(),
                  prior = prior_obj)
    return(structure(list(), class = "dry"))
  }
  fit <- brm(
    formula_obj, data = data_df, data2 = data2_list, family = gaussian(), prior = prior_obj,
    chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS, threads = threading(THREADS),
    seed = SEED, control = list(adapt_delta = adapt, max_treedepth = 12),
    file = file.path(fits_dir, name), file_refit = "on_change", refresh = 0, silent = 2
  )
  if (!smoke) {
    sm_chk <- posterior::summarise_draws(posterior::as_draws_df(fit))
    max_rh_chk <- suppressWarnings(max(sm_chk$rhat, na.rm = TRUE))
    if (is.finite(max_rh_chk) && max_rh_chk > 1.20) {
      message("  [refit high-rhat] ", name, " (max_rhat=", round(max_rh_chk, 3),
              ") -> iter=6000, adapt_delta=0.999, max_treedepth=14")
      fit <- brm(
        formula_obj, data = data_df, data2 = data2_list, family = gaussian(), prior = prior_obj,
        chains = CHAINS, iter = 6000L, warmup = 3000L, cores = CHAINS, threads = threading(THREADS),
        seed = SEED + 101L, control = list(adapt_delta = 0.999, max_treedepth = 14),
        file = file.path(fits_dir, name), file_refit = "always", refresh = 0, silent = 2
      )
    }
  }
  fit
}

extract_stage1_draws_matrix <- function(fit_rds_path, langs, n_imp, seed_imp = 20261112L) {
  if (dry) {
    set.seed(seed_imp)
    mat <- matrix(rnorm(n_imp * length(langs)), nrow = n_imp, ncol = length(langs))
    colnames(mat) <- langs
    return(mat)
  }
  if (!file.exists(fit_rds_path)) {
    stop("Stage-1 fit RDS missing for Berkson 20-draw propagation check: ", fit_rds_path)
  }
  fit <- readRDS(fit_rds_path)
  dr <- as_draws_df(fit)
  n_tot <- nrow(dr)
  idx <- unique(round(seq(1L, n_tot, length.out = min(n_imp, n_tot))))
  eta_mat <- sapply(langs, function(l) {
    dr$b_Intercept[idx] + dr[[sprintf("r_lang[%s,Intercept]", l)]][idx]
  })
  if (is.null(dim(eta_mat))) {
    eta_mat <- matrix(eta_mat, nrow = 1L)
  }
  colnames(eta_mat) <- langs
  eta_mat
}

standardize_outcome_and_covs <- function(df, y_col, se_col, use_demog = FALSE, use_cdist = FALSE) {
  sub <- df[!is.na(df[[y_col]]) & is.finite(df[[y_col]]), , drop = FALSE]
  if (isTRUE(use_demog)) {
    sub <- sub[!is.na(sub$has_area) & sub$has_area == 1L &
               is.finite(sub$log_pop) & is.finite(sub$log_area), , drop = FALSE]
  }
  if (nrow(sub) < 6L) return(NULL)

  y_raw <- as.numeric(sub[[y_col]])
  se_raw <- as.numeric(sub[[se_col]])
  se_raw[!is.finite(se_raw) | se_raw <= 0] <- median(se_raw[is.finite(se_raw) & se_raw > 0], na.rm = TRUE)
  if (!all(is.finite(se_raw))) se_raw <- rep(0.05, nrow(sub))

  mu_y <- mean(y_raw)
  sd_y <- sd(y_raw)
  if (is.na(sd_y) || sd_y < 1e-8) return(NULL)
  sub$y <- (y_raw - mu_y) / sd_y
  sub$se_y <- pmax(se_raw / sd_y, 1e-4)

  # Standardize continuous paradigm covariates on the exact active subset
  ln_av <- as.numeric(sub$log_navail)
  sub$z_log_navail <- if (sd(ln_av) > 1e-8) as.numeric(scale(ln_av)) else 0.0

  if (isTRUE(use_cdist)) {
    lc <- as.numeric(sub$log_C_distinct)
    sub$z_log_C_distinct <- if (sd(lc) > 1e-8) as.numeric(scale(lc)) else 0.0
  }
  if (isTRUE(use_demog)) {
    lp <- as.numeric(sub$log_pop)
    la <- as.numeric(sub$log_area)
    sub$z_pop <- if (sd(lp) > 1e-8) as.numeric(scale(lp)) else 0.0
    sub$z_area <- if (sd(la) > 1e-8) as.numeric(scale(la)) else 0.0
  }
  sub$source_prov <- as.character(sub$source_prov)
  sub$lang <- sub$glottocode
  sub$fam_eff <- ifelse(
    is.na(sub$family) | !nzchar(sub$family) | sub$family == "isolate",
    paste0("isolate__", sub$glottocode),
    as.character(sub$family)
  )
  attr(sub, "y_scale_info") <- list(y_center = mu_y, y_scale = sd_y)
  sub
}

build_cov_terms <- function(df, use_demog = FALSE, use_cdist = FALSE) {
  terms <- character(0)
  if (isTRUE(use_demog)) {
    if (sd(df$z_pop) > 1e-8) terms <- c(terms, "z_pop")
    if (sd(df$z_area) > 1e-8) terms <- c(terms, "z_area")
  }
  if (sd(df$z_log_navail) > 1e-8) terms <- c(terms, "z_log_navail")
  if (isTRUE(use_cdist) && sd(df$z_log_C_distinct) > 1e-8) {
    terms <- c(terms, "z_log_C_distinct")
  }
  # Only include source_prov factor when >= 2 levels are each attested on the active subset
  prov_tab <- table(df$source_prov)
  if (length(prov_tab[prov_tab >= 1L]) >= 2L) {
    terms <- c(terms, "source_prov")
  }
  terms
}

# --- Stage-2 Runner Per POS & Link Set --------------------------------------------------------

run_single_spec <- function(sp) {
  t0 <- Sys.time()
  f_obj <- bf(as.formula(sp$formula))
  pr_obj <- make_stage2_prior(sp$has_b, sp$has_sd, sp$has_gp)
  fit <- tryCatch(
    fit_stage2_brm(sp$name, f_obj, sp$data, pr_obj, data2_list = sp$data2, adapt = sp$adapt),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(name = sp$name, error = conditionMessage(fit)))
  }
  base_info <- list(
    name = sp$name,
    pos = sp$pos,
    set = sp$set,
    spec_group = sp$spec_group,
    outcome = sp$outcome,
    measure = sp$measure,
    ctrl = sp$ctrl,
    xmode = sp$xmode,
    n_lang = nrow(sp$data)
  )
  if (dry) return(base_info)

  fit <- tryCatch(add_criterion(fit, "loo", file = file.path(fits_dir, sp$name)), error = function(e) fit)
  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)
  loo_est <- if (!is.null(fit$criteria$loo)) fit$criteria$loo$estimates["elpd_loo", "Estimate"] else NA_real_
  pk_bad <- if (!is.null(fit$criteria$loo)) sum(fit$criteria$loo$diagnostics$pareto_k > 0.7) else NA_integer_

  out <- c(base_info, list(
    max_rhat = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
    sigma_mean = mean(dr$sigma),
    elpd_loo = loo_est,
    pareto_k_gt_07 = pk_bad,
    minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))

  if (isTRUE(sp$is_mundlak)) {
    if ("b_x_within" %in% names(dr) && "b_x_family_mean" %in% names(dr)) {
      bw <- dr$b_x_within
      bb <- dr$b_x_family_mean
      b_diff <- bw - bb
      idx_w <- match("b_x_within", sm$variable)
      idx_b <- match("b_x_family_mean", sm$variable)
      out <- c(
        out,
        summarise_slope_equivalence(
          bw, pfx = "b",
          b_rhat_val = if (!is.na(idx_w)) sm$rhat[idx_w] else NA_real_,
          b_ess_val = if (!is.na(idx_w)) sm$ess_bulk[idx_w] else NA_real_
        ),
        summarise_slope_equivalence(
          bb, pfx = "b_between",
          b_rhat_val = if (!is.na(idx_b)) sm$rhat[idx_b] else NA_real_,
          b_ess_val = if (!is.na(idx_b)) sm$ess_bulk[idx_b] else NA_real_
        ),
        list(
          b_wb_diff_mean = mean(b_diff),
          b_wb_diff_q025 = unname(quantile(b_diff, 0.025)),
          b_wb_diff_q975 = unname(quantile(b_diff, 0.975)),
          p_wb_diff_pos = mean(b_diff > 0.0)
        )
      )
    }
  } else if ("b_x" %in% names(dr)) {
    idx_x <- match("b_x", sm$variable)
    out <- c(
      out,
      summarise_slope_equivalence(
        dr$b_x, pfx = "b",
        b_rhat_val = if (!is.na(idx_x)) sm$rhat[idx_x] else NA_real_,
        b_ess_val = if (!is.na(idx_x)) sm$ess_bulk[idx_x] else NA_real_
      )
    )
  }

  if (!is.null(sp$scale_info)) out <- c(out, sp$scale_info)
  out
}

run_berkson_20draw_check <- function(pos_code, set_name, dd_base, eta_draws_mat, A_grafen) {
  spec_name <- paste0("s2_", pos_code, "__", set_name, "__berkson_20draw__ent_P")
  langs <- dd_base$glottocode
  if (!all(langs %in% colnames(eta_draws_mat))) return(NULL)
  eta_sub <- eta_draws_mat[, langs, drop = FALSE]
  n_d <- nrow(eta_sub)

  cov_terms <- build_cov_terms(dd_base, use_demog = "z_pop" %in% names(dd_base), use_cdist = FALSE)
  rhs_fixed <- paste(c("1", "x", cov_terms), collapse = " + ")
  form_str <- paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_fixed, " + (1 | gr(lang, cov = A))")
  f_obj <- bf(as.formula(form_str))
  pr_obj <- make_stage2_prior(has_b = TRUE, has_sd = TRUE, has_gp = FALSE)
  d2 <- list(A = A_grafen[langs, langs, drop = FALSE])

  t0 <- Sys.time()
  pooled_b <- numeric(0)
  max_rh <- 1.0
  max_b_rh <- 1.0
  min_b_ess <- Inf
  tot_div <- 0L

  for (d_idx in seq_len(n_d)) {
    dd_i <- dd_base
    e_vec <- as.numeric(eta_sub[d_idx, ])
    s_e <- sd(e_vec)
    if (is.na(s_e) || s_e < 1e-8) next
    dd_i$x <- (e_vec - mean(e_vec)) / s_e
    nm_i <- sprintf("%s__imp%02d", spec_name, d_idx)
    fit_i <- tryCatch(
      fit_stage2_brm(nm_i, f_obj, dd_i, pr_obj, data2_list = d2, adapt = 0.99),
      error = function(e) e
    )
    if (inherits(fit_i, "error")) {
      return(list(name = spec_name, error = conditionMessage(fit_i)))
    }
    if (!dry) {
      dr_i <- as_draws_df(fit_i)
      sm_i <- summarise_draws(dr_i)
      pooled_b <- c(pooled_b, dr_i$b_x)
      max_rh <- max(max_rh, max(sm_i$rhat, na.rm = TRUE))
      idx_bx <- match("b_x", sm_i$variable)
      if (!is.na(idx_bx)) {
        max_b_rh <- max(max_b_rh, sm_i$rhat[idx_bx], na.rm = TRUE)
        min_b_ess <- min(min_b_ess, sm_i$ess_bulk[idx_bx], na.rm = TRUE)
      }
      tot_div <- tot_div + sum(subset(nuts_params(fit_i), Parameter == "divergent__")$Value)
    }
  }

  base_info <- list(
    name = spec_name, pos = pos_code, set = set_name,
    spec_group = "berkson_20draw", outcome = "h_tree_bits",
    measure = "ent", ctrl = "P", xmode = sprintf("berkson_%ddraw", n_d),
    n_lang = nrow(dd_base)
  )
  if (dry) return(base_info)

  c(
    base_info,
    list(
      max_rhat = max_rh,
      min_ess_bulk = NA_real_,
      divergences = tot_div,
      sigma_mean = NA_real_,
      elpd_loo = NA_real_,
      pareto_k_gt_07 = NA_integer_,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ),
    summarise_slope_equivalence(
      pooled_b, pfx = "b",
      b_rhat_val = if (is.finite(max_b_rh)) max_b_rh else NA_real_,
      b_ess_val = if (is.finite(min_b_ess)) min_b_ess else NA_real_
    )
  )
}

run_pos_pipeline <- function(pos_code) {
  ld_file <- file.path(data_dir, sprintf("language_data_%s.csv", pos_code))
  ind_file <- file.path(data_dir, sprintf("individuals_%s.csv", pos_code))
  pop_file <- file.path(data_dir, sprintf("populations_%s.csv", pos_code))
  nwk_file <- file.path(data_dir, sprintf("phylo_tree_%s.nwk", pos_code))
  if (!all(file.exists(c(ld_file, ind_file, pop_file, nwk_file)))) {
    message("Skipping POS=", pos_code, " (prepared files not found in ", data_dir, ")")
    return(NULL)
  }

  d0 <- read.csv(ld_file, stringsAsFactors = FALSE)
  ind0 <- read.csv(ind_file, stringsAsFactors = FALSE)
  pop0 <- read.csv(pop_file, stringsAsFactors = FALSE)
  pop0 <- augment_population_f3_columns(pop0, k5_file = k5_file)
  pop0$f3k_z[is.na(pop0$f3_z)] <- NA_real_
  pop0$f3k_est[is.na(pop0$f3_z)] <- NA_real_
  pop0$f3k_est_pos[is.na(pop0$f3_z)] <- NA_real_
  stopifnot(!any(duplicated(d0$glottocode)))

  # Compute bounded logit-reliability outcome + delta-method SE
  rel_clip <- pmin(pmax(as.numeric(d0$rel_tree_mean), 1e-3), 1.0 - 1e-3)
  d0$logit_rel_tree <- qlogis(rel_clip)
  d0$logit_rel_tree_se <- as.numeric(d0$rel_tree_se) / pmax(rel_clip * (1.0 - rel_clip), 0.02)

  to_rad <- pi / 180.0
  d0$sx <- cos(d0$lat * to_rad) * cos(d0$lon * to_rad)
  d0$sy <- cos(d0$lat * to_rad) * sin(d0$lon * to_rad)
  d0$sz <- sin(d0$lat * to_rad)

  raw_tree <- read.tree(nwk_file)
  d_main_all <- d0[d0$in_main == 1L, , drop = FALSE]
  A_grafen_all <- build_grafen_corr(raw_tree, d_main_all$glottocode)
  bouckaert_tr <- prepare_bouckaert_for_sample(d_main_all$glottocode)

  all_pos_results <- list()

  for (set_name in sets_active) {
    d_set <- d0[d0[[paste0("in_", set_name)]] == 1L, , drop = FALSE]
    if (nrow(d_set) < 6L) {
      message("Skipping POS=", pos_code, " set=", set_name, " (only ", nrow(d_set), " langs)")
      next
    }
    pop_set <- pop0[pop0[[paste0("in_", set_name)]] == 1L & pop0$glottocode %in% d_set$glottocode, , drop = FALSE]

    # 1. Stage 1 outcome-blind hierarchical shrinkage
    s1_list <- lapply(measures_active, function(m) {
      fit_stage1_exposure(
        measure = m, ind_df = ind0, pop_df = pop_set,
        set_name = set_name, fits_dir = fits_dir,
        fit_prefix = paste0("para_", pos_code, "_"),
        dry = dry, chains = CHAINS, iter = ITER, warmup = WARMUP,
        threads = THREADS, seed = SEED
      )
    })
    expo_df <- do.call(rbind, lapply(s1_list, `[[`, "e"))
    diag_df <- do.call(rbind, lapply(s1_list, `[[`, "diag"))
    if (!dry) {
      write.csv(expo_df, file.path(res_dir, sprintf("exposure_%s_%s.csv", set_name, pos_code)), row.names = FALSE)
      write.csv(diag_df, file.path(res_dir, sprintf("stage1_diagnostics_%s_%s.csv", set_name, pos_code)), row.names = FALSE)
    }

    has_col <- function(col) intersect(d_set$glottocode, unique(pop_set$glottocode[!is.na(pop_set[[col]])]))
    subs_map <- list(
      all  = d_set$glottocode,
      neigh = has_col("neigh_flow"),
      het  = has_col("het"),
      f3   = has_col("f3_z"),
      f3b3 = has_col("f3b3_z")
    )

    specs <- list()

    # Helper to register Berkson `plain` Stage-2 model specs: standardizes y, se_y, x,
    # and covariates strictly on the exact per-measure language subset `dd` (`req_gcs`).
    add_with_x_specs <- function(d_raw_base, spec_group, out_col, se_col, ctrls,
                                 use_demog_flag = FALSE, use_cdist_flag = FALSE,
                                 allow_mundlak = FALSE, custom_A = NULL, extra_re = "") {
      if (is.null(d_raw_base) || nrow(d_raw_base) < 6L || !(out_col %in% names(d_raw_base))) return()
      d_cand <- d_raw_base[!is.na(d_raw_base[[out_col]]) & is.finite(d_raw_base[[out_col]]), , drop = FALSE]
      if (isTRUE(use_demog_flag)) {
        d_cand <- d_cand[!is.na(d_cand$has_area) & d_cand$has_area == 1L &
                         is.finite(d_cand$log_pop) & is.finite(d_cand$log_area), , drop = FALSE]
      }
      if (nrow(d_cand) < 6L) return()

      for (m in measures_active) {
        sb_code <- subset_of_measure(m)
        req_gcs <- intersect(d_cand$glottocode, subs_map[[sb_code]])
        if (length(req_gcs) < 6L) next
        e_m <- expo_df[expo_df$measure == m & expo_df$glottocode %in% req_gcs, , drop = FALSE]
        stopifnot(setequal(e_m$glottocode, req_gcs))

        d_sub_m <- standardize_outcome_and_covs(
          d_cand[d_cand$glottocode %in% req_gcs, , drop = FALSE],
          out_col, se_col, use_demog = use_demog_flag, use_cdist = use_cdist_flag
        )
        if (is.null(d_sub_m)) next
        y_sc <- attr(d_sub_m, "y_scale_info")

        dd <- merge(d_sub_m,
                    e_m[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
                    by = "glottocode")
        dd <- dd[order(dd$glottocode), , drop = FALSE]
        stopifnot(setequal(dd$glottocode, req_gcs))

        cov_t <- build_cov_terms(dd, use_demog = use_demog_flag, use_cdist = use_cdist_flag)

        mu_x <- mean(dd$eta_mean)
        sd_x <- sd(dd$eta_mean)
        if (is.na(sd_x) || sd_x < 1e-8) next
        dd$x <- (dd$eta_mean - mu_x) / sd_x
        sc_info <- c(y_sc, list(x_center = mu_x, x_scale = sd_x, s1_y_mean = dd$y_mean[1], s1_y_sd = dd$y_sd[1]))

        for (ctrl in ctrls) {
          A_use <- if (!is.null(custom_A)) {
            custom_A[dd$lang, dd$lang, drop = FALSE]
          } else {
            A_grafen_all[dd$lang, dd$lang, drop = FALSE]
          }
          has_phylo <- ctrl %in% c("P", "F", "P_bouckaert_stem", "P_bouckaert_crown")
          has_space <- ctrl %in% c("S", "F")
          re_str <- paste0(
            if (has_phylo) " + (1 | gr(lang, cov = A))" else "",
            extra_re,
            if (has_space) " + gp(sx, sy, sz)" else ""
          )
          rhs_plain <- paste(c("1", "x", cov_t), collapse = " + ")
          specs[[length(specs) + 1L]] <<- list(
            name = paste0("s2_", pos_code, "__", set_name, "__", spec_group, "__", out_col, "__", m, "_", ctrl),
            pos = pos_code, set = set_name, spec_group = spec_group, outcome = out_col,
            measure = m, ctrl = ctrl, xmode = "plain",
            data = dd, data2 = if (has_phylo) list(A = A_use) else NULL,
            formula = paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_plain, re_str),
            has_b = TRUE, has_sd = (has_phylo || nzchar(extra_re)), has_gp = has_space,
            is_mundlak = FALSE, adapt = if (has_space) 0.995 else 0.99,
            scale_info = sc_info
          )

          # Mundlak within- vs. between-family decomposition (properly zeroing singleton families)
          if (isTRUE(allow_mundlak) && ctrl %in% c("P", "F")) {
            fam_sz <- as.integer(ave(seq_len(nrow(dd)), dd$fam_eff, FUN = length))
            if (sum(fam_sz >= 2L) >= 4L && length(unique(dd$fam_eff)) >= 2L) {
              dd_m <- dd
              dd_m$x_family_mean <- as.numeric(ave(dd_m$x, dd_m$fam_eff, FUN = mean))
              dd_m$x_within <- as.numeric(dd_m$x - dd_m$x_family_mean)
              dd_m$x_within[fam_sz == 1L] <- 0.0
              if (sd(dd_m$x_within) > 1e-8 && sd(dd_m$x_family_mean) > 1e-8) {
                rhs_mun <- paste(c("1", "x_within", "x_family_mean", cov_t), collapse = " + ")
                specs[[length(specs) + 1L]] <<- list(
                  name = paste0("s2_", pos_code, "__", set_name, "__", spec_group, "_mundlak__", out_col, "__", m, "_", ctrl),
                  pos = pos_code, set = set_name, spec_group = paste0(spec_group, "_mundlak"), outcome = out_col,
                  measure = m, ctrl = ctrl, xmode = "plain_mundlak",
                  data = dd_m, data2 = if (has_phylo) list(A = A_use) else NULL,
                  formula = paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_mun, re_str),
                  has_b = TRUE, has_sd = (has_phylo || nzchar(extra_re)), has_gp = has_space,
                  is_mundlak = TRUE, adapt = if (has_space) 0.995 else 0.99,
                  scale_info = sc_info
                )
              }
            }
          }
        }
      }
    }

    # (a) Primary full-sample & demographic models on `h_tree_bits`
    d_full <- standardize_outcome_and_covs(d_set, "h_tree_bits", "h_tree_se", use_demog = FALSE)

    if (set_name == "main") {
      # Null baseline (`P` and `F`) on `main` for LOO delta_elpd checks
      if (!is.null(d_full)) {
        cov_b <- build_cov_terms(d_full, use_demog = FALSE)
        rhs_b <- paste(c("1", cov_b), collapse = " + ")
        for (b_ctrl in c("P", "F")) {
          has_sp <- (b_ctrl == "F")
          specs[[length(specs) + 1L]] <- list(
            name = paste0("s2_", pos_code, "__main__base__h_tree_bits__none_", b_ctrl),
            pos = pos_code, set = "main", spec_group = "baseline", outcome = "h_tree_bits",
            measure = "none", ctrl = b_ctrl, xmode = "none",
            data = d_full, data2 = list(A = A_grafen_all[d_full$lang, d_full$lang, drop = FALSE]),
            formula = paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_b,
                             " + (1 | gr(lang, cov = A))", if (has_sp) " + gp(sx, sy, sz)" else ""),
            has_b = (length(cov_b) > 0L), has_sd = TRUE, has_gp = has_sp,
            is_mundlak = FALSE, adapt = if (has_sp) 0.995 else 0.99,
            scale_info = attr(d_full, "y_scale_info")
          )
        }
      }

      add_with_x_specs(d_set, "full_covs", "h_tree_bits", "h_tree_se", ctrls = c("N", "P", "S", "F"),
                       use_demog_flag = FALSE, allow_mundlak = TRUE)
      add_with_x_specs(d_set, "primary_demog", "h_tree_bits", "h_tree_se", ctrls = c("P", "F"),
                       use_demog_flag = TRUE, allow_mundlak = TRUE)

      # (b) Time-calibrated Bouckaert v2.0.0 VCV (`P_bouckaert_stem` & `P_bouckaert_crown`)
      if (!is.null(bouckaert_tr)) {
        b_gcs <- intersect(d_set$glottocode, bouckaert_tr$tip.label)
        if (length(b_gcs) >= 6L) {
          d_bk_raw <- d_set[d_set$glottocode %in% b_gcs, , drop = FALSE]
          A_bk_stem <- build_bouckaert_corr(
            d_bk_raw$glottocode, families = d_bk_raw$family, tree = bouckaert_tr,
            within_family = TRUE, crown_only = FALSE
          )
          A_bk_crown <- build_bouckaert_corr(
            d_bk_raw$glottocode, families = d_bk_raw$family, tree = bouckaert_tr,
            within_family = TRUE, crown_only = TRUE, scale_by_age = FALSE
          )
          add_with_x_specs(d_bk_raw, "bouckaert", "h_tree_bits", "h_tree_se", ctrls = "P_bouckaert_stem",
                           use_demog_flag = TRUE, custom_A = A_bk_stem)
          add_with_x_specs(d_bk_raw, "bouckaert", "h_tree_bits", "h_tree_se", ctrls = "P_bouckaert_crown",
                           use_demog_flag = TRUE, custom_A = A_bk_crown, extra_re = " + (1 | fam_eff)")
        }
      }

      # (c) Paradigm-size control (`+ z_log_C_distinct`)
      add_with_x_specs(d_set, "ctrl_C_distinct", "h_tree_bits", "h_tree_se", ctrls = "F",
                       use_demog_flag = TRUE, use_cdist_flag = TRUE)

      # (d) Subsampling & secondary metric checks (`uniform480`, `topfreq120`, `k1`, `k3`, `rel`, `neural`)
      sec_specs <- list(
        list(grp = "tier2_uniform480", y = "h_tree_480_bits",     se = "h_tree_480_se"),
        list(grp = "topfreq120",       y = "h_tree_topfreq_bits", se = "h_tree_topfreq_se"),
        list(grp = "pcfp_k1",          y = "h_k1_bits",           se = "h_k1_se"),
        list(grp = "pcfp_k3",          y = "h_k3_bits",           se = "h_k3_se"),
        list(grp = "reliability",      y = "logit_rel_tree",      se = "logit_rel_tree_se"),
        list(grp = "neural_monotonic", y = "h_tree_neural_bits",  se = "h_tree_neural_se")
      )
      for (sc in sec_specs) {
        if (sc$y %in% names(d_set)) {
          add_with_x_specs(d_set, sc$grp, sc$y, sc$se, ctrls = "P", use_demog_flag = FALSE)
        }
      }

      # (e) Inner Asian exclusions (`no_turk_mong` & `no_inner_asia`)
      d_ntm <- d_set[!(d_set$family %in% c("Turkic", "Mongolic-Khitan") |
                       d_set$family_id %in% c("turk1311", "mong1329")), , drop = FALSE]
      d_nia <- d_set[!(d_set$family %in% c("Turkic", "Mongolic-Khitan", "Tungusic") |
                       d_set$family_id %in% c("turk1311", "mong1329", "tung1282")), , drop = FALSE]
      add_with_x_specs(d_ntm, "no_turk_mong", "h_tree_bits", "h_tree_se", ctrls = "P", use_demog_flag = FALSE)
      add_with_x_specs(d_nia, "no_inner_asia", "h_tree_bits", "h_tree_se", ctrls = "P", use_demog_flag = FALSE)
    } else {
      # Sensitivity link sets (`no_doubtful_language`, `no_other_language_stand_ins`, `strict`)
      add_with_x_specs(d_set, "link_sensitivity", "h_tree_bits", "h_tree_se", ctrls = "F", use_demog_flag = FALSE)
      add_with_x_specs(d_set, "link_sensitivity_demog", "h_tree_bits", "h_tree_se", ctrls = "F", use_demog_flag = TRUE)
    }

    # Execute Stage-2 specs (sequential in dry mode or PAR_FITS == 1, PSOCK parallel otherwise)
    if (dry || PAR_FITS <= 1L || length(specs) <= 1L) {
      res_set <- lapply(specs, run_single_spec)
    } else {
      cl <- makePSOCKcluster(min(length(specs), PAR_FITS), outfile = "")
      res_set <- tryCatch({
        clusterEvalQ(cl, {
          suppressPackageStartupMessages({ library(brms); library(ape); library(loo); library(posterior) })
          options(brms.backend = "cmdstanr"); NULL
        })
        clusterExport(cl, ls(envir = .GlobalEnv), envir = .GlobalEnv)
        parLapplyLB(cl, specs, run_single_spec)
      }, finally = {
        try(stopCluster(cl), silent = TRUE)
      })
    }

    # 20-draw posterior-imputation Berkson propagation check on `main` (`ent`)
    if (set_name == "main" && "ent" %in% measures_active && !is.null(d_full)) {
      ent_rds <- file.path(fits_dir, paste0("exposure_para_", pos_code, "_main__ent.rds"))
      eta_draws <- extract_stage1_draws_matrix(ent_rds, d_full$glottocode, n_imp = N_IMP, seed_imp = SEED)
      berk_res <- run_berkson_20draw_check(pos_code, "main", d_full, eta_draws, A_grafen_all)
      if (!is.null(berk_res)) res_set[[length(res_set) + 1L]] <- berk_res
    }

    errs <- Filter(function(r) !is.list(r) || !is.null(r$error), res_set)
    for (er in errs) {
      message("ERROR [POS=", pos_code, " set=", set_name, "]: ",
              if (is.list(er)) paste(er$name, er$error) else as.character(er))
    }
    if (dry && length(errs) > 0L) {
      stop("PARA_DRY validation failed on ", length(errs), " model(s) for POS=", pos_code)
    }
    ok_set <- Filter(function(r) is.list(r) && is.null(r$error), res_set)
    all_pos_results <- c(all_pos_results, ok_set)
    message(format(Sys.time(), "%H:%M"), " [POS=", pos_code, " set=", set_name, "] ",
            length(ok_set), " models ", if (dry) "validated (DRY)" else "completed")
  }

  if (length(all_pos_results) > 0L) {
    all_cols <- unique(unlist(lapply(all_pos_results, names)))
    tab_pos <- do.call(rbind, lapply(all_pos_results, function(r) {
      as.data.frame(lapply(setNames(all_cols, all_cols), function(k) {
        if (!is.null(r[[k]]) && length(r[[k]]) == 1L) r[[k]] else NA
      }), stringsAsFactors = FALSE)
    }))
    out_file <- file.path(res_dir, paste0("stage2_results_", pos_code, if (dry) "_dry" else "", ".csv"))
    write.csv(tab_pos, out_file, row.names = FALSE)
    return(tab_pos)
  }
  NULL
}

# --- Cross-Track & Within-Track Construct Validation Correlations -----------------------------

compute_construct_correlations <- function() {
  rows <- list()

  safe_phylo_gls <- function(df_xy, x_col, y_col, raw_tree) {
    sub <- df_xy[is.finite(df_xy[[x_col]]) & is.finite(df_xy[[y_col]]), , drop = FALSE]
    n <- nrow(sub)
    if (n < 5L) return(NULL)
    x_v <- as.numeric(sub[[x_col]])
    y_v <- as.numeric(sub[[y_col]])
    if (sd(x_v) < 1e-8 || sd(y_v) < 1e-8) return(NULL)

    ct <- suppressWarnings(cor.test(x_v, y_v, method = "pearson"))
    rho_sp <- suppressWarnings(cor(x_v, y_v, method = "spearman"))

    pgls_beta <- NA_real_; pgls_p <- NA_real_
    if (!is.null(raw_tree) && all(sub$glottocode %in% raw_tree$tip.label)) {
      tr_sub <- compute.brlen(collapse.singles(keep.tip(raw_tree, sub$glottocode)), method = "Grafen")
      df_g <- data.frame(
        glottocode = sub$glottocode,
        zx = as.numeric(scale(x_v)),
        zy = as.numeric(scale(y_v)),
        row.names = sub$glottocode
      )
      fit_g <- tryCatch(
        nlme::gls(zy ~ zx, data = df_g, correlation = ape::corBrownian(1, phy = tr_sub, form = ~glottocode)),
        error = function(e) NULL
      )
      if (!is.null(fit_g)) {
        tt <- summary(fit_g)$tTable
        pgls_beta <- unname(tt["zx", "Value"])
        pgls_p <- unname(tt["zx", "p-value"])
      }
    }

    data.frame(
      n_overlap = n,
      pearson_r = unname(ct$estimate),
      pearson_p = ct$p.value,
      spearman_rho = rho_sp,
      pgls_std_beta = pgls_beta,
      pgls_p = pgls_p,
      stringsAsFactors = FALSE
    )
  }

  loaded_ld <- list()
  loaded_tr <- list()
  for (p in c("V", "N")) {
    lf <- file.path(data_dir, sprintf("language_data_%s.csv", p))
    nf <- file.path(data_dir, sprintf("phylo_tree_%s.nwk", p))
    if (file.exists(lf) && file.exists(nf)) {
      loaded_ld[[p]] <- read.csv(lf, stringsAsFactors = FALSE)
      loaded_tr[[p]] <- read.tree(nf)
    }
  }

  for (p in names(loaded_ld)) {
    df_p <- loaded_ld[[p]]
    tr_p <- loaded_tr[[p]]
    t1_pairs <- list(
      c("h_tree_bits", "irreg13_prop",      sprintf("H_tree_%s_vs_Track1_irreg13_prop", p)),
      c("h_tree_bits", "theta_irreg13_eap", sprintf("H_tree_%s_vs_Track1_theta_irreg13_EAP", p)),
      c("h_tree_bits", "irreg_verb_prop",   sprintf("H_tree_%s_vs_Track1_irreg_verb_prop", p)),
      c("h_tree_bits", "irreg_noun_prop",   sprintf("H_tree_%s_vs_Track1_irreg_noun_prop", p)),
      c("h_tree_bits", "h_tree_neural_bits", sprintf("H_tree_%s_MGL_vs_Neural_120", p)),
      c("h_tree_bits", "h_tree_480_bits",    sprintf("H_tree_%s_120_vs_480", p)),
      c("h_tree_bits", "h_tree_topfreq_bits", sprintf("H_tree_%s_uniform_vs_topfreq", p)),
      c("h_tree_bits", "h_tree_phon_bits",   sprintf("H_tree_%s_orth_vs_phon", p)),
      c("h_tree_bits", "h_tree_malouf_bits", sprintf("H_tree_%s_orth_vs_malouf", p))
    )
    for (pr in t1_pairs) {
      if (all(pr[1:2] %in% names(df_p))) {
        st <- safe_phylo_gls(df_p, pr[1], pr[2], tr_p)
        if (!is.null(st)) {
          rows[[length(rows) + 1L]] <- cbind(
            data.frame(comparison = pr[3], pos = p, x_var = pr[1], y_var = pr[2], stringsAsFactors = FALSE),
            st
          )
        }
      }
    }
  }

  # Cross-POS correlation between H_tree_V and H_tree_N on shared languages
  if (all(c("V", "N") %in% names(loaded_ld))) {
    m_vn <- merge(
      loaded_ld[["V"]][, c("glottocode", "h_tree_bits")],
      loaded_ld[["N"]][, c("glottocode", "h_tree_bits")],
      by = "glottocode", suffixes = c("_V", "_N")
    )
    st_vn <- safe_phylo_gls(m_vn, "h_tree_bits_V", "h_tree_bits_N", loaded_tr[["V"]])
    if (!is.null(st_vn)) {
      rows[[length(rows) + 1L]] <- cbind(
        data.frame(comparison = "H_tree_V_vs_H_tree_N", pos = "V_N",
                   x_var = "h_tree_bits_V", y_var = "h_tree_bits_N", stringsAsFactors = FALSE),
        st_vn
      )
    }
  }

  if (length(rows) > 0L) {
    corr_df <- do.call(rbind, rows)
    tmp_corr <- file.path(res_dir, sprintf(".cross_track_correlations_%d.tmp", Sys.getpid()))
    write.csv(corr_df, tmp_corr, row.names = FALSE)
    file.rename(tmp_corr, file.path(res_dir, "cross_track_correlations.csv"))
    return(corr_df)
  }
  NULL
}

# --- Main Execution ---------------------------------------------------------------------------

for (p_code in pos_list) {
  run_pos_pipeline(p_code)
}
compute_construct_correlations()
message("STAGE 2 ", if (dry) "DRY VALIDATION " else "", "COMPLETE")
