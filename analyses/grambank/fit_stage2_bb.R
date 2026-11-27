# Harmonized Stage-2 Beta-Binomial Runner for Study 3 Track 1 (Grambank 5 Domains)
#
# Implements the two-block specification grid across all 5 clean morphological domains:
#   - `v_breadth` (k = 9, eligible >= 6)
#   - `n_breadth` (k = 5, eligible >= 3)
#   - `v_irreg`   (k = 5 domain-conditioned, eligible >= 3)
#   - `n_irreg`   (k = 3 domain-conditioned, eligible >= 2)
#   - `np_agr`    (k = 7, eligible >= 5)
#
# Environment variables:
#   BB_DRY=1:              Validate Stage-1 & Stage-2 Stan code and data blocks; zero MCMC sampling
#   BB_SMOKE=1:            Fast 400-iter 2-chain smoke fit into baseline/fits/bb_smoke/ & results_smoke/
#   BB_CORES=4:            Dynamic CPU core budget across chains/workers (or --cores=N; default 4)
#   BB_THREADS=1:          Within-chain Stan threads (or --threads=N; default 1)
#   BB_OUTCOMES=...:       Comma-separated outcomes (default: v_breadth,n_breadth,v_irreg,n_irreg,np_agr)
#   BB_BERKSON_OUTCOMES=.: Comma-separated outcomes for Berkson 20-draw check (default: v_breadth)
#   BB_N_IMP=20:           Number of Stage-1 posterior draws for Berkson check (2 dry, 4 smoke, 20 full)
#   BB_OUT_DIR=...:        Output directory for CSV results (default: analyses/grambank/results)
#   BB_FITS_DIR=...:       Output directory for cached RDS fits (default: analyses/grambank/baseline/fits/bb)
#   BB_TAG=...:            Optional suffix appended to result filenames

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
})
options(brms.backend = "cmdstanr")

cmd_args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_all, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else getwd()
if (!dir.exists("analyses/grambank/data") &&
    dir.exists(file.path(script_dir, "..", "..", "analyses", "grambank", "data"))) {
  setwd(normalizePath(file.path(script_dir, "..", "..")))
}

common_dir <- normalizePath("analyses/common", mustWork = TRUE)
source(file.path(common_dir, "phylogeny.R"))
source(file.path(common_dir, "stage1_exposure.R"))

parse_cli_int <- function(flag_name, default_val = NA_integer_) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- grep(paste0("^--", flag_name, "="), args, value = TRUE)
  if (length(hit) > 0L) {
    val <- suppressWarnings(as.integer(sub(paste0("^--", flag_name, "="), "", hit[length(hit)])))
    if (is.finite(val) && val >= 1L) return(val)
  }
  default_val
}

dry <- nzchar(Sys.getenv("BB_DRY"))
smoke <- nzchar(Sys.getenv("BB_SMOKE"))

cores_env <- Sys.getenv("BB_CORES", "")
CORES <- max(1L, as.integer(if (nzchar(cores_env)) cores_env else parse_cli_int("cores", 4L)))
threads_env <- Sys.getenv("BB_THREADS", "")
THREADS <- max(1L, as.integer(if (nzchar(threads_env)) threads_env else parse_cli_int("threads", 1L)))
if (THREADS > CORES) THREADS <- CORES

if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- min(2L, max(1L, CORES %/% THREADS))
} else {
  ITER <- 3000L; WARMUP <- 1500L; CHAINS <- min(4L, max(1L, CORES %/% THREADS))
}
PAR_FITS <- max(1L, CORES %/% (CHAINS * THREADS))
stopifnot(PAR_FITS * CHAINS * THREADS <= CORES)

SEED <- 20261126L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
ROPE_HALF_WIDTH <- 0.15

N_IMP_DEFAULT <- if (dry) 2L else if (smoke) 4L else 20L
N_IMP <- max(2L, as.integer(Sys.getenv("BB_N_IMP", as.character(N_IMP_DEFAULT))))

data_dir <- Sys.getenv("BB_DATA_DIR", "analyses/grambank/data")
base_out_dir <- Sys.getenv("BB_OUT_DIR", "analyses/grambank/results")
base_fits_dir <- Sys.getenv("BB_FITS_DIR", "analyses/grambank/baseline/fits/bb")
fits_dir <- if (dry) tempdir() else if (smoke) paste0(base_fits_dir, "_smoke") else base_fits_dir
res_dir <- if (dry) tempdir() else if (smoke) paste0(base_out_dir, "_smoke") else base_out_dir
stan_cache_dir <- file.path("analyses/grambank/baseline/fits", "stan_cache_bb")
tag_raw <- Sys.getenv("BB_TAG", "")
tag <- if (nzchar(tag_raw)) paste0("_", tag_raw) else ""

dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(stan_cache_dir, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = stan_cache_dir)
}

OUTCOME_SPECS <- list(
  v_breadth = list(pres = "v_breadth_present", coded = "v_breadth_coded", min_coded = 6L, k_items = 9L,
                   raw_pres = NULL, raw_coded = NULL, raw_min = NULL),
  n_breadth = list(pres = "n_breadth_present", coded = "n_breadth_coded", min_coded = 3L, k_items = 5L,
                   raw_pres = NULL, raw_coded = NULL, raw_min = NULL),
  v_irreg   = list(pres = "v_irreg_present",   coded = "v_irreg_coded",   min_coded = 3L, k_items = 5L,
                   raw_pres = "v_irreg_raw_present", raw_coded = "v_irreg_raw_coded", raw_min = 3L),
  n_irreg   = list(pres = "n_irreg_present",   coded = "n_irreg_coded",   min_coded = 2L, k_items = 3L,
                   raw_pres = "n_irreg_raw_present", raw_coded = "n_irreg_raw_coded", raw_min = 2L),
  np_agr    = list(pres = "np_agr_present",    coded = "np_agr_coded",    min_coded = 5L, k_items = 7L,
                   raw_pres = NULL, raw_coded = NULL, raw_min = NULL)
)

outcomes_active <- trimws(strsplit(
  Sys.getenv("BB_OUTCOMES", paste(names(OUTCOME_SPECS), collapse = ",")),
  ","
)[[1]])
stopifnot(all(outcomes_active %in% names(OUTCOME_SPECS)))

berkson_env <- Sys.getenv("BB_BERKSON_OUTCOMES", "v_breadth")
berkson_outcomes <- if (nzchar(berkson_env)) {
  intersect(trimws(strsplit(berkson_env, ",")[[1]]), outcomes_active)
} else {
  character(0)
}

INNER_ASIA_FAMILIES <- c(
  "Turkic", "Mongolic-Khitan", "Tungusic",
  "turk1311", "mong1329", "tung1282"
)

# --- Helpers ----------------------------------------------------------------------------------

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
  bw_val <- tryCatch(stats::bw.SJ(draws_vec), error = function(e) stats::bw.nrd0(draws_vec))
  dens <- stats::density(
    draws_vec, bw = bw_val,
    from = min(draws_vec, 0.0) - 3.0 * s,
    to = max(draws_vec, 0.0) + 3.0 * s,
    n = 2048L
  )
  as.numeric(stats::approx(dens$x, dens$y, xout = 0.0)$y)
}

summarise_slope_metrics <- function(b_draws, pfx = "b", b_rhat = NA_real_, b_ess = NA_real_) {
  mu <- mean(b_draws)
  s <- sd(b_draws)
  d0_norm <- dnorm(0.0, mean = mu, sd = s)
  d0_kde <- kde_density_at_zero(b_draws)
  vals <- list(
    mean = mu,
    sd = s,
    q025 = unname(quantile(b_draws, 0.025)),
    q05  = unname(quantile(b_draws, 0.05)),
    q10  = unname(quantile(b_draws, 0.10)),
    q90  = unname(quantile(b_draws, 0.90)),
    q95  = unname(quantile(b_draws, 0.95)),
    q975 = unname(quantile(b_draws, 0.975)),
    rhat = b_rhat,
    ess_bulk = b_ess,
    p_pos = mean(b_draws > 0.0),
    p_neg = mean(b_draws < 0.0),
    bf01_norm = d0_norm / PRIOR_DENS_ZERO,
    bf01_kde  = if (is.finite(d0_kde)) d0_kde / PRIOR_DENS_ZERO else NA_real_,
    p_rope_015 = mean(abs(b_draws) <= ROPE_HALF_WIDTH)
  )
  setNames(vals, paste0(pfx, "_", names(vals)))
}

prepare_bouckaert_for_sample <- function(glottocodes) {
  tr <- tryCatch(load_bouckaert_tree(apply_split_map = FALSE), error = function(e) NULL)
  if (is.null(tr)) return(NULL)
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

# Identical regularized spatial GP prior across Track 1 & Track 2
# Note: brms assigns a coefficient-specific default prior on class="lscale", coef="gpsxsysz",
# so coef="gpsxsysz" MUST be specified explicitly or brms ignores the class-level prior.
make_bb_prior <- function(has_b = TRUE, has_sd = TRUE, has_gp = TRUE, has_me = FALSE) {
  pr <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(gamma(2, 0.1), class = phi)
  )
  if (isTRUE(has_b))  pr <- c(pr, prior(normal(0, 0.5), class = b))
  if (isTRUE(has_sd)) pr <- c(pr, prior(exponential(1), class = sd))
  if (isTRUE(has_gp)) {
    pr <- c(
      pr,
      prior(exponential(1), class = sdgp),
      prior(inv_gamma(3, 0.4), class = lscale, coef = gpsxsysz)
    )
  }
  if (isTRUE(has_me)) {
    pr <- c(
      pr,
      prior(normal(0, 1.5), class = meanme),
      prior(exponential(1), class = sdme)
    )
  }
  pr
}

fit_bb_brm <- function(name, formula_obj, data_df, prior_obj, data2_list = NULL, adapt = 0.995) {
  if (dry) {
    make_stancode(
      formula_obj, data = data_df, data2 = data2_list,
      family = beta_binomial(), prior = prior_obj, threads = threading(THREADS)
    )
    make_standata(
      formula_obj, data = data_df, data2 = data2_list,
      family = beta_binomial(), prior = prior_obj
    )
    return(structure(list(), class = "dry"))
  }
  # Default "never" avoids a brms 2.23.0 gp() bug where standata(fit) adds Xgp_1_old and triggers false on_change refits
  refit_mode <- Sys.getenv("BB_FILE_REFIT", "never")
  brm(
    formula_obj, data = data_df, data2 = data2_list,
    family = beta_binomial(), prior = prior_obj,
    chains = CHAINS, iter = ITER, warmup = WARMUP,
    cores = CHAINS, threads = threading(THREADS),
    seed = SEED, control = list(adapt_delta = adapt, max_treedepth = 12),
    file = file.path(fits_dir, name), file_refit = refit_mode,
    refresh = 0, silent = 2
  )
}

extract_stage1_draws_matrix <- function(fit_rds_path, langs, n_imp, seed_imp = 20261126L) {
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

# --- Load Canonical Data ----------------------------------------------------------------------

d0 <- read.csv(file.path(data_dir, "language_data.csv"), stringsAsFactors = FALSE)
ind0 <- read.csv(file.path(data_dir, "individuals.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(data_dir, "populations.csv"), stringsAsFactors = FALSE)
pop0 <- augment_population_f3_columns(
  pop0,
  k5_file = file.path(common_dir, "data/f3_scores_k5.csv"),
  f3_fixed_file = file.path(common_dir, "data/f3_fixed_scores.csv"),
  fst_regimes_file = file.path(common_dir, "data/fst_neighbor_regimes.csv"),
  discrete_k_file = file.path(common_dir, "data/discrete_k_diagnostics.csv")
)
stopifnot(!any(duplicated(d0$glottocode)))

to_rad <- pi / 180.0
d0$sx <- cos(d0$lat * to_rad) * cos(d0$lon * to_rad)
d0$sy <- cos(d0$lat * to_rad) * sin(d0$lon * to_rad)
d0$sz <- sin(d0$lat * to_rad)
d0$lang <- d0$glottocode
d0$fam_eff <- ifelse(
  is.na(d0$family) | !nzchar(d0$family) | d0$family == "isolate",
  paste0("isolate__", d0$glottocode),
  as.character(d0$family)
)

raw_tree <- read.tree(file.path(data_dir, "phylo_tree.nwk"))
A_grafen_all <- build_grafen_corr(raw_tree, d0$glottocode)
bouckaert_tr <- prepare_bouckaert_for_sample(d0$glottocode)

# Compute language-level median_fst_all and Barbieri Regime A/B flags for (set_name, min_n_ind)
lang_pop_flags <- function(set_name, min_n_ind) {
  ps <- pop0[
    pop0[[paste0("in_", set_name)]] == 1L &
      !is.na(pop0$n_individuals) &
      pop0$n_individuals >= as.integer(min_n_ind),
    , drop = FALSE
  ]
  gcs <- sort(unique(ps$glottocode))
  rows <- lapply(gcs, function(g) {
    sub <- ps[ps$glottocode == g, , drop = FALSE]
    mf <- sub$median_fst_all[is.finite(sub$median_fst_all)]
    ra <- sub$regime_a_enclave[!is.na(sub$regime_a_enclave)]
    rb <- sub$regime_b_shift[!is.na(sub$regime_b_shift)]
    data.frame(
      glottocode = g,
      median_fst_all = if (length(mf)) mean(mf) else NA_real_,
      regime_a = if (length(ra)) as.integer(any(ra == 1L)) else 0L,
      regime_b = if (length(rb)) as.integer(any(rb == 1L)) else 0L,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# Stage-1 exposure cache keyed by "<set>__n<min_n>__<measure>"
s1_cache <- new.env(parent = emptyenv())
s1_diag_list <- list()

get_stage1 <- function(set_name, min_n_ind, measure) {
  key <- sprintf("%s__n%d__%s", set_name, as.integer(min_n_ind), measure)
  if (exists(key, envir = s1_cache, inherits = FALSE)) {
    return(get(key, envir = s1_cache, inherits = FALSE))
  }
  pop_set <- pop0[pop0[[paste0("in_", set_name)]] == 1L, , drop = FALSE]
  res <- fit_stage1_exposure(
    measure = measure,
    ind_df = ind0,
    pop_df = pop_set,
    set_name = set_name,
    fits_dir = fits_dir,
    fit_prefix = "bb_",
    min_n_ind = as.integer(min_n_ind),
    dry = dry,
    chains = CHAINS,
    iter = ITER,
    warmup = WARMUP,
    threads = THREADS,
    seed = SEED
  )
  res$e$min_n_ind <- as.integer(min_n_ind)
  assign(key, res, envir = s1_cache)
  s1_diag_list[[length(s1_diag_list) + 1L]] <<- cbind(
    data.frame(min_n_ind = as.integer(min_n_ind), stringsAsFactors = FALSE),
    res$diag
  )
  res
}

# Standardize predictor, covariates, and optional Mundlak decomposition strictly on the active subset
prepare_active_df <- function(
  d_base,
  pres_col,
  coded_col,
  min_coded,
  set_name,
  min_n_ind,
  measure,
  require_demog = TRUE,
  require_median_fst = FALSE,
  drop_inner_asia = FALSE,
  drop_regime_a = FALSE,
  drop_regime_b = FALSE,
  restrict_bouckaert = FALSE
) {
  sub <- d_base[
    d_base[[paste0("in_", set_name)]] == 1L &
      !is.na(d_base[[coded_col]]) &
      d_base[[coded_col]] >= as.integer(min_coded),
    , drop = FALSE
  ]
  if (isTRUE(require_demog)) {
    sub <- sub[
      !is.na(sub$has_area) & sub$has_area == 1L &
        is.finite(as.numeric(sub$log_pop)) &
        is.finite(as.numeric(sub$log_area)),
      , drop = FALSE
    ]
  }
  if (isTRUE(drop_inner_asia)) {
    sub <- sub[!(sub$family %in% INNER_ASIA_FAMILIES), , drop = FALSE]
  }
  if (isTRUE(restrict_bouckaert)) {
    if (is.null(bouckaert_tr)) return(NULL)
    sub <- sub[sub$glottocode %in% bouckaert_tr$tip.label, , drop = FALSE]
  }

  s1 <- get_stage1(set_name, min_n_ind, measure)
  e_df <- s1$e
  sub <- merge(
    sub,
    e_df[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
    by = "glottocode"
  )
  flags_df <- lang_pop_flags(set_name, min_n_ind)
  sub <- merge(sub, flags_df, by = "glottocode", all.x = TRUE)
  sub$regime_a[is.na(sub$regime_a)] <- 0L
  sub$regime_b[is.na(sub$regime_b)] <- 0L

  if (isTRUE(drop_regime_a)) sub <- sub[sub$regime_a == 0L, , drop = FALSE]
  if (isTRUE(drop_regime_b)) sub <- sub[sub$regime_b == 0L, , drop = FALSE]
  if (isTRUE(require_median_fst)) sub <- sub[is.finite(sub$median_fst_all), , drop = FALSE]

  sub <- sub[order(sub$glottocode), , drop = FALSE]
  if (nrow(sub) < 10L) return(NULL)

  sub$n_pres <- as.integer(sub[[pres_col]])
  sub$n_cod <- as.integer(sub[[coded_col]])

  mu_x <- mean(sub$eta_mean)
  sd_x <- sd(sub$eta_mean)
  if (!is.finite(sd_x) || sd_x < 1e-8) return(NULL)
  sub$x <- (sub$eta_mean - mu_x) / sd_x
  sub$x_sd <- pmax(sub$eta_sd / sd_x, 1e-4)

  if (isTRUE(require_demog)) {
    lp <- as.numeric(sub$log_pop)
    la <- as.numeric(sub$log_area)
    sub$z_pop <- (lp - mean(lp)) / sd(lp)
    sub$z_area <- (la - mean(la)) / sd(la)
  } else {
    sub$z_pop <- NA_real_
    sub$z_area <- NA_real_
  }

  pdoc <- as.numeric(sub$prop_coded_nonmorph)
  sub$z_doc <- if (sd(pdoc) > 1e-8) (pdoc - mean(pdoc)) / sd(pdoc) else 0.0

  if (isTRUE(require_median_fst)) {
    mf <- as.numeric(sub$median_fst_all)
    sub$z_median_fst_all <- (mf - mean(mf)) / sd(mf)
  }

  # Mundlak within- vs. between-family decomposition (x_within = 0 for singleton families)
  fam_sz <- as.integer(ave(seq_len(nrow(sub)), sub$fam_eff, FUN = length))
  fam_mean <- as.numeric(ave(sub$x, sub$fam_eff, FUN = mean))
  sub$x_within <- ifelse(fam_sz >= 2L, sub$x - fam_mean, 0.0)
  sub$x_family_mean <- fam_mean

  attr(sub, "scale_info") <- list(
    x_center = mu_x, x_scale = sd_x,
    s1_y_mean = sub$y_mean[1], s1_y_sd = sub$y_sd[1]
  )
  sub
}

# --- Spec Runner ------------------------------------------------------------------------------

run_single_bb_spec <- function(sp) {
  t0 <- Sys.time()
  f_obj <- bf(as.formula(sp$formula))
  pr_obj <- make_bb_prior(
    has_b = sp$has_b, has_sd = sp$has_sd, has_gp = sp$has_gp,
    has_me = identical(sp$xmode, "me")
  )
  fit <- tryCatch(
    fit_bb_brm(sp$name, f_obj, sp$data, pr_obj, data2_list = sp$data2, adapt = sp$adapt),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(name = sp$name, error = conditionMessage(fit)))
  }

  base_row <- list(
    name = sp$name,
    outcome = sp$outcome,
    block = sp$block,
    spec = sp$spec,
    measure = sp$measure,
    set = sp$set,
    min_n_ind = sp$min_n_ind,
    xmode = sp$xmode,
    n_lang = nrow(sp$data)
  )
  if (dry) return(base_row)

  fit <- tryCatch(add_criterion(fit, "loo", file = file.path(fits_dir, sp$name)), error = function(e) fit)
  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)
  loo_est <- if (!is.null(fit$criteria$loo)) fit$criteria$loo$estimates["elpd_loo", "Estimate"] else NA_real_
  pk_bad <- if (!is.null(fit$criteria$loo)) sum(fit$criteria$loo$diagnostics$pareto_k > 0.7) else NA_integer_

  out <- c(base_row, list(
    max_rhat = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
    phi_mean = mean(dr$phi),
    elpd_loo = loo_est,
    pareto_k_gt_07 = pk_bad,
    minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))

  if (isTRUE(sp$is_mundlak)) {
    bw <- dr$b_x_within
    bb <- dr$b_x_family_mean
    b_diff <- bw - bb
    idx_w <- match("b_x_within", sm$variable)
    idx_b <- match("b_x_family_mean", sm$variable)
    out <- c(
      out,
      summarise_slope_metrics(bw, pfx = "b", b_rhat = sm$rhat[idx_w], b_ess = sm$ess_bulk[idx_w]),
      summarise_slope_metrics(bb, pfx = "b_between", b_rhat = sm$rhat[idx_b], b_ess = sm$ess_bulk[idx_b]),
      list(
        b_wb_diff_mean = mean(b_diff),
        b_wb_diff_q025 = unname(quantile(b_diff, 0.025)),
        b_wb_diff_q975 = unname(quantile(b_diff, 0.975)),
        p_wb_diff_pos = mean(b_diff > 0.0)
      )
    )
  } else {
    b_col <- if (identical(sp$xmode, "me")) grep("^bsp_me", names(dr), value = TRUE)[1] else "b_x"
    if (!is.na(b_col) && b_col %in% names(dr)) {
      idx_b <- match(b_col, sm$variable)
      out <- c(
        out,
        summarise_slope_metrics(
          dr[[b_col]], pfx = "b",
          b_rhat = if (!is.na(idx_b)) sm$rhat[idx_b] else NA_real_,
          b_ess  = if (!is.na(idx_b)) sm$ess_bulk[idx_b] else NA_real_
        )
      )
    }
  }
  out
}

run_bb_psock_batch <- function(jobs, worker_fn) {
  if (length(jobs) == 0L) return(list())
  if (dry || PAR_FITS <= 1L || length(jobs) <= 1L) {
    return(lapply(jobs, worker_fn))
  }
  n_w <- min(length(jobs), PAR_FITS)
  cl <- makePSOCKcluster(n_w, outfile = "")
  tryCatch({
    export_syms <- c(
      "dry", "smoke", "CHAINS", "ITER", "WARMUP", "THREADS", "SEED",
      "PRIOR_SD_B", "PRIOR_DENS_ZERO", "ROPE_HALF_WIDTH",
      "fits_dir", "stan_cache_dir",
      "make_bb_prior", "fit_bb_brm",
      "kde_density_at_zero", "summarise_slope_metrics",
      "run_single_bb_spec", "run_berkson_subfit"
    )
    clusterExport(cl, export_syms, envir = .GlobalEnv)
    clusterEvalQ(cl, {
      Sys.setenv(
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
        OMP_NUM_THREADS = "1",
        MKL_NUM_THREADS = "1",
        OPENBLAS_NUM_THREADS = "1",
        VECLIB_MAXIMUM_THREADS = "1"
      )
      suppressPackageStartupMessages({
        library(brms); library(loo); library(posterior)
      })
      w_cache <- file.path(stan_cache_dir, paste0("worker_", Sys.getpid()))
      dir.create(w_cache, showWarnings = FALSE, recursive = TRUE)
      options(brms.backend = "cmdstanr", cmdstanr_write_stan_file_dir = w_cache)
      NULL
    })
    parLapplyLB(cl, jobs, worker_fn)
  }, finally = {
    try(stopCluster(cl), silent = TRUE)
  })
}

run_berkson_subfit <- function(job) {
  f_obj <- bf(as.formula(job$form_str))
  pr_obj <- make_bb_prior(has_b = TRUE, has_sd = TRUE, has_gp = TRUE)
  fit_i <- tryCatch(
    fit_bb_brm(job$nm_i, f_obj, job$dd_i, pr_obj, data2_list = job$d2, adapt = 0.995),
    error = function(e) e
  )
  if (inherits(fit_i, "error")) {
    return(list(name = job$nm_i, error = conditionMessage(fit_i)))
  }
  if (dry) return(list(ok = TRUE))
  dr_i <- as_draws_df(fit_i)
  sm_i <- summarise_draws(dr_i)
  idx_bx <- match("b_x", sm_i$variable)
  list(
    b_x = dr_i$b_x,
    max_rhat = max(sm_i$rhat, na.rm = TRUE),
    b_rhat = if (!is.na(idx_bx)) sm_i$rhat[idx_bx] else NA_real_,
    b_ess = if (!is.na(idx_bx)) sm_i$ess_bulk[idx_bx] else NA_real_,
    divergences = sum(subset(nuts_params(fit_i), Parameter == "divergent__")$Value)
  )
}

run_berkson_20draw <- function(outcome_name, dd_base) {
  berkson_tag <- sprintf("berkson_%ddraw", N_IMP)
  spec_name <- paste0("bb__", outcome_name, "__", berkson_tag, "__f3_fixed")
  langs <- dd_base$glottocode
  s1_rds <- file.path(fits_dir, "exposure_bb_main_n3__f3_fixed.rds")
  eta_mat <- extract_stage1_draws_matrix(s1_rds, langs, N_IMP)
  eta_sub <- eta_mat[, langs, drop = FALSE]
  n_d <- nrow(eta_sub)

  form_str <- "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)"
  d2 <- list(A = A_grafen_all[langs, langs, drop = FALSE])

  t0 <- Sys.time()
  sub_jobs <- list()
  for (d_idx in seq_len(n_d)) {
    dd_i <- dd_base
    e_vec <- as.numeric(eta_sub[d_idx, ])
    s_e <- sd(e_vec)
    if (!is.finite(s_e) || s_e < 1e-8) next
    dd_i$x <- (e_vec - mean(e_vec)) / s_e
    sub_jobs[[length(sub_jobs) + 1L]] <- list(
      nm_i = sprintf("%s__imp%02d", spec_name, d_idx),
      form_str = form_str,
      dd_i = dd_i,
      d2 = d2
    )
  }

  sub_res <- run_bb_psock_batch(sub_jobs, run_berkson_subfit)
  sub_errs <- Filter(function(r) !is.list(r) || !is.null(r$error), sub_res)
  if (length(sub_errs)) {
    return(list(name = spec_name, error = sub_errs[[1]]$error))
  }

  base_row <- list(
    name = spec_name,
    outcome = outcome_name,
    block = "block2_genetics",
    spec = sprintf("berkson_%ddraw", n_d),
    measure = "f3_fixed",
    set = "main",
    min_n_ind = 3L,
    xmode = sprintf("berkson_%ddraw", n_d),
    n_lang = nrow(dd_base)
  )
  if (dry) return(base_row)

  pooled_b <- unlist(lapply(sub_res, `[[`, "b_x"), use.names = FALSE)
  max_rh <- max(vapply(sub_res, `[[`, numeric(1), "max_rhat"), na.rm = TRUE)
  b_rhs <- vapply(sub_res, `[[`, numeric(1), "b_rhat")
  b_ess <- vapply(sub_res, `[[`, numeric(1), "b_ess")
  tot_div <- sum(vapply(sub_res, `[[`, numeric(1), "divergences"), na.rm = TRUE)

  c(
    base_row,
    list(
      max_rhat = max_rh,
      min_ess_bulk = NA_real_,
      divergences = as.integer(tot_div),
      phi_mean = NA_real_,
      elpd_loo = NA_real_,
      pareto_k_gt_07 = NA_integer_,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ),
    summarise_slope_metrics(
      pooled_b, pfx = "b",
      b_rhat = if (any(is.finite(b_rhs))) max(b_rhs, na.rm = TRUE) else NA_real_,
      b_ess  = if (any(is.finite(b_ess))) min(b_ess, na.rm = TRUE) else NA_real_
    )
  )
}

# --- Build & Execute Two-Block Specification Grid --------------------------------------------

all_specs <- list()
berkson_jobs <- list()

add_spec <- function(outcome, block, spec, measure, set_name, min_n_ind, dd, formula_str,
                     has_sd = TRUE, has_gp = TRUE, custom_A = NULL, xmode = "plain", is_mundlak = FALSE) {
  if (is.null(dd)) return(invisible(NULL))
  A_use <- if (!is.null(custom_A)) {
    custom_A[dd$lang, dd$lang, drop = FALSE]
  } else {
    A_grafen_all[dd$lang, dd$lang, drop = FALSE]
  }
  all_specs[[length(all_specs) + 1L]] <<- list(
    name = sprintf("bb__%s__%s__%s_%s_n%d", outcome, spec, measure, set_name, as.integer(min_n_ind)),
    outcome = outcome,
    block = block,
    spec = spec,
    measure = measure,
    set = set_name,
    min_n_ind = as.integer(min_n_ind),
    xmode = xmode,
    is_mundlak = is_mundlak,
    data = dd,
    data2 = if (has_sd) list(A = A_use) else NULL,
    formula = formula_str,
    has_b = TRUE,
    has_sd = has_sd,
    has_gp = has_gp,
    adapt = if (has_gp) 0.995 else 0.99
  )
}

for (out_nm in outcomes_active) {
  os <- OUTCOME_SPECS[[out_nm]]

  # --- Block 1: Locked complete-case Maximal Model & LOO Ablations (f3_fixed, main, n_A >= 3) ---
  dd_b1 <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE
  )
  stopifnot(!is.null(dd_b1))

  add_spec(out_nm, "block1", "maximal", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)
  add_spec(out_nm, "block1", "loo_no_space", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A))",
           has_sd = TRUE, has_gp = FALSE)
  add_spec(out_nm, "block1", "loo_no_demog", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)
  add_spec(out_nm, "block1", "loo_no_doccov", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)
  add_spec(out_nm, "block1", "mundlak_wb", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x_within + x_family_mean + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE, is_mundlak = TRUE)

  # --- Block 2: Sample, Phylogeny & Masking Robustness Checks ---
  # 6. full_sample_PS (all eligible n_A >= 3 languages, no has_area filter)
  dd_full <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = FALSE
  )
  add_spec(out_nm, "block2_sample", "full_sample_PS", "f3_fixed", "main", 3L, dd_full,
           "n_pres | trials(n_cod) ~ 1 + x + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  # 7. cutoff_n5 (maximal at min_n_ind = 5L)
  dd_n5 <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 5L, measure = "f3_fixed",
    require_demog = TRUE
  )
  add_spec(out_nm, "block2_sample", "cutoff_n5", "f3_fixed", "main", 5L, dd_n5,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  # 8. strict_links (maximal on set = "strict", min_n_ind = 3L)
  dd_strict <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "strict", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE
  )
  add_spec(out_nm, "block2_sample", "strict_links", "f3_fixed", "strict", 3L, dd_strict,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  # 9. bouckaert_stem & bouckaert_crown (+ (1 | fam_eff))
  dd_bouck <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE, restrict_bouckaert = TRUE
  )
  if (!is.null(dd_bouck)) {
    A_bstem <- build_bouckaert_corr(
      dd_bouck$glottocode, families = dd_bouck$family,
      tree = bouckaert_tr, within_family = TRUE, crown_only = FALSE
    )
    A_bcrown <- build_bouckaert_corr(
      dd_bouck$glottocode, families = dd_bouck$family,
      tree = bouckaert_tr, within_family = TRUE, crown_only = TRUE
    )
    add_spec(out_nm, "block2_phylo", "bouckaert_stem", "f3_fixed", "main", 3L, dd_bouck,
             "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
             has_sd = TRUE, has_gp = TRUE, custom_A = A_bstem)
    add_spec(out_nm, "block2_phylo", "bouckaert_crown", "f3_fixed", "main", 3L, dd_bouck,
             "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + (1 | fam_eff) + gp(sx, sy, sz)",
             has_sd = TRUE, has_gp = TRUE, custom_A = A_bcrown)
  }

  # 10. no_inner_asia
  dd_noia <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE, drop_inner_asia = TRUE
  )
  add_spec(out_nm, "block2_sample", "no_inner_asia", "f3_fixed", "main", 3L, dd_noia,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  # 11. raw_unmasked_zeros (ONLY for v_irreg and n_irreg)
  if (!is.null(os$raw_pres)) {
    dd_raw <- prepare_active_df(
      d0, os$raw_pres, os$raw_coded, os$raw_min,
      set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
      require_demog = TRUE
    )
    add_spec(out_nm, "block2_masking", "raw_unmasked_zeros", "f3_fixed", "main", 3L, dd_raw,
             "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
             has_sd = TRUE, has_gp = TRUE)
  }

  # 12. Genetics SI checks (maximal formula)
  for (gm in c("f3_all_fixed", "f3_fixed_iso", "f3_fixed_maxp", "f4_est")) {
    dd_g <- prepare_active_df(
      d0, os$pres, os$coded, os$min_coded,
      set_name = "main", min_n_ind = 3L, measure = gm,
      require_demog = TRUE
    )
    add_spec(out_nm, "block2_genetics", gm, gm, "main", 3L, dd_g,
             "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
             has_sd = TRUE, has_gp = TRUE)
  }

  # Drift-controlled exposures (+ z_median_fst_all): f3_fixed_raw and neigh_flow_out
  for (gm in c("f3_fixed_raw", "neigh_flow_out")) {
    dd_g <- prepare_active_df(
      d0, os$pres, os$coded, os$min_coded,
      set_name = "main", min_n_ind = 3L, measure = gm,
      require_demog = TRUE, require_median_fst = TRUE
    )
    add_spec(out_nm, "block2_genetics", gm, gm, "main", 3L, dd_g,
             "n_pres | trials(n_cod) ~ 1 + x + z_median_fst_all + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
             has_sd = TRUE, has_gp = TRUE)
  }

  # Barbieri Regime A / B checks
  dd_dra <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE, drop_regime_a = TRUE
  )
  add_spec(out_nm, "block2_genetics", "drop_regime_a", "f3_fixed", "main", 3L, dd_dra,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  dd_drb <- prepare_active_df(
    d0, os$pres, os$coded, os$min_coded,
    set_name = "main", min_n_ind = 3L, measure = "f3_fixed",
    require_demog = TRUE, drop_regime_b = TRUE
  )
  add_spec(out_nm, "block2_genetics", "drop_regime_b", "f3_fixed", "main", 3L, dd_drb,
           "n_pres | trials(n_cod) ~ 1 + x + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  add_spec(out_nm, "block2_genetics", "plus_regime_b", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + x + regime_b + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE)

  # Single classical-ME double-correction comparison: me(x, x_sd)
  add_spec(out_nm, "block2_genetics", "me_baseline", "f3_fixed", "main", 3L, dd_b1,
           "n_pres | trials(n_cod) ~ 1 + me(x, x_sd) + z_pop + z_area + z_doc + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)",
           has_sd = TRUE, has_gp = TRUE, xmode = "me")

  # 20-draw Stage-1 Berkson variance propagation check (selected via BB_BERKSON_OUTCOMES, default: v_breadth)
  if (out_nm %in% berkson_outcomes) {
    berkson_jobs[[length(berkson_jobs) + 1L]] <- list(outcome = out_nm, dd_base = dd_b1)
  }
}

bind_bb_rows <- function(res_list) {
  all_cols <- unique(unlist(lapply(res_list, names)))
  do.call(rbind, lapply(res_list, function(r) {
    as.data.frame(
      lapply(setNames(all_cols, all_cols), function(k) if (length(r[[k]])) r[[k]] else NA),
      stringsAsFactors = FALSE
    )
  }))
}

write_bb_outputs <- function(res_list, s1_diag_df) {
  res_df <- bind_bb_rows(res_list)
  out_csv <- file.path(res_dir, paste0("stage2_bb_results", tag, ".csv"))
  s1_csv <- file.path(res_dir, paste0("stage1_bb_diagnostics", tag, ".csv"))
  expo_csv <- file.path(res_dir, paste0("exposure_bb_all", tag, ".csv"))
  write.csv(res_df, out_csv, row.names = FALSE)
  write.csv(s1_diag_df, s1_csv, row.names = FALSE)
  s1_keys <- sort(ls(envir = s1_cache))
  if (length(s1_keys) > 0L) {
    expo_df <- do.call(rbind, lapply(s1_keys, function(k) get(k, envir = s1_cache, inherits = FALSE)$e))
    write.csv(expo_df, expo_csv, row.names = FALSE)
  }
  res_df
}

is_core_spec <- function(sp) {
  identical(sp$block, "block1") ||
    identical(sp$spec, "cutoff_n5") ||
    identical(sp$spec, "raw_unmasked_zeros")
}

core_idx <- which(vapply(all_specs, is_core_spec, logical(1)))
si_idx <- setdiff(seq_along(all_specs), core_idx)

s1_diag_df <- do.call(rbind, s1_diag_list)

message(sprintf(
  "Track 1 Beta-Binomial (%s, cores=%d, chains=%d, threads=%d, par_fits=%d): %d Stage-1 fits cached, %d core + %d sensitivity Stage-2 specs + %d Berkson %d-draw jobs queued across %d outcomes.",
  if (dry) "DRY" else if (smoke) "SMOKE" else "FULL",
  CORES, CHAINS, THREADS, PAR_FITS,
  nrow(s1_diag_df), length(core_idx), length(si_idx), length(berkson_jobs), N_IMP, length(outcomes_active)
))

res_core <- run_bb_psock_batch(all_specs[core_idx], run_single_bb_spec)
errs_core <- Filter(function(r) !is.list(r) || !is.null(r$error), res_core)
if (length(errs_core)) {
  for (e in errs_core) message("ERROR (core): ", e$name, " -> ", e$error)
  stop(sprintf("%d core Beta-Binomial specification(s) failed.", length(errs_core)))
}
if (!dry && length(res_core) > 0L) {
  core_df <- write_bb_outputs(res_core, s1_diag_df)
  message("Wrote ", nrow(core_df), " core Beta-Binomial rows to ", res_dir)
}

res_si <- run_bb_psock_batch(all_specs[si_idx], run_single_bb_spec)
errs_si <- Filter(function(r) !is.list(r) || !is.null(r$error), res_si)
if (length(errs_si)) {
  for (e in errs_si) message("ERROR (sensitivity): ", e$name, " -> ", e$error)
  stop(sprintf("%d sensitivity Beta-Binomial specification(s) failed.", length(errs_si)))
}

res_main <- vector("list", length(all_specs))
res_main[core_idx] <- res_core
res_main[si_idx] <- res_si

res_berk <- lapply(berkson_jobs, function(j) run_berkson_20draw(j$outcome, j$dd_base))
errs_b <- Filter(function(r) !is.list(r) || !is.null(r$error), res_berk)
if (length(errs_b)) {
  for (e in errs_b) message("ERROR (Berkson): ", e$name, " -> ", e$error)
  stop(sprintf("%d Berkson Beta-Binomial job(s) failed.", length(errs_b)))
}

res_all <- c(res_main, res_berk)

if (dry) {
  res_df <- bind_bb_rows(res_all)
  message(sprintf(
    "BB_DRY=1 PASSED: %d Stage-1 models + %d Stage-2 models validated (0 errors).",
    nrow(s1_diag_df), nrow(res_df)
  ))
  print(res_df[, c("outcome", "block", "spec", "measure", "set", "min_n_ind", "xmode", "n_lang")], row.names = FALSE)
} else {
  res_df <- write_bb_outputs(res_all, s1_diag_df)
  message("Wrote ", nrow(res_df), " Beta-Binomial results to: ", file.path(res_dir, paste0("stage2_bb_results", tag, ".csv")))
}
