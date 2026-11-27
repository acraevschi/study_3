# Stage-2 Bayesian Berkson, Equivalence (BF_01 / ROPE), and Cross-Track Validation
# for Study 3 Track 2: Morphological Paradigm Predictability (`V` and `N` separate).
#
# Harmonized with Track 1 (`analyses/grambank/fit_stage2_bb.R` & `analyses/common/stage1_exposure.R`):
#   - Strictly separate 1-row-per-`glottocode` Verb (`V`) and Noun (`N`) pipelines.
#   - Primary exposure: corrected cross-family `f3_fixed` (`min_n_ind = 3L` primary, `5L` check),
#     pooled via outcome-blind Stage-1 hierarchical shrinkage in Berkson `plain` mode
#     (`x = (eta_mean - mu) / sd` on the exact active Stage-2 language set), plus 1 `me(x, x_sd)`
#     benchmark and 1 20-draw Stage-1 posterior-imputation check.
#   - Precision-weighted Stage-2 Gaussian regressions `y | se(se_y, sigma = TRUE)`
#     (`se_y = se_raw / sd_raw`) with `Normal(0, 0.5)` slope priors and regularized spatial GP
#     priors `sdgp ~ exponential(1)`, `lscale ~ inv_gamma(3, 0.4)` on `gp(sx, sy, sz)`.
#   - Two-block specification grid:
#       Block 1: Maximal model & locked complete-case LOO ablations (identical N across all 7 specs)
#       Block 2: Sample, phylogeny, learner/regime, and genetics SI robustness checks
#   - Cross-track construct correlations against the 5 Track-1 Grambank domain proportions
#     (`v_breadth_prop`, `n_breadth_prop`, `v_irreg_prop`, `n_irreg_prop`, `np_agr_prop`).
#
# Environment variables:
#   PARA_DRY=1:          build Stan code & data for every Stage-1 and Stage-2 model; sample nothing
#   PARA_SMOKE=1:        fast 400-iteration 2-chain smoke fit
#   PARA_CORES=4:        dynamic CPU core budget across chains/workers (or --cores=N; default 4)
#   PARA_THREADS=1:      within-chain Stan threads (or --threads=N; default 1)
#   PARA_POS=V,N:        POS pipelines to run (default: V,N)
#   PARA_SETS=main:      link sets to run (default: main; strict runs as Block-2 `strict_links`)
#   PARA_MEASURES=...:   primary Stage-1 exposure(s) (default: f3_fixed)
#   PARA_MIN_N_IND=3:    primary population sample-size cutoff (default: 3)
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

parse_cli_int <- function(flag_name, default_val = NA_integer_) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- grep(paste0("^--", flag_name, "="), args, value = TRUE)
  if (length(hit) > 0L) {
    val <- suppressWarnings(as.integer(sub(paste0("^--", flag_name, "="), "", hit[length(hit)])))
    if (is.finite(val) && val >= 1L) return(val)
  }
  default_val
}

parse_cli_str <- function(flag_name, default_val = "") {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- grep(paste0("^--", flag_name, "="), args, value = TRUE)
  if (length(hit) > 0L) {
    val <- trimws(sub(paste0("^--", flag_name, "="), "", hit[length(hit)]))
    if (nzchar(val)) return(val)
  }
  idx <- which(args == paste0("--", flag_name))
  if (length(idx) > 0L && idx[length(idx)] < length(args)) {
    val <- trimws(args[idx[length(idx)] + 1L])
    if (nzchar(val)) return(val)
  }
  default_val
}

dry <- nzchar(Sys.getenv("PARA_DRY"))
smoke <- nzchar(Sys.getenv("PARA_SMOKE"))

cores_env <- Sys.getenv("PARA_CORES", "")
CORES <- max(1L, as.integer(if (nzchar(cores_env)) cores_env else parse_cli_int("cores", 4L)))
threads_env <- Sys.getenv("PARA_THREADS", "")
THREADS <- max(1L, as.integer(if (nzchar(threads_env)) threads_env else parse_cli_int("threads", 1L)))
if (THREADS > CORES) THREADS <- CORES

if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- min(2L, max(1L, CORES %/% THREADS))
} else {
  ITER <- 3000L; WARMUP <- 1500L; CHAINS <- min(4L, max(1L, CORES %/% THREADS))
}
PAR_FITS <- max(1L, CORES %/% (CHAINS * THREADS))
stopifnot(PAR_FITS * CHAINS * THREADS <= CORES)

SEED <- 20261112L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
TRACK1_BREADTH_REF <- 0.44
ROPE_HALF_WIDTH <- 0.15

PRIMARY_MIN_N <- max(1L, as.integer(Sys.getenv("PARA_MIN_N_IND", "3")))
SENS_MIN_N <- 5L

N_IMP_DEFAULT <- if (dry) 2L else if (smoke) 4L else 20L
N_IMP <- max(2L, as.integer(Sys.getenv("PARA_N_IMP", as.character(N_IMP_DEFAULT))))

data_dir <- Sys.getenv("PARA_DATA_DIR", "analyses/paradigm/results/stage2/prepared")
out_dir_env <- Sys.getenv("PARA_OUT_DIR", "")
base_out_dir <- if (nzchar(out_dir_env)) out_dir_env else "analyses/paradigm/results/stage2"
fits_dir <- if (dry) tempdir() else file.path(base_out_dir, if (smoke) "fits_smoke" else "fits")
res_dir <- if (dry) {
  if (nzchar(out_dir_env)) out_dir_env else tempdir()
} else {
  file.path(base_out_dir, if (smoke) "results_smoke" else ".")
}
ALL_SETS <- c("main", "no_doubtful_language", "no_other_language_stand_ins", "strict")
pos_env <- Sys.getenv("PARA_POS", "")
pos_raw <- if (nzchar(pos_env)) pos_env else parse_cli_str("pos", "V,N")
if (toupper(trimws(pos_raw)) == "BOTH") pos_raw <- "V,N"
pos_list <- trimws(strsplit(pos_raw, ",")[[1]])
sets_active <- trimws(strsplit(Sys.getenv("PARA_SETS", "main"), ",")[[1]])
measures_active <- trimws(strsplit(Sys.getenv("PARA_MEASURES", "f3_fixed"), ",")[[1]])
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
f3_fixed_file <- file.path(common_dir, "data/f3_fixed_scores.csv")
fst_regimes_file <- file.path(common_dir, "data/fst_neighbor_regimes.csv")

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
  bw_val <- tryCatch(stats::bw.SJ(draws_vec), error = function(e) stats::bw.nrd0(draws_vec))
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
    q10  = unname(quantile(b_draws, 0.10)),
    q90  = unname(quantile(b_draws, 0.90)),
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

make_stage2_prior <- function(has_b, has_sd, has_gp, has_me = FALSE) {
  pr <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(exponential(1), class = sigma)
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

fit_stage2_brm <- function(name, formula_obj, data_df, prior_obj, data2_list = NULL, adapt = 0.99) {
  if (dry) {
    make_stancode(formula_obj, data = data_df, data2 = data2_list, family = gaussian(),
                  prior = prior_obj, threads = threading(THREADS))
    make_standata(formula_obj, data = data_df, data2 = data2_list, family = gaussian(),
                  prior = prior_obj)
    return(structure(list(), class = "dry"))
  }
  # Default "never" avoids a brms 2.23.0 gp() bug where standata(fit) adds Xgp_1_old and triggers false on_change refits
  refit_mode <- Sys.getenv("PARA_FILE_REFIT", "never")
  adapt_override <- suppressWarnings(as.numeric(Sys.getenv("PARA_ADAPT_DELTA", "")))
  eff_adapt <- if (is.finite(adapt_override) && adapt_override > 0 && adapt_override < 1) adapt_override else adapt
  init_env <- Sys.getenv("PARA_INIT", "")
  brm_args <- list(
    formula = formula_obj, data = data_df, data2 = data2_list, family = gaussian(), prior = prior_obj,
    chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS, threads = threading(THREADS),
    seed = SEED, control = list(adapt_delta = eff_adapt, max_treedepth = 12),
    file = file.path(fits_dir, name), file_refit = refit_mode, refresh = 0, silent = 2
  )
  if (nzchar(init_env)) {
    init_num <- suppressWarnings(as.numeric(init_env))
    brm_args$init <- if (is.finite(init_num)) init_num else init_env
  }
  do.call(brm, brm_args)
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

attach_lang_pop_flags <- function(d_df, pop_df, min_n_ind = 3L) {
  drop_cols <- intersect(
    c("median_fst_all", "regime_a_enclave", "regime_b_shift", "barbieri_mismatch"),
    names(d_df)
  )
  out <- if (length(drop_cols)) d_df[, setdiff(names(d_df), drop_cols), drop = FALSE] else d_df
  pop_u <- pop_df[!is.na(pop_df$n_individuals) & pop_df$n_individuals >= min_n_ind, , drop = FALSE]

  gcs <- out$glottocode
  med_fst <- rep(NA_real_, length(gcs))
  reg_a <- rep(NA_integer_, length(gcs))
  reg_b <- rep(NA_integer_, length(gcs))
  barb_mm <- rep(NA_integer_, length(gcs))

  for (i in seq_along(gcs)) {
    p_i <- pop_u[pop_u$glottocode == gcs[i], , drop = FALSE]
    if (nrow(p_i) == 0L) next
    if ("median_fst_all" %in% names(p_i)) {
      vals <- as.numeric(p_i$median_fst_all)
      vals <- vals[is.finite(vals)]
      if (length(vals) > 0L) med_fst[i] <- mean(vals)
    }
    if ("regime_a_enclave" %in% names(p_i)) {
      va <- as.integer(p_i$regime_a_enclave)
      va <- va[!is.na(va)]
      if (length(va) > 0L) reg_a[i] <- if (any(va == 1L)) 1L else 0L
    }
    if ("regime_b_shift" %in% names(p_i)) {
      vb <- as.integer(p_i$regime_b_shift)
      vb <- vb[!is.na(vb)]
      if (length(vb) > 0L) reg_b[i] <- if (any(vb == 1L)) 1L else 0L
    }
    if ("barbieri_mismatch" %in% names(p_i)) {
      vm <- as.integer(p_i$barbieri_mismatch)
      vm <- vm[!is.na(vm)]
      if (length(vm) > 0L) barb_mm[i] <- if (any(vm == 1L)) 1L else 0L
    }
  }
  out$median_fst_all <- med_fst
  out$regime_a_enclave <- reg_a
  out$regime_b_shift <- reg_b
  out$barbieri_mismatch <- barb_mm
  out
}

build_subs_map <- function(d_set, pop_set, min_n_ind = 3L) {
  pop_u <- pop_set[!is.na(pop_set$n_individuals) & pop_set$n_individuals >= min_n_ind, , drop = FALSE]
  has_col <- function(col) {
    if (!(col %in% names(pop_u))) return(character(0))
    intersect(d_set$glottocode, unique(pop_u$glottocode[!is.na(pop_u[[col]])]))
  }
  has_col_se <- function(col, se_col) {
    if (!all(c(col, se_col) %in% names(pop_u))) return(character(0))
    ok <- !is.na(pop_u[[col]]) & !is.na(pop_u[[se_col]]) & is.finite(pop_u[[se_col]]) & pop_u[[se_col]] > 0
    intersect(d_set$glottocode, unique(pop_u$glottocode[ok]))
  }
  pop_5 <- pop_set[!is.na(pop_set$n_individuals) & pop_set$n_individuals >= 5L, , drop = FALSE]
  list(
    all          = intersect(d_set$glottocode, unique(pop_u$glottocode)),
    f3_fixed     = has_col("f3_fixed"),
    f3_fixed_iso = has_col("f3_fixed_iso"),
    f4           = has_col_se("f4_est", "f4_se"),
    neigh_out    = has_col("neigh_flow_out"),
    rmsd5        = if ("rmsd_K12" %in% names(pop_5)) intersect(d_set$glottocode, unique(pop_5$glottocode[!is.na(pop_5$rmsd_K12)])) else character(0),
    neigh        = has_col("neigh_flow"),
    het          = has_col_se("het", "het_se"),
    f3           = has_col("f3_z"),
    f3b3         = has_col("f3b3_z")
  )
}

standardize_outcome_and_covs <- function(
  df,
  y_col,
  se_col,
  require_demog = FALSE,
  use_demog = FALSE,
  use_cdist = TRUE,
  use_lexprov = TRUE,
  use_med_fst = FALSE,
  use_regime_b = FALSE,
  min_n_lang = 6L
) {
  sub <- df[!is.na(df[[y_col]]) & is.finite(df[[y_col]]), , drop = FALSE]
  if (isTRUE(require_demog) || isTRUE(use_demog)) {
    sub <- sub[!is.na(sub$has_area) & sub$has_area == 1L &
               is.finite(sub$log_pop) & is.finite(sub$log_area), , drop = FALSE]
  }
  if (isTRUE(use_med_fst)) {
    if (!("median_fst_all" %in% names(sub))) return(NULL)
    sub <- sub[is.finite(sub$median_fst_all), , drop = FALSE]
  }
  if (isTRUE(use_regime_b)) {
    if (!("regime_b_shift" %in% names(sub))) return(NULL)
    sub <- sub[!is.na(sub$regime_b_shift), , drop = FALSE]
  }
  if (nrow(sub) < min_n_lang) return(NULL)

  y_raw <- as.numeric(sub[[y_col]])
  se_raw <- as.numeric(sub[[se_col]])
  se_raw[!is.finite(se_raw) | se_raw <= 0] <- median(se_raw[is.finite(se_raw) & se_raw > 0], na.rm = TRUE)
  if (!all(is.finite(se_raw))) se_raw <- rep(0.05, nrow(sub))

  mu_y <- mean(y_raw)
  sd_y <- sd(y_raw)
  if (is.na(sd_y) || sd_y < 1e-8) return(NULL)
  sub$y <- (y_raw - mu_y) / sd_y
  sub$se_y <- pmax(se_raw / sd_y, 1e-4)

  if (isTRUE(use_cdist)) {
    lc <- as.numeric(sub$log_C_distinct)
    sub$z_log_C_distinct <- if (sd(lc) > 1e-8) as.numeric(scale(lc)) else 0.0
  }
  if (isTRUE(use_lexprov)) {
    ln_av <- as.numeric(sub$log_navail)
    sub$z_log_navail <- if (sd(ln_av) > 1e-8) as.numeric(scale(ln_av)) else 0.0
    sub$source_prov <- as.character(sub$source_prov)
  }
  if (isTRUE(use_demog)) {
    lp <- as.numeric(sub$log_pop)
    la <- as.numeric(sub$log_area)
    sub$z_pop <- if (sd(lp) > 1e-8) as.numeric(scale(lp)) else 0.0
    sub$z_area <- if (sd(la) > 1e-8) as.numeric(scale(la)) else 0.0
  }
  if (isTRUE(use_med_fst)) {
    mf <- as.numeric(sub$median_fst_all)
    sub$z_median_fst_all <- if (sd(mf) > 1e-8) as.numeric(scale(mf)) else 0.0
  }
  if (isTRUE(use_regime_b)) {
    sub$regime_b <- as.numeric(sub$regime_b_shift == 1L)
  }

  sub$lang <- sub$glottocode
  sub$fam_eff <- ifelse(
    is.na(sub$family) | !nzchar(sub$family) | sub$family == "isolate",
    paste0("isolate__", sub$glottocode),
    as.character(sub$family)
  )
  attr(sub, "y_scale_info") <- list(y_center = mu_y, y_scale = sd_y)
  sub
}

build_cov_terms <- function(
  df,
  use_cdist = TRUE,
  use_lexprov = TRUE,
  use_demog = FALSE,
  use_med_fst = FALSE,
  use_regime_b = FALSE
) {
  terms <- character(0)
  if (isTRUE(use_cdist) && "z_log_C_distinct" %in% names(df) && sd(df$z_log_C_distinct) > 1e-8) {
    terms <- c(terms, "z_log_C_distinct")
  }
  if (isTRUE(use_lexprov)) {
    if ("z_log_navail" %in% names(df) && sd(df$z_log_navail) > 1e-8) {
      terms <- c(terms, "z_log_navail")
    }
    if ("source_prov" %in% names(df)) {
      prov_tab <- table(df$source_prov)
      if (length(prov_tab[prov_tab >= 1L]) >= 2L) {
        terms <- c(terms, "source_prov")
      }
    }
  }
  if (isTRUE(use_demog)) {
    if ("z_pop" %in% names(df) && sd(df$z_pop) > 1e-8) terms <- c(terms, "z_pop")
    if ("z_area" %in% names(df) && sd(df$z_area) > 1e-8) terms <- c(terms, "z_area")
  }
  if (isTRUE(use_med_fst) && "z_median_fst_all" %in% names(df) && sd(df$z_median_fst_all) > 1e-8) {
    terms <- c(terms, "z_median_fst_all")
  }
  if (isTRUE(use_regime_b) && "regime_b" %in% names(df) && sd(df$regime_b) > 1e-8) {
    terms <- c(terms, "regime_b")
  }
  terms
}

# --- Stage-2 Runner Per POS & Link Set --------------------------------------------------------

run_single_spec <- function(sp) {
  t0 <- Sys.time()
  f_obj <- bf(as.formula(sp$formula))
  pr_obj <- make_stage2_prior(sp$has_b, sp$has_sd, sp$has_gp, has_me = isTRUE(sp$has_me))
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
    block = sp$block,
    set = sp$set,
    min_n_ind = sp$min_n_ind,
    spec_group = sp$spec_group,
    outcome = sp$outcome,
    measure = sp$measure,
    ctrl = sp$ctrl,
    xmode = sp$xmode,
    n_lang = nrow(sp$data),
    formula = sp$formula
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
  } else if (isTRUE(sp$has_me) && "bsp_mexx_sd" %in% names(dr)) {
    idx_x <- match("bsp_mexx_sd", sm$variable)
    out <- c(
      out,
      summarise_slope_equivalence(
        dr$bsp_mexx_sd, pfx = "b",
        b_rhat_val = if (!is.na(idx_x)) sm$rhat[idx_x] else NA_real_,
        b_ess_val = if (!is.na(idx_x)) sm$ess_bulk[idx_x] else NA_real_
      )
    )
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

run_stage2_berkson_subfit <- function(job) {
  f_obj <- bf(as.formula(job$form_str))
  pr_obj <- make_stage2_prior(has_b = TRUE, has_sd = TRUE, has_gp = TRUE)
  fit_i <- tryCatch(
    fit_stage2_brm(job$nm_i, f_obj, job$dd_i, pr_obj, data2_list = job$d2, adapt = 0.995),
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

run_berkson_20draw_check <- function(
  pos_code, set_name, min_n_ind, measure_name, dd_base, eta_draws_mat, A_grafen
) {
  spec_name <- paste0("s2_", pos_code, "__", set_name, "__berkson_20draw__", measure_name, "_PS")
  langs <- dd_base$glottocode
  if (!all(langs %in% colnames(eta_draws_mat))) return(NULL)
  eta_sub <- eta_draws_mat[, langs, drop = FALSE]
  n_d <- nrow(eta_sub)

  cov_terms <- build_cov_terms(
    dd_base,
    use_cdist = "z_log_C_distinct" %in% names(dd_base),
    use_lexprov = "z_log_navail" %in% names(dd_base),
    use_demog = "z_pop" %in% names(dd_base)
  )
  rhs_fixed <- paste(c("1", "x", cov_terms), collapse = " + ")
  form_str <- paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_fixed,
                     " + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)")
  d2 <- list(A = A_grafen[langs, langs, drop = FALSE])

  t0 <- Sys.time()
  sub_jobs <- list()
  for (d_idx in seq_len(n_d)) {
    dd_i <- dd_base
    e_vec <- as.numeric(eta_sub[d_idx, ])
    s_e <- sd(e_vec)
    if (is.na(s_e) || s_e < 1e-8) next
    dd_i$x <- (e_vec - mean(e_vec)) / s_e
    sub_jobs[[length(sub_jobs) + 1L]] <- list(
      nm_i = sprintf("%s__imp%02d", spec_name, d_idx),
      form_str = form_str,
      dd_i = dd_i,
      d2 = d2
    )
  }

  if (dry || PAR_FITS <= 1L || length(sub_jobs) <= 1L) {
    sub_res <- lapply(sub_jobs, run_stage2_berkson_subfit)
  } else {
    cl <- makePSOCKcluster(min(length(sub_jobs), PAR_FITS), outfile = "")
    sub_res <- tryCatch({
      clusterExport(cl, ls(envir = .GlobalEnv), envir = .GlobalEnv)
      clusterEvalQ(cl, {
        Sys.setenv(
          OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
          OMP_NUM_THREADS = "1",
          MKL_NUM_THREADS = "1",
          OPENBLAS_NUM_THREADS = "1",
          VECLIB_MAXIMUM_THREADS = "1"
        )
        suppressPackageStartupMessages({ library(brms); library(ape); library(loo); library(posterior) })
        w_cache <- file.path(stan_cache_dir, paste0("worker_", Sys.getpid()))
        dir.create(w_cache, showWarnings = FALSE, recursive = TRUE)
        options(brms.backend = "cmdstanr", cmdstanr_write_stan_file_dir = w_cache)
        NULL
      })
      parLapplyLB(cl, sub_jobs, run_stage2_berkson_subfit)
    }, finally = {
      try(stopCluster(cl), silent = TRUE)
    })
  }

  sub_errs <- Filter(function(r) !is.list(r) || !is.null(r$error), sub_res)
  if (length(sub_errs)) {
    return(list(name = spec_name, error = sub_errs[[1]]$error))
  }

  base_info <- list(
    name = spec_name, pos = pos_code, block = "block2", set = set_name,
    min_n_ind = min_n_ind, spec_group = "berkson_20draw", outcome = "h_tree_bits",
    measure = measure_name, ctrl = "PS", xmode = sprintf("berkson_%ddraw", n_d),
    n_lang = nrow(dd_base), formula = form_str
  )
  if (dry) return(base_info)

  pooled_b <- unlist(lapply(sub_res, `[[`, "b_x"), use.names = FALSE)
  max_rh <- max(vapply(sub_res, `[[`, numeric(1), "max_rhat"), na.rm = TRUE)
  b_rhs <- vapply(sub_res, `[[`, numeric(1), "b_rhat")
  b_ess <- vapply(sub_res, `[[`, numeric(1), "b_ess")
  tot_div <- sum(vapply(sub_res, `[[`, numeric(1), "divergences"), na.rm = TRUE)

  c(
    base_info,
    list(
      max_rhat = max_rh,
      min_ess_bulk = NA_real_,
      divergences = as.integer(tot_div),
      sigma_mean = NA_real_,
      elpd_loo = NA_real_,
      pareto_k_gt_07 = NA_integer_,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ),
    summarise_slope_equivalence(
      pooled_b, pfx = "b",
      b_rhat_val = if (any(is.finite(b_rhs))) max(b_rhs, na.rm = TRUE) else NA_real_,
      b_ess_val = if (any(is.finite(b_ess))) min(b_ess, na.rm = TRUE) else NA_real_
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
  pop0 <- augment_population_f3_columns(
    pop0,
    k5_file = k5_file,
    f3_fixed_file = f3_fixed_file,
    fst_regimes_file = fst_regimes_file
  )
  if ("f3_z" %in% names(pop0)) {
    pop0$f3k_z[is.na(pop0$f3_z)] <- NA_real_
    pop0$f3k_est[is.na(pop0$f3_z)] <- NA_real_
    pop0$f3k_est_pos[is.na(pop0$f3_z)] <- NA_real_
  }
  stopifnot(!any(duplicated(d0$glottocode)))
  stopifnot(!("mean_token_freq" %in% names(d0)))

  # Bounded logit-reliability outcome + delta-method SE
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

  # Stage-1 exposure cache keyed by "<set>__n<min_n>__<measure>"
  s1_cache <- new.env(parent = emptyenv())
  s1_diag_list <- list()

  get_stage1 <- function(set_name, min_n_ind, measure_name) {
    key <- sprintf("%s__n%d__%s", set_name, as.integer(min_n_ind), measure_name)
    if (exists(key, envir = s1_cache, inherits = FALSE)) {
      return(get(key, envir = s1_cache, inherits = FALSE))
    }
    d_s <- d0[d0[[paste0("in_", set_name)]] == 1L, , drop = FALSE]
    pop_s <- pop0[pop0[[paste0("in_", set_name)]] == 1L & pop0$glottocode %in% d_s$glottocode, , drop = FALSE]
    res_s1 <- fit_stage1_exposure(
      measure = measure_name, ind_df = ind0, pop_df = pop_s,
      set_name = set_name, fits_dir = fits_dir,
      fit_prefix = paste0("para_", pos_code, "_"),
      min_n_ind = as.integer(min_n_ind),
      dry = dry, chains = CHAINS, iter = ITER, warmup = WARMUP,
      threads = THREADS, seed = SEED
    )
    res_s1$e$min_n_ind <- as.integer(min_n_ind)
    res_s1$diag$min_n_ind <- as.integer(min_n_ind)
    assign(key, res_s1$e, envir = s1_cache)
    s1_diag_list[[key]] <<- res_s1$diag
    res_s1$e
  }

  # Helper to construct a single Stage-2 spec (returns NULL if sample < min_n_lang)
  make_spec <- function(
    d_base_raw,
    pop_base_raw,
    block_label,
    set_name,
    min_n_ind,
    spec_group,
    out_col,
    se_col,
    measure_name,
    ctrl_mode = "PS",
    require_demog = TRUE,
    use_demog = TRUE,
    use_cdist = TRUE,
    use_lexprov = TRUE,
    use_med_fst = FALSE,
    use_regime_b = FALSE,
    xmode = "plain",
    custom_A = NULL,
    extra_re = "",
    min_n_lang = 6L,
    drop_space_if_small = FALSE
  ) {
    if (is.null(d_base_raw) || nrow(d_base_raw) < min_n_lang || !(out_col %in% names(d_base_raw))) {
      return(NULL)
    }
    d_flagged <- attach_lang_pop_flags(d_base_raw, pop_base_raw, min_n_ind = min_n_ind)
    subs <- build_subs_map(d_flagged, pop_base_raw, min_n_ind = min_n_ind)

    if (identical(xmode, "none")) {
      # Null baseline locked on the same eligibility subset as `measure_name`
      sb_code <- subset_of_measure(measure_name)
      req_gcs <- intersect(d_flagged$glottocode, subs[[sb_code]])
      dd <- standardize_outcome_and_covs(
        d_flagged[d_flagged$glottocode %in% req_gcs, , drop = FALSE],
        out_col, se_col,
        require_demog = require_demog, use_demog = use_demog,
        use_cdist = use_cdist, use_lexprov = use_lexprov,
        use_med_fst = use_med_fst, use_regime_b = use_regime_b,
        min_n_lang = min_n_lang
      )
      if (is.null(dd)) return(NULL)
      dd <- dd[order(dd$glottocode), , drop = FALSE]
      sc_info <- attr(dd, "y_scale_info")
      cov_t <- build_cov_terms(
        dd, use_cdist = use_cdist, use_lexprov = use_lexprov,
        use_demog = use_demog, use_med_fst = use_med_fst, use_regime_b = use_regime_b
      )
      rhs_fixed <- paste(c("1", cov_t), collapse = " + ")
      eff_ctrl <- if (isTRUE(drop_space_if_small) && nrow(dd) < 20L && ctrl_mode == "PS") "P" else ctrl_mode
      has_phylo <- eff_ctrl %in% c("P", "PS", "PS_bouckaert_stem", "PS_bouckaert_crown", "P_bouckaert_stem", "P_bouckaert_crown")
      has_space <- eff_ctrl %in% c("S", "PS", "PS_bouckaert_stem", "PS_bouckaert_crown")
      A_use <- if (!is.null(custom_A)) custom_A[dd$lang, dd$lang, drop = FALSE] else A_grafen_all[dd$lang, dd$lang, drop = FALSE]
      re_str <- paste0(
        if (has_phylo) " + (1 | gr(lang, cov = A))" else "",
        extra_re,
        if (has_space) " + gp(sx, sy, sz)" else ""
      )
      return(list(
        name = paste0("s2_", pos_code, "__", set_name, "__", spec_group, "__", out_col, "__none_", eff_ctrl),
        pos = pos_code, block = block_label, set = set_name, min_n_ind = as.integer(min_n_ind),
        spec_group = spec_group, outcome = out_col, measure = "none",
        ctrl = eff_ctrl, xmode = "none",
        data = dd, data2 = if (has_phylo) list(A = A_use) else NULL,
        formula = paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_fixed, re_str),
        has_b = (length(cov_t) > 0L), has_sd = (has_phylo || nzchar(extra_re)),
        has_gp = has_space, has_me = FALSE, is_mundlak = FALSE,
        adapt = if (has_space) 0.995 else 0.99, scale_info = sc_info
      ))
    }

    sb_code <- subset_of_measure(measure_name)
    req_gcs <- intersect(d_flagged$glottocode, subs[[sb_code]])
    if (length(req_gcs) < min_n_lang) return(NULL)

    e_m <- tryCatch(get_stage1(set_name, min_n_ind, measure_name), error = function(e) NULL)
    if (is.null(e_m)) return(NULL)
    e_sub <- e_m[e_m$glottocode %in% req_gcs, , drop = FALSE]
    req_gcs <- intersect(req_gcs, e_sub$glottocode)
    if (length(req_gcs) < min_n_lang) return(NULL)

    d_sub_m <- standardize_outcome_and_covs(
      d_flagged[d_flagged$glottocode %in% req_gcs, , drop = FALSE],
      out_col, se_col,
      require_demog = require_demog, use_demog = use_demog,
      use_cdist = use_cdist, use_lexprov = use_lexprov,
      use_med_fst = use_med_fst, use_regime_b = use_regime_b,
      min_n_lang = min_n_lang
    )
    if (is.null(d_sub_m)) return(NULL)
    y_sc <- attr(d_sub_m, "y_scale_info")

    dd <- merge(
      d_sub_m,
      e_sub[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
      by = "glottocode"
    )
    dd <- dd[order(dd$glottocode), , drop = FALSE]
    if (nrow(dd) < min_n_lang) return(NULL)

    mu_x <- mean(dd$eta_mean)
    sd_x <- sd(dd$eta_mean)
    if (is.na(sd_x) || sd_x < 1e-8) return(NULL)
    dd$x <- (dd$eta_mean - mu_x) / sd_x
    dd$x_sd <- pmax(as.numeric(dd$eta_sd) / sd_x, 1e-4)
    sc_info <- c(y_sc, list(x_center = mu_x, x_scale = sd_x, s1_y_mean = dd$y_mean[1], s1_y_sd = dd$y_sd[1]))

    cov_t <- build_cov_terms(
      dd, use_cdist = use_cdist, use_lexprov = use_lexprov,
      use_demog = use_demog, use_med_fst = use_med_fst, use_regime_b = use_regime_b
    )

    is_mun <- identical(xmode, "plain_mundlak")
    is_me <- identical(xmode, "me")

    if (is_mun) {
      fam_sz <- as.integer(ave(seq_len(nrow(dd)), dd$fam_eff, FUN = length))
      if (sum(fam_sz >= 2L) < 4L || length(unique(dd$fam_eff)) < 2L) return(NULL)
      dd$x_family_mean <- as.numeric(ave(dd$x, dd$fam_eff, FUN = mean))
      dd$x_within <- as.numeric(dd$x - dd$x_family_mean)
      dd$x_within[fam_sz == 1L] <- 0.0
      if (sd(dd$x_within) <= 1e-8 || sd(dd$x_family_mean) <= 1e-8) return(NULL)
      x_terms <- c("x_within", "x_family_mean")
    } else if (is_me) {
      if (!all(is.finite(dd$x_sd))) return(NULL)
      x_terms <- "me(x, x_sd)"
    } else {
      x_terms <- "x"
    }

    eff_ctrl <- if (isTRUE(drop_space_if_small) && nrow(dd) < 20L && ctrl_mode == "PS") "P" else ctrl_mode
    has_phylo <- eff_ctrl %in% c("P", "PS", "PS_bouckaert_stem", "PS_bouckaert_crown", "P_bouckaert_stem", "P_bouckaert_crown")
    has_space <- eff_ctrl %in% c("S", "PS", "PS_bouckaert_stem", "PS_bouckaert_crown")
    A_use <- if (!is.null(custom_A)) custom_A[dd$lang, dd$lang, drop = FALSE] else A_grafen_all[dd$lang, dd$lang, drop = FALSE]
    re_str <- paste0(
      if (has_phylo) " + (1 | gr(lang, cov = A))" else "",
      extra_re,
      if (has_space) " + gp(sx, sy, sz)" else ""
    )
    rhs_fixed <- paste(c("1", x_terms, cov_t), collapse = " + ")
    n_tag <- if (as.integer(min_n_ind) != PRIMARY_MIN_N) paste0("_n", as.integer(min_n_ind)) else ""

    list(
      name = paste0("s2_", pos_code, "__", set_name, n_tag, "__", spec_group, "__", out_col, "__", measure_name, "_", eff_ctrl),
      pos = pos_code, block = block_label, set = set_name, min_n_ind = as.integer(min_n_ind),
      spec_group = spec_group, outcome = out_col, measure = measure_name,
      ctrl = eff_ctrl, xmode = xmode,
      data = dd, data2 = if (has_phylo) list(A = A_use) else NULL,
      formula = paste0("y | se(se_y, sigma = TRUE) ~ ", rhs_fixed, re_str),
      has_b = TRUE, has_sd = (has_phylo || nzchar(extra_re)),
      has_gp = has_space, has_me = is_me, is_mundlak = is_mun,
      adapt = if (has_space || is_me) 0.995 else 0.99, scale_info = sc_info
    )
  }

  specs <- list()
  push_spec <- function(sp) {
    if (!is.null(sp)) specs[[length(specs) + 1L]] <<- sp
  }

  berkson_jobs <- list()

  for (set_name in sets_active) {
    d_set <- d0[d0[[paste0("in_", set_name)]] == 1L, , drop = FALSE]
    if (nrow(d_set) < 6L) {
      message("Skipping POS=", pos_code, " set=", set_name, " (only ", nrow(d_set), " langs)")
      next
    }
    pop_set <- pop0[pop0[[paste0("in_", set_name)]] == 1L & pop0$glottocode %in% d_set$glottocode, , drop = FALSE]

    if (set_name == "main") {
      primary_m <- measures_active[1]

      # --- BLOCK 1: Maximal Model & Locked Complete-Case LOO Ablations -----------------------
      for (m in measures_active) {
        sp_max <- make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "maximal",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        )
        push_spec(sp_max)
        push_spec(make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "loo_no_space",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "P",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        ))
        push_spec(make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "loo_no_cdist",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = FALSE, use_lexprov = TRUE
        ))
        push_spec(make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "loo_no_demog",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = FALSE, use_cdist = TRUE, use_lexprov = TRUE
        ))
        push_spec(make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "loo_no_lexprov",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = FALSE
        ))
        push_spec(make_spec(
          d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "mundlak_wb",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
          xmode = "plain_mundlak"
        ))
        if (m == primary_m) {
          push_spec(make_spec(
            d_set, pop_set, "block1", "main", PRIMARY_MIN_N, "null_baseline",
            "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
            xmode = "none"
          ))
        }

        # --- BLOCK 2A: Sample, Cutoff, Links, Phylogeny & Inner Asia Checks ------------------
        push_spec(make_spec(
          d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "full_sample_PS",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = FALSE, use_demog = FALSE, use_cdist = TRUE, use_lexprov = TRUE
        ))
        push_spec(make_spec(
          d_set, pop_set, "block2", "main", SENS_MIN_N, "cutoff_n5",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        ))

        d_strict <- d0[d0$in_strict == 1L, , drop = FALSE]
        pop_strict <- pop0[pop0$in_strict == 1L & pop0$glottocode %in% d_strict$glottocode, , drop = FALSE]
        push_spec(make_spec(
          d_strict, pop_strict, "block2", "strict", PRIMARY_MIN_N, "strict_links",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        ))

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
            bk_ctrl_stem <- if (nrow(d_bk_raw) >= 20L) "PS_bouckaert_stem" else "P_bouckaert_stem"
            bk_ctrl_crown <- if (nrow(d_bk_raw) >= 20L) "PS_bouckaert_crown" else "P_bouckaert_crown"
            push_spec(make_spec(
              d_bk_raw, pop_set, "block2", "main", PRIMARY_MIN_N, "bouckaert_stem",
              "h_tree_bits", "h_tree_se", m, ctrl_mode = bk_ctrl_stem,
              require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
              custom_A = A_bk_stem
            ))
            push_spec(make_spec(
              d_bk_raw, pop_set, "block2", "main", PRIMARY_MIN_N, "bouckaert_crown",
              "h_tree_bits", "h_tree_se", m, ctrl_mode = bk_ctrl_crown,
              require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
              custom_A = A_bk_crown, extra_re = " + (1 | fam_eff)"
            ))
          }
        }

        d_nia <- d_set[!(d_set$family %in% c("Turkic", "Mongolic-Khitan", "Tungusic") |
                         d_set$family_id %in% c("turk1311", "mong1329", "tung1282")), , drop = FALSE]
        push_spec(make_spec(
          d_nia, pop_set, "block2", "main", PRIMARY_MIN_N, "no_inner_asia",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        ))

        # --- BLOCK 2B: Learner, Regime & Metric Robustness Checks ----------------------------
        sec_specs <- list(
          list(grp = "tier2_uniform480", y = "h_tree_480_bits",     se = "h_tree_480_se",     min_n = 6L),
          list(grp = "topfreq120",       y = "h_tree_topfreq_bits", se = "h_tree_topfreq_se", min_n = 6L),
          list(grp = "pcfp_k1",          y = "h_k1_bits",           se = "h_k1_se",           min_n = 6L),
          list(grp = "pcfp_k3",          y = "h_k3_bits",           se = "h_k3_se",           min_n = 6L),
          list(grp = "reliability",      y = "logit_rel_tree",      se = "logit_rel_tree_se", min_n = 6L),
          list(grp = "malouf_harmony",   y = "h_tree_malouf_bits",  se = "h_tree_malouf_se",  min_n = 6L),
          list(grp = "gold_ipa",         y = "h_tree_phon_bits",    se = "h_tree_phon_se",    min_n = 8L),
          list(grp = "neural_monotonic", y = "h_tree_neural_bits",  se = "h_tree_neural_se",  min_n = 6L)
        )
        for (sc in sec_specs) {
          if (sc$y %in% names(d_set)) {
            push_spec(make_spec(
              d_set, pop_set, "block2", "main", PRIMARY_MIN_N, sc$grp,
              sc$y, sc$se, m, ctrl_mode = "PS",
              require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
              min_n_lang = sc$min_n, drop_space_if_small = TRUE
            ))
          }
        }

        # --- BLOCK 2C: Genetics SI Checks (run for primary `f3_fixed`) -----------------------
        if (m == "f3_fixed") {
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "f3_all_fixed",
            "h_tree_bits", "h_tree_se", "f3_all_fixed", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "f3_fixed_iso",
            "h_tree_bits", "h_tree_se", "f3_fixed_iso", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "f3_fixed_maxp",
            "h_tree_bits", "h_tree_se", "f3_fixed_maxp", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "f3_fixed_raw",
            "h_tree_bits", "h_tree_se", "f3_fixed_raw", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
            use_med_fst = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "f4_est",
            "h_tree_bits", "h_tree_se", "f4_est", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "neigh_flow_out",
            "h_tree_bits", "h_tree_se", "neigh_flow_out", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
            use_med_fst = TRUE
          ))

          d_flagged_n3 <- attach_lang_pop_flags(d_set, pop_set, min_n_ind = PRIMARY_MIN_N)
          d_no_rega <- d_flagged_n3[is.na(d_flagged_n3$regime_a_enclave) | d_flagged_n3$regime_a_enclave != 1L, , drop = FALSE]
          d_no_regb <- d_flagged_n3[is.na(d_flagged_n3$regime_b_shift) | d_flagged_n3$regime_b_shift != 1L, , drop = FALSE]

          push_spec(make_spec(
            d_no_rega, pop_set, "block2", "main", PRIMARY_MIN_N, "drop_regime_a",
            "h_tree_bits", "h_tree_se", "f3_fixed", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_no_regb, pop_set, "block2", "main", PRIMARY_MIN_N, "drop_regime_b",
            "h_tree_bits", "h_tree_se", "f3_fixed", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "plus_regime_b",
            "h_tree_bits", "h_tree_se", "f3_fixed", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
            use_regime_b = TRUE
          ))
          push_spec(make_spec(
            d_set, pop_set, "block2", "main", PRIMARY_MIN_N, "me_baseline",
            "h_tree_bits", "h_tree_se", "f3_fixed", ctrl_mode = "PS",
            require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE,
            xmode = "me"
          ))

          if (!is.null(sp_max)) {
            berkson_jobs[[length(berkson_jobs) + 1L]] <- list(
              measure = "f3_fixed",
              min_n_ind = PRIMARY_MIN_N,
              dd_base = sp_max$data
            )
          }
        }
      }
    } else if (!(set_name == "strict" && "main" %in% sets_active)) {
      for (m in measures_active) {
        push_spec(make_spec(
          d_set, pop_set, "block2", set_name, PRIMARY_MIN_N, "link_sensitivity",
          "h_tree_bits", "h_tree_se", m, ctrl_mode = "PS",
          require_demog = TRUE, use_demog = TRUE, use_cdist = TRUE, use_lexprov = TRUE
        ))
      }
    }
  }

  # Save Stage-1 exposures & diagnostics (grouped by set and cutoff) when not in dry mode
  if (!dry && length(ls(envir = s1_cache)) > 0L) {
    all_e <- do.call(rbind, mget(ls(envir = s1_cache), envir = s1_cache))
    all_diag <- do.call(rbind, s1_diag_list)
    write.csv(all_e, file.path(res_dir, sprintf("exposure_all_%s.csv", pos_code)), row.names = FALSE)
    write.csv(all_diag, file.path(res_dir, sprintf("stage1_diagnostics_%s.csv", pos_code)), row.names = FALSE)
    e_main_n3 <- all_e[all_e$set == "main" & all_e$min_n_ind == PRIMARY_MIN_N, , drop = FALSE]
    if (nrow(e_main_n3) > 0L) {
      write.csv(e_main_n3, file.path(res_dir, sprintf("exposure_main_%s.csv", pos_code)), row.names = FALSE)
    }
  }

  # Execute Stage-2 specs (sequential in dry mode or PAR_FITS == 1, PSOCK parallel otherwise)
  if (dry || PAR_FITS <= 1L || length(specs) <= 1L) {
    res_all <- lapply(specs, run_single_spec)
  } else {
    cl <- makePSOCKcluster(min(length(specs), PAR_FITS), outfile = "")
    res_all <- tryCatch({
      clusterExport(cl, ls(envir = .GlobalEnv), envir = .GlobalEnv)
      clusterEvalQ(cl, {
        Sys.setenv(
          OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
          OMP_NUM_THREADS = "1",
          MKL_NUM_THREADS = "1",
          OPENBLAS_NUM_THREADS = "1",
          VECLIB_MAXIMUM_THREADS = "1"
        )
        suppressPackageStartupMessages({ library(brms); library(ape); library(loo); library(posterior) })
        w_cache <- file.path(stan_cache_dir, paste0("worker_", Sys.getpid()))
        dir.create(w_cache, showWarnings = FALSE, recursive = TRUE)
        options(brms.backend = "cmdstanr", cmdstanr_write_stan_file_dir = w_cache)
        NULL
      })
      parLapplyLB(cl, specs, run_single_spec)
    }, finally = {
      try(stopCluster(cl), silent = TRUE)
    })
  }

  # 20-draw Stage-1 Berkson variance-propagation check on `f3_fixed`
  for (bj in berkson_jobs) {
    s1_rds <- file.path(
      fits_dir,
      sprintf("exposure_para_%s_main_n%d__%s.rds", pos_code, as.integer(bj$min_n_ind), bj$measure)
    )
    eta_draws <- extract_stage1_draws_matrix(s1_rds, bj$dd_base$glottocode, n_imp = N_IMP, seed_imp = SEED)
    berk_res <- run_berkson_20draw_check(
      pos_code, "main", bj$min_n_ind, bj$measure, bj$dd_base, eta_draws, A_grafen_all
    )
    if (!is.null(berk_res)) res_all[[length(res_all) + 1L]] <- berk_res
  }

  errs <- Filter(function(r) !is.list(r) || !is.null(r$error), res_all)
  for (er in errs) {
    message("ERROR [POS=", pos_code, "]: ",
            if (is.list(er)) paste(er$name, er$error) else as.character(er))
  }
  if (dry && length(errs) > 0L) {
    stop("PARA_DRY validation failed on ", length(errs), " model(s) for POS=", pos_code)
  }
  ok_res <- Filter(function(r) is.list(r) && is.null(r$error), res_all)
  message(format(Sys.time(), "%H:%M"), " [POS=", pos_code, "] ",
          length(ok_res), " models ", if (dry) "validated (DRY)" else "completed")

  if (length(ok_res) > 0L) {
    all_cols <- unique(unlist(lapply(ok_res, names)))
    tab_pos <- do.call(rbind, lapply(ok_res, function(r) {
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

  track1_domains <- c("v_breadth_prop", "n_breadth_prop", "v_irreg_prop", "n_irreg_prop", "np_agr_prop")

  sem_eap_path <- Sys.getenv("PARA_SEM_EAP_CSV", "analyses/grambank/latent_sem/results_domain/sem_latent_eap.csv")
  sem_eap_df <- if (nzchar(sem_eap_path) && file.exists(sem_eap_path)) {
    tryCatch(read.csv(sem_eap_path, stringsAsFactors = FALSE), error = function(e) NULL)
  } else {
    NULL
  }
  sem_eap_cols <- if (!is.null(sem_eap_df) && "glottocode" %in% names(sem_eap_df)) {
    grep("^theta0?_.+_eap$", names(sem_eap_df), value = TRUE)
  } else {
    character(0)
  }

  for (p in names(loaded_ld)) {
    df_p <- loaded_ld[[p]]
    tr_p <- loaded_tr[[p]]
    if (length(sem_eap_cols) > 0L) {
      drop_old <- intersect(sem_eap_cols, names(df_p))
      if (length(drop_old)) df_p <- df_p[, setdiff(names(df_p), drop_old), drop = FALSE]
      df_p <- merge(df_p, sem_eap_df[, c("glottocode", sem_eap_cols), drop = FALSE], by = "glottocode", all.x = TRUE)
    }

    # 1. Track-2 `h_tree_bits` and `log_C_distinct` vs Track-1 Grambank domain proportions & SEM EAPs
    t1_pairs <- list()
    for (dom in c(track1_domains, sem_eap_cols)) {
      t1_pairs[[length(t1_pairs) + 1L]] <- c(
        "h_tree_bits", dom, sprintf("H_tree_%s_vs_Track1_%s", p, dom)
      )
      t1_pairs[[length(t1_pairs) + 1L]] <- c(
        "log_C_distinct", dom, sprintf("log_C_distinct_%s_vs_Track1_%s", p, dom)
      )
    }
    # 2. Within-Track-2 learner / regime / representation convergence checks
    internal_pairs <- list(
      c("h_tree_bits", "h_tree_neural_bits",  sprintf("H_tree_%s_MGL_vs_Neural_120", p)),
      c("h_tree_bits", "h_tree_480_bits",     sprintf("H_tree_%s_120_vs_480", p)),
      c("h_tree_bits", "h_tree_topfreq_bits", sprintf("H_tree_%s_uniform_vs_topfreq", p)),
      c("h_tree_bits", "h_tree_phon_bits",    sprintf("H_tree_%s_orth_vs_phon", p)),
      c("h_tree_bits", "h_tree_malouf_bits",  sprintf("H_tree_%s_orth_vs_malouf", p))
    )
    for (pr in c(t1_pairs, internal_pairs)) {
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

  # Cross-POS correlations on shared languages (`h_tree_bits` and `log_C_distinct`)
  if (all(c("V", "N") %in% names(loaded_ld))) {
    m_vn <- merge(
      loaded_ld[["V"]][, c("glottocode", "h_tree_bits", "log_C_distinct")],
      loaded_ld[["N"]][, c("glottocode", "h_tree_bits", "log_C_distinct")],
      by = "glottocode", suffixes = c("_V", "_N")
    )
    for (pair_vn in list(
      c("h_tree_bits_V", "h_tree_bits_N", "H_tree_V_vs_H_tree_N"),
      c("log_C_distinct_V", "log_C_distinct_N", "log_C_distinct_V_vs_log_C_distinct_N")
    )) {
      st_vn <- safe_phylo_gls(m_vn, pair_vn[1], pair_vn[2], loaded_tr[["V"]])
      if (!is.null(st_vn)) {
        rows[[length(rows) + 1L]] <- cbind(
          data.frame(comparison = pair_vn[3], pos = "V_N",
                     x_var = pair_vn[1], y_var = pair_vn[2], stringsAsFactors = FALSE),
          st_vn
        )
      }
    }
  }

  if (length(rows) > 0L) {
    corr_df <- do.call(rbind, rows)
    corr_fname <- if (dry) "cross_track_correlations_dry.csv" else "cross_track_correlations.csv"
    tmp_corr <- file.path(res_dir, sprintf(".cross_track_correlations_%d.tmp", Sys.getpid()))
    write.csv(corr_df, tmp_corr, row.names = FALSE)
    file.rename(tmp_corr, file.path(res_dir, corr_fname))
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
