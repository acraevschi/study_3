# Focal Stan 2PL IRT-SEM & Within-POS 2-Factor SEM Runner (Study 3 Track 1)
#
# Fits:
#   1. 1-Factor 2PL IRT-SEM (`stan/item_domain_sem.stan`) for each of the 5 clean domains
#      (`v_breadth`, `n_breadth`, `v_irreg`, `n_irreg`, `np_agr`):
#        - Block 1 locked complete-case Maximal Model & LOO ablations (`maximal`, `loo_no_space`,
#          `loo_no_demog`, `loo_no_doccov`, `mundlak_wb`) using `f3_fixed`, `main`, `min_n_ind = 3L`
#        - Key Block 2 checks (`full_sample_PS`, `cutoff_n5`, `strict_links`, `bouckaert_stem`,
#          `bouckaert_crown`, `no_inner_asia`)
#        - 1 Architecture-B Item-Varying-Slope DIF fit (`MB_2pl` `maximal`) per domain extracting
#          item discriminations `lambda_k` and item-specific admixture slope offsets `beta_item[k]`
#   2. Joint Within-POS 2-Factor 2PL SEMs (`stan/item_pos_2factor_sem.stan`) for `Verb_2fac`
#      (`v_breadth` + `v_irreg`) and `Noun_2fac` (`n_breadth` + `n_irreg`) at `n_A >= 3` & `n_A >= 5`,
#      estimating `b_breadth`, `b_irreg`, `b_diff`, and `rho_latent`.
#
# Environment variables:
#   SEM_DRY=1:        Compile both `.stan` models via cmdstanr and validate ALL Stan data blocks;
#                     zero MCMC sampling
#   SEM_SMOKE=1:      Fast 400-iter 2-chain smoke fit into fits/domain_smoke/ & results_domain_smoke/
#   SEM_CORES=4:      Dynamic CPU core budget across chains/workers (or --cores=N; default 4)
#   SEM_THREADS=1:    Within-chain Stan threads (or --threads=N; default 1)
#   SEM_DOMAINS=...:  Comma-separated domains (default: v_breadth,n_breadth,v_irreg,n_irreg,np_agr)
#   SEM_RUN_2FAC=1:   Whether to run joint Verb_2fac / Noun_2fac 2-factor models (default: 1)
#   SEM_TAG=...:      Optional suffix appended to output CSVs

Sys.setenv(
  OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
  OMP_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1"
)
invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(brms)
  library(ape)
  library(loo)
  library(posterior)
  library(parallel)
  library(tools)
})
options(brms.backend = "cmdstanr")

cmd_args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_all, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else getwd()
if (!dir.exists("analyses/grambank/latent_sem") &&
    dir.exists(file.path(script_dir, "..", "..", "..", "analyses", "grambank", "latent_sem"))) {
  setwd(normalizePath(file.path(script_dir, "..", "..", "..")))
}

here <- "analyses/grambank/latent_sem"
data_dir <- "analyses/grambank/data"
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

dry <- nzchar(Sys.getenv("SEM_DRY"))
smoke <- nzchar(Sys.getenv("SEM_SMOKE"))
run_2fac <- as.integer(Sys.getenv("SEM_RUN_2FAC", "1")) == 1L

cores_env <- Sys.getenv("SEM_CORES", "")
CORES <- max(1L, as.integer(if (nzchar(cores_env)) cores_env else parse_cli_int("cores", 4L)))
threads_env <- Sys.getenv("SEM_THREADS", "")
THREADS <- max(1L, as.integer(if (nzchar(threads_env)) threads_env else parse_cli_int("threads", 1L)))
if (THREADS > CORES) THREADS <- CORES

if (smoke) {
  ITER_SAMPLING <- 200L; ITER_WARMUP <- 200L; CHAINS <- min(2L, max(1L, CORES %/% THREADS))
} else {
  ITER_SAMPLING <- 1500L; ITER_WARMUP <- 1500L; CHAINS <- min(4L, max(1L, CORES %/% THREADS))
}
PAR_FITS <- max(1L, CORES %/% (CHAINS * THREADS))
stopifnot(PAR_FITS * CHAINS * THREADS <= CORES)

SEED <- 20261126L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
ROPE_HALF_WIDTH <- 0.15

# Identical regularized spatial GP lengthscale prior across Track 1 & Track 2
LSCALE_PRIOR_SHAPE <- 3.0
LSCALE_PRIOR_SCALE <- 0.4

stan_bin_dir <- file.path(here, "fits")
fits_dir <- if (dry) tempdir() else file.path(here, "fits", if (smoke) "domain_smoke" else "domain")
res_dir <- if (dry) tempdir() else file.path(here, if (smoke) "results_domain_smoke" else "results_domain")
s1_fits_dir <- if (dry) tempdir() else file.path("analyses/grambank/baseline/fits", if (smoke) "bb_smoke" else "bb")
s1_stan_cache <- file.path("analyses/grambank/baseline/fits", "stan_cache_bb")
tag_raw <- Sys.getenv("SEM_TAG", "")
tag <- if (nzchar(tag_raw)) paste0("_", tag_raw) else ""

dir.create(stan_bin_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(s1_fits_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(s1_stan_cache, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = s1_stan_cache)
}

DOMAIN_FLOORS <- list(
  v_breadth = list(coded = "v_breadth_coded", min_coded = 6L, k_items = 9L),
  n_breadth = list(coded = "n_breadth_coded", min_coded = 3L, k_items = 5L),
  v_irreg   = list(coded = "v_irreg_coded",   min_coded = 3L, k_items = 5L),
  n_irreg   = list(coded = "n_irreg_coded",   min_coded = 2L, k_items = 3L),
  np_agr    = list(coded = "np_agr_coded",    min_coded = 5L, k_items = 7L)
)
domains_active <- trimws(strsplit(
  Sys.getenv("SEM_DOMAINS", paste(names(DOMAIN_FLOORS), collapse = ",")),
  ","
)[[1]])
stopifnot(all(domains_active %in% names(DOMAIN_FLOORS)))

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

build_gp_coords <- function(d_lang) {
  coords <- as.matrix(d_lang[, c("sx", "sy", "sz"), drop = FALSE])
  max_d <- max(as.matrix(dist(coords)))
  stopifnot(is.finite(max_d), max_d > 0)
  Xgp <- coords / max_d
  lapply(seq_len(nrow(Xgp)), function(i) as.numeric(Xgp[i, ]))
}

# --- Compile Both Stan Models (Mandatory Even in SEM_DRY=1) ----------------------------------

stan_1f_path <- file.path(here, "stan", "item_domain_sem.stan")
stan_2f_path <- file.path(here, "stan", "item_pos_2factor_sem.stan")
stopifnot(file.exists(stan_1f_path), file.exists(stan_2f_path))

message("Compiling / verifying Stan models via cmdstanr into ", stan_bin_dir, " ...")
cpp_opts <- if (THREADS > 1L) list(stan_threads = TRUE) else list()
mod_1f <- cmdstan_model(stan_1f_path, dir = stan_bin_dir, cpp_options = cpp_opts)
mod_2f <- cmdstan_model(stan_2f_path, dir = stan_bin_dir, cpp_options = cpp_opts)
message("Both Stan models compiled cleanly: ", basename(stan_1f_path), ", ", basename(stan_2f_path))

# --- Load Canonical Prepared Data -------------------------------------------------------------

d0 <- read.csv(file.path(data_dir, "language_data.csv"), stringsAsFactors = FALSE)
items_df <- read.csv(file.path(data_dir, "items_metadata.csv"), stringsAsFactors = FALSE)
obs_df <- read.csv(file.path(data_dir, "item_observations.csv"), stringsAsFactors = FALSE)
ind0 <- read.csv(file.path(data_dir, "individuals.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(data_dir, "populations.csv"), stringsAsFactors = FALSE)
pop0 <- augment_population_f3_columns(
  pop0,
  k5_file = file.path(common_dir, "data/f3_scores_k5.csv"),
  f3_fixed_file = file.path(common_dir, "data/f3_fixed_scores.csv"),
  fst_regimes_file = file.path(common_dir, "data/fst_neighbor_regimes.csv"),
  discrete_k_file = file.path(common_dir, "data/discrete_k_diagnostics.csv")
)

to_rad <- pi / 180.0
d0$sx <- cos(d0$lat * to_rad) * cos(d0$lon * to_rad)
d0$sy <- cos(d0$lat * to_rad) * sin(d0$lon * to_rad)
d0$sz <- sin(d0$lat * to_rad)
d0$fam_eff <- ifelse(
  is.na(d0$family) | !nzchar(d0$family) | d0$family == "isolate",
  paste0("isolate__", d0$glottocode),
  as.character(d0$family)
)

raw_tree <- read.tree(file.path(data_dir, "phylo_tree.nwk"))
A_grafen_all <- build_grafen_corr(raw_tree, d0$glottocode)
bouckaert_tr <- prepare_bouckaert_for_sample(d0$glottocode)

# Stage-1 exposure cache (shares fit prefix "bb_" with fit_stage2_bb.R so Stage-1 fits are reused!)
s1_cache <- new.env(parent = emptyenv())
get_stage1_f3 <- function(set_name, min_n_ind) {
  key <- sprintf("%s__n%d__f3_fixed", set_name, as.integer(min_n_ind))
  if (exists(key, envir = s1_cache, inherits = FALSE)) {
    return(get(key, envir = s1_cache, inherits = FALSE))
  }
  pop_set <- pop0[pop0[[paste0("in_", set_name)]] == 1L, , drop = FALSE]
  res <- fit_stage1_exposure(
    measure = "f3_fixed",
    ind_df = ind0,
    pop_df = pop_set,
    set_name = set_name,
    fits_dir = s1_fits_dir,
    fit_prefix = "bb_",
    min_n_ind = as.integer(min_n_ind),
    dry = dry,
    chains = CHAINS,
    iter = if (smoke) 400L else 3000L,
    warmup = if (smoke) 200L else 1500L,
    threads = THREADS,
    seed = SEED
  )
  res$e$min_n_ind <- as.integer(min_n_ind)
  assign(key, res, envir = s1_cache)
  res
}

# Fit or load Stage-1 exposures in the main process before parallel Stage-2 sampling
for (pw in list(list("main", 3L), list("main", 5L), list("strict", 3L))) {
  invisible(get_stage1_f3(pw[[1]], pw[[2]]))
}
if (!dry) {
  s1_keys <- sort(ls(envir = s1_cache))
  expo_sem_df <- do.call(rbind, lapply(s1_keys, function(k) get(k, envir = s1_cache, inherits = FALSE)$e))
  write.csv(expo_sem_df, file.path(res_dir, paste0("exposure_sem_f3_fixed", tag, ".csv")), row.names = FALSE)
}

# Build standardized active language table for 1-factor domain or 2-factor POS union
prepare_sem_langs <- function(
  eligible_mask,
  set_name,
  min_n_ind,
  require_demog = TRUE,
  drop_inner_asia = FALSE,
  restrict_bouckaert = FALSE
) {
  sub <- d0[d0[[paste0("in_", set_name)]] == 1L & eligible_mask, , drop = FALSE]
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
  e_df <- get_stage1_f3(set_name, min_n_ind)$e
  sub <- merge(sub, e_df[, c("glottocode", "eta_mean", "eta_sd")], by = "glottocode")
  sub <- sub[order(sub$glottocode), , drop = FALSE]
  if (nrow(sub) < 10L) return(NULL)

  sub$x <- (sub$eta_mean - mean(sub$eta_mean)) / sd(sub$eta_mean)
  fam_sz <- as.integer(ave(seq_len(nrow(sub)), sub$fam_eff, FUN = length))
  fam_mean <- as.numeric(ave(sub$x, sub$fam_eff, FUN = mean))
  sub$x_within <- ifelse(fam_sz >= 2L, sub$x - fam_mean, 0.0)
  sub$x_between <- fam_mean
  sub$fam_size <- fam_sz
  sub$fam_id <- match(sub$fam_eff, unique(sub$fam_eff))

  if (isTRUE(require_demog)) {
    lp <- as.numeric(sub$log_pop)
    la <- as.numeric(sub$log_area)
    sub$z_pop <- (lp - mean(lp)) / sd(lp)
    sub$z_area <- (la - mean(la)) / sd(la)
  } else {
    sub$z_pop <- 0.0
    sub$z_area <- 0.0
  }
  pdoc <- as.numeric(sub$prop_coded_nonmorph)
  sub$z_doc <- if (sd(pdoc) > 1e-8) (pdoc - mean(pdoc)) / sd(pdoc) else 0.0
  sub
}

# Build & validate Stan data for 1-factor domain model
build_1f_standata <- function(
  d_lang,
  dom_name,
  cov_cols = c("z_pop", "z_area", "z_doc"),
  use_2pl = 1L,
  use_var_slopes = 0L,
  use_phylo = 1L,
  use_space = 1L,
  use_testlets = 1L,
  use_mundlak = 0L,
  use_fam_int = 0L,
  custom_A = NULL
) {
  langs <- d_lang$glottocode
  N_lang <- length(langs)
  lang_map <- setNames(seq_len(N_lang), langs)

  im <- items_df[items_df$item_group == dom_name, , drop = FALSE]
  im <- im[order(im$item_order), , drop = FALSE]
  K_item <- nrow(im)
  stopifnot(K_item >= 2L)
  item_map <- setNames(seq_len(K_item), im$item_id)

  testlet_levels <- unique(im$testlet[im$testlet != "none" & nzchar(im$testlet)])
  item_testlet <- ifelse(im$testlet %in% testlet_levels, match(im$testlet, testlet_levels), 0L)
  T_test <- length(testlet_levels)

  ob <- obs_df[obs_df$glottocode %in% langs & obs_df$item_id %in% im$item_id, , drop = FALSE]
  ob$lang_idx <- unname(lang_map[ob$glottocode])
  ob$item_idx <- unname(item_map[ob$item_id])
  ob <- ob[order(ob$lang_idx, ob$item_idx), , drop = FALSE]
  stopifnot(nrow(ob) > 0L, all(!is.na(ob$lang_idx)), all(!is.na(ob$item_idx)), all(ob$value %in% c(0L, 1L)))

  A_sub <- if (!is.null(custom_A)) custom_A[langs, langs, drop = FALSE] else A_grafen_all[langs, langs, drop = FALSE]
  L_phylo <- t(chol(A_sub))

  P_cov <- length(cov_cols)
  Z_cov <- if (P_cov > 0L) {
    mat <- as.matrix(d_lang[, cov_cols, drop = FALSE])
    storage.mode(mat) <- "double"
    stopifnot(all(is.finite(mat)))
    mat
  } else {
    matrix(0.0, nrow = N_lang, ncol = 0L)
  }

  sdata <- list(
    N_lang = as.integer(N_lang),
    K_item = as.integer(K_item),
    T_test = as.integer(T_test),
    N_obs = as.integer(nrow(ob)),
    N_fam = as.integer(max(d_lang$fam_id)),
    P_cov = as.integer(P_cov),
    lang_id = as.integer(ob$lang_idx),
    item_id = as.integer(ob$item_idx),
    Y = as.integer(ob$value),
    item_testlet = as.integer(item_testlet),
    fam_id = as.integer(d_lang$fam_id),
    fam_size = as.integer(d_lang$fam_size),
    x = as.numeric(d_lang$x),
    x_within = as.numeric(d_lang$x_within),
    x_between = as.numeric(d_lang$x_between),
    Z_cov = Z_cov,
    with_x = 1L,
    use_2pl = as.integer(use_2pl),
    use_var_slopes = as.integer(use_var_slopes),
    use_phylo = as.integer(use_phylo),
    use_space = as.integer(use_space),
    use_testlets = as.integer(use_testlets),
    use_fam_int = as.integer(use_fam_int),
    use_mundlak = as.integer(use_mundlak),
    L_phylo = L_phylo,
    Xgp = build_gp_coords(d_lang),
    lscale_prior_shape = LSCALE_PRIOR_SHAPE,
    lscale_prior_scale = LSCALE_PRIOR_SCALE
  )
  # Strict data-block validation
  stopifnot(
    length(sdata$x) == N_lang, all(is.finite(sdata$x)),
    length(sdata$x_within) == N_lang, all(is.finite(sdata$x_within)),
    length(sdata$x_between) == N_lang, all(is.finite(sdata$x_between)),
    nrow(sdata$L_phylo) == N_lang, ncol(sdata$L_phylo) == N_lang, all(is.finite(sdata$L_phylo)),
    length(sdata$Xgp) == N_lang
  )
  list(stan_data = sdata, items = im$item_id, cov_cols = cov_cols)
}

# Build & validate Stan data for joint within-POS 2-factor model
# Languages passing breadth >= 60% supply Factor-1 observations;
# languages passing irreg >= 60% supply Factor-2 observations.
build_2f_pos_standata <- function(
  d_lang,
  dom1,
  dom2,
  cov_cols = c("z_pop", "z_area", "z_doc"),
  use_phylo = 1L,
  use_space = 1L,
  use_testlets = 1L
) {
  langs <- d_lang$glottocode
  N_lang <- length(langs)
  lang_map <- setNames(seq_len(N_lang), langs)

  im1 <- items_df[items_df$item_group == dom1, , drop = FALSE]
  im1 <- im1[order(im1$item_order), , drop = FALSE]
  im2 <- items_df[items_df$item_group == dom2, , drop = FALSE]
  im2 <- im2[order(im2$item_order), , drop = FALSE]
  im <- rbind(im1, im2)
  K1 <- nrow(im1)
  K2 <- nrow(im2)
  K_item <- K1 + K2
  item_map <- setNames(seq_len(K_item), im$item_id)

  testlet_levels <- unique(im$testlet[im$testlet != "none" & nzchar(im$testlet)])
  item_testlet <- ifelse(im$testlet %in% testlet_levels, match(im$testlet, testlet_levels), 0L)
  T_test <- length(testlet_levels)
  # Ensure no testlet crosses factors
  for (t in seq_len(T_test)) {
    f_mem <- ifelse(which(item_testlet == t) <= K1, 1L, 2L)
    stopifnot(length(unique(f_mem)) == 1L)
  }

  f1 <- DOMAIN_FLOORS[[dom1]]
  f2 <- DOMAIN_FLOORS[[dom2]]
  elig1_gcs <- d_lang$glottocode[d_lang[[f1$coded]] >= f1$min_coded]
  elig2_gcs <- d_lang$glottocode[d_lang[[f2$coded]] >= f2$min_coded]

  ob1 <- obs_df[obs_df$glottocode %in% elig1_gcs & obs_df$item_id %in% im1$item_id, , drop = FALSE]
  ob2 <- obs_df[obs_df$glottocode %in% elig2_gcs & obs_df$item_id %in% im2$item_id, , drop = FALSE]
  ob <- rbind(ob1, ob2)
  ob$lang_idx <- unname(lang_map[ob$glottocode])
  ob$item_idx <- unname(item_map[ob$item_id])
  ob <- ob[order(ob$lang_idx, ob$item_idx), , drop = FALSE]
  stopifnot(nrow(ob) > 0L, all(!is.na(ob$lang_idx)), all(!is.na(ob$item_idx)), all(ob$value %in% c(0L, 1L)))

  A_sub <- A_grafen_all[langs, langs, drop = FALSE]
  L_phylo <- t(chol(A_sub))

  P_cov <- length(cov_cols)
  Z_cov <- if (P_cov > 0L) {
    mat <- as.matrix(d_lang[, cov_cols, drop = FALSE])
    storage.mode(mat) <- "double"
    stopifnot(all(is.finite(mat)))
    mat
  } else {
    matrix(0.0, nrow = N_lang, ncol = 0L)
  }

  sdata <- list(
    N_lang = as.integer(N_lang),
    K1 = as.integer(K1),
    K2 = as.integer(K2),
    K_item = as.integer(K_item),
    T_test = as.integer(T_test),
    N_obs = as.integer(nrow(ob)),
    N_fam = as.integer(max(d_lang$fam_id)),
    P_cov = as.integer(P_cov),
    lang_id = as.integer(ob$lang_idx),
    item_id = as.integer(ob$item_idx),
    Y = as.integer(ob$value),
    item_testlet = as.integer(item_testlet),
    fam_id = as.integer(d_lang$fam_id),
    fam_size = as.integer(d_lang$fam_size),
    x = as.numeric(d_lang$x),
    x_within = as.numeric(d_lang$x_within),
    x_between = as.numeric(d_lang$x_between),
    Z_cov = Z_cov,
    with_x = 1L,
    use_2pl = 1L,
    use_phylo = as.integer(use_phylo),
    use_space = as.integer(use_space),
    use_testlets = as.integer(use_testlets),
    use_fam_int = 0L,
    use_mundlak = 0L,
    L_phylo = L_phylo,
    Xgp = build_gp_coords(d_lang),
    lscale_prior_shape = LSCALE_PRIOR_SHAPE,
    lscale_prior_scale = LSCALE_PRIOR_SCALE
  )
  list(
    stan_data = sdata,
    items = im$item_id,
    n_lang_f1 = length(elig1_gcs),
    n_lang_f2 = length(elig2_gcs)
  )
}

# --- Queue All 1-Factor & 2-Factor SEM Specifications ----------------------------------------

specs_1f <- list()

add_1f_spec <- function(dom_name, block, spec, arch, set_name, min_n_ind, d_lang,
                        cov_cols, use_var_slopes = 0L, use_phylo = 1L, use_space = 1L,
                        use_mundlak = 0L, use_fam_int = 0L, custom_A = NULL) {
  if (is.null(d_lang)) return(invisible(NULL))
  built <- build_1f_standata(
    d_lang = d_lang, dom_name = dom_name, cov_cols = cov_cols,
    use_2pl = 1L, use_var_slopes = use_var_slopes,
    use_phylo = use_phylo, use_space = use_space, use_testlets = 1L,
    use_mundlak = use_mundlak, use_fam_int = use_fam_int, custom_A = custom_A
  )
  specs_1f[[length(specs_1f) + 1L]] <<- list(
    name = sprintf("sem1f__%s__%s__%s_%s_n%d", dom_name, arch, spec, set_name, as.integer(min_n_ind)),
    domain = dom_name,
    block = block,
    spec = spec,
    arch = arch,
    set = set_name,
    min_n_ind = as.integer(min_n_ind),
    n_lang = built$stan_data$N_lang,
    k_item = built$stan_data$K_item,
    n_obs = built$stan_data$N_obs,
    t_test = built$stan_data$T_test,
    p_cov = built$stan_data$P_cov,
    use_mundlak = as.integer(use_mundlak),
    use_var_slopes = as.integer(use_var_slopes),
    cov_cols = built$cov_cols,
    glottocodes = d_lang$glottocode,
    stan_data = built$stan_data,
    items = built$items
  )
}

for (dom_nm in domains_active) {
  fl <- DOMAIN_FLOORS[[dom_nm]]
  elig_mask <- !is.na(d0[[fl$coded]]) & d0[[fl$coded]] >= fl$min_coded

  # Block 1 locked complete-case subset (main, f3_fixed, n_A >= 3, has_area == 1)
  d_b1 <- prepare_sem_langs(elig_mask, set_name = "main", min_n_ind = 3L, require_demog = TRUE)
  stopifnot(!is.null(d_b1))

  add_1f_spec(dom_nm, "block1", "maximal", "MA_2pl", "main", 3L, d_b1,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L)
  add_1f_spec(dom_nm, "block1", "loo_no_space", "MA_2pl", "main", 3L, d_b1,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 0L)
  add_1f_spec(dom_nm, "block1", "loo_no_demog", "MA_2pl", "main", 3L, d_b1,
              cov_cols = c("z_doc"), use_phylo = 1L, use_space = 1L)
  add_1f_spec(dom_nm, "block1", "loo_no_doccov", "MA_2pl", "main", 3L, d_b1,
              cov_cols = c("z_pop", "z_area"), use_phylo = 1L, use_space = 1L)
  add_1f_spec(dom_nm, "block1", "mundlak_wb", "MA_2pl", "main", 3L, d_b1,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L, use_mundlak = 1L)

  # Key Block 2 checks (MA_2pl)
  d_full <- prepare_sem_langs(elig_mask, set_name = "main", min_n_ind = 3L, require_demog = FALSE)
  add_1f_spec(dom_nm, "block2", "full_sample_PS", "MA_2pl", "main", 3L, d_full,
              cov_cols = c("z_doc"), use_phylo = 1L, use_space = 1L)

  d_n5 <- prepare_sem_langs(elig_mask, set_name = "main", min_n_ind = 5L, require_demog = TRUE)
  add_1f_spec(dom_nm, "block2", "cutoff_n5", "MA_2pl", "main", 5L, d_n5,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L)

  d_strict <- prepare_sem_langs(elig_mask, set_name = "strict", min_n_ind = 3L, require_demog = TRUE)
  add_1f_spec(dom_nm, "block2", "strict_links", "MA_2pl", "strict", 3L, d_strict,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L)

  d_bouck <- prepare_sem_langs(elig_mask, set_name = "main", min_n_ind = 3L,
                               require_demog = TRUE, restrict_bouckaert = TRUE)
  if (!is.null(d_bouck)) {
    A_bstem <- build_bouckaert_corr(
      d_bouck$glottocode, families = d_bouck$family,
      tree = bouckaert_tr, within_family = TRUE, crown_only = FALSE
    )
    A_bcrown <- build_bouckaert_corr(
      d_bouck$glottocode, families = d_bouck$family,
      tree = bouckaert_tr, within_family = TRUE, crown_only = TRUE
    )
    add_1f_spec(dom_nm, "block2", "bouckaert_stem", "MA_2pl", "main", 3L, d_bouck,
                cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L,
                custom_A = A_bstem)
    add_1f_spec(dom_nm, "block2", "bouckaert_crown", "MA_2pl", "main", 3L, d_bouck,
                cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L,
                use_fam_int = 1L, custom_A = A_bcrown)
  }

  d_noia <- prepare_sem_langs(elig_mask, set_name = "main", min_n_ind = 3L,
                              require_demog = TRUE, drop_inner_asia = TRUE)
  add_1f_spec(dom_nm, "block2", "no_inner_asia", "MA_2pl", "main", 3L, d_noia,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_phylo = 1L, use_space = 1L)

  # Item-varying-slope DIF diagnostic (MB_2pl maximal)
  add_1f_spec(dom_nm, "dif", "maximal", "MB_2pl", "main", 3L, d_b1,
              cov_cols = c("z_pop", "z_area", "z_doc"), use_var_slopes = 1L,
              use_phylo = 1L, use_space = 1L)
}

specs_2f <- list()
if (run_2fac) {
  pos_pairs <- list(
    Verb_2fac = c("v_breadth", "v_irreg"),
    Noun_2fac = c("n_breadth", "n_irreg")
  )
  for (p_nm in names(pos_pairs)) {
    dom1 <- pos_pairs[[p_nm]][1]
    dom2 <- pos_pairs[[p_nm]][2]
    f1 <- DOMAIN_FLOORS[[dom1]]
    f2 <- DOMAIN_FLOORS[[dom2]]
    union_mask <- (!is.na(d0[[f1$coded]]) & d0[[f1$coded]] >= f1$min_coded) |
                  (!is.na(d0[[f2$coded]]) & d0[[f2$coded]] >= f2$min_coded)
    for (mn in c(3L, 5L)) {
      d_u <- prepare_sem_langs(union_mask, set_name = "main", min_n_ind = mn, require_demog = TRUE)
      built2 <- build_2f_pos_standata(
        d_u, dom1 = dom1, dom2 = dom2,
        cov_cols = c("z_pop", "z_area", "z_doc"),
        use_phylo = 1L, use_space = 1L, use_testlets = 1L
      )
      specs_2f[[length(specs_2f) + 1L]] <- list(
        name = sprintf("sem2f__%s__maximal_main_n%d", p_nm, mn),
        pos_pair = p_nm,
        dom1 = dom1,
        dom2 = dom2,
        spec = sprintf("maximal_n%d", mn),
        set = "main",
        min_n_ind = mn,
        n_lang = built2$stan_data$N_lang,
        n_lang_f1 = built2$n_lang_f1,
        n_lang_f2 = built2$n_lang_f2,
        k1 = built2$stan_data$K1,
        k2 = built2$stan_data$K2,
        n_obs = built2$stan_data$N_obs,
        t_test = built2$stan_data$T_test,
        stan_data = built2$stan_data,
        items = built2$items
      )
    }
  }
}

message(sprintf(
  "Track 1 Stan 2PL IRT-SEM (%s): %d 1-factor specs + %d joint within-POS 2-factor specs validated.",
  if (dry) "DRY" else if (smoke) "SMOKE" else "FULL",
  length(specs_1f), length(specs_2f)
))

if (dry) {
  df1 <- do.call(rbind, lapply(specs_1f, function(s) {
    data.frame(
      name = s$name, domain = s$domain, block = s$block, spec = s$spec, arch = s$arch,
      set = s$set, min_n_ind = s$min_n_ind, n_lang = s$n_lang, k_item = s$k_item,
      n_obs = s$n_obs, t_test = s$t_test, p_cov = s$p_cov, stringsAsFactors = FALSE
    )
  }))
  message("SEM_DRY=1 1-Factor Summary (", nrow(df1), " specs):")
  print(df1, row.names = FALSE)
  if (length(specs_2f)) {
    df2 <- do.call(rbind, lapply(specs_2f, function(s) {
      data.frame(
        name = s$name, pos_pair = s$pos_pair, spec = s$spec, min_n_ind = s$min_n_ind,
        n_lang = s$n_lang, n_lang_f1 = s$n_lang_f1, n_lang_f2 = s$n_lang_f2,
        k1 = s$k1, k2 = s$k2, n_obs = s$n_obs, t_test = s$t_test, stringsAsFactors = FALSE
      )
    }))
    message("SEM_DRY=1 2-Factor POS Summary (", nrow(df2), " specs):")
    print(df2, row.names = FALSE)
  }
  message("SEM_DRY=1 PASSED (0 errors).")
  quit(save = "no", status = 0L)
}

# --- Sampling Execution (Smoke / Full) --------------------------------------------------------

fit_or_load_cmdstan <- function(mod, sdata, rds_path, adapt_delta = 0.99) {
  if (file.exists(rds_path)) {
    return(readRDS(rds_path))
  }
  sample_args <- list(
    data = sdata,
    chains = CHAINS,
    parallel_chains = CHAINS,
    iter_warmup = ITER_WARMUP,
    iter_sampling = ITER_SAMPLING,
    seed = SEED,
    adapt_delta = adapt_delta,
    max_treedepth = 12L,
    refresh = 0L,
    show_messages = FALSE
  )
  if (THREADS > 1L) sample_args$threads_per_chain <- THREADS
  fit <- do.call(mod$sample, sample_args)
  fit$save_object(file = rds_path)
  fit
}

run_single_1f <- function(sp) {
  t0 <- Sys.time()
  rds_p <- file.path(fits_dir, paste0(sp$name, ".rds"))
  mod_use <- if (exists("w_mod_1f", envir = .GlobalEnv, inherits = FALSE)) {
    get("w_mod_1f", envir = .GlobalEnv)
  } else {
    mod_1f
  }
  fit <- tryCatch(
    fit_or_load_cmdstan(mod_use, sp$stan_data, rds_p, adapt_delta = 0.99),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(name = sp$name, error = conditionMessage(fit)))
  }

  dr <- fit$draws(format = "df")
  key_vars <- intersect(
    c("b_x", "b_between", "b_within_minus_between", "sd_lang", "sd_phylo[1]", "sd_gp[1]", "lscale_gp[1]",
      "sd_item_b[1]", grep("^b_cov\\[", names(dr), value = TRUE)),
    names(dr)
  )
  sm_key <- summarise_draws(fit$draws(variables = key_vars))
  diag_sm <- fit$diagnostic_summary(quiet = TRUE)
  ll_tip <- fit$draws("log_lik_lang_tip", format = "matrix")
  loo_res <- tryCatch(loo::loo(ll_tip), error = function(e) NULL)

  idx_bx <- match("b_x", sm_key$variable)
  row_out <- c(
    list(
      name = sp$name, domain = sp$domain, block = sp$block, spec = sp$spec, arch = sp$arch,
      set = sp$set, min_n_ind = sp$min_n_ind, n_lang = sp$n_lang, k_item = sp$k_item,
      n_obs = sp$n_obs,
      max_rhat = max(sm_key$rhat, na.rm = TRUE),
      min_ess_bulk = min(sm_key$ess_bulk, na.rm = TRUE),
      divergences = sum(diag_sm$num_divergent),
      elpd_loo_tip = if (!is.null(loo_res)) loo_res$estimates["elpd_loo", "Estimate"] else NA_real_,
      pareto_k_gt_07 = if (!is.null(loo_res)) sum(loo_res$diagnostics$pareto_k > 0.7) else NA_integer_,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ),
    summarise_slope_metrics(
      dr$b_x, pfx = "b",
      b_rhat = if (!is.na(idx_bx)) sm_key$rhat[idx_bx] else NA_real_,
      b_ess  = if (!is.na(idx_bx)) sm_key$ess_bulk[idx_bx] else NA_real_
    )
  )

  if (sp$use_mundlak == 1L && "b_between" %in% names(dr)) {
    idx_bb <- match("b_between", sm_key$variable)
    row_out <- c(
      row_out,
      summarise_slope_metrics(
        dr$b_between, pfx = "b_between",
        b_rhat = if (!is.na(idx_bb)) sm_key$rhat[idx_bb] else NA_real_,
        b_ess  = if (!is.na(idx_bb)) sm_key$ess_bulk[idx_bb] else NA_real_
      )
    )
    if ("b_within_minus_between" %in% names(dr)) {
      b_diff <- dr$b_within_minus_between
      row_out <- c(row_out, list(
        b_wb_diff_mean = mean(b_diff),
        b_wb_diff_q025 = unname(quantile(b_diff, 0.025)),
        b_wb_diff_q10  = unname(quantile(b_diff, 0.10)),
        b_wb_diff_q90  = unname(quantile(b_diff, 0.90)),
        b_wb_diff_q975 = unname(quantile(b_diff, 0.975)),
        p_wb_diff_pos  = mean(b_diff > 0.0)
      ))
    }
  }

  cov_map <- c(z_pop = "b_pop", z_area = "b_area", z_doc = "b_doc")
  for (cc in names(cov_map)) {
    pfx <- unname(cov_map[cc])
    j <- match(cc, sp$cov_cols)
    col_j <- if (!is.na(j)) sprintf("b_cov[%d]", j) else ""
    if (!is.na(j) && col_j %in% names(dr)) {
      cv <- dr[[col_j]]
      row_out[[paste0(pfx, "_mean")]] <- mean(cv)
      row_out[[paste0(pfx, "_q025")]] <- unname(quantile(cv, 0.025))
      row_out[[paste0(pfx, "_q10")]]  <- unname(quantile(cv, 0.10))
      row_out[[paste0(pfx, "_q90")]]  <- unname(quantile(cv, 0.90))
      row_out[[paste0(pfx, "_q975")]] <- unname(quantile(cv, 0.975))
    } else {
      row_out[[paste0(pfx, "_mean")]] <- NA_real_
      row_out[[paste0(pfx, "_q025")]] <- NA_real_
      row_out[[paste0(pfx, "_q10")]]  <- NA_real_
      row_out[[paste0(pfx, "_q90")]]  <- NA_real_
      row_out[[paste0(pfx, "_q975")]] <- NA_real_
    }
  }

  dif_out <- list()
  if (sp$use_var_slopes == 1L) {
    for (k in seq_along(sp$items)) {
      alp_v <- dr[[sprintf("alpha[%d]", k)]]
      lam_v <- dr[[sprintf("lambda[%d]", k)]]
      bet_v <- dr[[sprintf("beta_item[%d]", k)]]
      del_v <- dr[[sprintf("delta_item_b[%d]", k)]]
      dif_out[[length(dif_out) + 1L]] <- data.frame(
        name = sp$name, domain = sp$domain, item_id = sp$items[k], item_idx = k,
        alpha_mean = mean(alp_v), alpha_q025 = unname(quantile(alp_v, 0.025)), alpha_q975 = unname(quantile(alp_v, 0.975)),
        lambda_mean = mean(lam_v), lambda_q025 = unname(quantile(lam_v, 0.025)),
        lambda_q10 = unname(quantile(lam_v, 0.10)), lambda_q90 = unname(quantile(lam_v, 0.90)),
        lambda_q975 = unname(quantile(lam_v, 0.975)),
        beta_mean = mean(bet_v), beta_q025 = unname(quantile(bet_v, 0.025)),
        beta_q10 = unname(quantile(bet_v, 0.10)), beta_q90 = unname(quantile(bet_v, 0.90)),
        beta_q975 = unname(quantile(bet_v, 0.975)),
        beta_p_pos = mean(bet_v > 0.0), beta_p_neg = mean(bet_v < 0.0),
        delta_mean = mean(del_v), delta_q025 = unname(quantile(del_v, 0.025)),
        delta_q10 = unname(quantile(del_v, 0.10)), delta_q90 = unname(quantile(del_v, 0.90)),
        delta_q975 = unname(quantile(del_v, 0.975)),
        stringsAsFactors = FALSE
      )
    }
  }

  eap_df <- NULL
  if (identical(sp$arch, "MA_2pl") && identical(sp$spec, "maximal") &&
      identical(sp$set, "main") && identical(sp$min_n_ind, 3L)) {
    th_cols <- sprintf("theta[%d]", seq_along(sp$glottocodes))
    th0_cols <- sprintf("theta_0[%d]", seq_along(sp$glottocodes))
    th_mat <- fit$draws("theta", format = "matrix")[, th_cols, drop = FALSE]
    th0_mat <- fit$draws("theta_0", format = "matrix")[, th0_cols, drop = FALSE]
    eap_df <- data.frame(glottocode = sp$glottocodes, stringsAsFactors = FALSE)
    eap_df[[paste0("theta_", sp$domain, "_eap")]]  <- as.numeric(colMeans(th_mat))
    eap_df[[paste0("theta_", sp$domain, "_sd")]]   <- as.numeric(apply(th_mat, 2, sd))
    eap_df[[paste0("theta0_", sp$domain, "_eap")]] <- as.numeric(colMeans(th0_mat))
  }

  list(row_1f = row_out, dif_rows = dif_out, eap_df = eap_df)
}

run_single_2f <- function(sp) {
  t0 <- Sys.time()
  rds_p <- file.path(fits_dir, paste0(sp$name, ".rds"))
  mod_use <- if (exists("w_mod_2f", envir = .GlobalEnv, inherits = FALSE)) {
    get("w_mod_2f", envir = .GlobalEnv)
  } else {
    mod_2f
  }
  fit <- tryCatch(
    fit_or_load_cmdstan(mod_use, sp$stan_data, rds_p, adapt_delta = 0.99),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(name = sp$name, error = conditionMessage(fit)))
  }

  dr <- fit$draws(format = "df")
  key_vars <- intersect(
    c("b_breadth", "b_irreg", "b_diff", "rho_latent", "rho_theta0",
      "sd_lang[1]", "sd_lang[2]",
      "sd_phylo[1]", "sd_phylo[2]",
      "sd_gp[1]", "sd_gp[2]",
      "lscale_gp[1]", "lscale_gp[2]",
      grep("^b_cov\\[", names(dr), value = TRUE)),
    names(dr)
  )
  sm_key <- summarise_draws(fit$draws(variables = key_vars))
  diag_sm <- fit$diagnostic_summary(quiet = TRUE)
  ll_tip <- fit$draws("log_lik_lang_tip", format = "matrix")
  loo_res <- tryCatch(loo::loo(ll_tip), error = function(e) NULL)

  idx_bb <- match("b_breadth", sm_key$variable)
  idx_bi <- match("b_irreg", sm_key$variable)
  idx_bd <- match("b_diff", sm_key$variable)

  c(
    list(
      name = sp$name, pos_pair = sp$pos_pair, dom1 = sp$dom1, dom2 = sp$dom2,
      spec = sp$spec, set = sp$set, min_n_ind = sp$min_n_ind,
      n_lang = sp$n_lang, n_lang_f1 = sp$n_lang_f1, n_lang_f2 = sp$n_lang_f2,
      n_obs = sp$n_obs,
      max_rhat = max(sm_key$rhat, na.rm = TRUE),
      min_ess_bulk = min(sm_key$ess_bulk, na.rm = TRUE),
      divergences = sum(diag_sm$num_divergent),
      elpd_loo_tip = if (!is.null(loo_res)) loo_res$estimates["elpd_loo", "Estimate"] else NA_real_,
      pareto_k_gt_07 = if (!is.null(loo_res)) sum(loo_res$diagnostics$pareto_k > 0.7) else NA_integer_,
      rho_latent_mean = mean(dr$rho_latent),
      rho_latent_q025 = unname(quantile(dr$rho_latent, 0.025)),
      rho_latent_q10  = unname(quantile(dr$rho_latent, 0.10)),
      rho_latent_q90  = unname(quantile(dr$rho_latent, 0.90)),
      rho_latent_q975 = unname(quantile(dr$rho_latent, 0.975)),
      rho_theta0_mean = mean(dr$rho_theta0),
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ),
    summarise_slope_metrics(
      dr$b_breadth, pfx = "b_breadth",
      b_rhat = if (!is.na(idx_bb)) sm_key$rhat[idx_bb] else NA_real_,
      b_ess  = if (!is.na(idx_bb)) sm_key$ess_bulk[idx_bb] else NA_real_
    ),
    summarise_slope_metrics(
      dr$b_irreg, pfx = "b_irreg",
      b_rhat = if (!is.na(idx_bi)) sm_key$rhat[idx_bi] else NA_real_,
      b_ess  = if (!is.na(idx_bi)) sm_key$ess_bulk[idx_bi] else NA_real_
    ),
    summarise_slope_metrics(
      dr$b_diff, pfx = "b_diff",
      b_rhat = if (!is.na(idx_bd)) sm_key$rhat[idx_bd] else NA_real_,
      b_ess  = if (!is.na(idx_bd)) sm_key$ess_bulk[idx_bd] else NA_real_
    )
  )
}

run_sem_psock_batch <- function(jobs, worker_fn, load_1f = TRUE, load_2f = FALSE) {
  if (length(jobs) == 0L) return(list())
  if (PAR_FITS <= 1L || length(jobs) <= 1L) {
    return(lapply(jobs, worker_fn))
  }
  n_w <- min(length(jobs), PAR_FITS)
  cl <- makePSOCKcluster(n_w, outfile = "")
  tryCatch({
    export_syms <- c(
      "smoke", "CHAINS", "THREADS", "ITER_WARMUP", "ITER_SAMPLING", "SEED",
      "PRIOR_SD_B", "PRIOR_DENS_ZERO", "ROPE_HALF_WIDTH",
      "fits_dir", "stan_bin_dir", "stan_1f_path", "stan_2f_path", "cpp_opts",
      "kde_density_at_zero", "summarise_slope_metrics",
      "fit_or_load_cmdstan", "run_single_1f", "run_single_2f"
    )
    clusterExport(cl, export_syms, envir = .GlobalEnv)
    clusterExport(cl, c("load_1f", "load_2f"), envir = environment())
    clusterEvalQ(cl, {
      Sys.setenv(
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
        OMP_NUM_THREADS = "1",
        MKL_NUM_THREADS = "1",
        OPENBLAS_NUM_THREADS = "1",
        VECLIB_MAXIMUM_THREADS = "1"
      )
      suppressPackageStartupMessages({
        library(cmdstanr); library(loo); library(posterior)
      })
      if (isTRUE(load_1f)) {
        w_mod_1f <<- cmdstan_model(stan_1f_path, dir = stan_bin_dir, cpp_options = cpp_opts)
      }
      if (isTRUE(load_2f)) {
        w_mod_2f <<- cmdstan_model(stan_2f_path, dir = stan_bin_dir, cpp_options = cpp_opts)
      }
      NULL
    })
    parLapplyLB(cl, jobs, worker_fn)
  }, finally = {
    try(stopCluster(cl), silent = TRUE)
  })
}

bind_sem_rows <- function(rows_list) {
  all_cols <- unique(unlist(lapply(rows_list, names)))
  do.call(rbind, lapply(rows_list, function(r) {
    as.data.frame(
      lapply(setNames(all_cols, all_cols), function(k) if (length(r[[k]])) r[[k]] else NA),
      stringsAsFactors = FALSE
    )
  }))
}

write_1f_outputs <- function(res_1f_list) {
  rows_1f <- lapply(res_1f_list, `[[`, "row_1f")
  df_1f <- bind_sem_rows(rows_1f)
  write.csv(df_1f, file.path(res_dir, paste0("sem_1factor_results", tag, ".csv")), row.names = FALSE)

  dif_rows <- unlist(lapply(res_1f_list, `[[`, "dif_rows"), recursive = FALSE)
  if (length(dif_rows)) {
    write.csv(do.call(rbind, dif_rows), file.path(res_dir, paste0("sem_item_dif", tag, ".csv")), row.names = FALSE)
  }

  eap_parts <- Filter(Negate(is.null), lapply(res_1f_list, `[[`, "eap_df"))
  if (length(eap_parts)) {
    eap_all <- data.frame(glottocode = sort(unique(d0$glottocode)), stringsAsFactors = FALSE)
    for (ep in eap_parts) {
      eap_all <- merge(eap_all, ep, by = "glottocode", all.x = TRUE, sort = FALSE)
    }
    eap_all <- eap_all[order(eap_all$glottocode), , drop = FALSE]
    write.csv(eap_all, file.path(res_dir, paste0("sem_latent_eap", tag, ".csv")), row.names = FALSE)
  }
  df_1f
}

is_core_spec <- function(sp) {
  identical(sp$block, "block1") ||
    identical(sp$spec, "cutoff_n5") ||
    identical(sp$block, "dif")
}

core_idx <- which(vapply(specs_1f, is_core_spec, logical(1)))
si_idx <- setdiff(seq_along(specs_1f), core_idx)

message(sprintf(
  "Running SEM (%s, cores=%d, chains=%d, threads=%d, par_fits=%d): %d core + %d sensitivity 1-factor specs, %d 2-factor specs.",
  if (smoke) "SMOKE" else "FULL", CORES, CHAINS, THREADS, PAR_FITS,
  length(core_idx), length(si_idx), length(specs_2f)
))

# --- Core 1-Factor & 2-Factor Models (Block 1, cutoff_n5, DIF, and within-POS 2-factor) ------
res_core_1f <- run_sem_psock_batch(specs_1f[core_idx], run_single_1f, load_1f = TRUE, load_2f = FALSE)
errs_core <- Filter(function(r) !is.list(r) || !is.null(r$error), res_core_1f)
if (length(errs_core)) {
  for (e in errs_core) message("ERROR (core 1f): ", e$name, " -> ", e$error)
  stop(sprintf("%d core 1-factor SEM model(s) failed.", length(errs_core)))
}
df_core_1f <- write_1f_outputs(res_core_1f)
message("Wrote ", nrow(df_core_1f), " core 1-factor rows + DIF + EAP to ", res_dir)

if (length(specs_2f)) {
  res_2f <- run_sem_psock_batch(specs_2f, run_single_2f, load_1f = FALSE, load_2f = TRUE)
  errs_2f <- Filter(function(r) !is.list(r) || !is.null(r$error), res_2f)
  if (length(errs_2f)) {
    for (e in errs_2f) message("ERROR (2f): ", e$name, " -> ", e$error)
    stop(sprintf("%d 2-factor SEM model(s) failed.", length(errs_2f)))
  }
  df_2f <- bind_sem_rows(res_2f)
  write.csv(df_2f, file.path(res_dir, paste0("sem_2factor_pos_results", tag, ".csv")), row.names = FALSE)
  message("Wrote ", nrow(df_2f), " 2-factor POS rows to ", res_dir)
}

# --- Sensitivity 1-Factor Models (Sample Scope & Phylogeny Checks) ----------------------------
res_si_1f <- run_sem_psock_batch(specs_1f[si_idx], run_single_1f, load_1f = TRUE, load_2f = FALSE)
errs_si <- Filter(function(r) !is.list(r) || !is.null(r$error), res_si_1f)
if (length(errs_si)) {
  for (e in errs_si) message("ERROR (sensitivity 1f): ", e$name, " -> ", e$error)
  stop(sprintf("%d sensitivity 1-factor SEM model(s) failed.", length(errs_si)))
}

res_all_1f <- vector("list", length(specs_1f))
res_all_1f[core_idx] <- res_core_1f
res_all_1f[si_idx] <- res_si_1f
df_final_1f <- write_1f_outputs(res_all_1f)
message("Done writing Track 1 Stan 2PL IRT-SEM results (", nrow(df_final_1f), " 1-factor rows) to: ", res_dir)
