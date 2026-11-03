# Step 2 Unifying Synthesis Checks for Track 1 (GeLaTo Admixture -> Grambank Morphology):
#   - Check 1.1:  Family-varying admixture slopes `(1 + x | family)` in the Latent SEM
#                 (`MA_2pl`, `MB_2pl` on `cat12`; `MC_cross` on `cat12_irreg13` & `cat12_flex26`)
#                 and Beta-Binomial baseline (`brms`, both centered `x_within` and uncentered `x`).
#   - Check 1.1b: Within- vs. Between-Family Mundlak decomposition (`x_within` + `x_between`)
#                 in the Latent SEM (`MA_2pl`, `MB_2pl`, `MC_cross`).
#   - Check 1.2:  Leave-Out-Inner-Asia sensitivity checks (`no_turk_mong` dropping Turkic +
#                 Mongolic-Khitan; `no_inner_asia` dropping Turkic + Mongolic-Khitan + Tungusic)
#                 in both the Latent SEM and Beta-Binomial baseline.
#   - Check 2:    Time-calibrated Bouckaert v2.0.0 within-family covariance (`P_b_stem`
#                 block-diagonal global-stem, `P_b_crown` unit crown-MRCA + `(1 | family)`, and
#                 `P_b_kyr` absolute-kyr crown-MRCA + `(1 | family)`) across `cat12` (N = 202),
#                 `cat12_irreg13` (N = 175), and `cat12_flex26` (N = 186).
#   - Check 3.1:  Speaker population size (`z_pop`) and interaction (`x * z_pop`) inside
#                 `MB_2pl` (`cat12`, N = 195) and `MC_cross` (`cat12_irreg13`, N = 171;
#                 `cat12_flex26`, N = 181) with matched `has_pop` baselines.
#   - Check 3.2:  Spatial Gaussian Process (`F`: Phylogeny + Exact Spatial GP) on the core
#                 winning SEM architectures (`M0_1pl`, `MA0_2pl`, `MA_2pl`, `MB_2pl` on `cat12`;
#                 `MC0_cross`, `MC_cross` on `cat12_irreg13` and `cat12_flex26`).
#
# Outputs written to `analyses/grambank/latent_sem/results/`:
#   - `synthesis_checks_main.csv`: all SEM & Beta-Binomial synthesis model summaries (with variance
#                                  components, diagnostics, per-SD slopes, and raw-unit slopes)
#   - `family_slopes_main.csv`:    family-specific posterior slopes beta_f across all multi-language families
#   - `loo_synthesis_main.csv`:    paired PSIS-LOO comparisons (obs, lang, and 31x31/21-pt tip-marginalized
#                                  lang_tip, with language-level SEs, family-clustered SEs, and Pareto-k counts)
#
# Run (from repo root):
#   SYN_DRY=1 Rscript analyses/grambank/latent_sem/fit_synthesis.R
#   SYN_PAR_FITS=4 SYN_MAX_CORES=16 Rscript analyses/grambank/latent_sem/fit_synthesis.R

Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")
invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))
suppressPackageStartupMessages({
  library(brms); library(ape); library(loo); library(posterior); library(parallel); library(tools)
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
adm_dir <- "analyses/grambank/baseline"
common_data_dir <- "analyses/common/data"
source("analyses/common/phylogeny.R")
source("analyses/common/stage1_exposure.R")

dry <- nzchar(Sys.getenv("SYN_DRY"))
smoke <- nzchar(Sys.getenv("SYN_SMOKE"))
use_testlets <- 1L

real_fits_dir <- file.path(here, if (smoke) "fits_smoke" else "fits")
real_bb_fits_dir <- file.path(adm_dir, if (smoke) "fits_smoke" else "fits")
fits_dir <- if (dry) tempdir() else real_fits_dir
res_dir <- if (dry) tempdir() else file.path(here, if (smoke) "results_smoke" else "results")
bb_fits_dir <- if (dry) tempdir() else real_bb_fits_dir
stan_cache_dir <- file.path(adm_dir, "stan_cache")
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(bb_fits_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(stan_cache_dir, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = stan_cache_dir)
}

if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- 2L; PAR_FITS <- as.integer(Sys.getenv("SYN_PAR_FITS", "2"))
} else {
  ITER <- 2000L; WARMUP <- 1000L; CHAINS <- 4L; PAR_FITS <- as.integer(Sys.getenv("SYN_PAR_FITS", "4"))
}
max_cores <- as.integer(Sys.getenv("SYN_MAX_CORES", "16"))
stopifnot(PAR_FITS * CHAINS <= max_cores)
SEED <- 20261030L
BB_SEED <- 20261021L
BB_ITER <- if (smoke) 400L else 3000L
BB_WARMUP <- if (smoke) 200L else 1500L

measures <- strsplit(Sys.getenv("SYN_MEASURES", "ent,sqrt_het,f3k_est_pos"), ",")[[1]]
checks_active <- strsplit(Sys.getenv("SYN_CHECKS", "c1_famslope,c1_mundlak,c1_inner_asia,c2_bouckaert,c3_demog,c3_space_F,c1_bb"), ",")[[1]]

# --- 1. Load canonical data tables & phylogenetic correlation matrices -----------------------
langs_df <- read.csv(file.path(data_dir, "language_table.csv"), stringsAsFactors = FALSE)
ld_full <- read.csv(file.path(data_dir, "language_data.csv"), stringsAsFactors = FALSE)
demog_cols <- c("iso_639_3", "population_l1", "population_source", "area_km2", "area_source",
                "log_pop", "log_area", "has_pop", "has_area")
for (dc in intersect(demog_cols, names(ld_full))) {
  langs_df[[dc]] <- ld_full[[dc]][match(langs_df$glottocode, ld_full$glottocode)]
}
items_df <- read.csv(file.path(data_dir, "items_metadata.csv"), stringsAsFactors = FALSE)
obs_df <- read.csv(file.path(data_dir, "item_observations.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(data_dir, "populations.csv"), stringsAsFactors = FALSE)

k5_file <- file.path(common_data_dir, "f3_scores_k5.csv")
if (file.exists(k5_file)) {
  k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
  idx_k5 <- match(pop0$population, k5$population)
  pop0$f3k_z <- -k5$min_z[idx_k5]
  pop0$f3k_est <- -k5$min_est[idx_k5]
} else {
  pop0$f3k_z <- pop0$f3k_est <- NA_real_
}
pop0$f3k_est_pos <- pmax(0, pop0$f3k_est)

to_rad <- pi / 180
langs_df$sx <- cos(langs_df$lat * to_rad) * cos(langs_df$lon * to_rad)
langs_df$sy <- cos(langs_df$lat * to_rad) * sin(langs_df$lon * to_rad)
langs_df$sz <- sin(langs_df$lat * to_rad)
langs_df$fam_eff <- ifelse(is.na(langs_df$family) | !nzchar(langs_df$family) | langs_df$family == "isolate",
                           paste0("isolate__", langs_df$glottocode), langs_df$family)

d_main_202 <- langs_df[langs_df$in_main == 1 & langs_df$n_coded >= 8, ]
d_main_202 <- d_main_202[order(d_main_202$glottocode), ]
fam_counts_202 <- table(d_main_202$fam_eff)
min5_fams <- names(fam_counts_202[fam_counts_202 >= 5])
min3_fams <- names(fam_counts_202[fam_counts_202 >= 3])
min2_fams <- names(fam_counts_202[fam_counts_202 >= 2])

# Four phylogenetic covariance/correlation matrices on the 202 main languages:
#   1. A_grafen:  Glottolog 5.3 topology with Grafen branch lengths
#   2. A_b_stem:  Bouckaert v2.0.0 block-diagonal global-stem correlation matrix (kyr)
#   3. A_b_crown: Bouckaert v2.0.0 block-diagonal unit crown-MRCA correlation matrix (paired with (1 | family))
#   4. A_b_kyr:   Bouckaert v2.0.0 block-diagonal absolute-kyr crown-MRCA covariance matrix (paired with (1 | family))
raw_tree <- read.tree(file.path(data_dir, "phylo_tree.nwk"))
A_grafen <- build_grafen_corr(raw_tree, d_main_202$glottocode)
bouckaert_tree <- load_bouckaert_tree()
A_b_stem <- build_bouckaert_corr(d_main_202$glottocode, families = d_main_202$family,
                                 tree = bouckaert_tree, within_family = TRUE, crown_only = FALSE)
A_b_crown <- build_bouckaert_corr(d_main_202$glottocode, families = d_main_202$family,
                                  tree = bouckaert_tree, within_family = TRUE, crown_only = TRUE, scale_by_age = FALSE)
A_b_kyr <- build_bouckaert_corr(d_main_202$glottocode, families = d_main_202$family,
                                tree = bouckaert_tree, within_family = TRUE, crown_only = TRUE, scale_by_age = TRUE)

load_stage1_exposures <- function(set = "main") {
  adm_res <- file.path(adm_dir, "results")
  files <- c(
    file.path(adm_res, paste0("exposure_", set, ".csv")),
    file.path(adm_res, paste0("exposure_", set, "_f3check.csv")),
    file.path(adm_res, paste0("exposure_", set, "_tailcheck.csv"))
  )
  files <- files[file.exists(files)]
  if (!length(files)) stop("No saved Stage-1 exposure tables found for set: ", set)
  tabs <- lapply(files, read.csv, stringsAsFactors = FALSE)
  expo <- do.call(rbind, tabs)
  expo[!duplicated(expo[, c("measure", "glottocode")], fromLast = TRUE), ]
}
expo_main <- load_stage1_exposures("main")

subset_of <- function(m) {
  switch(m,
    neigh = "neigh",
    het = , sqrt_het = "het",
    f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
    f3b3_z = , f3b3_est = "f3b3",
    "all"
  )
}

select_languages <- function(df, set, iset, drop_fams = character(0), require_pop = FALSE) {
  sub <- df[df[[paste0("in_", set)]] == 1 & df$n_coded >= 8, , drop = FALSE]
  if (length(drop_fams)) {
    sub <- sub[!(sub$family %in% drop_fams), , drop = FALSE]
  }
  if (isTRUE(require_pop)) {
    sub <- sub[!is.na(sub$has_pop) & sub$has_pop == 1L, , drop = FALSE]
  }
  if (iset == "cat12_irreg13") {
    sub <- sub[sub$irreg13_coded >= 8, , drop = FALSE]
  } else if (iset == "cat12_flex26") {
    sub <- sub[sub$flex26_coded >= 16, , drop = FALSE]
  }
  sub[order(sub$glottocode), , drop = FALSE]
}

build_gp_inputs <- function(d_lang) {
  coords <- as.matrix(d_lang[, c("sx", "sy", "sz")])
  dmat <- as.matrix(dist(coords))
  max_d <- max(dmat)
  Xgp <- coords / max_d
  tmp <- d_lang
  tmp$y_dummy <- c(0L, rep(1L, nrow(tmp) - 1L))
  pr_df <- default_prior(y_dummy ~ 1 + gp(sx, sy, sz), data = tmp, family = bernoulli())
  ls_str <- pr_df$prior[pr_df$class == "lscale" & nzchar(pr_df$coef)][1]
  nums <- as.numeric(regmatches(ls_str, gregexpr("[0-9]+\\.[0-9]+|[0-9]+", ls_str))[[1]])
  stopifnot(length(nums) == 2L)
  list(
    Xgp = lapply(seq_len(nrow(Xgp)), function(i) as.numeric(Xgp[i, ])),
    shape = nums[1],
    scale = nums[2]
  )
}

get_phylo_matrix <- function(ctrl_code, langs) {
  if (ctrl_code %in% c("N", "S", "P", "F")) {
    return(A_grafen[langs, langs, drop = FALSE])
  } else if (ctrl_code == "P_b_stem") {
    return(A_b_stem[langs, langs, drop = FALSE])
  } else if (ctrl_code == "P_b_crown") {
    return(A_b_crown[langs, langs, drop = FALSE])
  } else if (ctrl_code == "P_b_kyr") {
    return(A_b_kyr[langs, langs, drop = FALSE])
  }
  stop("Unknown ctrl_code: ", ctrl_code)
}

build_standata_syn <- function(d_lang, iset, x_vec, arch, ctrl_code,
                               use_fam_slopes = 0L, use_mundlak = 0L,
                               with_pop = 0L, with_int_pop = 0L,
                               z_pop_vec = NULL, legacy_stan = FALSE) {
  langs <- d_lang$glottocode
  N_lang <- length(langs)
  lang_map <- setNames(seq_along(langs), langs)

  if (iset == "cat12") {
    im <- items_df[items_df$in_cat12 == 1, , drop = FALSE]
  } else if (iset == "cat12_irreg13") {
    im <- items_df[items_df$in_cat12 == 1 | items_df$in_irreg13 == 1, , drop = FALSE]
  } else if (iset == "cat12_flex26") {
    im <- items_df[items_df$in_cat12 == 1 | items_df$in_flex26 == 1, , drop = FALSE]
  }
  im <- im[order(im$item_order), , drop = FALSE]
  K_item <- nrow(im)
  item_map <- setNames(seq_len(K_item), im$item_id)

  dom_levels <- unique(im$domain)
  item_dom <- match(im$domain, dom_levels)
  dom_tab <- table(im$domain)
  multi_doms <- names(dom_tab[dom_tab >= 2])
  item_testlet <- ifelse(im$domain %in% multi_doms, match(im$domain, multi_doms), 0L)
  T_test <- length(multi_doms)

  ob <- obs_df[obs_df$glottocode %in% langs & obs_df$item_id %in% im$item_id, , drop = FALSE]
  ob$lang_idx <- unname(lang_map[ob$glottocode])
  ob$item_idx <- unname(item_map[ob$item_id])
  ob <- ob[order(ob$lang_idx, ob$item_idx), , drop = FALSE]
  stopifnot(nrow(ob) > 0, all(!is.na(ob$lang_idx)), all(!is.na(ob$item_idx)))

  A_sub <- get_phylo_matrix(ctrl_code, langs)
  L_phylo <- t(chol(A_sub))
  gp_in <- build_gp_inputs(d_lang)

  use_phylo <- as.integer(ctrl_code %in% c("P", "F", "P_b_stem", "P_b_crown", "P_b_kyr"))
  use_space <- as.integer(ctrl_code %in% c("S", "F"))
  use_fam_int <- as.integer(ctrl_code %in% c("P_b_crown", "P_b_kyr"))

  fam_levels <- unique(d_lang$fam_eff)
  fam_id <- match(d_lang$fam_eff, fam_levels)
  fam_tab <- table(fam_id)
  fam_size <- as.integer(unname(fam_tab[as.character(fam_id)]))
  fam_means <- tapply(x_vec, fam_id, mean)
  x_between <- as.numeric(unname(fam_means[as.character(fam_id)]))
  x_within <- as.numeric(x_vec - x_between)
  x_within[fam_size == 1L] <- 0.0

  if (is.null(z_pop_vec)) {
    z_pop_vec <- rep(0.0, N_lang)
  }

  meta <- list(
    items = im$item_id,
    domains = dom_levels,
    item_dom_names = im$domain,
    langs = langs,
    fam_eff = d_lang$fam_eff,
    fam_levels = fam_levels,
    fam_id = fam_id,
    fam_size_by_level = as.integer(unname(table(factor(d_lang$fam_eff, levels = fam_levels)))),
    obs_fam_id = fam_id[ob$lang_idx]
  )

  if (iset != "cat12") {
    K1 <- sum(im$in_cat12 == 1)
    K2 <- K_item - K1
    stopifnot(K1 == 12L, K2 %in% c(13L, 26L))
    for (t in seq_len(T_test)) {
      stopifnot(length(unique(im$in_cat12[item_testlet == t])) == 1L)
    }
  }

  if (legacy_stan) {
    if (iset == "cat12") {
      with_x <- as.integer(arch %in% c("M1_1pl", "MA_2pl", "MB_1pl", "MB_2pl"))
      use_2pl <- as.integer(arch %in% c("MA0_2pl", "MA_2pl", "MB_2pl"))
      use_var_slopes <- as.integer(arch %in% c("MB_1pl", "MB_2pl"))
      return(list(
        N_lang = N_lang, K_item = K_item, D_dom = length(dom_levels), T_test = T_test,
        N_obs = nrow(ob), lang_id = as.integer(ob$lang_idx), item_id = as.integer(ob$item_idx),
        Y = as.integer(ob$value), item_dom = as.integer(item_dom), item_testlet = as.integer(item_testlet),
        item_is_agree = as.integer(im$is_agreement), x = as.numeric(x_vec),
        with_x = with_x, use_2pl = use_2pl, use_var_slopes = use_var_slopes,
        use_phylo = use_phylo, use_space = use_space, use_testlets = use_testlets,
        L_phylo = L_phylo, Xgp = gp_in$Xgp, lscale_prior_shape = gp_in$shape,
        lscale_prior_scale = gp_in$scale, .meta = meta
      ))
    } else {
      K1 <- sum(im$in_cat12 == 1)
      K2 <- K_item - K1
      agree_pos <- which(im$is_agreement == 1)
      item_agree_idx <- integer(K_item)
      item_agree_idx[agree_pos] <- seq_along(agree_pos)
      with_x <- as.integer(arch %in% c("MC_strict", "MC_cross"))
      allow_agree_cross <- as.integer(arch %in% c("MC0_cross", "MC_cross"))
      meta$agree_items <- im$item_id[agree_pos]
      return(list(
        N_lang = N_lang, K1 = K1, K2 = K2, K_item = K_item, T_test = T_test,
        K_agree = length(agree_pos), N_obs = nrow(ob), lang_id = as.integer(ob$lang_idx),
        item_id = as.integer(ob$item_idx), Y = as.integer(ob$value),
        item_testlet = as.integer(item_testlet), item_agree_idx = as.integer(item_agree_idx),
        x = as.numeric(x_vec), with_x = with_x, use_2pl = 1L, allow_agree_cross = allow_agree_cross,
        use_phylo = use_phylo, use_space = use_space, use_testlets = use_testlets,
        L_phylo = L_phylo, Xgp = gp_in$Xgp, lscale_prior_shape = gp_in$shape,
        lscale_prior_scale = gp_in$scale, .meta = meta
      ))
    }
  }

  if (iset == "cat12") {
    with_x <- as.integer(arch %in% c("M1_1pl", "MA_2pl", "MB_1pl", "MB_2pl"))
    use_2pl <- as.integer(arch %in% c("MA0_2pl", "MA_2pl", "MB_2pl"))
    use_var_slopes <- as.integer(arch %in% c("MB_1pl", "MB_2pl"))
    list(
      N_lang = N_lang, K_item = K_item, D_dom = length(dom_levels), T_test = T_test,
      N_obs = nrow(ob), N_fam = length(fam_levels),
      lang_id = as.integer(ob$lang_idx), item_id = as.integer(ob$item_idx), Y = as.integer(ob$value),
      item_dom = as.integer(item_dom), item_testlet = as.integer(item_testlet),
      item_is_agree = as.integer(im$is_agreement),
      fam_id = as.integer(fam_id), fam_size = as.integer(fam_size),
      x = as.numeric(x_vec), x_within = as.numeric(x_within), x_between = as.numeric(x_between),
      z_pop = as.numeric(z_pop_vec),
      with_x = with_x, use_2pl = use_2pl, use_var_slopes = use_var_slopes,
      use_phylo = use_phylo, use_space = use_space, use_testlets = use_testlets,
      use_fam_int = use_fam_int, use_fam_slopes = as.integer(use_fam_slopes),
      use_mundlak = as.integer(use_mundlak),
      with_pop = as.integer(with_pop), with_int_pop = as.integer(with_int_pop),
      L_phylo = L_phylo, Xgp = gp_in$Xgp,
      lscale_prior_shape = gp_in$shape, lscale_prior_scale = gp_in$scale,
      .meta = meta
    )
  } else {
    K1 <- sum(im$in_cat12 == 1)
    K2 <- K_item - K1
    agree_pos <- which(im$is_agreement == 1)
    item_agree_idx <- integer(K_item)
    item_agree_idx[agree_pos] <- seq_along(agree_pos)
    with_x <- as.integer(arch %in% c("MC_strict", "MC_cross"))
    allow_agree_cross <- as.integer(arch %in% c("MC0_cross", "MC_cross"))
    meta$agree_items <- im$item_id[agree_pos]
    list(
      N_lang = N_lang, K1 = K1, K2 = K2, K_item = K_item, T_test = T_test,
      K_agree = length(agree_pos), N_obs = nrow(ob), N_fam = length(fam_levels),
      lang_id = as.integer(ob$lang_idx), item_id = as.integer(ob$item_idx), Y = as.integer(ob$value),
      item_testlet = as.integer(item_testlet), item_agree_idx = as.integer(item_agree_idx),
      fam_id = as.integer(fam_id), fam_size = as.integer(fam_size),
      x = as.numeric(x_vec), x_within = as.numeric(x_within), x_between = as.numeric(x_between),
      z_pop = as.numeric(z_pop_vec),
      with_x = with_x, use_2pl = 1L, allow_agree_cross = allow_agree_cross,
      use_phylo = use_phylo, use_space = use_space, use_testlets = use_testlets,
      use_fam_int = use_fam_int, use_fam_slopes = as.integer(use_fam_slopes),
      use_mundlak = as.integer(use_mundlak),
      with_pop = as.integer(with_pop), with_int_pop = as.integer(with_int_pop),
      L_phylo = L_phylo, Xgp = gp_in$Xgp,
      lscale_prior_shape = gp_in$shape, lscale_prior_scale = gp_in$scale,
      .meta = meta
    )
  }
}

# --- 2. Compile Stan models ------------------------------------------------------------------
stan_1f_leg_path <- file.path(here, "stan", "item_1factor.stan")
stan_2f_leg_path <- file.path(here, "stan", "item_2factor.stan")
stan_1f_syn_path <- file.path(here, "stan", "item_1factor_synthesis.stan")
stan_2f_syn_path <- file.path(here, "stan", "item_2factor_synthesis.stan")

stan_1f_leg_code <- readLines(stan_1f_leg_path, warn = FALSE)
stan_2f_leg_code <- readLines(stan_2f_leg_path, warn = FALSE)
stan_1f_syn_code <- readLines(stan_1f_syn_path, warn = FALSE)
stan_2f_syn_code <- readLines(stan_2f_syn_path, warn = FALSE)

compute_cache_hash <- function(sdata, stan_code) {
  clean_code <- trimws(sub("//.*$", "", stan_code))
  clean_code <- clean_code[nzchar(clean_code)]
  tf <- tempfile()
  on.exit(unlink(tf), add = TRUE)
  saveRDS(list(sdata = sdata, stan_code = clean_code, seed = SEED,
               iter = ITER, warmup = WARMUP, chains = CHAINS), file = tf)
  unname(md5sum(tf))
}

if (dry) {
  for (p in c(stan_1f_leg_path, stan_2f_leg_path, stan_1f_syn_path, stan_2f_syn_path)) {
    chk <- rstan::stanc(file = p)
    stopifnot(isTRUE(chk$status))
  }
  message("DRY RUN: Stan syntax check passed for all 4 Stan files.")
  exe_1f_leg <- exe_2f_leg <- exe_1f_syn <- exe_2f_syn <- NULL
} else {
  suppressPackageStartupMessages(library(cmdstanr))
  exe_1f_leg <- cmdstan_model(stan_1f_leg_path)$exe_file()
  exe_2f_leg <- cmdstan_model(stan_2f_leg_path)$exe_file()
  exe_1f_syn <- cmdstan_model(stan_1f_syn_path)$exe_file()
  exe_2f_syn <- cmdstan_model(stan_2f_syn_path)$exe_file()
}

par_run <- function(specs, fn) {
  if (!length(specs)) return(list())
  if (dry || PAR_FITS <= 1L || length(specs) == 1L) {
    res <- lapply(specs, fn)
  } else {
    cl <- makePSOCKcluster(min(length(specs), PAR_FITS), outfile = "")
    on.exit(stopCluster(cl), add = TRUE)
    clusterEvalQ(cl, {
      invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))
      suppressPackageStartupMessages({
        library(brms); library(ape); library(cmdstanr); library(loo); library(posterior); library(tools)
      })
      options(brms.backend = "cmdstanr")
      NULL
    })
    clusterExport(cl, ls(envir = .GlobalEnv), envir = .GlobalEnv)
    clusterExport(cl, ls(envir = parent.frame()), envir = parent.frame())
    res <- parLapplyLB(cl, specs, fn)
  }
  for (i in seq_along(res)) {
    if (!is.list(res[[i]])) {
      res[[i]] <- list(name = specs[[i]]$name, error = "worker died or returned non-list")
    }
  }
  res
}

is_err <- function(r) !is.list(r) || !is.null(r$error)
err_text <- function(r) if (is.list(r)) paste(r$name, r$error) else as.character(r)

summ_vec <- function(v, pfx) {
  if (is.null(v) || !length(v)) return(list())
  setNames(
    list(mean(v), sd(v), unname(quantile(v, 0.025)), unname(quantile(v, 0.05)),
         unname(quantile(v, 0.95)), unname(quantile(v, 0.975)), mean(v < 0), mean(v > 0)),
    c(paste0(pfx, "_mean"), paste0(pfx, "_sd"), paste0(pfx, "_q025"), paste0(pfx, "_q05"),
      paste0(pfx, "_q95"), paste0(pfx, "_q975"), paste0("p_", sub("^b_", "", pfx), "_neg"),
      paste0("p_", sub("^b_", "", pfx), "_pos"))
  )
}

summ_sd <- function(dr, col, pfx) {
  if (!col %in% names(dr)) return(list())
  v <- dr[[col]]
  setNames(
    list(mean(v), sd(v), unname(quantile(v, 0.025)), unname(quantile(v, 0.975))),
    c(paste0(pfx, "_mean"), paste0(pfx, "_sd"), paste0(pfx, "_q025"), paste0(pfx, "_q975"))
  )
}

add_raw_slopes <- function(row_out) {
  if (!is.null(row_out$x_scale) && !is.na(row_out$x_scale) && row_out$x_scale > 0) {
    xs <- row_out$x_scale
    if (!is.null(row_out$b_mean) && !is.na(row_out$b_mean)) {
      row_out$raw_b_mean <- row_out$b_mean / xs
      row_out$raw_b_q025 <- row_out$b_q025 / xs
      row_out$raw_b_q975 <- row_out$b_q975 / xs
    }
    if (!is.null(row_out$b_irreg_mean) && !is.na(row_out$b_irreg_mean)) {
      row_out$raw_b_irreg_mean <- row_out$b_irreg_mean / xs
      row_out$raw_b_irreg_q025 <- row_out$b_irreg_q025 / xs
      row_out$raw_b_irreg_q975 <- row_out$b_irreg_q975 / xs
    }
    if (!is.null(row_out$b_diff_mean) && !is.na(row_out$b_diff_mean)) {
      row_out$raw_b_diff_mean <- row_out$b_diff_mean / xs
      row_out$raw_b_diff_q025 <- row_out$b_diff_q025 / xs
      row_out$raw_b_diff_q975 <- row_out$b_diff_q975 / xs
    }
  }
  row_out
}

# --- 3. Fit a single SEM specification and extract summaries + LOO ---------------------------
run_sem_spec <- function(s) {
  tryCatch(run_sem_spec_inner(s), error = function(e) list(name = s$name, error = conditionMessage(e)))
}

run_sem_spec_inner <- function(s) {
  t0 <- Sys.time()
  sdata <- s$standata
  meta <- sdata$.meta
  sdata$.meta <- NULL
  is_leg <- isTRUE(s$legacy_stan)

  stan_code <- if (s$item_set == "cat12") {
    if (is_leg) stan_1f_leg_code else stan_1f_syn_code
  } else {
    if (is_leg) stan_2f_leg_code else stan_2f_syn_code
  }
  chash <- compute_cache_hash(sdata, stan_code)
  rds_check <- file.path(real_fits_dir, paste0(s$name, ".rds"))
  is_cached <- FALSE
  if (file.exists(rds_check)) {
    saved <- readRDS(rds_check)
    is_cached <- identical(saved$cache_hash, chash)
  }

  if (dry) {
    return(list(
      row = data.frame(
        name = s$name, check = s$check, set = s$set, item_set = s$item_set,
        arch = s$arch, measure = s$measure, ctrl_code = s$ctrl, dspec = s$dspec,
        n_lang = sdata$N_lang, n_items = sdata$K_item, n_obs = sdata$N_obs,
        cached = is_cached, stringsAsFactors = FALSE
      ),
      fam_rows = data.frame()
    ))
  }

  rds_file <- file.path(fits_dir, paste0(s$name, ".rds"))
  ctrl_label <- c(
    N = "none", P = "phylogeny", S = "space", F = "phylogeny+space",
    P_b_stem = "bouckaert_stem", P_b_crown = "bouckaert_crown_fam",
    P_b_kyr = "bouckaert_kyr_fam"
  )

  if (is_cached) {
    message(format(Sys.time(), "%H:%M"), " cached ", s$name)
    out <- saved$summary_out
    out$row$check <- s$check
    out$row$dspec <- s$dspec
    out$row$ctrl_code <- s$ctrl
    out$row$ctrl <- unname(ctrl_label[s$ctrl])
    out$row <- add_raw_slopes(out$row)
    if (is.null(out$fam_rows)) out$fam_rows <- data.frame()
    return(out)
  }

  exe_path <- if (s$item_set == "cat12") {
    if (is_leg) exe_1f_leg else exe_1f_syn
  } else {
    if (is_leg) exe_2f_leg else exe_2f_syn
  }
  mod <- cmdstan_model(exe_file = exe_path)
  adapt_d <- if (s$ctrl == "F") 0.99 else 0.98
  fit <- mod$sample(
    data = sdata,
    seed = SEED,
    chains = CHAINS,
    parallel_chains = CHAINS,
    iter_warmup = WARMUP,
    iter_sampling = ITER,
    adapt_delta = adapt_d,
    max_treedepth = 12L,
    refresh = 0,
    show_messages = FALSE,
    show_exceptions = FALSE
  )

  dr <- fit$draws(format = "df")
  keep_pars <- grep("^(mu_alpha|sd_|lscale_|b_|lambda|alpha|beta_|gamma_cross|rho_latent|rho_theta0)", names(dr), value = TRUE)
  sm <- summarise_draws(dr[, keep_pars, drop = FALSE])
  np <- nuts_params(fit)
  divs <- sum(np$Value[np$Parameter == "divergent__"])
  td_hits <- sum(np$Value[np$Parameter == "treedepth__"] >= 12L)
  sampler_df <- fit$sampler_diagnostics(format = "df")
  ebfmi_by_chain <- tapply(sampler_df$energy__, sampler_df$.chain, function(e) sum(diff(e)^2) / ((length(e) - 1) * var(e)))
  min_ebfmi <- min(ebfmi_by_chain, na.rm = TRUE)

  ll_obs <- fit$draws("log_lik_obs", format = "matrix")
  ll_lang <- fit$draws("log_lik_lang", format = "matrix")
  chain_ids <- rep(seq_len(CHAINS), each = ITER)
  r_eff_obs <- relative_eff(exp(ll_obs), chain_id = chain_ids)
  r_eff_lang <- relative_eff(exp(ll_lang), chain_id = chain_ids)
  loo_obs <- loo(ll_obs, r_eff = r_eff_obs)
  loo_lang <- loo(ll_lang, r_eff = r_eff_lang)

  loo_lang_tip <- NULL
  if (!is_leg) {
    ll_tip <- fit$draws("log_lik_lang_tip", format = "matrix")
    r_eff_tip <- relative_eff(exp(ll_tip), chain_id = chain_ids)
    loo_lang_tip <- loo(ll_tip, r_eff = r_eff_tip)
  }

  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  row_out <- list(
    name = s$name, check = s$check, set = s$set, item_set = s$item_set, arch = s$arch,
    measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]), ctrl_code = s$ctrl,
    dspec = s$dspec, subset = s$subset,
    n_lang = sdata$N_lang, n_items = sdata$K_item, n_obs = sdata$N_obs,
    max_rhat = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(sm$ess_tail, na.rm = TRUE),
    divergences = divs, treedepth_hits = td_hits, min_ebfmi = min_ebfmi, minutes = mins,
    elpd_loo_obs = loo_obs$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_obs = loo_obs$estimates["elpd_loo", "SE"],
    pareto_k_obs_gt_07 = sum(loo_obs$diagnostics$pareto_k > 0.7),
    max_pareto_k_obs = max(loo_obs$diagnostics$pareto_k),
    elpd_loo_lang = loo_lang$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_lang = loo_lang$estimates["elpd_loo", "SE"],
    pareto_k_lang_gt_07 = sum(loo_lang$diagnostics$pareto_k > 0.7),
    max_pareto_k_lang = max(loo_lang$diagnostics$pareto_k),
    elpd_loo_lang_tip = if (!is.null(loo_lang_tip)) loo_lang_tip$estimates["elpd_loo", "Estimate"] else NA_real_,
    se_elpd_loo_lang_tip = if (!is.null(loo_lang_tip)) loo_lang_tip$estimates["elpd_loo", "SE"] else NA_real_,
    pareto_k_lang_tip_gt_07 = if (!is.null(loo_lang_tip)) sum(loo_lang_tip$diagnostics$pareto_k > 0.7) else NA_integer_,
    max_pareto_k_lang_tip = if (!is.null(loo_lang_tip)) max(loo_lang_tip$diagnostics$pareto_k) else NA_real_
  )

  fam_rows <- list()

  if (s$item_set == "cat12") {
    # Save structural variance components
    row_out <- c(row_out,
      summ_sd(dr, "sd_lang", "sd_lang"),
      summ_sd(dr, "sd_phylo[1]", "sd_phylo"),
      summ_sd(dr, "sd_fam[1]", "sd_fam"),
      summ_sd(dr, "sd_gp[1]", "sd_gp"),
      summ_sd(dr, "lscale_gp[1]", "lscale_gp"),
      summ_sd(dr, "sd_testlet[1]", "sd_testlet"),
      summ_sd(dr, "sd_lambda[1]", "sd_lambda"),
      summ_sd(dr, "sd_dom_b[1]", "sd_dom_b"),
      summ_sd(dr, "sd_item_b[1]", "sd_item_b")
    )
    if (sdata$with_x == 1L) {
      row_out <- c(row_out, summ_vec(dr$b_x, "b"))
      if (!is_leg && sdata$use_mundlak == 1L) {
        row_out <- c(row_out,
          summ_vec(dr$b_between, "b_between"),
          summ_vec(dr$b_within_minus_between, "b_within_minus_between")
        )
      }
      if (sdata$use_var_slopes == 1L) {
        binh <- dr$b_inherent; bagr <- dr$b_agreement; bdiff <- dr$b_inherent_minus_agreement
        row_out <- c(row_out, list(
          b_inherent_mean = mean(binh), b_inherent_q025 = unname(quantile(binh, 0.025)), b_inherent_q975 = unname(quantile(binh, 0.975)),
          b_agree_mean = mean(bagr), b_agree_q025 = unname(quantile(bagr, 0.025)), b_agree_q975 = unname(quantile(bagr, 0.975)),
          b_diff_mean = mean(bdiff), b_diff_q025 = unname(quantile(bdiff, 0.025)), b_diff_q975 = unname(quantile(bdiff, 0.975)),
          p_diff_pos = mean(bdiff > 0), p_agree_neg = mean(bagr < 0)
        ))
      }
    }
    if (!is_leg && sdata$with_pop == 1L) {
      row_out <- c(row_out, summ_vec(dr$b_pop, "b_pop"))
    }
    if (!is_leg && sdata$with_int_pop == 1L) {
      row_out <- c(row_out, summ_vec(dr$b_x_pop, "b_x_pop"))
    }
    if (!is_leg && sdata$use_fam_slopes == 1L) {
      row_out <- c(row_out, summ_sd(dr, "sd_fam_b[1]", "sd_fam_b"))
      for (f_idx in seq_along(meta$fam_levels)) {
        fnm <- meta$fam_levels[f_idx]
        nf <- meta$fam_size_by_level[f_idx]
        if (nf >= 2L) {
          bf_draws <- dr[[sprintf("beta_fam[%d]", f_idx)]]
          df_draws <- dr[[sprintf("delta_fam_b[%d]", f_idx)]]
          x_f <- sdata$x[meta$fam_id == f_idx]
          fam_rows[[length(fam_rows) + 1L]] <- data.frame(
            model = s$name, framework = "latent_sem", item_set = s$item_set, arch = s$arch,
            measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]),
            family = fnm, n_langs_model = nf,
            n_langs_cat12 = as.integer(fam_counts_202[fnm]),
            in_min3 = as.integer(fnm %in% min3_fams),
            in_min5 = as.integer(fnm %in% min5_fams),
            x_fam_mean_std = mean(x_f), x_fam_sd_within = sd(x_f),
            factor = "breadth",
            beta_mean = mean(bf_draws), beta_sd = sd(bf_draws),
            beta_q025 = unname(quantile(bf_draws, 0.025)), beta_q05 = unname(quantile(bf_draws, 0.05)),
            beta_q95 = unname(quantile(bf_draws, 0.95)), beta_q975 = unname(quantile(bf_draws, 0.975)),
            p_beta_pos = mean(bf_draws > 0), p_beta_neg = mean(bf_draws < 0),
            delta_mean = mean(df_draws), delta_q025 = unname(quantile(df_draws, 0.025)), delta_q975 = unname(quantile(df_draws, 0.975)),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  } else {
    rho <- dr$rho_latent
    rho0 <- dr$rho_theta0
    row_out <- c(row_out,
      list(
        rho_latent_mean = mean(rho), rho_latent_q025 = unname(quantile(rho, 0.025)), rho_latent_q975 = unname(quantile(rho, 0.975)),
        rho_theta0_mean = mean(rho0), rho_theta0_q025 = unname(quantile(rho0, 0.025)), rho_theta0_q975 = unname(quantile(rho0, 0.975))
      ),
      summ_sd(dr, "sd_lang[1]", "sd_lang"),
      summ_sd(dr, "sd_lang[2]", "sd_lang_irreg"),
      summ_sd(dr, "sd_phylo[1]", "sd_phylo"),
      summ_sd(dr, "sd_phylo[2]", "sd_phylo_irreg"),
      summ_sd(dr, "sd_fam[1]", "sd_fam"),
      summ_sd(dr, "sd_fam[2]", "sd_fam_irreg"),
      summ_sd(dr, "sd_gp[1]", "sd_gp"),
      summ_sd(dr, "sd_gp[2]", "sd_gp_irreg"),
      summ_sd(dr, "lscale_gp[1]", "lscale_gp"),
      summ_sd(dr, "lscale_gp[2]", "lscale_gp_irreg"),
      summ_sd(dr, "sd_testlet[1]", "sd_testlet")
    )
    if (sdata$with_x == 1L) {
      b1 <- dr$b_breadth; b2 <- dr$b_irreg; bdiff <- dr$b_diff
      row_out <- c(row_out,
        summ_vec(b1, "b"),
        summ_vec(b2, "b_irreg"),
        list(
          b_diff_mean = mean(bdiff), b_diff_sd = sd(bdiff),
          b_diff_q025 = unname(quantile(bdiff, 0.025)), b_diff_q975 = unname(quantile(bdiff, 0.975)),
          p_diff_pos = mean(bdiff > 0)
        )
      )
      if (!is_leg && sdata$use_mundlak == 1L) {
        row_out <- c(row_out,
          summ_vec(dr$b_bet_breadth, "b_between"),
          summ_vec(dr$b_bet_irreg, "b_between_irreg"),
          summ_vec(dr$b_bet_diff, "b_between_diff")
        )
      }
    }
    if (sdata$allow_agree_cross == 1L) {
      g1 <- dr[["gamma_cross[1]"]]; g2 <- dr[["gamma_cross[2]"]]
      row_out <- c(row_out, list(
        gamma_cross_gender_mean = mean(g1), gamma_cross_gender_q025 = unname(quantile(g1, 0.025)), gamma_cross_gender_q975 = unname(quantile(g1, 0.975)),
        gamma_cross_number_mean = mean(g2), gamma_cross_number_q025 = unname(quantile(g2, 0.025)), gamma_cross_number_q975 = unname(quantile(g2, 0.975))
      ))
    }
    if (!is_leg && sdata$with_pop == 1L) {
      row_out <- c(row_out,
        summ_vec(dr$b_pop_breadth, "b_pop"),
        summ_vec(dr$b_pop_irreg, "b_pop_irreg")
      )
    }
    if (!is_leg && sdata$with_int_pop == 1L) {
      row_out <- c(row_out,
        summ_vec(dr$b_x_pop_breadth, "b_x_pop"),
        summ_vec(dr$b_x_pop_irreg, "b_x_pop_irreg")
      )
    }
    if (!is_leg && sdata$use_fam_slopes == 1L) {
      row_out <- c(row_out,
        summ_sd(dr, "sd_fam_b[1]", "sd_fam_b"),
        summ_sd(dr, "sd_fam_b[2]", "sd_fam_b_irreg")
      )
      for (f_idx in seq_along(meta$fam_levels)) {
        fnm <- meta$fam_levels[f_idx]
        nf <- meta$fam_size_by_level[f_idx]
        if (nf >= 2L) {
          x_f <- sdata$x[meta$fam_id == f_idx]
          fac_map <- list(
            breadth = list(b = dr[[sprintf("beta_fam_breadth[%d]", f_idx)]], d = dr[[sprintf("delta_fam_b1[%d]", f_idx)]]),
            irreg   = list(b = dr[[sprintf("beta_fam_irreg[%d]", f_idx)]],   d = dr[[sprintf("delta_fam_b2[%d]", f_idx)]]),
            diff    = list(b = dr[[sprintf("beta_fam_diff[%d]", f_idx)]],    d = dr[[sprintf("delta_fam_b1[%d]", f_idx)]] - dr[[sprintf("delta_fam_b2[%d]", f_idx)]])
          )
          for (fac_nm in names(fac_map)) {
            bf_draws <- fac_map[[fac_nm]]$b
            df_draws <- fac_map[[fac_nm]]$d
            fam_rows[[length(fam_rows) + 1L]] <- data.frame(
              model = s$name, framework = "latent_sem", item_set = s$item_set, arch = s$arch,
              measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]),
              family = fnm, n_langs_model = nf,
              n_langs_cat12 = as.integer(fam_counts_202[fnm]),
              in_min3 = as.integer(fnm %in% min3_fams),
              in_min5 = as.integer(fnm %in% min5_fams),
              x_fam_mean_std = mean(x_f), x_fam_sd_within = sd(x_f),
              factor = fac_nm,
              beta_mean = mean(bf_draws), beta_sd = sd(bf_draws),
              beta_q025 = unname(quantile(bf_draws, 0.025)), beta_q05 = unname(quantile(bf_draws, 0.05)),
              beta_q95 = unname(quantile(bf_draws, 0.95)), beta_q975 = unname(quantile(bf_draws, 0.975)),
              p_beta_pos = mean(bf_draws > 0), p_beta_neg = mean(bf_draws < 0),
              delta_mean = mean(df_draws), delta_q025 = unname(quantile(df_draws, 0.025)), delta_q975 = unname(quantile(df_draws, 0.975)),
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
  }

  if (!is.null(s$scale_info)) row_out <- c(row_out, s$scale_info)
  row_out <- add_raw_slopes(row_out)

  summary_out <- list(
    name = s$name,
    row = row_out,
    items = data.frame(),
    fam_rows = if (length(fam_rows)) do.call(rbind, fam_rows) else data.frame(),
    param_summary = sm
  )
  saveRDS(list(
    cache_hash = chash, summary_out = summary_out,
    loo_obs = loo_obs, loo_lang = loo_lang, loo_lang_tip = loo_lang_tip,
    meta = list(fam_id = meta$fam_id, obs_fam_id = meta$obs_fam_id, langs = meta$langs)
  ), rds_file)
  message(format(Sys.time(), "%H:%M"), " done ", s$name, sprintf(" (%.1f min, Rhat=%.3f, div=%d)", mins, row_out$max_rhat, divs))
  summary_out
}

# --- 4. Build all SEM synthesis specifications -----------------------------------------------
sem_specs <- list()

add_sem_group <- function(check_name, set_label, drop_fams = character(0), require_pop = FALSE,
                          iset_list = c("cat12", "cat12_irreg13", "cat12_flex26"),
                          base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
                          base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
                          ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 0L,
                          dspec_mode = "standard", legacy_stan = FALSE, name_prefix = "syn") {
  for (iset in iset_list) {
    d_set <- select_languages(langs_df, "main", iset, drop_fams = drop_fams, require_pop = require_pop)
    inset <- pop0[pop0$in_main == 1 & pop0$glottocode %in% d_set$glottocode, ]
    has_col <- function(col) intersect(d_set$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
    subs <- list(
      all = d_set$glottocode,
      het = has_col("het"),
      f3  = has_col("f3_z")
    )
    active_subs <- unique(vapply(measures, subset_of, ""))
    subs <- subs[intersect(names(subs), active_subs)]

    b_archs <- if (iset == "cat12") base_archs_1f else base_archs_2f
    x_archs <- if (iset == "cat12") x_archs_1f else x_archs_2f

    sub_zpop <- list(); sub_pop_scale <- list()
    for (sb in names(subs)) {
      d_sb <- d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE]
      d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
      if (require_pop) {
        lp <- as.numeric(d_sb$log_pop)
        pmu <- mean(lp); psd <- sd(lp)
        sub_zpop[[sb]] <- (lp - pmu) / psd
        sub_pop_scale[[sb]] <- list(pop_center = pmu, pop_scale = psd)
      } else {
        sub_zpop[[sb]] <- rep(0.0, nrow(d_sb))
        sub_pop_scale[[sb]] <- list()
      }
    }

    # Baselines (no-x)
    for (sb in names(subs)) {
      d_sb <- d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE]
      d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
      x_zero <- rep(0.0, nrow(d_sb))
      base_dspecs <- if (dspec_mode == "demog") c("base", "pop") else "base"
      for (ctrl in ctrl_list) for (arch in b_archs) for (bds in base_dspecs) {
        w_pop <- as.integer(bds == "pop")
        nm <- if (legacy_stan) {
          paste0(iset, "__", set_label, "__", arch, "__base_", ctrl, "_", sb)
        } else {
          paste0(name_prefix, "__", iset, "__", set_label, "__", arch, "__", bds, "_", ctrl, "_", sb)
        }
        sem_specs[[length(sem_specs) + 1L]] <<- list(
          name = nm, check = check_name, set = set_label, item_set = iset, arch = arch,
          measure = "none", ctrl = ctrl, dspec = bds, subset = sb, legacy_stan = legacy_stan,
          standata = build_standata_syn(
            d_sb, iset, x_zero, arch, ctrl,
            use_fam_slopes = 0L, use_mundlak = 0L, with_pop = w_pop, with_int_pop = 0L,
            z_pop_vec = sub_zpop[[sb]], legacy_stan = legacy_stan
          ),
          scale_info = sub_pop_scale[[sb]]
        )
      }
    }

    # With-x models
    for (m in measures) {
      sb <- subset_of(m)
      e_m <- expo_main[expo_main$measure == m & expo_main$glottocode %in% subs[[sb]], , drop = FALSE]
      stopifnot(setequal(e_m$glottocode, subs[[sb]]))
      d_sb <- merge(d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE],
                    e_m[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
                    by = "glottocode")
      d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
      mu <- mean(d_sb$eta_mean); s <- sd(d_sb$eta_mean)
      x_std <- (d_sb$eta_mean - mu) / s
      scale_info <- c(list(x_center = mu, x_scale = s, y_mean = d_sb$y_mean[1], y_sd = d_sb$y_sd[1]),
                      sub_pop_scale[[sb]])

      x_dspecs <- if (dspec_mode == "demog") {
        c("x", "x_pop", "x_int_pop")
      } else if (use_fam_slopes == 1L) {
        "famslope"
      } else if (use_mundlak == 1L) {
        "mundlak"
      } else {
        "x"
      }

      for (ctrl in ctrl_list) for (arch in x_archs) for (xds in x_dspecs) {
        w_pop <- as.integer(xds %in% c("x_pop", "x_int_pop"))
        w_int <- as.integer(xds == "x_int_pop")
        nm <- if (legacy_stan) {
          paste0(iset, "__", set_label, "__", arch, "__", m, "_", ctrl)
        } else {
          paste0(name_prefix, "__", iset, "__", set_label, "__", arch, "__", m, "__", xds, "_", ctrl)
        }
        sem_specs[[length(sem_specs) + 1L]] <<- list(
          name = nm, check = check_name, set = set_label, item_set = iset, arch = arch,
          measure = m, ctrl = ctrl, dspec = xds, subset = sb, legacy_stan = legacy_stan,
          standata = build_standata_syn(
            d_sb, iset, x_std, arch, ctrl,
            use_fam_slopes = use_fam_slopes, use_mundlak = use_mundlak,
            with_pop = w_pop, with_int_pop = w_int,
            z_pop_vec = sub_zpop[[sb]], legacy_stan = legacy_stan
          ),
          scale_info = scale_info
        )
      }
    }
  }
}

# 0. Reference fixed-slope Grafen P models in synthesis Stan
if (any(c("c1_famslope", "c1_mundlak", "c2_bouckaert") %in% checks_active)) {
  add_sem_group(
    check_name = "ref_grafen_P", set_label = "main",
    base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 0L, dspec_mode = "standard",
    legacy_stan = FALSE, name_prefix = "syn_ref"
  )
}

# Check 1.1: Family-varying slopes in SEM (`MA_2pl`, `MB_2pl` on `cat12`; `MC_cross` on `cat12_irreg13`, `cat12_flex26`)
if ("c1_famslope" %in% checks_active) {
  add_sem_group(
    check_name = "check1_family_slopes", set_label = "main",
    base_archs_1f = character(0), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = character(0), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 1L, use_mundlak = 0L, dspec_mode = "standard",
    legacy_stan = FALSE, name_prefix = "syn_c1"
  )
}

# Check 1.1b: Mundlak within- vs. between-family decomposition (`x_within` + `x_between`)
if ("c1_mundlak" %in% checks_active) {
  add_sem_group(
    check_name = "check1_mundlak_wb", set_label = "main",
    base_archs_1f = character(0), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = character(0), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 1L, dspec_mode = "standard",
    legacy_stan = FALSE, name_prefix = "syn_c1"
  )
}

# Check 1.2: Leave-Out-Inner-Asia sensitivity checks (`no_turk_mong` and `no_inner_asia`)
if ("c1_inner_asia" %in% checks_active) {
  add_sem_group(
    check_name = "check1_no_turk_mong", set_label = "no_turk_mong",
    drop_fams = c("Turkic", "Mongolic-Khitan"),
    base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 0L, dspec_mode = "standard",
    legacy_stan = FALSE, name_prefix = "syn_c1"
  )
  add_sem_group(
    check_name = "check1_no_inner_asia", set_label = "no_inner_asia",
    drop_fams = c("Turkic", "Mongolic-Khitan", "Tungusic"),
    base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 0L, dspec_mode = "standard",
    legacy_stan = FALSE, name_prefix = "syn_c1"
  )
}

# Check 2: Time-calibrated Bouckaert v2.0.0 covariance (`P_b_stem`, `P_b_crown`, and `P_b_kyr`)
if ("c2_bouckaert" %in% checks_active) {
  add_sem_group(
    check_name = "check2_bouckaert", set_label = "main",
    base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = c("P_b_stem", "P_b_crown", "P_b_kyr"), use_fam_slopes = 0L, use_mundlak = 0L,
    dspec_mode = "standard", legacy_stan = FALSE, name_prefix = "syn_c2"
  )
}

# Check 3.1: Demographic controls (`z_pop` and `x * z_pop`) on `has_pop == 1` sample
if ("c3_demog" %in% checks_active) {
  add_sem_group(
    check_name = "check3_demog_pop", set_label = "main_has_pop",
    require_pop = TRUE,
    base_archs_1f = c("MA0_2pl"), x_archs_1f = c("MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = "P", use_fam_slopes = 0L, use_mundlak = 0L, dspec_mode = "demog",
    legacy_stan = FALSE, name_prefix = "syn_c3"
  )
}

# Check 3.2: Spatial GP (`F`: Phylogeny + Exact Spatial GP) + matched cached legacy `P` specs
if ("c3_space_F" %in% checks_active) {
  add_sem_group(
    check_name = "check3_space_F", set_label = "main",
    base_archs_1f = c("M0_1pl", "MA0_2pl"), x_archs_1f = c("MA_2pl", "MB_2pl"),
    base_archs_2f = c("MC0_cross"), x_archs_2f = c("MC_cross"),
    ctrl_list = c("P", "F"), use_fam_slopes = 0L, use_mundlak = 0L, dspec_mode = "standard",
    legacy_stan = TRUE, name_prefix = ""
  )
}

# --- 5. Build Beta-Binomial (`brms`) specifications for Check 1.1 & Check 1.2 ----------------
bb_specs <- list()
bb_outcomes <- list(
  cat12   = list(pres = "n_present",       coded = "n_coded",       min_coded = 8L),
  irreg13 = list(pres = "irreg13_present", coded = "irreg13_coded", min_coded = 8L),
  flex26  = list(pres = "flex26_present",  coded = "flex26_coded",  min_coded = 16L)
)

if ("c1_bb" %in% checks_active) {
  for (out_nm in names(bb_outcomes)) {
    osp <- bb_outcomes[[out_nm]]
    for (set_cfg in list(
      list(set_label = "main",          drop_fams = character(0),                        modes = c("base", "x", "famslope_centered", "famslope_uncentered")),
      list(set_label = "no_turk_mong",  drop_fams = c("Turkic", "Mongolic-Khitan"),      modes = c("base", "x")),
      list(set_label = "no_inner_asia", drop_fams = c("Turkic", "Mongolic-Khitan", "Tungusic"), modes = c("base", "x"))
    )) {
      set_lbl <- set_cfg$set_label
      d_sub <- langs_df[langs_df$in_main == 1 & langs_df[[osp$coded]] >= osp$min_coded, , drop = FALSE]
      if (length(set_cfg$drop_fams)) {
        d_sub <- d_sub[!(d_sub$family %in% set_cfg$drop_fams), , drop = FALSE]
      }
      d_sub <- d_sub[order(d_sub$glottocode), , drop = FALSE]
      d_sub$y_pres <- d_sub[[osp$pres]]
      d_sub$y_coded <- d_sub[[osp$coded]]
      d_sub$lang <- d_sub$glottocode

      inset <- pop0[pop0$in_main == 1 & pop0$glottocode %in% d_sub$glottocode, ]
      has_col <- function(col) intersect(d_sub$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
      subs <- list(all = d_sub$glottocode, het = has_col("het"), f3 = has_col("f3_z"))

      if ("base" %in% set_cfg$modes) {
        for (sb in names(subs)) {
          dd <- d_sub[d_sub$glottocode %in% subs[[sb]], , drop = FALSE]
          dd <- dd[order(dd$glottocode), , drop = FALSE]
          nm <- paste0("bb_syn__", out_nm, "__", set_lbl, "__base_P_", sb)
          bb_specs[[length(bb_specs) + 1L]] <- list(
            name = nm, check = paste0("bb_", set_lbl), set = set_lbl, item_set = paste0("bb_", out_nm),
            arch = "BetaBinom", measure = "none", ctrl = "P", dspec = "base", subset = sb,
            data = dd, formula = "y_pres | trials(y_coded) ~ 1 + (1 | gr(lang, cov = A))",
            with_x = FALSE, fam_mode = "none", scale_info = NULL
          )
        }
      }

      for (m in measures) {
        sb <- subset_of(m)
        e_m <- expo_main[expo_main$measure == m & expo_main$glottocode %in% subs[[sb]], , drop = FALSE]
        dd <- merge(d_sub[d_sub$glottocode %in% subs[[sb]], , drop = FALSE],
                    e_m[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
                    by = "glottocode")
        dd <- dd[order(dd$glottocode), , drop = FALSE]
        mu <- mean(dd$eta_mean); s <- sd(dd$eta_mean)
        dd$x <- (dd$eta_mean - mu) / s
        fam_means <- tapply(dd$x, dd$fam_eff, mean)
        dd$x_within <- as.numeric(dd$x - unname(fam_means[dd$fam_eff]))
        ftab <- table(dd$fam_eff)
        dd$x_within[unname(ftab[dd$fam_eff]) == 1L] <- 0.0
        scale_info <- list(x_center = mu, x_scale = s, y_mean = dd$y_mean[1], y_sd = dd$y_sd[1])

        for (md in setdiff(set_cfg$modes, "base")) {
          rhs_f <- switch(md,
            x                   = "1 + x + (1 | gr(lang, cov = A))",
            famslope_centered   = "1 + x + (0 + x_within | fam_eff) + (1 | gr(lang, cov = A))",
            famslope_uncentered = "1 + x + (0 + x | fam_eff) + (1 | gr(lang, cov = A))"
          )
          nm <- paste0("bb_syn__", out_nm, "__", set_lbl, "__", m, "__", md, "_P")
          bb_specs[[length(bb_specs) + 1L]] <- list(
            name = nm,
            check = if (grepl("^famslope", md)) "check1_bb_family_slopes" else paste0("check1_bb_", set_lbl),
            set = set_lbl, item_set = paste0("bb_", out_nm), arch = paste0("BetaBinom_", md),
            measure = m, ctrl = "P", dspec = md, subset = sb,
            data = dd, formula = paste0("y_pres | trials(y_coded) ~ ", rhs_f),
            with_x = TRUE, fam_mode = md, scale_info = scale_info
          )
        }
      }
    }
  }
}

run_bb_spec <- function(s) {
  tryCatch(run_bb_spec_inner(s), error = function(e) list(name = s$name, error = conditionMessage(e)))
}

run_bb_spec_inner <- function(s) {
  t0 <- Sys.time()
  A_sub <- A_grafen[s$data$lang, s$data$lang, drop = FALSE]
  pr <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(gamma(2, 0.1), class = phi),
    prior(exponential(1), class = sd, group = lang)
  )
  if (s$with_x) {
    pr <- c(pr, prior(normal(0, 0.5), class = b))
  }
  if (grepl("^famslope", s$fam_mode)) {
    pr <- c(pr, prior(exponential(2), class = sd, group = fam_eff))
  }

  if (dry) {
    make_stancode(bf(as.formula(s$formula)), data = s$data, data2 = list(A = A_sub), family = beta_binomial(), prior = pr)
    make_standata(bf(as.formula(s$formula)), data = s$data, data2 = list(A = A_sub), family = beta_binomial(), prior = pr)
    return(list(
      row = data.frame(
        name = s$name, check = s$check, set = s$set, item_set = s$item_set,
        arch = s$arch, measure = s$measure, ctrl_code = s$ctrl, dspec = s$dspec,
        n_lang = nrow(s$data), n_items = 1L, n_obs = nrow(s$data),
        cached = file.exists(file.path(real_bb_fits_dir, paste0(s$name, ".rds"))),
        stringsAsFactors = FALSE
      ),
      fam_rows = data.frame()
    ))
  }

  fit_file <- file.path(bb_fits_dir, s$name)
  fit <- brm(
    bf(as.formula(s$formula)), data = s$data, data2 = list(A = A_sub),
    family = beta_binomial(), prior = pr,
    chains = CHAINS, iter = BB_ITER, warmup = BB_WARMUP, cores = CHAINS,
    seed = BB_SEED, control = list(adapt_delta = 0.99, max_treedepth = 12),
    file = fit_file, file_refit = "on_change", refresh = 0, silent = 2
  )
  fit <- add_criterion(fit, "loo", file = fit_file)
  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)
  divs <- sum(subset(nuts_params(fit), Parameter == "divergent__")$Value)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  row_out <- list(
    name = s$name, check = s$check, set = s$set, item_set = s$item_set, arch = s$arch,
    measure = s$measure, ctrl = "phylogeny", ctrl_code = s$ctrl, dspec = s$dspec, subset = s$subset,
    n_lang = nrow(s$data), n_items = 1L, n_obs = nrow(s$data),
    max_rhat = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(sm$ess_tail, na.rm = TRUE),
    divergences = divs, minutes = mins,
    elpd_loo_obs = fit$criteria$loo$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_obs = fit$criteria$loo$estimates["elpd_loo", "SE"],
    pareto_k_obs_gt_07 = sum(fit$criteria$loo$diagnostics$pareto_k > 0.7),
    max_pareto_k_obs = max(fit$criteria$loo$diagnostics$pareto_k),
    elpd_loo_lang = fit$criteria$loo$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_lang = fit$criteria$loo$estimates["elpd_loo", "SE"],
    pareto_k_lang_gt_07 = sum(fit$criteria$loo$diagnostics$pareto_k > 0.7),
    max_pareto_k_lang = max(fit$criteria$loo$diagnostics$pareto_k),
    elpd_loo_lang_tip = NA_real_, se_elpd_loo_lang_tip = NA_real_,
    pareto_k_lang_tip_gt_07 = NA_integer_, max_pareto_k_lang_tip = NA_real_
  )

  fam_rows <- list()
  if (s$with_x) {
    row_out <- c(row_out, summ_vec(dr$b_x, "b"))
    if (grepl("^famslope", s$fam_mode)) {
      sd_col <- grep("^sd_fam_eff__", names(dr), value = TRUE)[1]
      sdfb <- dr[[sd_col]]
      row_out <- c(row_out, list(
        sd_fam_b_mean = mean(sdfb), sd_fam_b_sd = sd(sdfb),
        sd_fam_b_q025 = unname(quantile(sdfb, 0.025)), sd_fam_b_q975 = unname(quantile(sdfb, 0.975))
      ))
      r_term <- if (s$fam_mode == "famslope_centered") "x_within" else "x"
      ftab_model <- table(s$data$fam_eff)
      for (fnm in names(ftab_model[ftab_model >= 2L])) {
        r_col <- sprintf("r_fam_eff[%s,%s]", gsub(" ", ".", fnm), r_term)
        if (!r_col %in% names(dr)) {
          r_col <- sprintf("r_fam_eff[%s,%s]", fnm, r_term)
        }
        if (r_col %in% names(dr)) {
          df_draws <- dr[[r_col]]
          bf_draws <- dr$b_x + df_draws
          x_f <- s$data$x[s$data$fam_eff == fnm]
          fam_rows[[length(fam_rows) + 1L]] <- data.frame(
            model = s$name, framework = paste0("brms_", s$fam_mode), item_set = s$item_set, arch = s$arch,
            measure = s$measure, ctrl = "phylogeny",
            family = fnm, n_langs_model = as.integer(ftab_model[[fnm]]),
            n_langs_cat12 = as.integer(fam_counts_202[fnm]),
            in_min3 = as.integer(fnm %in% min3_fams),
            in_min5 = as.integer(fnm %in% min5_fams),
            x_fam_mean_std = mean(x_f), x_fam_sd_within = sd(x_f),
            factor = sub("^bb_", "", s$item_set),
            beta_mean = mean(bf_draws), beta_sd = sd(bf_draws),
            beta_q025 = unname(quantile(bf_draws, 0.025)), beta_q05 = unname(quantile(bf_draws, 0.05)),
            beta_q95 = unname(quantile(bf_draws, 0.95)), beta_q975 = unname(quantile(bf_draws, 0.975)),
            p_beta_pos = mean(bf_draws > 0), p_beta_neg = mean(bf_draws < 0),
            delta_mean = mean(df_draws), delta_q025 = unname(quantile(df_draws, 0.025)), delta_q975 = unname(quantile(df_draws, 0.975)),
            stringsAsFactors = FALSE
          )
        } else {
          stop("Missing r_fam_eff column in brms draws: ", r_col)
        }
      }
    }
    if (!is.null(s$scale_info)) row_out <- c(row_out, s$scale_info)
    row_out <- add_raw_slopes(row_out)
  }

  message(format(Sys.time(), "%H:%M"), " done BB ", s$name, sprintf(" (%.1f min)", mins))
  list(
    name = s$name,
    row = row_out,
    fam_rows = if (length(fam_rows)) do.call(rbind, fam_rows) else data.frame()
  )
}

# --- 6. Execute all SEM and Beta-Binomial specifications -------------------------------------
message(format(Sys.time(), "%H:%M"), " Running ", length(sem_specs), " SEM specs and ", length(bb_specs), " Beta-Binomial specs...")
sem_res <- par_run(sem_specs, run_sem_spec)
sem_errs <- Filter(is_err, sem_res)
for (r in sem_errs) message("ERROR SEM: ", err_text(r))
sem_ok <- Filter(Negate(is_err), sem_res)

bb_res <- par_run(bb_specs, run_bb_spec)
bb_errs <- Filter(is_err, bb_res)
for (r in bb_errs) message("ERROR BB: ", err_text(r))
bb_ok <- Filter(Negate(is_err), bb_res)

if (length(sem_errs) || length(bb_errs)) {
  stop("Synthesis run encountered ", length(sem_errs) + length(bb_errs), " model error(s).")
}

all_ok <- c(sem_ok, bb_ok)

if (dry) {
  rows_df <- do.call(rbind, lapply(all_ok, `[[`, "row"))
  message("DRY RUN COMPLETE: ", nrow(rows_df), " / ", length(sem_specs) + length(bb_specs),
          " total synthesis models validated (", sum(rows_df$cached), " already cached), 0 failed.")
  print(table(rows_df$check, rows_df$cached))
  quit(save = "no", status = 0L)
}

# --- 7. Save `synthesis_checks_main.csv` and `family_slopes_main.csv` ------------------------
raw_rows <- lapply(all_ok, `[[`, "row")
all_cols <- unique(unlist(lapply(raw_rows, names)))
syn_tab <- do.call(rbind, lapply(raw_rows, function(r) {
  as.data.frame(lapply(setNames(all_cols, all_cols), function(k) if (length(r[[k]])) r[[k]] else NA),
                stringsAsFactors = FALSE)
}))
write.csv(syn_tab, file.path(res_dir, "synthesis_checks_main.csv"), row.names = FALSE)
message("Wrote ", nrow(syn_tab), " rows to ", file.path(res_dir, "synthesis_checks_main.csv"))

fam_list <- Filter(function(df) !is.null(df) && nrow(df) > 0, lapply(all_ok, `[[`, "fam_rows"))
if (length(fam_list)) {
  fam_tab <- do.call(rbind, fam_list)
  fam_tab <- fam_tab[order(fam_tab$framework, fam_tab$item_set, fam_tab$arch, fam_tab$measure,
                           fam_tab$factor, -fam_tab$n_langs_cat12, fam_tab$family), ]
  write.csv(fam_tab, file.path(res_dir, "family_slopes_main.csv"), row.names = FALSE)
  message("Wrote ", nrow(fam_tab), " rows to ", file.path(res_dir, "family_slopes_main.csv"))
}

# --- 8. Compute paired PSIS-LOO comparisons (with language-level & family-clustered SEs) -----
cluster_se_diff <- function(d_vec, cluster_id) {
  stopifnot(length(d_vec) == length(cluster_id))
  s_g <- as.numeric(tapply(d_vec, cluster_id, sum))
  G <- length(s_g)
  if (G <= 1L) return(NA_real_)
  sqrt(G) * sd(s_g)
}

get_clusters <- function(iset, set_lbl, sb, require_pop = FALSE, drop_fams = character(0)) {
  d_set <- select_languages(langs_df, "main", iset, drop_fams = drop_fams, require_pop = require_pop)
  inset <- pop0[pop0$in_main == 1 & pop0$glottocode %in% d_set$glottocode, ]
  has_col <- function(col) intersect(d_set$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
  subs <- list(all = d_set$glottocode, het = has_col("het"), f3 = has_col("f3_z"))
  d_sb <- d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE]
  d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
  langs <- d_sb$glottocode
  lang_map <- setNames(seq_along(langs), langs)
  if (iset == "cat12") {
    im <- items_df[items_df$in_cat12 == 1, , drop = FALSE]
  } else if (iset == "cat12_irreg13") {
    im <- items_df[items_df$in_cat12 == 1 | items_df$in_irreg13 == 1, , drop = FALSE]
  } else {
    im <- items_df[items_df$in_cat12 == 1 | items_df$in_flex26 == 1, , drop = FALSE]
  }
  im <- im[order(im$item_order), , drop = FALSE]
  item_map <- setNames(seq_len(nrow(im)), im$item_id)
  ob <- obs_df[obs_df$glottocode %in% langs & obs_df$item_id %in% im$item_id, , drop = FALSE]
  ob$lang_idx <- unname(lang_map[ob$glottocode])
  ob$item_idx <- unname(item_map[ob$item_id])
  ob <- ob[order(ob$lang_idx, ob$item_idx), , drop = FALSE]
  list(lang_fam = d_sb$fam_eff, obs_fam = d_sb$fam_eff[ob$lang_idx])
}

load_sem_rds <- function(nm) {
  if (!nm %in% syn_tab$name) return(NULL)
  f <- file.path(fits_dir, paste0(nm, ".rds"))
  if (!file.exists(f)) return(NULL)
  readRDS(f)
}

pair_sem_loo <- function(nm1, nm0, cmp_type, check_nm, iset, set_lbl, arch1, arch0, m, ctrl1, ctrl0, sb,
                         require_pop = FALSE, drop_fams = character(0)) {
  o1 <- load_sem_rds(nm1); o0 <- load_sem_rds(nm0)
  if (is.null(o1) || is.null(o0)) return(NULL)
  tryCatch({
    cl_info <- get_clusters(iset, set_lbl, sb, require_pop = require_pop, drop_fams = drop_fams)
    pw_obs_1 <- o1$loo_obs$pointwise[, "elpd_loo"]
    pw_obs_0 <- o0$loo_obs$pointwise[, "elpd_loo"]
    stopifnot(length(pw_obs_1) == length(pw_obs_0), length(pw_obs_1) == length(cl_info$obs_fam))
    d_obs_vec <- pw_obs_1 - pw_obs_0

    pw_lang_1 <- o1$loo_lang$pointwise[, "elpd_loo"]
    pw_lang_0 <- o0$loo_lang$pointwise[, "elpd_loo"]
    stopifnot(length(pw_lang_1) == length(pw_lang_0), length(pw_lang_1) == length(cl_info$lang_fam))
    d_lang_vec <- pw_lang_1 - pw_lang_0

    has_tip <- !is.null(o1$loo_lang_tip) && !is.null(o0$loo_lang_tip)
    d_tip_vec <- if (has_tip) {
      pt1 <- o1$loo_lang_tip$pointwise[, "elpd_loo"]
      pt0 <- o0$loo_lang_tip$pointwise[, "elpd_loo"]
      stopifnot(length(pt1) == length(pt0), length(pt1) == length(cl_info$lang_fam))
      pt1 - pt0
    } else NULL

    data.frame(
      check = check_nm, comparison = cmp_type, set = set_lbl, item_set = iset,
      model_1 = nm1, model_0 = nm0, arch_1 = arch1, arch_0 = arch0,
      measure = m, ctrl_1 = ctrl1, ctrl_0 = ctrl0, subset = sb,
      delta_elpd_obs = sum(d_obs_vec),
      se_obs = sqrt(length(d_obs_vec)) * sd(d_obs_vec),
      se_obs_fam = cluster_se_diff(d_obs_vec, cl_info$obs_fam),
      k_obs_1_gt07 = sum(o1$loo_obs$diagnostics$pareto_k > 0.7),
      k_obs_0_gt07 = sum(o0$loo_obs$diagnostics$pareto_k > 0.7),
      delta_elpd_lang = sum(d_lang_vec),
      se_lang = sqrt(length(d_lang_vec)) * sd(d_lang_vec),
      se_lang_fam = cluster_se_diff(d_lang_vec, cl_info$lang_fam),
      k_lang_1_gt07 = sum(o1$loo_lang$diagnostics$pareto_k > 0.7),
      k_lang_0_gt07 = sum(o0$loo_lang$diagnostics$pareto_k > 0.7),
      delta_elpd_lang_tip = if (has_tip) sum(d_tip_vec) else NA_real_,
      se_lang_tip = if (has_tip) sqrt(length(d_tip_vec)) * sd(d_tip_vec) else NA_real_,
      se_lang_tip_fam = if (has_tip) cluster_se_diff(d_tip_vec, cl_info$lang_fam) else NA_real_,
      k_tip_1_gt07 = if (has_tip) sum(o1$loo_lang_tip$diagnostics$pareto_k > 0.7) else NA_integer_,
      k_tip_0_gt07 = if (has_tip) sum(o0$loo_lang_tip$diagnostics$pareto_k > 0.7) else NA_integer_,
      max_k_tip_1 = if (has_tip) max(o1$loo_lang_tip$diagnostics$pareto_k) else NA_real_,
      max_k_tip_0 = if (has_tip) max(o0$loo_lang_tip$diagnostics$pareto_k) else NA_real_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    message("SKIPPED LOO ", nm1, " vs ", nm0, ": ", conditionMessage(e))
    NULL
  })
}

loo_rows <- list()

for (iset in c("cat12", "cat12_irreg13", "cat12_flex26")) {
  x_archs <- if (iset == "cat12") c("MA_2pl", "MB_2pl") else "MC_cross"
  base_of <- function(a) if (a %in% c("MA_2pl", "MB_2pl")) "MA0_2pl" else "MC0_cross"

  for (m in measures) {
    sb <- subset_of(m)
    for (arch in x_archs) {
      b_arch <- base_of(arch)
      nm_fs   <- paste0("syn_c1__", iset, "__main__", arch, "__", m, "__famslope_P")
      nm_mun  <- paste0("syn_c1__", iset, "__main__", arch, "__", m, "__mundlak_P")
      nm_ref  <- paste0("syn_ref__", iset, "__main__", arch, "__", m, "__x_P")
      nm_bref <- paste0("syn_ref__", iset, "__main__", b_arch, "__base_P_", sb)

      # 1. Check 1.1 & 1.1b: Family-varying slopes & Mundlak vs fixed-slope reference & vs baseline
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_fs, nm_ref, "famslope_vs_fixed", "check1_family_slopes", iset, "main", arch, arch, m, "P", "P", sb
      )
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_fs, nm_bref, "famslope_vs_base", "check1_family_slopes", iset, "main", arch, b_arch, m, "P", "P", sb
      )
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_mun, nm_ref, "mundlak_vs_pooled", "check1_mundlak_wb", iset, "main", arch, arch, m, "P", "P", sb
      )
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_mun, nm_bref, "mundlak_vs_base", "check1_mundlak_wb", iset, "main", arch, b_arch, m, "P", "P", sb
      )
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_ref, nm_bref, "x_vs_base_ref", "ref_grafen_P", iset, "main", arch, b_arch, m, "P", "P", sb
      )

      # 2. Check 1.2: Leave-Out-Inner-Asia x vs baseline
      for (ia_cfg in list(
        list(lbl = "no_turk_mong",  df = c("Turkic", "Mongolic-Khitan")),
        list(lbl = "no_inner_asia", df = c("Turkic", "Mongolic-Khitan", "Tungusic"))
      )) {
        nm_ia_x <- paste0("syn_c1__", iset, "__", ia_cfg$lbl, "__", arch, "__", m, "__x_P")
        nm_ia_b <- paste0("syn_c1__", iset, "__", ia_cfg$lbl, "__", b_arch, "__base_P_", sb)
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_ia_x, nm_ia_b, "x_vs_base", paste0("check1_", ia_cfg$lbl), iset, ia_cfg$lbl, arch, b_arch, m, "P", "P", sb,
          drop_fams = ia_cfg$df
        )
      }

      # 3. Check 2: Bouckaert (P_b_stem, P_b_crown, P_b_kyr) x_vs_base AND Bouckaert vs Grafen P
      for (bctrl in c("P_b_stem", "P_b_crown", "P_b_kyr")) {
        nm_bx <- paste0("syn_c2__", iset, "__main__", arch, "__", m, "__x_", bctrl)
        nm_bb <- paste0("syn_c2__", iset, "__main__", b_arch, "__base_", bctrl, "_", sb)
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_bx, nm_bb, "x_vs_base", "check2_bouckaert", iset, "main", arch, b_arch, m, bctrl, bctrl, sb
        )
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_bx, nm_ref, paste0(bctrl, "_vs_Grafen_P"), "check2_bouckaert", iset, "main", arch, arch, m, bctrl, "P", sb
        )
      }

      # 4. Check 3.1: Demographic controls (`z_pop` and `x * z_pop`) on `main_has_pop`
      if (arch %in% c("MB_2pl", "MC_cross")) {
        nm_d_base <- paste0("syn_c3__", iset, "__main_has_pop__", b_arch, "__base_P_", sb)
        nm_d_pop  <- paste0("syn_c3__", iset, "__main_has_pop__", b_arch, "__pop_P_", sb)
        nm_d_x    <- paste0("syn_c3__", iset, "__main_has_pop__", arch, "__", m, "__x_P")
        nm_d_xp   <- paste0("syn_c3__", iset, "__main_has_pop__", arch, "__", m, "__x_pop_P")
        nm_d_xip  <- paste0("syn_c3__", iset, "__main_has_pop__", arch, "__", m, "__x_int_pop_P")

        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_d_pop, nm_d_base, "pop_vs_base", "check3_demog_pop", iset, "main_has_pop", b_arch, b_arch, "none", "P", "P", sb, require_pop = TRUE
        )
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_d_x, nm_d_base, "x_vs_base_has_pop", "check3_demog_pop", iset, "main_has_pop", arch, b_arch, m, "P", "P", sb, require_pop = TRUE
        )
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_d_xp, nm_d_pop, "x_pop_vs_pop", "check3_demog_pop", iset, "main_has_pop", arch, b_arch, m, "P", "P", sb, require_pop = TRUE
        )
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_d_xp, nm_d_x, "x_pop_vs_x", "check3_demog_pop", iset, "main_has_pop", arch, arch, m, "P", "P", sb, require_pop = TRUE
        )
        loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
          nm_d_xip, nm_d_xp, "x_int_pop_vs_x_pop", "check3_demog_pop", iset, "main_has_pop", arch, arch, m, "P", "P", sb, require_pop = TRUE
        )
      }

      # 5. Check 3.2: Spatial GP `F` (`x_vs_base_F` comparing with-x vs baseline under the SAME F control)
      nm_fx <- paste0(iset, "__main__", arch, "__", m, "_F")
      nm_fb <- paste0(iset, "__main__", b_arch, "__base_F_", sb)
      loo_rows[[length(loo_rows) + 1L]] <- pair_sem_loo(
        nm_fx, nm_fb, "x_vs_base_F", "check3_space_F", iset, "main", arch, b_arch, m, "F", "F", sb
      )
    }
  }
}

# 6. Beta-Binomial paired LOO comparisons (famslope vs fixed & x vs base across sets)
pair_bb_loo <- function(nm1, nm0, cmp_type, check_nm, out_nm, set_lbl, arch1, arch0, m, sb, fam_vec) {
  if (!all(c(nm1, nm0) %in% syn_tab$name)) return(NULL)
  f1 <- file.path(bb_fits_dir, paste0(nm1, ".rds"))
  f0 <- file.path(bb_fits_dir, paste0(nm0, ".rds"))
  if (!file.exists(f1) || !file.exists(f0)) return(NULL)
  tryCatch({
    l1 <- readRDS(f1)$criteria$loo
    l0 <- readRDS(f0)$criteria$loo
    pw1 <- l1$pointwise[, "elpd_loo"]
    pw0 <- l0$pointwise[, "elpd_loo"]
    stopifnot(length(pw1) == length(pw0), length(pw1) == length(fam_vec))
    d_vec <- pw1 - pw0
    se_val <- sqrt(length(d_vec)) * sd(d_vec)
    se_fam <- cluster_se_diff(d_vec, fam_vec)
    data.frame(
      check = check_nm, comparison = cmp_type, set = set_lbl, item_set = paste0("bb_", out_nm),
      model_1 = nm1, model_0 = nm0, arch_1 = arch1, arch_0 = arch0,
      measure = m, ctrl_1 = "P", ctrl_0 = "P", subset = sb,
      delta_elpd_obs = sum(d_vec), se_obs = se_val, se_obs_fam = se_fam,
      k_obs_1_gt07 = sum(l1$diagnostics$pareto_k > 0.7),
      k_obs_0_gt07 = sum(l0$diagnostics$pareto_k > 0.7),
      delta_elpd_lang = sum(d_vec), se_lang = se_val, se_lang_fam = se_fam,
      k_lang_1_gt07 = sum(l1$diagnostics$pareto_k > 0.7),
      k_lang_0_gt07 = sum(l0$diagnostics$pareto_k > 0.7),
      delta_elpd_lang_tip = NA_real_, se_lang_tip = NA_real_, se_lang_tip_fam = NA_real_,
      k_tip_1_gt07 = NA_integer_, k_tip_0_gt07 = NA_integer_,
      max_k_tip_1 = NA_real_, max_k_tip_0 = NA_real_,
      stringsAsFactors = FALSE
    )
  }, error = function(e) NULL)
}

for (out_nm in names(bb_outcomes)) {
  osp <- bb_outcomes[[out_nm]]
  for (set_cfg in list(
    list(lbl = "main",          df = character(0)),
    list(lbl = "no_turk_mong",  df = c("Turkic", "Mongolic-Khitan")),
    list(lbl = "no_inner_asia", df = c("Turkic", "Mongolic-Khitan", "Tungusic"))
  )) {
    d_sub <- langs_df[langs_df$in_main == 1 & langs_df[[osp$coded]] >= osp$min_coded, , drop = FALSE]
    if (length(set_cfg$df)) d_sub <- d_sub[!(d_sub$family %in% set_cfg$df), , drop = FALSE]
    d_sub <- d_sub[order(d_sub$glottocode), , drop = FALSE]
    inset <- pop0[pop0$in_main == 1 & pop0$glottocode %in% d_sub$glottocode, ]
    has_col <- function(col) intersect(d_sub$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
    subs <- list(all = d_sub$glottocode, het = has_col("het"), f3 = has_col("f3_z"))

    for (m in measures) {
      sb <- subset_of(m)
      dd <- d_sub[d_sub$glottocode %in% subs[[sb]], , drop = FALSE]
      dd <- dd[order(dd$glottocode), , drop = FALSE]
      nm_b <- paste0("bb_syn__", out_nm, "__", set_cfg$lbl, "__base_P_", sb)
      nm_x <- paste0("bb_syn__", out_nm, "__", set_cfg$lbl, "__", m, "__x_P")
      loo_rows[[length(loo_rows) + 1L]] <- pair_bb_loo(
        nm_x, nm_b, "x_vs_base", paste0("check1_bb_", set_cfg$lbl), out_nm, set_cfg$lbl,
        "BetaBinom_x", "BetaBinom", m, sb, dd$fam_eff
      )
      if (set_cfg$lbl == "main") {
        for (fmd in c("famslope_centered", "famslope_uncentered")) {
          nm_fs <- paste0("bb_syn__", out_nm, "__main__", m, "__", fmd, "_P")
          loo_rows[[length(loo_rows) + 1L]] <- pair_bb_loo(
            nm_fs, nm_x, paste0(fmd, "_vs_fixed"), "check1_bb_family_slopes", out_nm, "main",
            paste0("BetaBinom_", fmd), "BetaBinom_x", m, sb, dd$fam_eff
          )
        }
      }
    }
  }
}

loo_rows <- Filter(Negate(is.null), loo_rows)
if (length(loo_rows)) {
  loo_tab <- do.call(rbind, loo_rows)
  loo_tab <- loo_tab[!duplicated(loo_tab[, c("comparison", "model_1", "model_0")]), ]
  write.csv(loo_tab, file.path(res_dir, "loo_synthesis_main.csv"), row.names = FALSE)
  message("Wrote ", nrow(loo_tab), " rows to ", file.path(res_dir, "loo_synthesis_main.csv"))
}

message(format(Sys.time(), "%H:%M"), " ALL SYNTHESIS CHECKS DONE")
