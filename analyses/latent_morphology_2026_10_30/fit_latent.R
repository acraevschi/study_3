# Latent-variable and multi-item morphological models on the 2026-10-20 GeLaTo-Grambank links
# (EXPLORATORY; see README.md).
#
# Fits and compares three model architectures across the item-level binary response matrix:
#   1. 1-Factor models on the 12 GBI inflectional categories (`cat12`, stan/item_1factor.stan):
#        - M0_1pl:  baseline (no x), equal item weights (lambda_k = 1)
#        - M1_1pl:  with x, equal item weights (lambda_k = 1)
#        - MA0_2pl: baseline (no x), learned 2PL item weights (lambda_k > 0, arithmetic mean = 1)
#        - MA_2pl:  with x, learned 2PL item weights (Architecture A: 2PL IRT / MIMIC SEM)
#        - MB_1pl:  with x, hierarchical varying admixture slopes on 1PL baseline (Arch B, 1PL)
#        - MB_2pl:  with x, hierarchical varying admixture slopes on 2PL baseline (Arch B, 2PL)
#   2. 2-Factor joint models of Enumerative Breadth (`cat12`) + Irregularity/Flexivity
#      (`cat12_irreg13` [25 items] and `cat12_flex26` [38 items], stan/item_2factor.stan):
#        - MC0_strict: baseline (no x), strict factor assignment
#        - MC_strict:  with x (b_breadth, b_irreg), strict factor assignment (Architecture C)
#        - MC0_cross:  baseline (no x), agreement categories cross-load onto Factor 2
#        - MC_cross:   with x (b_breadth, b_irreg), agreement categories cross-load onto Factor 2
#
# Run (from repo or worktree root):
#   /usr/bin/python3 analyses/latent_morphology_2026_10_30/prepare.py
#   LAT_DRY=1 Rscript analyses/latent_morphology_2026_10_30/fit_latent.R
#   Rscript analyses/latent_morphology_2026_10_30/fit_latent.R [set ...]
#
# Environment variables:
#   LAT_DRY=1:            validate Stan syntax and all model data structures; fit nothing
#   LAT_SMOKE=1:          fast smoke run (400 iter, 2 chains) into fits_smoke/ and results_smoke/
#   LAT_MEASURES=a,b,...: exposures to fit (default: ent,sqrt_het,f3k_est_pos)
#   LAT_CTRLS=P,F,...:    control sets (default: P,F; supports N,P,S,F)
#   LAT_ITEM_SETS=...:    item sets (default: cat12,cat12_irreg13,cat12_flex26)
#   LAT_ARCHS=...:        subset of architectures to run (default: all)
#   LAT_SAMPLE=intersect (default) | all_cat12: language sample rule for 2-factor models
#   LAT_TESTLETS=1 (default) | 0: include within-language domain testlet random effects
#   LAT_PAR_FITS=k:       parallel PSOCK workers (default: 1 x 4 chains = 4 cores max)
#   LAT_TAG=x:            optional suffix for result CSVs

Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")
suppressPackageStartupMessages({
  library(brms); library(ape); library(loo); library(posterior); library(parallel); library(tools)
})

cmd_args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args_all, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)) else getwd()
if (!dir.exists("analyses/latent_morphology_2026_10_30") &&
    dir.exists(file.path(script_dir, "..", "..", "analyses", "latent_morphology_2026_10_30"))) {
  setwd(normalizePath(file.path(script_dir, "..", "..")))
}
here <- "analyses/latent_morphology_2026_10_30"
adm_dir <- "analyses/admixture_links_2026_10_21"
dry <- nzchar(Sys.getenv("LAT_DRY"))
smoke <- nzchar(Sys.getenv("LAT_SMOKE"))
use_testlets <- as.integer(Sys.getenv("LAT_TESTLETS", "1"))
stopifnot(use_testlets %in% c(0L, 1L))
sample_mode <- Sys.getenv("LAT_SAMPLE", "intersect")
stopifnot(sample_mode %in% c("intersect", "all_cat12"))

real_fits_dir <- file.path(here, if (smoke) "fits_smoke" else "fits")
fits_dir <- if (dry) tempdir() else real_fits_dir
res_dir <- if (dry) tempdir() else file.path(here, if (smoke) "results_smoke" else "results")
dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- 2L; PAR_FITS <- as.integer(Sys.getenv("LAT_PAR_FITS", "2"))
} else {
  ITER <- 2000L; WARMUP <- 1000L; CHAINS <- 4L; PAR_FITS <- as.integer(Sys.getenv("LAT_PAR_FITS", "1"))
}
stopifnot(PAR_FITS * CHAINS <= 4L)
SEED <- 20261030L

ALL_SETS <- c("main", "no_doubtful_language", "no_other_language_stand_ins", "strict")
args <- commandArgs(trailingOnly = TRUE)
sets <- if (length(args)) args else "main"
stopifnot(all(sets %in% ALL_SETS))

measures <- strsplit(Sys.getenv("LAT_MEASURES", "ent,sqrt_het,f3k_est_pos"), ",")[[1]]
ctrls <- strsplit(Sys.getenv("LAT_CTRLS", "P,F"), ",")[[1]]
stopifnot(all(ctrls %in% c("N", "P", "S", "F")))

item_sets <- strsplit(Sys.getenv("LAT_ITEM_SETS", "cat12,cat12_irreg13,cat12_flex26"), ",")[[1]]
stopifnot(all(item_sets %in% c("cat12", "cat12_irreg13", "cat12_flex26")))

tag_raw <- Sys.getenv("LAT_TAG", "")
tag <- if (nzchar(tag_raw)) paste0("_", tag_raw) else ""

# --- 1. Load prepared item and language tables -----------------------------------------------
langs_df <- read.csv(file.path(here, "language_table.csv"), stringsAsFactors = FALSE)
items_df <- read.csv(file.path(here, "items_metadata.csv"), stringsAsFactors = FALSE)
obs_df <- read.csv(file.path(here, "item_observations.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(adm_dir, "populations.csv"), stringsAsFactors = FALSE)

k5_file <- file.path(adm_dir, "f3_scores_k5.csv")
if (file.exists(k5_file)) {
  k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
  idx_k5 <- match(pop0$population, k5$population)
  pop0$f3k_z <- -k5$min_z[idx_k5]
  pop0$f3k_est <- -k5$min_est[idx_k5]
} else {
  pop0$f3k_z <- pop0$f3k_est <- NA_real_
}
pop0$f3k_est_pos <- pmax(0, pop0$f3k_est)
pop0$f3b3_z <- ifelse(pop0$n_B >= 3, pop0$f3_z, NA_real_)
pop0$f3b3_est <- ifelse(pop0$n_B >= 3, pop0$f3_est, NA_real_)

# Phylogenetic correlation matrix on the 202 main languages (matching fit.R)
raw_tree <- read.tree(file.path(adm_dir, "phylo_tree.nwk"))
d_main_202 <- langs_df[langs_df$in_main == 1 & langs_df$n_coded >= 8, ]
tree <- compute.brlen(collapse.singles(keep.tip(raw_tree, d_main_202$glottocode)), method = "Grafen")
A_all <- vcv(tree, corr = TRUE)
stopifnot(all(d_main_202$glottocode %in% rownames(A_all)))

to_rad <- pi / 180
langs_df$sx <- cos(langs_df$lat * to_rad) * cos(langs_df$lon * to_rad)
langs_df$sy <- cos(langs_df$lat * to_rad) * sin(langs_df$lon * to_rad)
langs_df$sz <- sin(langs_df$lat * to_rad)

# --- 2. Load saved Stage-1 exposures from admixture_links_2026_10_21 -------------------------
load_stage1_exposures <- function(set) {
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
  expo <- expo[!duplicated(expo[, c("measure", "glottocode")], fromLast = TRUE), ]
  expo
}

subset_of <- function(m) {
  switch(m,
    neigh = "neigh",
    het = , sqrt_het = "het",
    f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
    f3b3_z = , f3b3_est = "f3b3",
    "all"
  )
}

select_languages <- function(df, set, iset) {
  sub <- df[df[[paste0("in_", set)]] == 1 & df$n_coded >= 8, , drop = FALSE]
  if (iset == "cat12" || sample_mode == "all_cat12") {
    if (iset == "cat12_irreg13") sub <- sub[sub$irreg13_coded > 0, , drop = FALSE]
    if (iset == "cat12_flex26") sub <- sub[sub$flex26_coded > 0, , drop = FALSE]
    return(sub[order(sub$glottocode), , drop = FALSE])
  }
  if (iset == "cat12_irreg13") {
    sub <- sub[sub$irreg13_coded >= 8, , drop = FALSE]
  } else if (iset == "cat12_flex26") {
    sub <- sub[sub$flex26_coded >= 16, , drop = FALSE]
  }
  sub[order(sub$glottocode), , drop = FALSE]
}

# Compute exact brms-equivalent GP coordinates Xgp and inv_gamma prior parameters for lscale
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

# Build Stan data list for 1-factor (cat12) or 2-factor (cat12_irreg13 / cat12_flex26)
build_standata <- function(d_lang, iset, x_vec, arch, ctrl) {
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

  # Domain indices (1..D_dom) and multi-item testlet indices (0 or 1..T_test)
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

  A_sub <- A_all[langs, langs, drop = FALSE]
  L_phylo <- t(chol(A_sub))
  gp_in <- build_gp_inputs(d_lang)

  use_phylo <- as.integer(ctrl %in% c("P", "F"))
  use_space <- as.integer(ctrl %in% c("S", "F"))

  if (iset == "cat12") {
    with_x <- as.integer(arch %in% c("M1_1pl", "MA_2pl", "MB_1pl", "MB_2pl"))
    use_2pl <- as.integer(arch %in% c("MA0_2pl", "MA_2pl", "MB_2pl"))
    use_var_slopes <- as.integer(arch %in% c("MB_1pl", "MB_2pl"))
    list(
      N_lang = N_lang,
      K_item = K_item,
      D_dom = length(dom_levels),
      T_test = T_test,
      N_obs = nrow(ob),
      lang_id = as.integer(ob$lang_idx),
      item_id = as.integer(ob$item_idx),
      Y = as.integer(ob$value),
      item_dom = as.integer(item_dom),
      item_testlet = as.integer(item_testlet),
      item_is_agree = as.integer(im$is_agreement),
      x = as.numeric(x_vec),
      with_x = with_x,
      use_2pl = use_2pl,
      use_var_slopes = use_var_slopes,
      use_phylo = use_phylo,
      use_space = use_space,
      use_testlets = use_testlets,
      L_phylo = L_phylo,
      Xgp = gp_in$Xgp,
      lscale_prior_shape = gp_in$shape,
      lscale_prior_scale = gp_in$scale,
      .meta = list(items = im$item_id, domains = dom_levels, item_dom_names = im$domain)
    )
  } else {
    K1 <- sum(im$in_cat12 == 1)
    K2 <- K_item - K1
    stopifnot(K1 == 12L, K2 %in% c(13L, 26L))
    agree_pos <- which(im$is_agreement == 1)
    item_agree_idx <- integer(K_item)
    item_agree_idx[agree_pos] <- seq_along(agree_pos)
    with_x <- as.integer(arch %in% c("MC_strict", "MC_cross"))
    allow_agree_cross <- as.integer(arch %in% c("MC0_cross", "MC_cross"))
    list(
      N_lang = N_lang,
      K1 = K1,
      K2 = K2,
      K_item = K_item,
      T_test = T_test,
      K_agree = length(agree_pos),
      N_obs = nrow(ob),
      lang_id = as.integer(ob$lang_idx),
      item_id = as.integer(ob$item_idx),
      Y = as.integer(ob$value),
      item_testlet = as.integer(item_testlet),
      item_agree_idx = as.integer(item_agree_idx),
      x = as.numeric(x_vec),
      with_x = with_x,
      use_2pl = 1L,
      allow_agree_cross = allow_agree_cross,
      use_phylo = use_phylo,
      use_space = use_space,
      use_testlets = use_testlets,
      L_phylo = L_phylo,
      Xgp = gp_in$Xgp,
      lscale_prior_shape = gp_in$shape,
      lscale_prior_scale = gp_in$scale,
      .meta = list(items = im$item_id, domains = dom_levels, item_dom_names = im$domain,
                   agree_items = im$item_id[agree_pos])
    )
  }
}

# --- 3. Validate or compile Stan models ------------------------------------------------------
stan_1f_path <- file.path(here, "stan", "item_1factor.stan")
stan_2f_path <- file.path(here, "stan", "item_2factor.stan")
stan_1f_code <- readLines(stan_1f_path, warn = FALSE)
stan_2f_code <- readLines(stan_2f_path, warn = FALSE)

compute_cache_hash <- function(sdata, stan_code) {
  tf <- tempfile()
  on.exit(unlink(tf), add = TRUE)
  saveRDS(list(sdata = sdata, stan_code = stan_code, seed = SEED,
               iter = ITER, warmup = WARMUP, chains = CHAINS), file = tf)
  unname(md5sum(tf))
}

if (dry) {
  s1_chk <- rstan::stanc(file = stan_1f_path)
  s2_chk <- rstan::stanc(file = stan_2f_path)
  stopifnot(isTRUE(s1_chk$status), isTRUE(s2_chk$status))
  message("DRY RUN: Stan syntax check passed for item_1factor.stan and item_2factor.stan")
  exe_1f <- exe_2f <- NULL
} else {
  suppressPackageStartupMessages(library(cmdstanr))
  mod_1f <- cmdstan_model(stan_1f_path)
  mod_2f <- cmdstan_model(stan_2f_path)
  exe_1f <- mod_1f$exe_file()
  exe_2f <- mod_2f$exe_file()
}

par_run <- function(specs, fn) {
  if (!length(specs)) return(list())
  if (dry || PAR_FITS <= 1L || length(specs) == 1L) {
    res <- lapply(specs, fn)
  } else {
    cl <- makePSOCKcluster(min(length(specs), PAR_FITS), outfile = "")
    on.exit(stopCluster(cl), add = TRUE)
    clusterEvalQ(cl, {
      suppressPackageStartupMessages({
        library(cmdstanr); library(loo); library(posterior); library(tools)
      })
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

# --- 4. Fit a single model specification and extract summaries + LOO -------------------------
run_spec <- function(s) {
  tryCatch(run_spec_inner(s), error = function(e) list(name = s$name, error = conditionMessage(e)))
}

run_spec_inner <- function(s) {
  t0 <- Sys.time()
  sdata <- s$standata
  meta <- sdata$.meta
  sdata$.meta <- NULL

  stan_code <- if (s$item_set == "cat12") stan_1f_code else stan_2f_code
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
        name = s$name, set = s$set, item_set = s$item_set, arch = s$arch,
        measure = s$measure, ctrl = s$ctrl, subset = s$subset,
        n_lang = sdata$N_lang, n_items = sdata$K_item, n_obs = sdata$N_obs,
        cached = is_cached,
        stringsAsFactors = FALSE
      ),
      items = data.frame()
    ))
  }

  rds_file <- file.path(fits_dir, paste0(s$name, ".rds"))
  if (is_cached) {
    message(format(Sys.time(), "%H:%M"), " cached ", s$name)
    return(saved$summary_out)
  }

  exe_path <- if (s$item_set == "cat12") exe_1f else exe_2f
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

  ll_obs <- fit$draws("log_lik_obs", format = "matrix")
  ll_lang <- fit$draws("log_lik_lang", format = "matrix")
  loo_obs <- loo(ll_obs)
  loo_lang <- loo(ll_lang)

  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ctrl_label <- c(N = "none", P = "phylogeny", S = "space", F = "phylogeny+space")

  row_out <- list(
    name = s$name, set = s$set, item_set = s$item_set, arch = s$arch,
    measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]), ctrl_code = s$ctrl,
    subset = s$subset, n_lang = sdata$N_lang, n_items = sdata$K_item, n_obs = sdata$N_obs,
    max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    divergences = divs, minutes = mins,
    elpd_loo_obs = loo_obs$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_obs = loo_obs$estimates["elpd_loo", "SE"],
    pareto_k_obs_gt_07 = sum(loo_obs$diagnostics$pareto_k > 0.7),
    elpd_loo_lang = loo_lang$estimates["elpd_loo", "Estimate"],
    se_elpd_loo_lang = loo_lang$estimates["elpd_loo", "SE"],
    pareto_k_lang_gt_07 = sum(loo_lang$diagnostics$pareto_k > 0.7)
  )

  item_rows <- list()

  if (s$item_set == "cat12") {
    if (sdata$with_x == 1L) {
      bx <- dr$b_x
      row_out <- c(row_out, list(
        b_mean = mean(bx), b_sd = sd(bx),
        b_q025 = unname(quantile(bx, 0.025)), b_q05 = unname(quantile(bx, 0.05)),
        b_q95 = unname(quantile(bx, 0.95)), b_q975 = unname(quantile(bx, 0.975)),
        p_b_neg = mean(bx < 0), p_b_pos = mean(bx > 0)
      ))
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
    for (k in seq_len(sdata$K_item)) {
      ak <- dr[[sprintf("alpha[%d]", k)]]
      lk <- dr[[sprintf("lambda[%d]", k)]]
      bk <- dr[[sprintf("beta_item[%d]", k)]]
      item_rows[[length(item_rows) + 1L]] <- data.frame(
        model = s$name, set = s$set, item_set = s$item_set, arch = s$arch,
        measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]),
        item_id = meta$items[k], domain = meta$item_dom_names[k],
        alpha_mean = mean(ak), alpha_q025 = unname(quantile(ak, 0.025)), alpha_q975 = unname(quantile(ak, 0.975)),
        lambda_mean = mean(lk), lambda_q025 = unname(quantile(lk, 0.025)), lambda_q975 = unname(quantile(lk, 0.975)),
        p_lambda_gt_1 = mean(lk > 1.0),
        beta_mean = mean(bk), beta_q025 = unname(quantile(bk, 0.025)), beta_q975 = unname(quantile(bk, 0.975)),
        p_beta_neg = mean(bk < 0), p_beta_pos = mean(bk > 0),
        cross_mean = NA_real_, cross_q025 = NA_real_, cross_q975 = NA_real_,
        stringsAsFactors = FALSE
      )
    }
  } else {
    rho <- dr$rho_latent
    rho0 <- dr$rho_theta0
    row_out <- c(row_out, list(
      rho_latent_mean = mean(rho),
      rho_latent_q025 = unname(quantile(rho, 0.025)),
      rho_latent_q975 = unname(quantile(rho, 0.975)),
      rho_theta0_mean = mean(rho0),
      rho_theta0_q025 = unname(quantile(rho0, 0.025)),
      rho_theta0_q975 = unname(quantile(rho0, 0.975))
    ))
    if (sdata$with_x == 1L) {
      b1 <- dr$b_breadth; b2 <- dr$b_irreg; bdiff <- dr$b_diff
      row_out <- c(row_out, list(
        b_mean = mean(b1), b_sd = sd(b1),
        b_q025 = unname(quantile(b1, 0.025)), b_q05 = unname(quantile(b1, 0.05)),
        b_q95 = unname(quantile(b1, 0.95)), b_q975 = unname(quantile(b1, 0.975)),
        p_b_neg = mean(b1 < 0), p_b_pos = mean(b1 > 0),
        b_irreg_mean = mean(b2), b_irreg_sd = sd(b2),
        b_irreg_q025 = unname(quantile(b2, 0.025)), b_irreg_q05 = unname(quantile(b2, 0.05)),
        b_irreg_q95 = unname(quantile(b2, 0.95)), b_irreg_q975 = unname(quantile(b2, 0.975)),
        p_irreg_neg = mean(b2 < 0), p_irreg_pos = mean(b2 > 0),
        b_diff_mean = mean(bdiff), b_diff_q025 = unname(quantile(bdiff, 0.025)), b_diff_q975 = unname(quantile(bdiff, 0.975)),
        p_diff_pos = mean(bdiff > 0)
      ))
    }
    if (sdata$allow_agree_cross == 1L) {
      g1 <- dr[["gamma_cross[1]"]]; g2 <- dr[["gamma_cross[2]"]]
      row_out <- c(row_out, list(
        gamma_cross_gender_mean = mean(g1), gamma_cross_gender_q025 = unname(quantile(g1, 0.025)), gamma_cross_gender_q975 = unname(quantile(g1, 0.975)),
        gamma_cross_number_mean = mean(g2), gamma_cross_number_q025 = unname(quantile(g2, 0.025)), gamma_cross_number_q975 = unname(quantile(g2, 0.975))
      ))
    }
    for (k in seq_len(sdata$K_item)) {
      ak <- dr[[sprintf("alpha[%d]", k)]]
      lk <- if (k <= sdata$K1) dr[[sprintf("lambda1[%d]", k)]] else dr[[sprintf("lambda2[%d]", k - sdata$K1)]]
      ag <- sdata$item_agree_idx[k]
      has_cr <- sdata$allow_agree_cross == 1L && ag > 0L
      ck <- if (has_cr) dr[[sprintf("gamma_cross[%d]", ag)]] else NULL
      bk <- if (k <= sdata$K1) {
        if (has_cr && sdata$with_x == 1L) lk * dr$b_breadth + ck * dr$b_irreg else lk * dr$b_breadth
      } else {
        lk * dr$b_irreg
      }
      item_rows[[length(item_rows) + 1L]] <- data.frame(
        model = s$name, set = s$set, item_set = s$item_set, arch = s$arch,
        measure = s$measure, ctrl = unname(ctrl_label[s$ctrl]),
        item_id = meta$items[k], domain = meta$item_dom_names[k],
        alpha_mean = mean(ak), alpha_q025 = unname(quantile(ak, 0.025)), alpha_q975 = unname(quantile(ak, 0.975)),
        lambda_mean = mean(lk), lambda_q025 = unname(quantile(lk, 0.025)), lambda_q975 = unname(quantile(lk, 0.975)),
        p_lambda_gt_1 = mean(lk > 1.0),
        beta_mean = mean(bk), beta_q025 = unname(quantile(bk, 0.025)), beta_q975 = unname(quantile(bk, 0.975)),
        p_beta_neg = mean(bk < 0), p_beta_pos = mean(bk > 0),
        cross_mean = if (has_cr) mean(ck) else NA_real_,
        cross_q025 = if (has_cr) unname(quantile(ck, 0.025)) else NA_real_,
        cross_q975 = if (has_cr) unname(quantile(ck, 0.975)) else NA_real_,
        stringsAsFactors = FALSE
      )
    }
  }

  if (!is.null(s$scale_info)) row_out <- c(row_out, s$scale_info)

  summary_out <- list(
    name = s$name,
    row = row_out,
    items = do.call(rbind, item_rows)
  )
  saveRDS(list(cache_hash = chash, summary_out = summary_out, loo_obs = loo_obs, loo_lang = loo_lang), rds_file)
  message(format(Sys.time(), "%H:%M"), " done ", s$name, sprintf(" (%.1f min, Rhat=%.3f, div=%d)", mins, row_out$max_rhat, divs))
  summary_out
}

# --- 5. Build and run all specifications for each set ----------------------------------------
archs_1f_base <- c("M0_1pl", "MA0_2pl")
archs_1f_x    <- c("M1_1pl", "MA_2pl", "MB_1pl", "MB_2pl")
archs_2f_base <- c("MC0_strict", "MC0_cross")
archs_2f_x    <- c("MC_strict", "MC_cross")

arch_filter <- if (nzchar(Sys.getenv("LAT_ARCHS"))) strsplit(Sys.getenv("LAT_ARCHS"), ",")[[1]] else NULL
mode_sfx <- paste0(
  if (use_testlets == 0L) "_notest" else "",
  if (sample_mode != "intersect") paste0("_", sample_mode) else ""
)

run_set <- function(set) {
  message(format(Sys.time(), "%H:%M"), " ===== set ", set, " =====")
  expo_all <- load_stage1_exposures(set)
  specs <- list()

  for (iset in item_sets) {
    d_set <- select_languages(langs_df, set, iset)
    inset <- pop0[pop0[[paste0("in_", set)]] == 1 & pop0$glottocode %in% d_set$glottocode, ]
    has_col <- function(col) intersect(d_set$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
    subs <- list(
      all = d_set$glottocode,
      neigh = has_col("neigh_flow"),
      het = has_col("het"),
      f3 = has_col("f3_z"),
      f3b3 = has_col("f3b3_z")
    )
    active_subs <- unique(vapply(measures, subset_of, ""))
    subs <- subs[intersect(names(subs), active_subs)]

    base_archs <- if (iset == "cat12") archs_1f_base else archs_2f_base
    x_archs <- if (iset == "cat12") archs_1f_x else archs_2f_x
    if (!is.null(arch_filter)) {
      base_archs <- intersect(base_archs, arch_filter)
      x_archs <- intersect(x_archs, arch_filter)
    }

    # Baseline (no-x) models per (subset, control, baseline architecture)
    for (sb in names(subs)) {
      d_sb <- d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE]
      d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
      x_zero <- rep(0.0, nrow(d_sb))
      for (ctrl in ctrls) for (arch in base_archs) {
        nm <- paste0(iset, mode_sfx, "__", set, "__", arch, "__base_", ctrl, "_", sb)
        specs[[length(specs) + 1L]] <- list(
          name = nm, set = set, item_set = iset, arch = arch,
          measure = "none", ctrl = ctrl, subset = sb,
          standata = build_standata(d_sb, iset, x_zero, arch, ctrl),
          scale_info = NULL
        )
      }
    }

    # With-x models per (measure, control, architecture)
    for (m in measures) {
      sb <- subset_of(m)
      e_m <- expo_all[expo_all$measure == m & expo_all$glottocode %in% subs[[sb]], , drop = FALSE]
      if (!setequal(e_m$glottocode, subs[[sb]])) {
        message("WARNING: measure ", m, " missing some languages in ", set, " / ", iset, "; skipped")
        next
      }
      d_sb <- merge(d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE],
                    e_m[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
                    by = "glottocode")
      d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
      mu <- mean(d_sb$eta_mean)
      s <- sd(d_sb$eta_mean)
      x_std <- (d_sb$eta_mean - mu) / s
      scale_info <- list(x_center = mu, x_scale = s, y_mean = d_sb$y_mean[1], y_sd = d_sb$y_sd[1])

      for (ctrl in ctrls) for (arch in x_archs) {
        nm <- paste0(iset, mode_sfx, "__", set, "__", arch, "__", m, "_", ctrl)
        specs[[length(specs) + 1L]] <- list(
          name = nm, set = set, item_set = iset, arch = arch,
          measure = m, ctrl = ctrl, subset = sb,
          standata = build_standata(d_sb, iset, x_std, arch, ctrl),
          scale_info = scale_info
        )
      }
    }
  }

  results <- par_run(specs, run_spec)
  for (r in Filter(is_err, results)) message("ERROR ", err_text(r))
  ok <- Filter(Negate(is_err), results)

  if (dry) {
    rows_df <- do.call(rbind, lapply(ok, `[[`, "row"))
    n_cached <- sum(rows_df$cached)
    message("DRY RUN [", set, "]: ", length(ok), " / ", length(specs),
            " models validated (", n_cached, " cached in fits/), 0 failed.")
    print(rows_df, row.names = FALSE)
    return(invisible(NULL))
  }

  # Save model_results_<set><tag>.csv
  raw_rows <- lapply(ok, `[[`, "row")
  all_cols <- unique(unlist(lapply(raw_rows, names)))
  tab <- do.call(rbind, lapply(raw_rows, function(r) {
    as.data.frame(lapply(setNames(all_cols, all_cols), function(k) if (length(r[[k]])) r[[k]] else NA),
                  stringsAsFactors = FALSE)
  }))
  write.csv(tab, file.path(res_dir, paste0("model_results_", set, mode_sfx, tag, ".csv")), row.names = FALSE)

  # Save item_parameters_<set><tag>.csv
  item_tab <- do.call(rbind, lapply(ok, `[[`, "items"))
  write.csv(item_tab, file.path(res_dir, paste0("item_parameters_", set, mode_sfx, tag, ".csv")), row.names = FALSE)

  # Compute pairwise LOO comparisons (both item-level and language-level)
  load_loos <- function(nm) {
    f <- file.path(fits_dir, paste0(nm, ".rds"))
    if (!file.exists(f)) stop("Missing fit RDS: ", nm)
    readRDS(f)
  }
  pair_loo <- function(nm1, nm0, cmp_type, iset, arch1, arch0, m, ctrl, sb) {
    if (!all(c(nm1, nm0) %in% tab$name)) return(NULL)
    tryCatch({
      o1 <- load_loos(nm1); o0 <- load_loos(nm0)
      d_obs <- o1$loo_obs$estimates["elpd_loo", "Estimate"] - o0$loo_obs$estimates["elpd_loo", "Estimate"]
      se_obs <- loo_compare(o1$loo_obs, o0$loo_obs)[2, "se_diff"]
      d_lang <- o1$loo_lang$estimates["elpd_loo", "Estimate"] - o0$loo_lang$estimates["elpd_loo", "Estimate"]
      se_lang <- loo_compare(o1$loo_lang, o0$loo_lang)[2, "se_diff"]
      data.frame(
        set = set, comparison = cmp_type, item_set = iset,
        model_1 = nm1, model_0 = nm0, arch_1 = arch1, arch_0 = arch0,
        measure = m, ctrl = ctrl, subset = sb,
        delta_elpd_obs = d_obs, se_obs = se_obs,
        delta_elpd_lang = d_lang, se_lang = se_lang,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      message("SKIPPED LOO ", nm1, " vs ", nm0, ": ", conditionMessage(e))
      NULL
    })
  }

  cmp_list <- list()
  x_rows <- tab[tab$measure != "none", , drop = FALSE]
  for (i in seq_len(nrow(x_rows))) {
    r <- x_rows[i, ]
    base_arch <- switch(r$arch,
      M1_1pl = "M0_1pl",
      MA_2pl = "MA0_2pl",
      MB_1pl = "M0_1pl",
      MB_2pl = "MA0_2pl",
      MC_strict = "MC0_strict",
      MC_cross = "MC0_cross"
    )
    nm_base <- paste0(r$item_set, mode_sfx, "__", r$set, "__", base_arch, "__base_", r$ctrl_code, "_", r$subset)
    cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
      r$name, nm_base, "x_vs_baseline", r$item_set, r$arch, base_arch, r$measure, r$ctrl, r$subset
    )
    # Architecture comparisons on the same (item_set, measure, ctrl)
    if (r$arch == "MA_2pl") {
      nm_1pl <- paste0(r$item_set, mode_sfx, "__", r$set, "__M1_1pl__", r$measure, "_", r$ctrl_code)
      cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
        r$name, nm_1pl, "2pl_vs_1pl", r$item_set, "MA_2pl", "M1_1pl", r$measure, r$ctrl, r$subset
      )
    }
    if (r$arch == "MB_1pl") {
      nm_1pl <- paste0(r$item_set, mode_sfx, "__", r$set, "__M1_1pl__", r$measure, "_", r$ctrl_code)
      cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
        r$name, nm_1pl, "varslopes_vs_1pl", r$item_set, "MB_1pl", "M1_1pl", r$measure, r$ctrl, r$subset
      )
    }
    if (r$arch == "MB_2pl") {
      nm_2pl <- paste0(r$item_set, mode_sfx, "__", r$set, "__MA_2pl__", r$measure, "_", r$ctrl_code)
      nm_mb1 <- paste0(r$item_set, mode_sfx, "__", r$set, "__MB_1pl__", r$measure, "_", r$ctrl_code)
      cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
        r$name, nm_2pl, "varslopes_vs_2pl", r$item_set, "MB_2pl", "MA_2pl", r$measure, r$ctrl, r$subset
      )
      cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
        r$name, nm_mb1, "2pl_vs_1pl_varslopes", r$item_set, "MB_2pl", "MB_1pl", r$measure, r$ctrl, r$subset
      )
    }
    if (r$arch == "MC_cross") {
      nm_strict <- paste0(r$item_set, mode_sfx, "__", r$set, "__MC_strict__", r$measure, "_", r$ctrl_code)
      cmp_list[[length(cmp_list) + 1L]] <- pair_loo(
        r$name, nm_strict, "cross_vs_strict", r$item_set, "MC_cross", "MC_strict", r$measure, r$ctrl, r$subset
      )
    }
  }
  cmp_list <- Filter(Negate(is.null), cmp_list)
  if (length(cmp_list)) {
    write.csv(do.call(rbind, cmp_list), file.path(res_dir, paste0("loo_comparisons_", set, mode_sfx, tag, ".csv")), row.names = FALSE)
  }
  message(format(Sys.time(), "%H:%M"), " ===== set ", set, " done =====")
}

for (s in sets) tryCatch(run_set(s), error = function(e) message("ERROR set ", s, ": ", conditionMessage(e)))
message("ALL DONE")
