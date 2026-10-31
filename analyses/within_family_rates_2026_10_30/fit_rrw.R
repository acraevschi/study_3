# Component 2: Two-Stage Within-Family Relaxed Random Walk (RRW) & Directional Drift in Stan
#
# Strictly respects the <= 4 CPU cores limit:
#   - No outer parallel cluster (models are fitted sequentially one at a time)
#   - Each Stan fit uses chains = 4, cores = 4, threads_per_chain = 1 (at most 4 cores active)
#
# Design:
#   Stage 1 (gen_rrw.stan, fitted on genetic data only):
#     - Fits the within-family RRW across the F multi-language families for each cutoff (min5, min3)
#       and each of the 6 genetic representations:
#         ent, sqrt_het, f3k_est_pos, q_within_d3, fst_within_d3, fst_global_d3
#     - Extracts posterior mean branch log-rate shifts J_hat_b = E[J_b | X] (and SD), plus
#       posterior summary of tau_J (branch rate heterogeneity SD) and sigma_0 (clock rate).
#   Stage 2 (morph_rrw.stan, fitted on morphological outcomes cat12, flex, irreg13):
#     - For each (cutoff, outcome), fits:
#       0. clock baseline (use_dir=0, use_rate=0)
#       1. rate           (use_dir=0, use_rate=1): supervisor's cov_rrw branch-rate coupling
#       2. dir_jump       (use_dir=1, use_rate=0): cumulative branch-jump directional shift
#       3. both           (use_dir=1, use_rate=1): both directional shift and rate coupling
#       4. dir_signed     (use_dir=1, use_rate=0): signed within-family genetic divergence from root
#     - Computes exact PSIS-LOO (loo::loo) and delta_elpd against the clock baseline on the
#       identical language set.

Sys.setenv(
  PATH = paste("/usr/bin:/bin:/usr/sbin:/sbin", Sys.getenv("PATH"), sep = ":"),
  MAKE = "/usr/bin/make",
  R_MAKEVARS_USER = "/dev/null"
)

suppressPackageStartupMessages({
  library(rstan)
  library(loo)
  library(posterior)
  library(ape)
})

# Enforce strict <= 4 CPU cores cap
MAX_CORES <- 4L
options(mc.cores = MAX_CORES)
rstan_options(auto_write = TRUE)
rstan_options(threads_per_chain = 1L)

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
} else if (file.exists("gen_rrw.stan")) {
  normalizePath(".", mustWork = FALSE)
} else if (dir.exists("analyses/within_family_rates_2026_10_30")) {
  normalizePath("analyses/within_family_rates_2026_10_30", mustWork = FALSE)
} else {
  normalizePath(".", mustWork = FALSE)
}
out_dir  <- Sys.getenv("RRW_OUT_DIR", script_dir)
res_dir  <- file.path(out_dir, "results")
fits_dir <- file.path(out_dir, "fits")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fits_dir, recursive = TRUE, showWarnings = FALSE)

CHAINS <- 4L
CORES  <- 4L
ITER_S1   <- as.integer(Sys.getenv("RRW_ITER_S1", "1600"))
WARMUP_S1 <- as.integer(Sys.getenv("RRW_WARMUP_S1", "800"))
ITER_S2   <- as.integer(Sys.getenv("RRW_ITER_S2", "2000"))
WARMUP_S2 <- as.integer(Sys.getenv("RRW_WARMUP_S2", "1000"))
SEED      <- 20261030L

cutoffs  <- if (nzchar(Sys.getenv("RRW_CUTOFFS")))  strsplit(Sys.getenv("RRW_CUTOFFS"), ",")[[1]]  else c("min5", "min3")
outcomes <- if (nzchar(Sys.getenv("RRW_OUTCOMES"))) strsplit(Sys.getenv("RRW_OUTCOMES"), ",")[[1]] else c("cat12", "flex", "irreg13")

gen_specs <- list(
  ent           = list(cols = "s1_ent_mean",                                           se = "s1_ent_sd",                                                  d = 1L, type = "scalar"),
  sqrt_het      = list(cols = "s1_sqrt_het_mean",                                      se = "s1_sqrt_het_sd",                                             d = 1L, type = "scalar"),
  f3k_est_pos   = list(cols = "s1_f3k_est_pos_mean",                                   se = "s1_f3k_est_pos_sd",                                          d = 1L, type = "scalar"),
  q_within_d3   = list(cols = c("q_within_pc1", "q_within_pc2", "q_within_pc3"),       se = c("q_within_pc1_se", "q_within_pc2_se", "q_within_pc3_se"),   d = 3L, type = "coord"),
  fst_within_d3 = list(cols = c("fst_within_pc1", "fst_within_pc2", "fst_within_pc3"), se = c("fst_within_pc1_se", "fst_within_pc2_se", "fst_within_pc3_se"), d = 3L, type = "coord"),
  fst_global_d3 = list(cols = c("fst_global_pc1", "fst_global_pc2", "fst_global_pc3"), se = c("fst_global_pc1_se", "fst_global_pc2_se", "fst_global_pc3_se"), d = 3L, type = "coord")
)
if (nzchar(Sys.getenv("RRW_MEASURES"))) {
  ms <- strsplit(Sys.getenv("RRW_MEASURES"), ",")[[1]]
  gen_specs <- gen_specs[ms]
}

sm_gen   <- NULL
sm_morph <- NULL
get_sm_gen <- function() {
  if (is.null(sm_gen)) {
    cat(format(Sys.time(), "%H:%M:%S"), "Compiling gen_rrw.stan...\n")
    sm_gen <<- stan_model(file = file.path(out_dir, "gen_rrw.stan"))
    cat(format(Sys.time(), "%H:%M:%S"), "gen_rrw.stan compilation complete.\n")
  }
  sm_gen
}
get_sm_morph <- function() {
  if (is.null(sm_morph)) {
    cat(format(Sys.time(), "%H:%M:%S"), "Compiling morph_rrw.stan...\n")
    sm_morph <<- stan_model(file = file.path(out_dir, "morph_rrw.stan"))
    cat(format(Sys.time(), "%H:%M:%S"), "morph_rrw.stan compilation complete.\n")
  }
  sm_morph
}

# Helper: build block-diagonal incidence matrix A_scaled and family index arrays from a dataset
pack_tree_blocks <- function(families_list) {
  F_cnt <- length(families_list)
  T_fam <- vapply(families_list, `[[`, 0L, "T_f")
  B_fam <- vapply(families_list, `[[`, 0L, "B_f")
  T_tot <- sum(T_fam)
  B_tot <- sum(B_fam)
  tip_start <- cumsum(c(1L, head(T_fam, -1L)))
  br_start  <- cumsum(c(1L, head(B_fam, -1L)))

  A_scaled <- matrix(0, nrow = T_tot, ncol = B_tot)
  A_time   <- matrix(0, nrow = T_tot, ncol = B_tot)
  for (f in seq_len(F_cnt)) {
    ts <- tip_start[f]; tf <- T_fam[f]
    bs <- br_start[f];  bf <- B_fam[f]
    A_scaled[ts:(ts + tf - 1L), bs:(bs + bf - 1L)] <- families_list[[f]]$A_scaled
    A_time[ts:(ts + tf - 1L),   bs:(bs + bf - 1L)] <- families_list[[f]]$A_time
  }
  list(
    T = T_tot, B = B_tot, F = F_cnt,
    tip_start = as.array(tip_start), T_fam = as.array(T_fam),
    br_start  = as.array(br_start),  B_fam = as.array(B_fam),
    A_scaled = A_scaled, A_time = A_time
  )
}

# --- STAGE 1: Fit Genetic RRW per (cutoff, genetic_measure) on the full cat12 family tree ---
run_stage1 <- function(cut_label, gname) {
  rds_file <- file.path(fits_dir, sprintf("s1_%s_%s.rds", cut_label, gname))
  if (file.exists(rds_file)) {
    cat(format(Sys.time(), "%H:%M:%S"), sprintf("[Stage 1] Reusing cached %s / %s\n", cut_label, gname))
    return(readRDS(rds_file))
  }
  t0 <- Sys.time()
  ds <- readRDS(file.path(out_dir, "data", sprintf("dataset_%s_cat12.rds", cut_label)))
  gs <- gen_specs[[gname]]
  D  <- gs$d

  # Build per-family blocks (handling any single missing tip in sqrt_het / f3k_est_pos by imputing
  # family mean with very wide SE = 10.0 so the tree topology and branch indices 1..B stay identical!)
  df_all <- ds$data
  X_mat    <- as.matrix(df_all[, gs$cols, drop = FALSE])
  X_se_mat <- as.matrix(df_all[, gs$se,   drop = FALSE])

  D_fam <- integer(ds$n_families)
  for (f in seq_len(ds$n_families)) {
    fam_obj <- ds$families[[f]]
    rows_f  <- which(df_all$family == fam_obj$family)
    D_fam[f] <- min(D, fam_obj$T_f - 1L)
    for (d in seq_len(D)) {
      if (d > D_fam[f]) {
        X_mat[rows_f, d]    <- 0.0
        X_se_mat[rows_f, d] <- 1.0
      } else {
        na_r <- rows_f[is.na(X_mat[rows_f, d])]
        if (length(na_r) > 0L) {
          X_mat[na_r, d]    <- mean(X_mat[rows_f, d], na.rm = TRUE)
          X_se_mat[na_r, d] <- 10.0  # Uninformative observation so missing tip does not pull branch rates
        }
      }
    }
  }

  blk <- pack_tree_blocks(ds$families)
  sdata <- c(blk[c("T", "B", "F", "tip_start", "T_fam", "br_start", "B_fam", "A_scaled")],
             list(D = D, D_fam = as.array(D_fam), X = X_mat, X_se = X_se_mat))

  fit <- sampling(
    get_sm_gen(), data = sdata,
    chains = CHAINS, cores = CORES, iter = ITER_S1, warmup = WARMUP_S1,
    seed = SEED, control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0
  )

  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)
  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  n_div <- sum(vapply(sp, function(x) sum(x[, "divergent__"]), 0))

  j_cols <- paste0("J[", seq_len(blk$B), "]")
  j_mat  <- as.matrix(dr[, j_cols, drop = FALSE])
  J_mean <- colMeans(j_mat)
  J_sd   <- apply(j_mat, 2, sd)

  # Map each branch 1..B to its family, edge length, and descendant tip(s)
  br_meta <- do.call(rbind, lapply(seq_len(ds$n_families), function(f) {
    fo <- ds$families[[f]]
    bs <- blk$br_start[f]
    child_node <- fo$tree$edge[, 2]
    is_term <- child_node <= fo$T_f
    tip_code <- ifelse(is_term, fo$tree$tip.label[child_node], NA_character_)
    tip_name <- ifelse(is_term, fo$data$name[child_node], "<internal clade>")
    data.frame(
      branch_id = bs:(bs + fo$B_f - 1L),
      family    = fo$family,
      is_terminal = is_term,
      glottocode  = tip_code,
      name        = tip_name,
      edge_kyr    = fo$edge_len,
      stringsAsFactors = FALSE
    )
  }))
  br_meta$J_mean    <- J_mean
  br_meta$J_sd      <- J_sd
  br_meta$rate_mult <- colMeans(exp(j_mat))

  # Also compute signed divergence from Stage-1 family root mu_fam[f, d] for each tip in ds$data
  mu_cols <- outer(seq_len(ds$n_families), seq_len(D), function(f, d) sprintf("mu_fam[%d,%d]", f, d))
  mu_hat  <- matrix(colMeans(as.matrix(dr[, as.vector(mu_cols), drop = FALSE])),
                    nrow = ds$n_families, ncol = D)
  signed_dev <- numeric(blk$T)
  for (f in seq_len(ds$n_families)) {
    ts <- blk$tip_start[f]; tf <- blk$T_fam[f]
    idx_f <- ts:(ts + tf - 1L)
    if (gs$type == "scalar") {
      signed_dev[idx_f] <- X_mat[idx_f, 1] - mu_hat[f, 1]
    } else {
      df_act <- D_fam[f]
      diff_mat <- sweep(X_mat[idx_f, seq_len(df_act), drop = FALSE], 2, mu_hat[f, seq_len(df_act)], `-`)
      signed_dev[idx_f] <- sqrt(rowSums(diff_mat^2))
    }
  }
  names(signed_dev) <- df_all$glottocode

  # Cumulative branch jump along root-to-tip path: A_scaled %*% J_mean
  cum_jump <- as.numeric(blk$A_scaled %*% J_mean)
  names(cum_jump) <- df_all$glottocode

  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  diag_row <- data.frame(
    cutoff       = cut_label,
    measure      = gname,
    n_families   = blk$F,
    n_tips       = blk$T,
    n_branches   = blk$B,
    tau_J_mean   = mean(dr$tau_J),
    tau_J_q025   = unname(quantile(dr$tau_J, 0.025)),
    tau_J_q975   = unname(quantile(dr$tau_J, 0.975)),
    sigma_0_mean = mean(dr$sigma_0),
    sigma_tip_extra_mean = mean(dr$sigma_tip_extra),
    max_J_mean   = max(J_mean),
    sd_J_mean    = sd(J_mean),
    max_rhat     = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    divergences  = n_div,
    minutes      = round(mins, 2),
    stringsAsFactors = FALSE
  )
  cat(format(Sys.time(), "%H:%M:%S"),
      sprintf("[Stage 1] Done %s / %s (tau_J=%.2f, max(J_hat)=%.2f, div=%d, %.1f min)\n",
              cut_label, gname, diag_row$tau_J_mean, diag_row$max_J_mean, n_div, mins))

  out <- list(
    cutoff = cut_label, measure = gname,
    J_mean = J_mean, J_sd = J_sd,
    cum_jump = cum_jump, signed_dev = signed_dev,
    branch_table = br_meta, diag = diag_row
  )
  saveRDS(out, rds_file)
  out
}

# --- STAGE 2: Fit Morphological Beta-Binomial RRW & Drift Models --------------------------
fit_one_s2 <- function(tag_name, sdata) {
  rds_f <- file.path(fits_dir, paste0("s2_", tag_name, ".rds"))
  if (file.exists(rds_f)) return(readRDS(rds_f))
  t0 <- Sys.time()
  fit <- sampling(
    get_sm_morph(), data = sdata,
    chains = CHAINS, cores = CORES, iter = ITER_S2, warmup = WARMUP_S2,
    seed = SEED, control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0
  )
  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)
  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  n_div <- sum(vapply(sp, function(x) sum(x[, "divergent__"]), 0))
  ll_mat <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  r_eff  <- relative_eff(exp(ll_mat))
  loo_ob <- suppressWarnings(loo(ll_mat, r_eff = r_eff))
  mins   <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  res <- list(
    name         = tag_name,
    loo          = loo_ob,
    elpd_loo     = loo_ob$estimates["elpd_loo", "Estimate"],
    se_elpd_loo  = loo_ob$estimates["elpd_loo", "SE"],
    pareto_k_gt07 = sum(loo_ob$diagnostics$pareto_k > 0.7),
    sigma_morph_mean = mean(dr$sigma_morph),
    sd_fam_mean      = mean(dr$sd_fam),
    phi_mean         = mean(dr$phi),
    beta_dir_mean    = if (sdata$use_dir == 1L) mean(dr$beta_dir) else NA_real_,
    beta_dir_q025    = if (sdata$use_dir == 1L) unname(quantile(dr$beta_dir, 0.025)) else NA_real_,
    beta_dir_q975    = if (sdata$use_dir == 1L) unname(quantile(dr$beta_dir, 0.975)) else NA_real_,
    p_beta_neg       = if (sdata$use_dir == 1L) mean(dr$beta_dir < 0) else NA_real_,
    gamma_rate_mean  = if (sdata$use_rate == 1L) mean(dr$gamma_rate) else NA_real_,
    gamma_rate_q025  = if (sdata$use_rate == 1L) unname(quantile(dr$gamma_rate, 0.025)) else NA_real_,
    gamma_rate_q975  = if (sdata$use_rate == 1L) unname(quantile(dr$gamma_rate, 0.975)) else NA_real_,
    p_gamma_pos      = if (sdata$use_rate == 1L) mean(dr$gamma_rate > 0) else NA_real_,
    max_rhat         = max(sm$rhat, na.rm = TRUE),
    min_ess_bulk     = min(sm$ess_bulk, na.rm = TRUE),
    divergences      = n_div,
    minutes          = round(mins, 2)
  )
  saveRDS(res, rds_f)
  res
}

s1_results <- list()
for (cut_label in cutoffs) {
  for (gname in names(gen_specs)) {
    s1_results[[paste(cut_label, gname, sep = "__")]] <- run_stage1(cut_label, gname)
  }
}

s1_diag_df <- do.call(rbind, lapply(s1_results, `[[`, "diag"))
write.csv(s1_diag_df, file.path(res_dir, "stage1_genetic_rrw_diagnostics.csv"), row.names = FALSE)
cat("\n=== Stage 1 Genetic RRW Summary ===\n")
print(s1_diag_df, row.names = FALSE)

# Save all Stage-1 branch jump tables
s1_branches_df <- do.call(rbind, lapply(names(s1_results), function(k) {
  bt <- s1_results[[k]]$branch_table
  bt$cutoff  <- s1_results[[k]]$cutoff
  bt$measure <- s1_results[[k]]$measure
  bt
}))
write.csv(s1_branches_df, file.path(res_dir, "stage1_branch_jumps.csv"), row.names = FALSE)

# Run Stage 2 across all (cutoff, outcome, measure)
s2_rows <- list()

for (cut_label in cutoffs) {
  ds_cat12 <- readRDS(file.path(out_dir, "data", sprintf("dataset_%s_cat12.rds", cut_label)))
  blk_cat12 <- pack_tree_blocks(ds_cat12$families)

  for (out_name in outcomes) {
    ds_out <- readRDS(file.path(out_dir, "data", sprintf("dataset_%s_%s.rds", cut_label, out_name)))
    # Slice the full family incidence matrix A_scaled to the outcome-coded tips in each family,
    # keeping all B_f branches of the family tree intact so Stage-1 J_hat maps 1-to-1!
    F_out <- ds_out$n_families
    T_fam_out <- vapply(ds_out$families, `[[`, 0L, "T_f")
    B_fam_out <- integer(F_out)
    tip_start_out <- cumsum(c(1L, head(T_fam_out, -1L)))
    T_tot_out <- sum(T_fam_out)

    # Identify matching family indices in ds_cat12
    fam_names_out <- names(ds_out$families)
    cat12_fam_idx <- match(fam_names_out, names(ds_cat12$families))
    for (f in seq_len(F_out)) {
      B_fam_out[f] <- ds_cat12$families[[cat12_fam_idx[f]]]$B_f
    }
    br_start_out <- cumsum(c(1L, head(B_fam_out, -1L)))
    B_tot_out <- sum(B_fam_out)

    A_scaled_out <- matrix(0, nrow = T_tot_out, ncol = B_tot_out)
    for (f in seq_len(F_out)) {
      f_c12 <- ds_cat12$families[[cat12_fam_idx[f]]]
      f_out <- ds_out$families[[f]]
      row_match <- match(f_out$data$glottocode, f_c12$data$glottocode)
      ts <- tip_start_out[f]; tf <- T_fam_out[f]
      bs <- br_start_out[f];  bf <- B_fam_out[f]
      A_scaled_out[ts:(ts + tf - 1L), bs:(bs + bf - 1L)] <- f_c12$A_scaled[row_match, , drop = FALSE]
    }

    base_sdata <- list(
      T = T_tot_out, B = B_tot_out, F = F_out,
      tip_start = as.array(tip_start_out), T_fam = as.array(T_fam_out),
      br_start  = as.array(br_start_out),  B_fam = as.array(B_fam_out),
      A_scaled  = A_scaled_out,
      y_pres    = as.array(ds_out$data$y_pres),
      y_coded   = as.array(ds_out$data$y_coded),
      J_hat     = rep(0, B_tot_out),
      z_jump    = rep(0, T_tot_out),
      use_dir   = 0L,
      use_rate  = 0L
    )

    # 0. Fit Strict-Clock Baseline once per (cutoff, outcome)
    base_tag <- sprintf("%s__%s__clock_baseline", cut_label, out_name)
    fit_base <- fit_one_s2(base_tag, base_sdata)
    cat(format(Sys.time(), "%H:%M:%S"),
        sprintf("[Stage 2] Baseline %s (elpd=%.1f, sigma_morph=%.2f, div=%d)\n",
                base_tag, fit_base$elpd_loo, fit_base$sigma_morph_mean, fit_base$divergences))

    s2_rows[[length(s2_rows) + 1L]] <- data.frame(
      cutoff = cut_label, outcome = out_name, measure = "none", spec = "clock",
      n_families = F_out, n_langs = T_tot_out,
      beta_dir_mean = NA_real_, beta_dir_q025 = NA_real_, beta_dir_q975 = NA_real_, p_beta_neg = NA_real_,
      gamma_rate_mean = NA_real_, gamma_rate_q025 = NA_real_, gamma_rate_q975 = NA_real_, p_gamma_pos = NA_real_,
      sigma_morph_mean = fit_base$sigma_morph_mean, sd_fam_mean = fit_base$sd_fam_mean, phi_mean = fit_base$phi_mean,
      elpd_loo = fit_base$elpd_loo, delta_elpd = 0.0, se_delta_elpd = 0.0,
      pareto_k_gt07 = fit_base$pareto_k_gt07, max_rhat = fit_base$max_rhat,
      min_ess_bulk = fit_base$min_ess_bulk, divergences = fit_base$divergences, minutes = fit_base$minutes,
      stringsAsFactors = FALSE
    )

    for (gname in names(gen_specs)) {
      s1 <- s1_results[[paste(cut_label, gname, sep = "__")]]
      # Extract branch jumps J_hat for the active families
      J_vec <- numeric(B_tot_out)
      for (f in seq_len(F_out)) {
        c12_f <- cat12_fam_idx[f]
        bs_c12 <- blk_cat12$br_start[c12_f]; bf_c12 <- blk_cat12$B_fam[c12_f]
        bs_out <- br_start_out[f]
        J_vec[bs_out:(bs_out + bf_c12 - 1L)] <- s1$J_mean[bs_c12:(bs_c12 + bf_c12 - 1L)]
      }
      # Standardize J_vec to unit SD across branches so gamma_rate is comparable across measures
      j_sd <- sd(J_vec)
      J_hat_std <- if (j_sd > 1e-6) (J_vec - mean(J_vec)) / j_sd else J_vec

      # Cumulative branch jump along root-to-tip path (standardized across outcome tips)
      cum_raw <- s1$cum_jump[ds_out$data$glottocode]
      z_jump_cum <- if (sd(cum_raw) > 1e-6) as.numeric(scale(cum_raw)) else rep(0, T_tot_out)

      # Signed within-family shift from family root (standardized across outcome tips)
      sgn_raw <- s1$signed_dev[ds_out$data$glottocode]
      z_jump_sgn <- if (sd(sgn_raw) > 1e-6) as.numeric(scale(sgn_raw)) else rep(0, T_tot_out)

      spec_list <- list(
        rate       = list(use_dir = 0L, use_rate = 1L, z = z_jump_cum),
        dir_jump   = list(use_dir = 1L, use_rate = 0L, z = z_jump_cum),
        both       = list(use_dir = 1L, use_rate = 1L, z = z_jump_cum),
        dir_signed = list(use_dir = 1L, use_rate = 0L, z = z_jump_sgn)
      )

      for (sp_name in names(spec_list)) {
        sp_cfg <- spec_list[[sp_name]]
        sdata_m <- base_sdata
        sdata_m$J_hat    <- as.array(J_hat_std)
        sdata_m$z_jump   <- as.array(sp_cfg$z)
        sdata_m$use_dir  <- sp_cfg$use_dir
        sdata_m$use_rate <- sp_cfg$use_rate

        m_tag <- sprintf("%s__%s__%s__%s", cut_label, out_name, gname, sp_name)
        fit_m <- fit_one_s2(m_tag, sdata_m)
        lc    <- suppressWarnings(loo_compare(fit_m$loo, fit_base$loo))
        delpd <- fit_m$elpd_loo - fit_base$elpd_loo
        se_d  <- as.numeric(lc[2, "se_diff"])

        s2_rows[[length(s2_rows) + 1L]] <- data.frame(
          cutoff = cut_label, outcome = out_name, measure = gname, spec = sp_name,
          n_families = F_out, n_langs = T_tot_out,
          beta_dir_mean = fit_m$beta_dir_mean, beta_dir_q025 = fit_m$beta_dir_q025,
          beta_dir_q975 = fit_m$beta_dir_q975, p_beta_neg = fit_m$p_beta_neg,
          gamma_rate_mean = fit_m$gamma_rate_mean, gamma_rate_q025 = fit_m$gamma_rate_q025,
          gamma_rate_q975 = fit_m$gamma_rate_q975, p_gamma_pos = fit_m$p_gamma_pos,
          sigma_morph_mean = fit_m$sigma_morph_mean, sd_fam_mean = fit_m$sd_fam_mean, phi_mean = fit_m$phi_mean,
          elpd_loo = fit_m$elpd_loo, delta_elpd = delpd, se_delta_elpd = se_d,
          pareto_k_gt07 = fit_m$pareto_k_gt07, max_rhat = fit_m$max_rhat,
          min_ess_bulk = fit_m$min_ess_bulk, divergences = fit_m$divergences, minutes = fit_m$minutes,
          stringsAsFactors = FALSE
        )
      }
      cat(format(Sys.time(), "%H:%M:%S"),
          sprintf("[Stage 2] Done %s / %s / %s\n", cut_label, out_name, gname))
    }
  }
}

s2_df <- do.call(rbind, s2_rows)
write.csv(s2_df, file.path(res_dir, "stage2_morph_rrw_results.csv"), row.names = FALSE)
cat("\n=== ALL STAGE 1 & STAGE 2 STAN FITS COMPLETE ===\n")
