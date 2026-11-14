# Two-Level Within-Language Token-Frequency Entrenchment & Cross-Level Admixture Interaction
# (`analyses/paradigm/stage2/fit_entrenchment.R`), fitted separately for Verbs (`V`) and Nouns (`N`).
#
# Tests:
#   1. Universal Irregularity Entrenchment (Cotterell et al. 2018):
#      `beta_freq > 0` (`z_freq_within` positively predicts lemma unpredictability `h_tree_lemma`,
#      meaning high-frequency lexical items sustain greater paradigm irregularity/suppletion).
#   2. Selective Contact Erosion (`Admixture x Token Frequency` interaction `beta_int`):
#      Tests whether population admixture selectively regularizes low-frequency lemmas while
#      high-frequency lemmas remain entrenched (plus Savage-Dickey `BF_01` & ROPE null tests).
#
# Model structure (with Normal(0, 0.5) fixed-effect priors matching Stage 2 & Track 1):
#   y_lemma ~ 1 + x_adm * z_freq_within + z_log_navail +
#     (1 | gr(glottocode, cov = A_phylo)) + (1 + z_freq_within | lang_id)
#
# Environment variables:
#   PARA_DRY=1:          validate Stan code & data via make_stancode/make_standata; sample nothing
#   PARA_SMOKE=1:        fast 400-iter 2-chain smoke fit
#   PARA_CORES=4:        CPU core budget (default 4, hard cap 8)
#   PARA_POS=V,N:        POS pipelines (default: V,N)
#   PARA_MEASURES=...:   admixture measures (default: ent,sqrt_het,f3k_est_pos)
#   PARA_DATA_DIR=...:   prepared Stage-2 directory
#   PARA_OUT_DIR=...:    Stage-2 results directory

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
  library(posterior)
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
  ITER <- 2500L; WARMUP <- 1250L; CHAINS <- min(4L, CORES)
}
THREADS <- 1L
stopifnot(CHAINS * THREADS <= 16L)

SEED <- 20261112L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
ROPE_HALF_WIDTH <- 0.15

data_dir <- Sys.getenv("PARA_DATA_DIR", "analyses/paradigm/results/stage2/prepared")
base_out_dir <- Sys.getenv("PARA_OUT_DIR", "analyses/paradigm/results/stage2")
fits_dir <- if (dry) tempdir() else file.path(base_out_dir, if (smoke) "fits_smoke" else "fits")
res_dir <- if (dry) base_out_dir else file.path(base_out_dir, if (smoke) "results_smoke" else ".")

pos_list <- strsplit(Sys.getenv("PARA_POS", "V,N"), ",")[[1]]
measures_active <- strsplit(Sys.getenv("PARA_MEASURES", "ent,sqrt_het,f3k_est_pos"), ",")[[1]]
stan_cache_dir <- file.path(base_out_dir, paste0("stan_cache_ent_", paste(pos_list, collapse = "_")))

dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(stan_cache_dir, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = stan_cache_dir)
}
k5_file <- file.path(common_dir, "data/f3_scores_k5.csv")

kde_at_zero <- function(v) {
  s <- sd(v)
  if (is.na(s) || s < 1e-12) return(NA_real_)
  bw_val <- tryCatch(stats::bw.SJ(v), error = function(e) stats::bw.nrd0(v))
  dens <- stats::density(v, bw = bw_val, from = min(v, 0) - 3 * s, to = max(v, 0) + 3 * s, n = 2048L)
  as.numeric(stats::approx(dens$x, dens$y, xout = 0.0)$y)
}

summ_term_eq <- function(draws_vec, pfx) {
  mu <- mean(draws_vec); s <- sd(draws_vec)
  d0_n <- dnorm(0.0, mean = mu, sd = s)
  d0_k <- kde_at_zero(draws_vec)
  vals <- list(
    mean = mu, sd = s,
    q025 = unname(quantile(draws_vec, 0.025)),
    q975 = unname(quantile(draws_vec, 0.975)),
    p_pos = mean(draws_vec > 0.0),
    p_neg = mean(draws_vec < 0.0),
    bf01_norm = d0_n / PRIOR_DENS_ZERO,
    bf01_kde = if (is.finite(d0_k)) d0_k / PRIOR_DENS_ZERO else NA_real_,
    p_rope_015 = mean(abs(draws_vec) <= ROPE_HALF_WIDTH)
  )
  setNames(vals, paste0(pfx, "_", names(vals)))
}

run_entrenchment_pos <- function(pos_code) {
  lf_file <- file.path(data_dir, sprintf("lemma_freq_data_%s.csv", pos_code))
  ld_file <- file.path(data_dir, sprintf("language_data_%s.csv", pos_code))
  ind_file <- file.path(data_dir, sprintf("individuals_%s.csv", pos_code))
  pop_file <- file.path(data_dir, sprintf("populations_%s.csv", pos_code))
  nwk_file <- file.path(data_dir, sprintf("phylo_tree_%s.nwk", pos_code))

  if (!all(file.exists(c(lf_file, ld_file, ind_file, pop_file, nwk_file)))) {
    message("Skipping entrenchment POS=", pos_code, " (prepared files absent)")
    return(NULL)
  }

  lf0 <- read.csv(lf_file, stringsAsFactors = FALSE)
  if (nrow(lf0) < 30L) {
    message("Skipping entrenchment POS=", pos_code, " (< 30 lemma rows)")
    return(NULL)
  }
  ld0 <- read.csv(ld_file, stringsAsFactors = FALSE)
  ind0 <- read.csv(ind_file, stringsAsFactors = FALSE)
  pop0 <- augment_population_f3_columns(read.csv(pop_file, stringsAsFactors = FALSE), k5_file = k5_file)
  pop0$f3k_z[is.na(pop0$f3_z)] <- NA_real_
  pop0$f3k_est[is.na(pop0$f3_z)] <- NA_real_
  pop0$f3k_est_pos[is.na(pop0$f3_z)] <- NA_real_
  raw_tree <- read.tree(nwk_file)

  d_main <- ld0[ld0$in_main == 1L, , drop = FALSE]
  A_all <- build_grafen_corr(raw_tree, d_main$glottocode)

  # Reuse saved Stage-1 exposures ONLY when exact language coverage (`setequal`) matches
  saved_expo_file <- file.path(res_dir, sprintf("exposure_main_%s.csv", pos_code))
  expo_cached <- if (!dry && file.exists(saved_expo_file)) read.csv(saved_expo_file, stringsAsFactors = FALSE) else NULL
  pop_main <- pop0[pop0$in_main == 1L & pop0$glottocode %in% d_main$glottocode, , drop = FALSE]

  has_col <- function(col) intersect(d_main$glottocode, unique(pop_main$glottocode[!is.na(pop_main[[col]])]))
  subs_map <- list(
    all   = d_main$glottocode,
    neigh = has_col("neigh_flow"),
    het   = has_col("het"),
    f3    = has_col("f3_z"),
    f3b3  = has_col("f3b3_z")
  )

  get_expo <- function(m) {
    sb_code <- subset_of_measure(m)
    if (!is.null(expo_cached)) {
      sub_e <- expo_cached[expo_cached$measure == m, , drop = FALSE]
      if (nrow(sub_e) > 0L && setequal(sub_e$glottocode, subs_map[[sb_code]])) {
        return(sub_e)
      }
    }
    fit_stage1_exposure(
      measure = m, ind_df = ind0, pop_df = pop_main, set_name = "main",
      fits_dir = fits_dir, fit_prefix = paste0("para_", pos_code, "_"),
      dry = dry, chains = CHAINS, iter = ITER, warmup = WARMUP,
      threads = THREADS, seed = SEED
    )$e
  }

  prior_ent <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(normal(0, 0.5), class = b),
    prior(exponential(1), class = sd),
    prior(exponential(1), class = sigma),
    prior(lkj(2), class = cor)
  )
  use_phylo_slope <- nzchar(Sys.getenv("PARA_ENT_PHYLO_SLOPE"))

  res_rows <- list()
  lang_slope_rows <- list()

  for (sample_tag in c("all_lemmas", "attested_only")) {
    lf_base <- if (sample_tag == "attested_only") lf0[lf0$is_attested == 1L, , drop = FALSE] else lf0
    if (nrow(lf_base) < 30L) next

    # Re-center and standardize log10_freq_lemma strictly within each language (>= 10 items)
    cnt_by_gc <- table(lf_base$glottocode)
    valid_freq_gcs <- names(cnt_by_gc[cnt_by_gc >= 10L])
    lf_base <- lf_base[lf_base$glottocode %in% valid_freq_gcs, , drop = FALSE]
    if (length(unique(lf_base$glottocode)) < 5L) next

    sd_by_gc <- tapply(lf_base$log10_freq_lemma, lf_base$glottocode, sd)
    nonconst_gcs <- names(sd_by_gc[!is.na(sd_by_gc) & sd_by_gc > 1e-8])
    lf_base <- lf_base[lf_base$glottocode %in% nonconst_gcs, , drop = FALSE]
    if (length(unique(lf_base$glottocode)) < 5L) next

    for (m in measures_active) {
      e_m <- get_expo(m)
      sb_code <- subset_of_measure(m)
      req_gcs <- intersect(unique(lf_base$glottocode), subs_map[[sb_code]])
      if (length(req_gcs) < 5L) next

      e_sub <- e_m[e_m$glottocode %in% req_gcs, , drop = FALSE]
      stopifnot(setequal(e_sub$glottocode, req_gcs))

      lang_sub <- merge(
        d_main[d_main$glottocode %in% req_gcs, c("glottocode", "family", "log_navail"), drop = FALSE],
        e_sub[, c("glottocode", "eta_mean")],
        by = "glottocode"
      )
      lang_sub <- lang_sub[order(lang_sub$glottocode), , drop = FALSE]
      mu_x <- mean(lang_sub$eta_mean); sd_x <- sd(lang_sub$eta_mean)
      if (is.na(sd_x) || sd_x < 1e-8) next
      lang_sub$x_adm <- (lang_sub$eta_mean - mu_x) / sd_x
      ln_av <- as.numeric(lang_sub$log_navail)
      lang_sub$z_log_navail <- if (sd(ln_av) > 1e-8) as.numeric(scale(ln_av)) else 0.0

      dat <- merge(
        lf_base[lf_base$glottocode %in% req_gcs, , drop = FALSE],
        lang_sub[, c("glottocode", "family", "x_adm", "z_log_navail")],
        by = "glottocode"
      )
      # Exact within-language frequency standardization on active sample
      f_mu <- ave(dat$log10_freq_lemma, dat$glottocode, FUN = mean)
      f_sd <- ave(dat$log10_freq_lemma, dat$glottocode, FUN = sd)
      dat$z_freq_within <- (dat$log10_freq_lemma - f_mu) / f_sd

      y_mu <- mean(dat$h_tree_lemma); y_sd <- sd(dat$h_tree_lemma)
      if (is.na(y_sd) || y_sd < 1e-8) next
      dat$y_lemma <- (dat$h_tree_lemma - y_mu) / y_sd
      dat$lang_id <- dat$glottocode

      rhs_re <- if (use_phylo_slope) {
        "(1 + z_freq_within | gr(glottocode, cov = A_phylo))"
      } else {
        "(1 | gr(glottocode, cov = A_phylo)) + (1 + z_freq_within | lang_id)"
      }
      nav_term <- if (sd(dat$z_log_navail) > 1e-8) " + z_log_navail" else ""
      form_ent <- bf(as.formula(paste0(
        "y_lemma ~ 1 + x_adm * z_freq_within", nav_term, " + ", rhs_re
      )))
      A_sub <- A_all[lang_sub$glottocode, lang_sub$glottocode, drop = FALSE]
      fit_name <- paste0("entrench_", pos_code, "__", sample_tag, "__", m)

      t0 <- Sys.time()
      if (dry) {
        make_stancode(form_ent, data = dat, data2 = list(A_phylo = A_sub),
                      family = gaussian(), prior = prior_ent, threads = threading(THREADS))
        make_standata(form_ent, data = dat, data2 = list(A_phylo = A_sub),
                      family = gaussian(), prior = prior_ent)
        res_rows[[length(res_rows) + 1L]] <- list(
          name = fit_name, pos = pos_code, sample = sample_tag, measure = m,
          n_lang = nrow(lang_sub), n_lemmas = nrow(dat)
        )
        next
      }

      fit <- brm(
        form_ent, data = dat, data2 = list(A_phylo = A_sub),
        family = gaussian(), prior = prior_ent,
        chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS,
        threads = threading(THREADS), seed = SEED,
        control = list(adapt_delta = 0.99, max_treedepth = 12),
        file = file.path(fits_dir, fit_name), file_refit = "on_change",
        refresh = 0, silent = 2
      )
      dr <- as_draws_df(fit)
      sm <- summarise_draws(dr)

      row_out <- c(
        list(
          name = fit_name, pos = pos_code, sample = sample_tag, measure = m,
          n_lang = nrow(lang_sub), n_lemmas = nrow(dat),
          max_rhat = max(sm$rhat, na.rm = TRUE),
          min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
          divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
          y_bits_mean = y_mu, y_bits_sd = y_sd,
          minutes = as.numeric(difftime(Sys.time(), t0, units = "mins"))
        ),
        summ_term_eq(dr$b_z_freq_within, "b_freq"),
        summ_term_eq(dr$b_x_adm, "b_adm"),
        summ_term_eq(dr$`b_x_adm:z_freq_within`, "b_int")
      )
      res_rows[[length(res_rows) + 1L]] <- row_out

      # Extract per-language posterior entrenchment slopes on `all_lemmas` + `ent`
      if (sample_tag == "all_lemmas" && m == "ent" && !use_phylo_slope) {
        for (gc in lang_sub$glottocode) {
          r_col <- sprintf("r_lang_id[%s,z_freq_within]", gc)
          if (r_col %in% names(dr)) {
            sl_draws <- dr$b_z_freq_within + dr$`b_x_adm:z_freq_within` * lang_sub$x_adm[lang_sub$glottocode == gc] + dr[[r_col]]
            lang_slope_rows[[length(lang_slope_rows) + 1L]] <- data.frame(
              pos = pos_code,
              glottocode = gc,
              family = lang_sub$family[lang_sub$glottocode == gc],
              x_adm = lang_sub$x_adm[lang_sub$glottocode == gc],
              slope_freq_mean = mean(sl_draws),
              slope_freq_q025 = unname(quantile(sl_draws, 0.025)),
              slope_freq_q975 = unname(quantile(sl_draws, 0.975)),
              p_slope_pos = mean(sl_draws > 0.0),
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
  }

  if (length(res_rows) > 0L) {
    cols <- unique(unlist(lapply(res_rows, names)))
    tab <- do.call(rbind, lapply(res_rows, function(r) {
      as.data.frame(lapply(setNames(cols, cols), function(k) {
        if (!is.null(r[[k]]) && length(r[[k]]) == 1L) r[[k]] else NA
      }), stringsAsFactors = FALSE)
    }))
    write.csv(tab, file.path(res_dir, paste0("entrenchment_results_", pos_code, if (dry) "_dry" else "", ".csv")),
              row.names = FALSE)
  }
  if (!dry && length(lang_slope_rows) > 0L) {
    write.csv(do.call(rbind, lang_slope_rows),
              file.path(res_dir, sprintf("language_entrenchment_slopes_%s.csv", pos_code)),
              row.names = FALSE)
  }
  message("Entrenchment [POS=", pos_code, "] ", length(res_rows), " models ",
          if (dry) "validated (DRY)" else "completed")
}

for (p_code in pos_list) {
  run_entrenchment_pos(p_code)
}
message("ENTRENCHMENT ", if (dry) "DRY VALIDATION " else "", "COMPLETE")
