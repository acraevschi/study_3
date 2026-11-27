# Two-Level Within-Language Token-Frequency Entrenchment & Cross-Level Admixture Interaction
# (`analyses/paradigm/stage2/fit_entrenchment.R`), fitted separately for Verbs (`V`) and Nouns (`N`).
#
# Tests:
#   1. Universal Irregularity Entrenchment (Cotterell et al. 2018):
#      `beta_freq > 0` (`z_freq_within` positively predicts lemma unpredictability `h_tree_lemma`,
#      meaning high-frequency lexical items sustain greater paradigm irregularity/suppletion).
#   2. Selective Contact Erosion (`Admixture x Token Frequency` interaction `beta_int`):
#      Tests whether population admixture (`f3_fixed`, `min_n_ind = 3L` primary + `5L` check)
#      selectively regularizes low-frequency lemmas while high-frequency lemmas remain entrenched
#      (plus Savage-Dickey `BF_01` & ROPE null tests).
#
# Model structure (with Normal(0, 0.5) fixed-effect priors matching Stage 2 & Track 1):
#   y_lemma ~ 1 + x_adm * z_freq_within + z_log_navail +
#     (1 | gr(glottocode, cov = A_phylo)) + (1 + z_freq_within | lang_id)
#
# Environment variables:
#   ENT_DRY=1 (or PARA_DRY=1):     validate Stan code & data via make_stancode/make_standata; sample nothing
#   ENT_SMOKE=1 (or PARA_SMOKE=1): fast 400-iter 2-chain smoke fit
#   ENT_CORES=4 (or PARA_CORES=4): dynamic CPU core budget across chains/workers (or --cores=N; default 4)
#   ENT_THREADS=1 (or PARA_THREADS): within-chain Stan threads (or --threads=N; default 1)
#   ENT_POS=V,N (or PARA_POS):     POS pipelines (default: V,N)
#   ENT_MEASURES=f3_fixed:         admixture measures (default: f3_fixed)
#   ENT_CUTOFFS=3,5:               population sample-size cutoffs (default: 3,5)
#   ENT_DATA_DIR / PARA_DATA_DIR:  prepared Stage-2 directory
#   ENT_OUT_DIR / PARA_OUT_DIR:    Stage-2 results directory

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
  library(parallel)
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

getenv_pref <- function(primary, fallback, default_val = "") {
  v1 <- Sys.getenv(primary, "")
  if (nzchar(v1)) return(v1)
  v2 <- Sys.getenv(fallback, "")
  if (nzchar(v2)) return(v2)
  default_val
}

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

dry <- nzchar(Sys.getenv("ENT_DRY")) || nzchar(Sys.getenv("PARA_DRY"))
smoke <- nzchar(Sys.getenv("ENT_SMOKE")) || nzchar(Sys.getenv("PARA_SMOKE"))

cores_env <- getenv_pref("ENT_CORES", "PARA_CORES", "")
CORES <- max(1L, as.integer(if (nzchar(cores_env)) cores_env else parse_cli_int("cores", 4L)))
threads_env <- getenv_pref("ENT_THREADS", "PARA_THREADS", "")
THREADS <- max(1L, as.integer(if (nzchar(threads_env)) threads_env else parse_cli_int("threads", 1L)))
if (THREADS > CORES) THREADS <- CORES

if (smoke) {
  ITER <- 400L; WARMUP <- 200L; CHAINS <- min(2L, max(1L, CORES %/% THREADS))
} else {
  ITER <- 2500L; WARMUP <- 1250L; CHAINS <- min(4L, max(1L, CORES %/% THREADS))
}
PAR_FITS <- max(1L, CORES %/% (CHAINS * THREADS))
stopifnot(PAR_FITS * CHAINS * THREADS <= CORES)

SEED <- 20261112L
PRIOR_SD_B <- 0.5
PRIOR_DENS_ZERO <- dnorm(0.0, mean = 0.0, sd = PRIOR_SD_B)
ROPE_HALF_WIDTH <- 0.15

data_dir <- getenv_pref("ENT_DATA_DIR", "PARA_DATA_DIR", "analyses/paradigm/results/stage2/prepared")
out_dir_env <- getenv_pref("ENT_OUT_DIR", "PARA_OUT_DIR", "")
base_out_dir <- if (nzchar(out_dir_env)) out_dir_env else "analyses/paradigm/results/stage2"
fits_dir <- if (dry) tempdir() else file.path(base_out_dir, if (smoke) "fits_smoke" else "fits")
res_dir <- if (dry) {
  if (nzchar(out_dir_env)) out_dir_env else tempdir()
} else {
  file.path(base_out_dir, if (smoke) "results_smoke" else ".")
}

pos_env <- getenv_pref("ENT_POS", "PARA_POS", "")
pos_raw <- if (nzchar(pos_env)) pos_env else parse_cli_str("pos", "V,N")
if (toupper(trimws(pos_raw)) == "BOTH") pos_raw <- "V,N"
pos_list <- trimws(strsplit(pos_raw, ",")[[1]])
measures_active <- trimws(strsplit(getenv_pref("ENT_MEASURES", "PARA_MEASURES", "f3_fixed"), ",")[[1]])
cutoffs_active <- as.integer(trimws(strsplit(Sys.getenv("ENT_CUTOFFS", "3,5"), ",")[[1]]))
stopifnot(all(pos_list %in% c("V", "N")))
stopifnot(all(measures_active %in% names(get_stage1_measure_specs())))
stopifnot(all(is.finite(cutoffs_active) & cutoffs_active >= 1L))

stan_cache_dir <- file.path(base_out_dir, paste0("stan_cache_ent_", paste(pos_list, collapse = "_")))

dir.create(fits_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)
if (!dry) {
  dir.create(stan_cache_dir, showWarnings = FALSE, recursive = TRUE)
  options(cmdstanr_write_stan_file_dir = stan_cache_dir)
}

k5_file <- file.path(common_dir, "data/f3_scores_k5.csv")
f3_fixed_file <- file.path(common_dir, "data/f3_fixed_scores.csv")
fst_regimes_file <- file.path(common_dir, "data/fst_neighbor_regimes.csv")

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
    q10  = unname(quantile(draws_vec, 0.10)),
    q90  = unname(quantile(draws_vec, 0.90)),
    q975 = unname(quantile(draws_vec, 0.975)),
    p_pos = mean(draws_vec > 0.0),
    p_neg = mean(draws_vec < 0.0),
    bf01_norm = d0_n / PRIOR_DENS_ZERO,
    bf01_kde = if (is.finite(d0_k)) d0_k / PRIOR_DENS_ZERO else NA_real_,
    p_rope_015 = mean(abs(draws_vec) <= ROPE_HALF_WIDTH)
  )
  setNames(vals, paste0(pfx, "_", names(vals)))
}

build_ent_subs_map <- function(d_main, pop_main, min_n_ind) {
  pop_u <- pop_main[!is.na(pop_main$n_individuals) & pop_main$n_individuals >= min_n_ind, , drop = FALSE]
  has_col <- function(col) {
    if (!(col %in% names(pop_u))) return(character(0))
    intersect(d_main$glottocode, unique(pop_u$glottocode[!is.na(pop_u[[col]])]))
  }
  has_col_se <- function(col, se_col) {
    if (!all(c(col, se_col) %in% names(pop_u))) return(character(0))
    ok <- !is.na(pop_u[[col]]) & !is.na(pop_u[[se_col]]) & is.finite(pop_u[[se_col]]) & pop_u[[se_col]] > 0
    intersect(d_main$glottocode, unique(pop_u$glottocode[ok]))
  }
  list(
    all          = intersect(d_main$glottocode, unique(pop_u$glottocode)),
    f3_fixed     = has_col("f3_fixed"),
    f3_fixed_iso = has_col("f3_fixed_iso"),
    f4           = has_col_se("f4_est", "f4_se"),
    neigh_out    = has_col("neigh_flow_out"),
    neigh        = has_col("neigh_flow"),
    het          = has_col_se("het", "het_se"),
    f3           = has_col("f3_z"),
    f3b3         = has_col("f3b3_z")
  )
}

run_single_ent_job <- function(job) {
  form_ent <- bf(as.formula(job$form_str))
  prior_ent <- c(
    prior(normal(0, 1.5), class = Intercept),
    prior(normal(0, 0.5), class = b),
    prior(exponential(1), class = sd),
    prior(exponential(1), class = sigma),
    prior(lkj(2), class = cor)
  )
  t0 <- Sys.time()
  if (dry) {
    make_stancode(form_ent, data = job$dat, data2 = list(A_phylo = job$A_sub),
                  family = gaussian(), prior = prior_ent, threads = threading(THREADS))
    make_standata(form_ent, data = job$dat, data2 = list(A_phylo = job$A_sub),
                  family = gaussian(), prior = prior_ent)
    return(list(
      pos = job$pos_code,
      row = list(
        name = job$fit_name, pos = job$pos_code, sample = job$sample_tag,
        measure = job$measure, min_n_ind = as.integer(job$cutoff),
        n_lang = nrow(job$lang_sub), n_lemmas = nrow(job$dat),
        formula = job$form_str
      ),
      lang_slopes = NULL
    ))
  }

  fit <- tryCatch(
    brm(
      form_ent, data = job$dat, data2 = list(A_phylo = job$A_sub),
      family = gaussian(), prior = prior_ent,
      chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS,
      threads = threading(THREADS), seed = SEED,
      control = list(adapt_delta = 0.99, max_treedepth = 12),
      file = file.path(fits_dir, job$fit_name), file_refit = "on_change",
      refresh = 0, silent = 2
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(name = job$fit_name, pos = job$pos_code, error = conditionMessage(fit)))
  }

  dr <- as_draws_df(fit)
  sm <- summarise_draws(dr)

  row_out <- c(
    list(
      name = job$fit_name, pos = job$pos_code, sample = job$sample_tag,
      measure = job$measure, min_n_ind = as.integer(job$cutoff),
      n_lang = nrow(job$lang_sub), n_lemmas = nrow(job$dat),
      max_rhat = max(sm$rhat, na.rm = TRUE),
      min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
      divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
      y_bits_mean = job$y_mu, y_bits_sd = job$y_sd,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")),
      formula = job$form_str
    ),
    summ_term_eq(dr$b_z_freq_within, "b_freq"),
    summ_term_eq(dr$b_x_adm, "b_adm"),
    summ_term_eq(dr$`b_x_adm:z_freq_within`, "b_int")
  )

  lang_slope_df <- NULL
  if (isTRUE(job$extract_lang_slopes)) {
    ls_rows <- list()
    for (gc in job$lang_sub$glottocode) {
      r_col <- sprintf("r_lang_id[%s,z_freq_within]", gc)
      if (r_col %in% names(dr)) {
        sl_draws <- dr$b_z_freq_within + dr$`b_x_adm:z_freq_within` * job$lang_sub$x_adm[job$lang_sub$glottocode == gc] + dr[[r_col]]
        ls_rows[[length(ls_rows) + 1L]] <- data.frame(
          pos = job$pos_code,
          glottocode = gc,
          family = job$lang_sub$family[job$lang_sub$glottocode == gc],
          x_adm = job$lang_sub$x_adm[job$lang_sub$glottocode == gc],
          slope_freq_mean = mean(sl_draws),
          slope_freq_q025 = unname(quantile(sl_draws, 0.025)),
          slope_freq_q10  = unname(quantile(sl_draws, 0.10)),
          slope_freq_q90  = unname(quantile(sl_draws, 0.90)),
          slope_freq_q975 = unname(quantile(sl_draws, 0.975)),
          p_slope_pos = mean(sl_draws > 0.0),
          stringsAsFactors = FALSE
        )
      }
    }
    if (length(ls_rows) > 0L) lang_slope_df <- do.call(rbind, ls_rows)
  }

  list(pos = job$pos_code, row = row_out, lang_slopes = lang_slope_df)
}

build_entrenchment_jobs_pos <- function(pos_code) {
  lf_file <- file.path(data_dir, sprintf("lemma_freq_data_%s.csv", pos_code))
  ld_file <- file.path(data_dir, sprintf("language_data_%s.csv", pos_code))
  ind_file <- file.path(data_dir, sprintf("individuals_%s.csv", pos_code))
  pop_file <- file.path(data_dir, sprintf("populations_%s.csv", pos_code))
  nwk_file <- file.path(data_dir, sprintf("phylo_tree_%s.nwk", pos_code))

  if (!all(file.exists(c(lf_file, ld_file, ind_file, pop_file, nwk_file)))) {
    message("Skipping entrenchment POS=", pos_code, " (prepared files absent)")
    return(list())
  }

  lf0 <- read.csv(lf_file, stringsAsFactors = FALSE)
  if (nrow(lf0) < 30L) {
    message("Skipping entrenchment POS=", pos_code, " (< 30 lemma rows)")
    return(list())
  }
  ld0 <- read.csv(ld_file, stringsAsFactors = FALSE)
  ind0 <- read.csv(ind_file, stringsAsFactors = FALSE)
  pop0 <- augment_population_f3_columns(
    read.csv(pop_file, stringsAsFactors = FALSE),
    k5_file = k5_file,
    f3_fixed_file = f3_fixed_file,
    fst_regimes_file = fst_regimes_file
  )
  if ("f3_z" %in% names(pop0)) {
    pop0$f3k_z[is.na(pop0$f3_z)] <- NA_real_
    pop0$f3k_est[is.na(pop0$f3_z)] <- NA_real_
    pop0$f3k_est_pos[is.na(pop0$f3_z)] <- NA_real_
  }
  raw_tree <- read.tree(nwk_file)

  d_main <- ld0[ld0$in_main == 1L, , drop = FALSE]
  A_all <- build_grafen_corr(raw_tree, d_main$glottocode)
  pop_main <- pop0[pop0$in_main == 1L & pop0$glottocode %in% d_main$glottocode, , drop = FALSE]

  # Reuse saved Stage-1 exposures ONLY when exact measure, cutoff, and language coverage match
  saved_expo_all <- file.path(res_dir, sprintf("exposure_all_%s.csv", pos_code))
  expo_cached <- if (!dry && file.exists(saved_expo_all)) read.csv(saved_expo_all, stringsAsFactors = FALSE) else NULL

  s1_cache <- new.env(parent = emptyenv())
  get_expo <- function(m, min_n_ind, subs_map) {
    key <- sprintf("%s__n%d", m, as.integer(min_n_ind))
    if (exists(key, envir = s1_cache, inherits = FALSE)) {
      return(get(key, envir = s1_cache, inherits = FALSE))
    }
    sb_code <- subset_of_measure(m)
    if (!is.null(expo_cached) && "min_n_ind" %in% names(expo_cached)) {
      sub_e <- expo_cached[
        expo_cached$set == "main" &
        expo_cached$measure == m &
        as.integer(expo_cached$min_n_ind) == as.integer(min_n_ind),
        , drop = FALSE
      ]
      if (nrow(sub_e) > 0L && setequal(sub_e$glottocode, subs_map[[sb_code]])) {
        assign(key, sub_e, envir = s1_cache)
        return(sub_e)
      }
    }
    e_fit <- fit_stage1_exposure(
      measure = m, ind_df = ind0, pop_df = pop_main, set_name = "main",
      fits_dir = fits_dir, fit_prefix = paste0("para_", pos_code, "_"),
      min_n_ind = as.integer(min_n_ind),
      dry = dry, chains = CHAINS, iter = ITER, warmup = WARMUP,
      threads = THREADS, seed = SEED
    )$e
    assign(key, e_fit, envir = s1_cache)
    e_fit
  }

  use_phylo_slope <- nzchar(Sys.getenv("PARA_ENT_PHYLO_SLOPE"))
  primary_m <- measures_active[1]
  primary_cutoff <- cutoffs_active[1]

  # Fit or load Stage-1 exposures in the main process before parallel Stage-2 sampling
  for (cutoff in cutoffs_active) {
    subs_map <- build_ent_subs_map(d_main, pop_main, min_n_ind = cutoff)
    for (m in measures_active) {
      sb_code <- subset_of_measure(m)
      if (length(subs_map[[sb_code]]) >= 5L) {
        invisible(get_expo(m, cutoff, subs_map))
      }
    }
  }

  jobs <- list()
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

    for (cutoff in cutoffs_active) {
      subs_map <- build_ent_subs_map(d_main, pop_main, min_n_ind = cutoff)
      for (m in measures_active) {
        sb_code <- subset_of_measure(m)
        req_gcs <- intersect(unique(lf_base$glottocode), subs_map[[sb_code]])
        if (length(req_gcs) < 5L) next

        e_m <- get_expo(m, cutoff, subs_map)
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
        form_str <- paste0("y_lemma ~ 1 + x_adm * z_freq_within", nav_term, " + ", rhs_re)
        A_sub <- A_all[lang_sub$glottocode, lang_sub$glottocode, drop = FALSE]
        fit_name <- sprintf("entrench_%s__%s__%s_n%d", pos_code, sample_tag, m, as.integer(cutoff))

        jobs[[length(jobs) + 1L]] <- list(
          pos_code = pos_code,
          sample_tag = sample_tag,
          measure = m,
          cutoff = as.integer(cutoff),
          fit_name = fit_name,
          form_str = form_str,
          dat = dat,
          lang_sub = lang_sub,
          A_sub = A_sub,
          y_mu = y_mu,
          y_sd = y_sd,
          extract_lang_slopes = (sample_tag == "all_lemmas" && m == primary_m && cutoff == primary_cutoff && !use_phylo_slope)
        )
      }
    }
  }
  jobs
}

all_ent_jobs <- unlist(lapply(pos_list, build_entrenchment_jobs_pos), recursive = FALSE)

message(sprintf(
  "Entrenchment (%s, cores=%d, chains=%d, threads=%d, par_fits=%d): %d models queued across POS=%s",
  if (dry) "DRY" else if (smoke) "SMOKE" else "FULL",
  CORES, CHAINS, THREADS, PAR_FITS, length(all_ent_jobs), paste(pos_list, collapse = ",")
))

if (length(all_ent_jobs) == 0L) {
  all_ent_res <- list()
} else if (dry || PAR_FITS <= 1L || length(all_ent_jobs) <= 1L) {
  all_ent_res <- lapply(all_ent_jobs, run_single_ent_job)
} else {
  cl <- makePSOCKcluster(min(length(all_ent_jobs), PAR_FITS), outfile = "")
  all_ent_res <- tryCatch({
    export_syms <- c(
      "dry", "smoke", "CHAINS", "ITER", "WARMUP", "THREADS", "SEED",
      "PRIOR_SD_B", "PRIOR_DENS_ZERO", "ROPE_HALF_WIDTH",
      "fits_dir", "stan_cache_dir",
      "kde_at_zero", "summ_term_eq", "run_single_ent_job"
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
      suppressPackageStartupMessages({ library(brms); library(ape); library(posterior) })
      w_cache <- file.path(stan_cache_dir, paste0("worker_", Sys.getpid()))
      dir.create(w_cache, showWarnings = FALSE, recursive = TRUE)
      options(brms.backend = "cmdstanr", cmdstanr_write_stan_file_dir = w_cache)
      NULL
    })
    parLapplyLB(cl, all_ent_jobs, run_single_ent_job)
  }, finally = {
    try(stopCluster(cl), silent = TRUE)
  })
}

errs <- Filter(function(r) !is.list(r) || !is.null(r$error), all_ent_res)
if (length(errs) > 0L) {
  for (er in errs) message("ERROR (Entrenchment): ", er$name, " -> ", er$error)
  stop(sprintf("%d entrenchment model(s) failed.", length(errs)))
}

for (p_code in pos_list) {
  p_res <- Filter(function(r) identical(r$pos, p_code), all_ent_res)
  res_rows <- lapply(p_res, `[[`, "row")
  if (length(res_rows) > 0L) {
    cols <- unique(unlist(lapply(res_rows, names)))
    tab <- do.call(rbind, lapply(res_rows, function(r) {
      as.data.frame(lapply(setNames(cols, cols), function(k) {
        if (!is.null(r[[k]]) && length(r[[k]]) == 1L) r[[k]] else NA
      }), stringsAsFactors = FALSE)
    }))
    write.csv(tab, file.path(res_dir, paste0("entrenchment_results_", p_code, if (dry) "_dry" else "", ".csv")),
              row.names = FALSE)
  }
  ls_parts <- Filter(Negate(is.null), lapply(p_res, `[[`, "lang_slopes"))
  if (!dry && length(ls_parts) > 0L) {
    write.csv(do.call(rbind, ls_parts),
              file.path(res_dir, sprintf("language_entrenchment_slopes_%s.csv", p_code)),
              row.names = FALSE)
  }
  message("Entrenchment [POS=", p_code, "] ", length(res_rows), " models ",
          if (dry) "validated (DRY)" else "completed")
}
message("ENTRENCHMENT ", if (dry) "DRY VALIDATION " else "", "COMPLETE")
