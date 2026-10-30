# Admixture -> Grambank inflection, hierarchical exposure models on the 2026-10-20 links
# (EXPLORATORY; see README.md). Same design as analyses/draft_admixture_model_2026_10_10/fit_hier.R,
# run once per analysis set (main + the three link sensitivity sets).
#
# Stage 1 (exposure), per set and measure:
#   individual-level measures:  value_i ~ 1 + (1 | lang) + (1 | lang:population)
#   population-level measures:  value_p | se(se_p, sigma = TRUE) ~ 1 + (1 | lang)
#                               (se_p: bootstrap SE over the population's individuals;
#                                no `se` for graff, neigh, f3_z, f3_est; graff: bernoulli)
#   language exposure eta_l = Intercept + r_lang[l]  ->  posterior mean and SD.
# Stage 2 (outcome), per set, measure and control set:
#   n_present | trials(n_coded) ~ 1 + me(x, x_sd) [+ phylogeny] [+ space], beta-binomial.
#   Control sets N, P, S, F on the main set; only F (phylogeny + space) on the sensitivity sets.
# The stages are fitted separately ("cut"): the outcome does not inform the exposure.
#
# Run (from the repo root):  Rscript analyses/admixture_links_2026_10_21/fit.R [set ...]
# Environment: ADM_XMODE=me (default) | plain | both: how stage 2 uses the exposure (see README §10).
#                me:    me(x, x_sd), the stage-1 SD enters as classical measurement error (the draft's choice)
#                plain: x only, the posterior mean as an ordinary predictor
#              ADM_DRY=1 fits NOTHING: it builds every model (data, formula, priors) through
#   brms::make_stancode/make_standata, lists the planned fits and stops.
#              ADM_SMOKE=1 runs a tiny version into fits_smoke/ (fits models).
#              ADM_MEASURES=a,b,...: fit only these measures (default: the 12 of the main run).
#              ADM_OUTCOME=cat12 (default) | flex | flex26 | irreg13: which Grambank outcome to fit.
#              ADM_MIN_COV=p: minimum share of outcome features coded (default: 0.6).
#              ADM_SAMPLE=own (default) | intersect:
#                own:       all languages in the set meeting ADM_OUTCOME's coverage threshold (e.g. 202 for cat12, 193 for flex)
#                intersect: only languages meeting both cat12 (>=60%) and Flexivity coverage (e.g. 189 for cat12 x flex)
#              ADM_INTERSECT_WITH=flex (default) | flex26 | irreg13: partner outcome when ADM_OUTCOME=cat12 and ADM_SAMPLE=intersect.
#              ADM_CTRLS=N,P,S,F: override control sets (default: N,P,S,F on main, F on sensitivity sets).
#              ADM_PAR_FITS=k: parallel fits (default: 4 x 4 chains = 16 cores; 2 in smoke mode).
#              ADM_REUSE_STAGE1=1: reuse saved stage-1 exposure tables in results/ when available.
#              ADM_TAG=x: result files get the suffix _x, so other runs' result files are kept.
#   f3 sensitivity check (2026-10-21, README section 4):
#     ADM_XMODE=both ADM_MEASURES=f3k_z,f3k_est,f3b3_z,f3b3_est ADM_TAG=f3check Rscript .../fit.R main
Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")
suppressPackageStartupMessages({
  library(brms); library(ape); library(loo); library(posterior); library(parallel)
})
options(brms.backend = "cmdstanr")
here <- "analyses/admixture_links_2026_10_21"
smoke <- nzchar(Sys.getenv("ADM_SMOKE"))
dry <- nzchar(Sys.getenv("ADM_DRY"))
reuse_s1 <- nzchar(Sys.getenv("ADM_REUSE_STAGE1"))
fits_dir <- if (dry) tempdir() else file.path(here, if (smoke) "fits_smoke" else "fits")
res_dir <- if (dry) tempdir() else file.path(here, if (smoke) "results_smoke" else "results")
dir.create(fits_dir, showWarnings = FALSE); dir.create(res_dir, showWarnings = FALSE)
if (smoke) {
  ITER <- 400; WARMUP <- 200; CHAINS <- 2; THREADS <- 1; PAR_FITS <- as.integer(Sys.getenv("ADM_PAR_FITS", "2"))
} else {
  ITER <- 3000; WARMUP <- 1500; CHAINS <- 4; THREADS <- 1; PAR_FITS <- as.integer(Sys.getenv("ADM_PAR_FITS", "4"))
}
SEED <- 20261021
ALL_SETS <- c("main", "no_doubtful_language", "no_other_language_stand_ins", "strict")
xmodes <- switch(Sys.getenv("ADM_XMODE", "me"), me = "me", plain = "plain", both = c("me", "plain"),
                 stop("ADM_XMODE must be me, plain or both"))
outcome_specs <- list(
  cat12   = list(pres = "n_present",       coded = "n_coded",       total = 12L),
  flex    = list(pres = "flex_present",    coded = "flex_coded",    total = 30L),
  flex26  = list(pres = "flex26_present",  coded = "flex26_coded",  total = 26L),
  irreg13 = list(pres = "irreg13_present", coded = "irreg13_coded", total = 13L)
)
outcome <- Sys.getenv("ADM_OUTCOME", "cat12")
if (!outcome %in% names(outcome_specs)) stop("ADM_OUTCOME must be one of: ", paste(names(outcome_specs), collapse = ", "))
ospec <- outcome_specs[[outcome]]
sample_mode <- Sys.getenv("ADM_SAMPLE", "own")
if (!sample_mode %in% c("own", "intersect")) stop("ADM_SAMPLE must be own or intersect")
other_outcome <- if (outcome == "cat12") Sys.getenv("ADM_INTERSECT_WITH", "flex") else "cat12"
if (outcome == "cat12" && !other_outcome %in% setdiff(names(outcome_specs), "cat12"))
  stop("ADM_INTERSECT_WITH must be one of: ", paste(setdiff(names(outcome_specs), "cat12"), collapse = ", "))
min_cov <- as.numeric(Sys.getenv("ADM_MIN_COV", "0.6"))
cat12_ix_cov <- as.numeric(Sys.getenv("ADM_CAT12_MIN_COV", "0.6"))
stopifnot(!is.na(min_cov), min_cov >= 0, min_cov <= 1, !is.na(cat12_ix_cov), cat12_ix_cov >= 0, cat12_ix_cov <= 1)
is_default_cat12 <- outcome == "cat12" && sample_mode == "own" && abs(min_cov - 0.6) < 1e-9
cov_tag <- if (min_cov > 0) paste0("_cov", round(min_cov * 100)) else "_all"
ix_pair <- if (outcome == "cat12") other_outcome else outcome
ix_partner_sfx <- if (ix_pair != "flex") paste0("_", ix_pair) else ""
ix_extra <- if (sample_mode == "intersect" && outcome == "cat12" && other_outcome != "flex") paste0("_", other_outcome) else ""
out_prefix <- if (is_default_cat12) "" else if (sample_mode == "intersect") paste0(outcome, cov_tag, "_ix", ix_extra, "__") else paste0(outcome, cov_tag, "__")
s1_prefix <- if (is_default_cat12) "" else if (sample_mode == "intersect") paste0("ix_", ix_pair, cov_tag, "_") else paste0(outcome, cov_tag, "_")
default_tag <- if (is_default_cat12) "" else if (sample_mode == "intersect") paste0(outcome, cov_tag, "_intersect", ix_extra) else paste0(outcome, cov_tag)
args <- commandArgs(trailingOnly = TRUE)
sets <- if (length(args)) args else ALL_SETS
stopifnot(all(sets %in% ALL_SETS))

d0 <- read.csv(file.path(here, "language_data.csv"), stringsAsFactors = FALSE)
ind0 <- read.csv(file.path(here, "individuals.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(here, "populations.csv"), stringsAsFactors = FALSE)
cov_ok <- function(df, out_name, mc) {
  sp <- outcome_specs[[out_name]]
  df[[sp$coded]] > 0 & (df[[sp$coded]] / sp$total >= mc - 1e-9)
}
# In intersect mode, cat12 uses cat12_ix_cov (default 0.6) and the Flexivity partner uses min_cov,
# so ADM_OUTCOME=cat12 and ADM_OUTCOME=flex/flex26/irreg13 always select the identical language set.
keep_lang <- function(df) {
  if (sample_mode == "own") {
    ok <- cov_ok(df, outcome, min_cov)
  } else {
    ok <- cov_ok(df, "cat12", cat12_ix_cov) & cov_ok(df, ix_pair, min_cov)
  }
  df[ok, , drop = FALSE]
}
# f3 sensitivity measures: f3k = only the 5 nearest B and C (f3_scores_k5.csv);
# f3b3 = the ordinary f3 scores, only for populations with at least 3 relatives in the panel
pop0$f3b3_z <- ifelse(pop0$n_B >= 3, pop0$f3_z, NA)
pop0$f3b3_est <- ifelse(pop0$n_B >= 3, pop0$f3_est, NA)
k5_file <- file.path(here, "f3_scores_k5.csv")
if (file.exists(k5_file)) {
  k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
  i <- match(pop0$population, k5$population)
  pop0$f3k_z <- -k5$min_z[i]; pop0$f3k_est <- -k5$min_est[i]
} else pop0$f3k_z <- pop0$f3k_est <- NA_real_
pop0$f3k_est_pos <- pmax(0, pop0$f3k_est)
d_main <- keep_lang(d0[d0$in_main == 1, ])
raw_tree <- read.tree(file.path(here, "phylo_tree.nwk"))
tree <- compute.brlen(collapse.singles(keep.tip(raw_tree, d_main$glottocode)), method = "Grafen")
A_all <- vcv(tree, corr = TRUE)
to_rad <- pi / 180
d0$sx <- cos(d0$lat * to_rad) * cos(d0$lon * to_rad)
d0$sy <- cos(d0$lat * to_rad) * sin(d0$lon * to_rad)
d0$sz <- sin(d0$lat * to_rad)
d0$lang <- d0$glottocode
stopifnot(all(d_main$glottocode %in% rownames(A_all)))

fit_brm <- function(name, formula, data, family, prior, data2 = NULL, adapt = 0.99) {
  if (dry) {                      # validate only: Stan code and data are built, nothing is sampled
    make_stancode(formula, data = data, data2 = data2, family = family, prior = prior, threads = threading(THREADS))
    make_standata(formula, data = data, data2 = data2, family = family, prior = prior)
    return(structure(list(), class = "dry"))
  }
  brm(formula, data = data, data2 = data2, family = family, prior = prior,
      chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS, threads = threading(THREADS),
      seed = SEED, control = list(adapt_delta = adapt, max_treedepth = 12),
      file = file.path(fits_dir, name), file_refit = "on_change", refresh = 0, silent = 2)
}
# PSOCK workers avoid macOS fork() crashes when CmdStan invokes the system C++/Swift toolchain
par_run <- function(specs, fn, name = function(s) if (is.list(s)) s$name else as.character(s)) {
  if (!length(specs)) return(list())
  if (dry || PAR_FITS <= 1L || length(specs) == 1L) {
    res <- lapply(specs, fn)
  } else {
    cl <- makePSOCKcluster(min(length(specs), PAR_FITS), outfile = "")
    on.exit(stopCluster(cl), add = TRUE)
    clusterEvalQ(cl, {
      suppressPackageStartupMessages({ library(brms); library(ape); library(loo); library(posterior) })
      options(brms.backend = "cmdstanr"); NULL
    })
    clusterExport(cl, ls(envir = .GlobalEnv), envir = .GlobalEnv)
    clusterExport(cl, ls(envir = parent.frame()), envir = parent.frame())
    res <- parLapplyLB(cl, specs, fn)
  }
  for (i in seq_along(res)) if (!is.list(res[[i]]))
    res[[i]] <- list(name = name(specs[[i]]), error = "no result (worker died or failed outside the model)")
  res
}

exposure_specs <- list(
  ent         = list(level = "ind", col = "ent"),
  nonmax      = list(level = "ind", col = "nonmax"),
  ent_K12     = list(level = "ind", col = "ent_K12"),
  ent_K30     = list(level = "ind", col = "ent_K30"),
  pop_ent     = list(level = "pop", col = "pop_ent", se = "pop_ent_se"),
  het         = list(level = "pop", col = "het", se = "het_se"),
  support     = list(level = "pop", col = "support", se = "support_se"),
  amount      = list(level = "pop", col = "amount", se = "amount_se"),
  graff       = list(level = "pop", col = "graff_target", bern = TRUE),
  neigh       = list(level = "pop", col = "neigh_flow"),
  f3_z        = list(level = "pop", col = "f3_z"),
  f3_est      = list(level = "pop", col = "f3_est"),
  f3k_z       = list(level = "pop", col = "f3k_z"),
  f3k_est     = list(level = "pop", col = "f3k_est"),
  f3b3_z      = list(level = "pop", col = "f3b3_z"),
  f3b3_est    = list(level = "pop", col = "f3b3_est"),
  sqrt_het    = list(level = "pop", col = "sqrt_het", se = "sqrt_het_se"),
  f3k_est_pos = list(level = "pop", col = "f3k_est_pos")
)
measures <- if (nzchar(Sys.getenv("ADM_MEASURES"))) strsplit(Sys.getenv("ADM_MEASURES"), ",")[[1]] else names(exposure_specs)[1:12]
stopifnot(all(measures %in% names(exposure_specs)))
exposure_specs <- exposure_specs[measures]
if (smoke) exposure_specs <- exposure_specs[intersect(c("ent", "het", "f3_z"), names(exposure_specs))]
tag_raw <- if (nzchar(Sys.getenv("ADM_TAG"))) Sys.getenv("ADM_TAG") else default_tag
tag <- if (nzchar(tag_raw)) paste0("_", tag_raw) else ""
# measures that are missing for some populations are fitted on the languages that have them
# (f3k has scores for the same populations as f3, so it shares the f3 subset; the fit checks this)
subset_of <- function(m) switch(m, neigh = "neigh", het = , sqrt_het = "het",
                                f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
                                f3b3_z = , f3b3_est = "f3b3", "all")

stage1 <- function(set, m) {
  s <- exposure_specs[[m]]
  d_set <- keep_lang(d0[d0[[paste0("in_", set)]] == 1, ])
  inset <- pop0[pop0[[paste0("in_", set)]] == 1 & pop0$glottocode %in% d_set$glottocode, ]
  dat <- if (s$level == "ind") ind0[ind0$population %in% inset$population, ] else inset
  dat <- dat[!is.na(dat[[s$col]]), ]
  req_langs <- sort(unique(dat$glottocode))
  diag_cols <- c("set", "measure", "level", "n_obs", "n_lang", "max_rhat", "min_ess_bulk", "divergences", "sd_lang", "sd_pop", "sd_resid")
  if (reuse_s1 && !dry) {
    cand_sfx <- unique(c(tag, if (sample_mode == "intersect") c(paste0("_cat12", cov_tag, "_intersect", ix_partner_sfx), paste0("_", ix_pair, cov_tag, "_intersect")) else NULL, "", "_f3check", "_tailcheck"))
    for (sfx in cand_sfx) {
      ef <- file.path(res_dir, paste0("exposure_", set, sfx, ".csv"))
      df <- file.path(res_dir, paste0("stage1_diagnostics_", set, sfx, ".csv"))
      if (file.exists(ef) && file.exists(df)) {
        e0 <- read.csv(ef, stringsAsFactors = FALSE); d0_s1 <- read.csv(df, stringsAsFactors = FALSE)
        e_sub <- e0[e0$measure == m, , drop = FALSE]
        d_sub <- d0_s1[d0_s1$measure == m, , drop = FALSE]
        if (nrow(e_sub) && nrow(d_sub) && d_sub$n_obs[1] == nrow(dat) && setequal(e_sub$glottocode, req_langs)) {
          if (!"sd_pop" %in% names(d_sub)) d_sub$sd_pop <- NA_real_
          d_sub <- d_sub[, diag_cols, drop = FALSE]
          message(format(Sys.time(), "%H:%M"), " [", set, "] exposure ", m, " reused from ", basename(ef))
          return(list(e = e_sub, diag = d_sub))
        }
      }
    }
  }
  dat$y <- dat[[s$col]]; dat$lang <- dat$glottocode
  bern <- isTRUE(s$bern)
  ymu <- NA; sdy <- NA
  if (!bern) { ymu <- mean(dat$y); sdy <- sd(dat$y); dat$y <- (dat$y - ymu) / sdy }   # unit scale for generic priors
  pr <- c(prior(normal(0, 1.5), class = Intercept), prior(exponential(1), class = sd))
  if (!bern) pr <- c(pr, prior(exponential(1), class = sigma))
  if (s$level == "ind") f <- bf(y ~ 1 + (1 | lang) + (1 | lang:population))
  else if (!is.null(s$se)) { dat$se_y <- dat[[s$se]] / sdy; f <- bf(y | se(se_y, sigma = TRUE) ~ 1 + (1 | lang)) }
  else f <- bf(y ~ 1 + (1 | lang))
  fit <- fit_brm(paste0("exposure_", s1_prefix, set, "__", m), f, dat, if (bern) bernoulli() else gaussian(), pr)
  langs <- sort(unique(dat$lang))
  if (dry) return(list(e = data.frame(set = set, glottocode = langs, eta_mean = rnorm(length(langs)), eta_sd = 0.3, measure = m,
                                       y_mean = ymu, y_sd = sdy),
                       diag = data.frame(set = set, measure = m, n_obs = nrow(dat), n_lang = length(langs))))
  dr <- as_draws_df(fit)
  eta <- sapply(langs, function(l) dr$b_Intercept + dr[[sprintf("r_lang[%s,Intercept]", l)]])
  sm <- summarise_draws(dr)
  message(format(Sys.time(), "%H:%M"), " [", set, "] exposure ", m, " done")
  # eta is on the scale of the standardised measure: raw = y_mean + y_sd * eta (y_mean, y_sd: this set's data)
  list(e = data.frame(set = set, glottocode = langs, eta_mean = colMeans(eta), eta_sd = apply(eta, 2, sd), measure = m,
                      y_mean = ymu, y_sd = sdy),
       diag = data.frame(set = set, measure = m, level = s$level, n_obs = nrow(dat), n_lang = length(langs),
         max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
         divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
         sd_lang = mean(dr$sd_lang__Intercept),
         sd_pop = if (s$level == "ind") mean(dr$`sd_lang:population__Intercept`) else NA,
         sd_resid = if (bern) NA else mean(dr$sigma)))
}

rhs <- list(N = "", P = " + (1 | gr(lang, cov = A))", S = " + gp(sx, sy, sz)",
            F = " + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)")
ctrls_env <- if (nzchar(Sys.getenv("ADM_CTRLS"))) strsplit(Sys.getenv("ADM_CTRLS"), ",")[[1]] else NULL
if (!is.null(ctrls_env)) stopifnot(all(ctrls_env %in% names(rhs)))
make_prior <- function(ctrl, with_x) {
  p <- c(prior(normal(0, 1.5), class = Intercept), prior(gamma(2, 0.1), class = phi))
  if (with_x) p <- c(p, prior(normal(0, 0.5), class = b))
  if (ctrl %in% c("P", "F")) p <- c(p, prior(exponential(1), class = sd))
  if (ctrl %in% c("S", "F")) p <- c(p, prior(exponential(1), class = sdgp))
  p
}
# a failed model (or a worker that died) is reported and skipped; it must never stop the run
is_err <- function(r) !is.list(r) || !is.null(r$error)
err_text <- function(r) if (is.list(r)) paste(r$name, r$error) else as.character(r)
run_spec <- function(s) {
  tryCatch(run_spec_inner(s), error = function(e) list(name = s$name, error = conditionMessage(e)))
}
run_spec_inner <- function(s) {
  t0 <- Sys.time()
  fit <- tryCatch(fit_brm(s$name, bf(as.formula(s$formula)), s$data, beta_binomial(),
                          make_prior(s$ctrl, s$with_x), data2 = list(A = A_all[s$data$lang, s$data$lang]),
                          adapt = if (s$ctrl == "F") 0.995 else 0.99),
                  error = function(e) e)
  if (inherits(fit, "error")) return(list(name = s$name, error = conditionMessage(fit)))
  if (dry) return(list(name = s$name, set = s$set, measure = s$measure, xmode = s$xmode, ctrl = s$ctrl, subset = s$subset, n = nrow(s$data)))
  fit <- add_criterion(fit, "loo", file = file.path(fits_dir, s$name))
  dr <- as_draws_df(fit); sm <- summarise_draws(dr)
  out <- list(name = s$name, set = s$set, measure = s$measure, xmode = s$xmode, ctrl = s$ctrl, subset = s$subset, n = nrow(s$data),
              max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
              divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
              elpd_loo = fit$criteria$loo$estimates["elpd_loo", "Estimate"],
              pareto_k_gt_07 = sum(fit$criteria$loo$diagnostics$pareto_k > 0.7),
              minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  if (s$with_x) {
    b <- dr[[grep(if (identical(s$xmode, "plain")) "^b_x$" else "^bsp_me", names(dr), value = TRUE)[1]]]
    out <- c(out, list(b_mean = mean(b), b_q025 = unname(quantile(b, .025)), b_q05 = unname(quantile(b, .05)),
                       b_q95 = unname(quantile(b, .95)), b_q975 = unname(quantile(b, .975)), p_b_neg = mean(b < 0)),
             s$scale_info)
  }
  message(format(Sys.time(), "%H:%M"), " done ", s$name, sprintf(" (%.1f min)", out$minutes))
  out
}

run_set <- function(set) {
  message(format(Sys.time(), "%H:%M"), " ===== set ", set, " (outcome=", outcome, ", min_cov=", min_cov, ", sample=", sample_mode, ") =====")
  ctrls <- if (!is.null(ctrls_env)) ctrls_env else if (set == "main") names(rhs) else "F"
  spref <- paste0(out_prefix, set)
  d <- keep_lang(d0[d0[[paste0("in_", set)]] == 1, ])
  d$n_present <- d[[ospec$pres]]; d$n_coded <- d[[ospec$coded]]

  # stage 1 (fitted on the active language sample for this set)
  s1 <- par_run(as.list(names(exposure_specs)),
                function(m) tryCatch(stage1(set, m), error = function(e) list(error = paste(m, conditionMessage(e)))))
  for (r in Filter(is_err, s1)) message("ERROR stage 1 ", err_text(r))
  s1 <- Filter(Negate(is_err), s1)
  expo <- do.call(rbind, lapply(s1, `[[`, "e"))
  write.csv(expo, file.path(res_dir, paste0("exposure_", set, tag, ".csv")), row.names = FALSE)
  write.csv(do.call(rbind, lapply(s1, `[[`, "diag")), file.path(res_dir, paste0("stage1_diagnostics_", set, tag, ".csv")), row.names = FALSE)

  # stage 2: baselines per (control set, language subset), then one model per measure and control set
  # a language is in a subset when it has at least one population with that measure in this set
  inset <- pop0[pop0[[paste0("in_", set)]] == 1 & pop0$glottocode %in% d$glottocode, ]
  if (any(names(exposure_specs) %in% c("f3_z", "f3_est", "f3k_z", "f3k_est", "f3k_est_pos", "f3b3_z", "f3b3_est"))) {
    if (any(is.na(inset$n_triples)))
      stop("populations in set '", set, "' are missing from f3_scores.csv; re-run f3.R and prepare.py first")
    if (any(names(exposure_specs) %in% c("f3k_z", "f3k_est", "f3k_est_pos")) && (!file.exists(k5_file) || any(!inset$population %in% k5$population)))
      stop("populations in set '", set, "' are missing from f3_scores_k5.csv; re-run 'Rscript .../f3.R <F2results> 3000 5' first")
  }
  has <- function(col) intersect(d$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
  subs <- list(all = d$glottocode, neigh = has("neigh_flow"), het = has("het"), f3 = has("f3_z"), f3b3 = has("f3b3_z"))
  stopifnot(setequal(has("f3_z"), has("f3_est")), setequal(has("f3b3_z"), has("f3b3_est")), setequal(has("het"), has("sqrt_het")))
  if (file.exists(k5_file) && all(inset$population %in% k5$population))
    stopifnot(setequal(has("f3_z"), has("f3k_est")), setequal(has("f3_z"), has("f3k_est_pos")))
  subs <- subs[intersect(names(subs), vapply(names(exposure_specs), subset_of, ""))]   # baselines only where needed
  specs <- list()
  for (sb in names(subs)) for (ctrl in ctrls) {
    specs[[length(specs) + 1]] <- list(name = paste0(spref, "__base_", ctrl, "_", sb), set = set, measure = "none", xmode = "none", ctrl = ctrl,
      subset = sb, data = d[d$glottocode %in% subs[[sb]], ],
      formula = paste0("n_present | trials(n_coded) ~ 1", rhs[[ctrl]]), with_x = FALSE)
  }
  for (m in names(exposure_specs)) {
    if (!m %in% expo$measure) next
    e <- expo[expo$measure == m, ]
    dd <- merge(d, e[, c("glottocode", "eta_mean", "eta_sd")], by = "glottocode")
    if (!setequal(dd$glottocode, subs[[subset_of(m)]])) {    # with-x and baseline must see the same languages
      message("ERROR ", set, " ", m, ": exposure languages differ from the ", subset_of(m), " subset; measure skipped"); next
    }
    mu <- mean(dd$eta_mean); s <- sd(dd$eta_mean)
    # the slope is per SD of eta over this set's languages; with y_sd it converts to raw units
    # (b / (s * y_sd)), so slopes of different sets can be put on one scale
    scale_info <- list(x_center = mu, x_scale = s, y_mean = e$y_mean[1], y_sd = e$y_sd[1])
    dd$x <- (dd$eta_mean - mu) / s
    dd$x_sd <- dd$eta_sd / s
    for (xm in xmodes) for (ctrl in ctrls) {
      specs[[length(specs) + 1]] <- list(name = paste0(spref, "__", m, "_", ctrl, if (xm == "plain") "_plain" else ""),
        set = set, measure = m, xmode = xm, ctrl = ctrl, subset = subset_of(m), data = dd, with_x = TRUE, scale_info = scale_info,
        formula = paste0("n_present | trials(n_coded) ~ 1 + ", if (xm == "me") "me(x, x_sd)" else "x", rhs[[ctrl]]))
    }
  }
  results <- par_run(specs, run_spec)
  for (r in Filter(is_err, results)) message("ERROR ", err_text(r))
  ok <- Filter(Negate(is_err), results)
  if (dry) {
    message("DRY RUN [", set, "]: ", length(s1), " exposure models, ", length(ok), " outcome models validated, ",
            length(results) - length(ok), " failed; languages: ", nrow(d))
    print(do.call(rbind, lapply(ok, as.data.frame)), row.names = FALSE)
    return(invisible(NULL))
  }
  cols <- unique(unlist(lapply(ok, names)))
  tab <- do.call(rbind, lapply(ok, function(r)
    as.data.frame(lapply(setNames(cols, cols), function(k) if (length(r[[k]])) r[[k]] else NA))))
  ctrl_label <- c(N = "none", P = "phylogeny", S = "space", F = "phylogeny+space")   # "F" alone is read back as FALSE
  tab_out <- tab; tab_out$ctrl <- unname(ctrl_label[as.character(tab$ctrl)])
  write.csv(tab_out, file.path(res_dir, paste0("model_results_", set, tag, ".csv")), row.names = FALSE)

  loo_of <- function(nm) {
    f <- file.path(fits_dir, paste0(nm, ".rds"))
    if (!file.exists(f)) stop("no fit ", nm)
    readRDS(f)$criteria$loo
  }
  safe <- function(expr_fn, what) tryCatch(expr_fn(), error = function(e) { message("SKIPPED ", what, ": ", conditionMessage(e)); NULL })
  cmp <- do.call(rbind, lapply(which(tab$measure != "none"), function(i) {
    r <- tab[i, ]; base <- paste0(spref, "__base_", r$ctrl, "_", r$subset)
    safe(function() {
      if (!all(c(r$name, base) %in% tab$name)) stop("model or baseline was not fitted in this run")
      l1 <- loo_of(r$name); l0 <- loo_of(base)
      data.frame(set = set, name = r$name, baseline = base,
                 delta_elpd_x = l1$estimates["elpd_loo", "Estimate"] - l0$estimates["elpd_loo", "Estimate"],
                 se_delta = loo_compare(l1, l0)[2, "se_diff"])
    }, paste("LOO comparison", r$name))
  }))
  if (!is.null(cmp)) write.csv(cmp, file.path(res_dir, paste0("loo_x_vs_baseline_", set, tag, ".csv")), row.names = FALSE)
  if (length(ctrls) > 1) {
    ctrl_cmp <- do.call(rbind, lapply(names(subs), function(sb) safe(function() {
      if (!all(paste0(spref, "__base_", ctrls, "_", sb) %in% tab$name)) stop("a baseline was not fitted in this run")
      ls <- lapply(ctrls, function(c) loo_of(paste0(spref, "__base_", c, "_", sb))); names(ls) <- paste0("base_", ctrls, "_", sb)
      lc <- loo_compare(ls)
      data.frame(set = set, subset = sb, model = rownames(lc), elpd_diff = lc[, "elpd_diff"], se_diff = lc[, "se_diff"])
    }, paste("control comparison", sb))))
    if (!is.null(ctrl_cmp)) write.csv(ctrl_cmp, file.path(res_dir, paste0("loo_controls_", set, tag, ".csv")), row.names = FALSE)
  }
  message(format(Sys.time(), "%H:%M"), " ===== set ", set, " done =====")
}
for (set in sets) tryCatch(run_set(set), error = function(e) message("ERROR set ", set, ": ", conditionMessage(e)))
message("ALL DONE")
