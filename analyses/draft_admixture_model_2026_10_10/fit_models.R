# Draft admixture -> Grambank inflection models (EXPLORATORY; see README.md).
#
# Outcome: n_present of the n_coded Grambank inflectional categories (beta-binomial, so
# languages with fewer coded categories carry less information instead of being rescaled).
# Controls: phylogeny (Glottolog tree from `low`, Grafen branch lengths, correlation
# matrix) and space (exact Gaussian process on the language's Glottolog point, unit-sphere
# coordinates so distances are chordal and the Pacific is not cut at 180 degrees).
#
# For each admixture measure x: x only (N), x + phylogeny (P), x + space (S), x + both (F);
# baselines without x on the same languages. Comparison by PSIS-LOO.
#
# Run: Rscript analyses/draft_admixture_model_2026_10_10/fit_models.R

suppressPackageStartupMessages({
  library(brms); library(ape); library(loo); library(posterior); library(parallel)
})
options(brms.backend = "cmdstanr", mc.cores = 4)
here <- "analyses/draft_admixture_model_2026_10_10"
fits_dir <- file.path(here, "fits"); dir.create(fits_dir, showWarnings = FALSE)
set.seed(20261010)

d <- read.csv(file.path(here, "language_data.csv"), stringsAsFactors = FALSE)
tree <- read.tree(file.path(here, "phylo_tree.nwk"))
tree <- collapse.singles(tree)
stopifnot(setequal(tree$tip.label, d$glottocode))
tree <- compute.brlen(tree, method = "Grafen")
A_all <- vcv(tree, corr = TRUE)

to_rad <- pi / 180
d$sx <- cos(d$lat * to_rad) * cos(d$lon * to_rad)
d$sy <- cos(d$lat * to_rad) * sin(d$lon * to_rad)
d$sz <- sin(d$lat * to_rad)
d$lang <- d$glottocode
z <- function(v) (v - mean(v, na.rm = TRUE)) / sd(v, na.rm = TRUE)

priors_x <- c(prior(normal(0, 1.5), class = Intercept), prior(normal(0, 0.5), class = b),
              prior(gamma(2, 0.1), class = phi))
pr_sd <- prior(exponential(1), class = sd)
pr_gp <- prior(exponential(1), class = sdgp)

rhs <- list(N = "", P = " + (1 | gr(lang, cov = A))", S = " + gp(sx, sy, sz)",
            F = " + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)")

make_prior <- function(ctrl, with_x) {
  p <- if (with_x) priors_x else priors_x[priors_x$class != "b", ]
  if (ctrl %in% c("P", "F")) p <- c(p, pr_sd)
  if (ctrl %in% c("S", "F")) p <- c(p, pr_gp)
  p
}

fit_one <- function(name, formula, data, ctrl, with_x) {
  f <- file.path(fits_dir, name)
  # GP + phylogeny models showed divergences at adapt_delta 0.95; the measurement-error
  # model mixes slowly: stricter adaptation and longer chains for those
  ad <- if (ctrl == "F") 0.995 else 0.95
  it <- if (grepl("me\\(", formula)) 4000 else if (ctrl == "F") 3000 else 2000
  A <- A_all[data$lang, data$lang]
  brm(bf(as.formula(formula)), data = data, data2 = list(A = A),
      family = beta_binomial(), prior = make_prior(ctrl, with_x),
      chains = 4, cores = 4, iter = it, warmup = it / 2, seed = 20261010,
      control = list(adapt_delta = ad, max_treedepth = 12),
      stan_model_args = list(cpp_options = list(PRECOMPILED_HEADERS = "false")),
      file = f, file_refit = "on_change", refresh = 0, silent = 2)
}

# --- specifications ------------------------------------------------------------------
measures <- c(ent = "ent", nonmax = "nonmax", pop_ent = "pop_ent",
              graff = "graff_target", neigh = "neigh_flow", ent_K12 = "ent_K12", ent_K30 = "ent_K30")
byk <- read.csv(file.path(here, "admixture_by_K.csv"))
d$ent_K12 <- byk$ent[byk$K == 12][match(d$glottocode, byk$glottocode[byk$K == 12])]
d$ent_K30 <- byk$ent[byk$K == 30][match(d$glottocode, byk$glottocode[byk$K == 30])]

specs <- list()
for (subset in c("all", "fst")) {
  dd <- if (subset == "all") d else d[!is.na(d$neigh_flow), ]
  for (ctrl in names(rhs)) {
    specs[[length(specs) + 1]] <- list(name = paste0("base_", ctrl, "_", subset), measure = "none",
      ctrl = ctrl, subset = subset, data = dd,
      formula = paste0("n_present | trials(n_coded) ~ 1", rhs[[ctrl]]), with_x = FALSE)
  }
}
for (m in names(measures)) {
  subset <- if (m == "neigh") "fst" else "all"
  dd <- if (subset == "all") d else d[!is.na(d$neigh_flow), ]
  col <- measures[[m]]
  dd$x <- if (m == "graff") dd[[col]] - mean(dd[[col]]) else z(dd[[col]])
  ctrls <- if (m %in% c("ent_K12", "ent_K30", "pop_ent", "nonmax")) c("N", "F") else names(rhs)
  for (ctrl in ctrls) {
    specs[[length(specs) + 1]] <- list(name = paste0(m, "_", ctrl), measure = m, ctrl = ctrl, subset = subset,
      data = dd, formula = paste0("n_present | trials(n_coded) ~ 1 + x", rhs[[ctrl]]), with_x = TRUE)
  }
}
# measurement-error sensitivity for the primary measure (sampling error of the language
# mean over individuals), full controls
dme <- d
dme$x <- z(d$ent); dme$x_se <- d$ent_se / sd(d$ent)
specs[[length(specs) + 1]] <- list(name = "ent_F_me", measure = "ent_me", ctrl = "F", subset = "all", data = dme,
  formula = "n_present | trials(n_coded) ~ 1 + me(x, x_se) + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)", with_x = TRUE)

# --- fit ----------------------------------------------------------------------------
# One model per distinct Stan program is fitted first, sequentially, so that the parallel
# workers reuse compiled executables instead of compiling the same file concurrently.
run_spec <- function(s) {
  t0 <- Sys.time()
  fit <- tryCatch(fit_one(s$name, s$formula, s$data, s$ctrl, s$with_x), error = function(e) e)
  if (inherits(fit, "error")) return(list(name = s$name, error = conditionMessage(fit)))
  fit <- add_criterion(fit, "loo", file = file.path(fits_dir, s$name))
  dr <- summarise_draws(as_draws_df(fit))
  np <- nuts_params(fit)
  out <- list(name = s$name, measure = s$measure, ctrl = s$ctrl, subset = s$subset, n = nrow(s$data),
              max_rhat = max(dr$rhat, na.rm = TRUE), min_ess_bulk = min(dr$ess_bulk, na.rm = TRUE),
              divergences = sum(np$Value[np$Parameter == "divergent__"]),
              elpd_loo = fit$criteria$loo$estimates["elpd_loo", "Estimate"],
              se_elpd = fit$criteria$loo$estimates["elpd_loo", "SE"],
              pareto_k_gt_07 = sum(fit$criteria$loo$diagnostics$pareto_k > 0.7),
              minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  bname <- if (s$measure == "ent_me") "bsp_mexx_se" else if (s$with_x) "b_x" else NA
  if (!is.na(bname)) {
    b <- as_draws_df(fit)[[bname]]
    out <- c(out, list(b_mean = mean(b), b_q025 = quantile(b, .025), b_q05 = quantile(b, .05),
                       b_q95 = quantile(b, .95), b_q975 = quantile(b, .975), p_b_neg = mean(b < 0)))
  }
  out
}
key <- vapply(specs, function(s) paste(s$formula, s$with_x), "")
first <- !duplicated(key)
log_line <- function(r) message(format(Sys.time(), "%H:%M"), " done ", r$name,
                                if (!is.null(r$error)) paste(" ERROR", r$error) else sprintf(" (%.1f min)", r$minutes))
results <- lapply(specs[first], function(s) { r <- run_spec(s); log_line(r); r })
results <- c(results, mclapply(specs[!first], function(s) { r <- run_spec(s); log_line(r); r },
                               mc.cores = 4, mc.preschedule = FALSE))

errs <- Filter(function(r) !is.null(r$error), results)
for (e in errs) message("ERROR ", e$name, ": ", e$error)
ok <- Filter(function(r) is.null(r$error), results)
cols <- unique(unlist(lapply(ok, names)))
tab <- do.call(rbind, lapply(ok, function(r) {
  as.data.frame(lapply(setNames(cols, cols), function(k) if (length(r[[k]])) unname(r[[k]]) else NA),
                stringsAsFactors = FALSE)
}))
write.csv(tab, file.path(here, "model_results_raw.csv"), row.names = FALSE)

# --- LOO comparison against the baseline with the same controls and languages ----------
cmp <- do.call(rbind, lapply(seq_len(nrow(tab)), function(i) {
  r <- tab[i, ]
  if (r$measure == "none") return(NULL)
  base <- paste0("base_", r$ctrl, "_", r$subset)
  f1 <- readRDS(file.path(fits_dir, paste0(r$name, ".rds")))
  f0 <- readRDS(file.path(fits_dir, paste0(base, ".rds")))
  lc <- loo_compare(f1$criteria$loo, f0$criteria$loo)   # only se_diff is used (symmetric)
  data.frame(name = r$name, baseline = base,
             delta_elpd_x = f1$criteria$loo$estimates["elpd_loo", "Estimate"] -
                            f0$criteria$loo$estimates["elpd_loo", "Estimate"],
             se_delta = lc[2, "se_diff"])
}))
write.csv(cmp, file.path(here, "loo_x_vs_baseline.csv"), row.names = FALSE)

ctrl_cmp <- lapply(c("all", "fst"), function(s) {
  fits <- lapply(names(rhs), function(c) readRDS(file.path(fits_dir, paste0("base_", c, "_", s, ".rds")))$criteria$loo)
  names(fits) <- paste0("base_", names(rhs), "_", s)
  lc <- loo_compare(fits)
  data.frame(subset = s, model = rownames(lc), elpd_diff = lc[, "elpd_diff"], se_diff = lc[, "se_diff"])
})
write.csv(do.call(rbind, ctrl_cmp), file.path(here, "loo_controls.csv"), row.names = FALSE)
print(tab[, c("name", "n", "b_mean", "b_q05", "b_q95", "p_b_neg", "max_rhat", "divergences", "pareto_k_gt_07", "minutes")])
print(cmp)
