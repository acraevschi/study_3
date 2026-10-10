# Draft admixture -> Grambank inflection models with a hierarchical exposure model
# (EXPLORATORY; see README.md section 7).
#
# Stage 1 (exposure), per measure:
#   individual-level measures:  value_i ~ 1 + (1 | lang) + (1 | lang:population)
#   population-level measures:  value_p ~ 1 + (1 | lang)       (graff: bernoulli, logit)
#   language exposure eta_l = Intercept + r_lang[l]  ->  posterior mean and SD.
# Stage 2 (outcome), per measure and control set:
#   n_present | trials(n_coded) ~ 1 + me(x, x_sd) [+ phylogeny] [+ space], beta-binomial,
#   x = eta_l standardised over languages, x_sd its posterior SD on the same scale.
# The two stages are fitted separately ("cut"): the outcome does not inform the exposure.
#
# Run: Rscript analyses/draft_admixture_model_2026_10_10/fit_hier.R

suppressPackageStartupMessages({
  library(brms); library(ape); library(loo); library(posterior); library(parallel)
})
options(brms.backend = "cmdstanr")
here <- "analyses/draft_admixture_model_2026_10_10"
fits_dir <- file.path(here, "fits_hier"); dir.create(fits_dir, showWarnings = FALSE)
ITER <- 3000; WARMUP <- 1500; CHAINS <- 4; THREADS <- 2; PAR_FITS <- 2
SEED <- 20261010

d <- read.csv(file.path(here, "language_data.csv"), stringsAsFactors = FALSE)
ind <- read.csv(file.path(here, "individuals.csv"), stringsAsFactors = FALSE)
pop <- read.csv(file.path(here, "populations.csv"), stringsAsFactors = FALSE)
tree <- compute.brlen(collapse.singles(read.tree(file.path(here, "phylo_tree.nwk"))), method = "Grafen")
A_all <- vcv(tree, corr = TRUE)
to_rad <- pi / 180
d$sx <- cos(d$lat * to_rad) * cos(d$lon * to_rad)
d$sy <- cos(d$lat * to_rad) * sin(d$lon * to_rad)
d$sz <- sin(d$lat * to_rad)
d$lang <- d$glottocode

fit_brm <- function(name, formula, data, family, prior, data2 = NULL, adapt = 0.99) {
  brm(formula, data = data, data2 = data2, family = family, prior = prior,
      chains = CHAINS, iter = ITER, warmup = WARMUP, cores = CHAINS, threads = threading(THREADS),
      seed = SEED, control = list(adapt_delta = adapt, max_treedepth = 12),
      file = file.path(fits_dir, name), file_refit = "on_change", refresh = 0, silent = 2)
}

# --- stage 1: exposure models --------------------------------------------------------
exposure_specs <- list(
  ent      = list(level = "ind", col = "ent"),
  nonmax   = list(level = "ind", col = "nonmax"),
  ent_K12  = list(level = "ind", col = "ent_K12"),
  ent_K30  = list(level = "ind", col = "ent_K30"),
  pop_ent  = list(level = "pop", col = "pop_ent"),
  support  = list(level = "pop", col = "support"),
  amount   = list(level = "pop", col = "amount"),
  graff    = list(level = "pop", col = "graff_target", bern = TRUE),
  neigh    = list(level = "pop", col = "neigh_flow")
)

exposure <- list(); stage1_diag <- list()
for (m in names(exposure_specs)) {
  s <- exposure_specs[[m]]
  dat <- if (s$level == "ind") ind else pop
  dat <- dat[!is.na(dat[[s$col]]), ]
  dat$y <- dat[[s$col]]
  dat$lang <- dat$glottocode
  bern <- isTRUE(s$bern)
  if (!bern) dat$y <- (dat$y - mean(dat$y)) / sd(dat$y)   # unit scale for generic priors
  f <- if (s$level == "ind") bf(y ~ 1 + (1 | lang) + (1 | lang:population)) else bf(y ~ 1 + (1 | lang))
  pr <- c(prior(normal(0, 1.5), class = Intercept), prior(exponential(1), class = sd))
  if (!bern) pr <- c(pr, prior(exponential(1), class = sigma))
  fit <- fit_brm(paste0("exposure_", m), f, dat, if (bern) bernoulli() else gaussian(), pr)
  dr <- as_draws_df(fit)
  langs <- sort(unique(dat$lang))
  eta <- sapply(langs, function(l) dr$b_Intercept + dr[[sprintf("r_lang[%s,Intercept]", l)]])
  e <- data.frame(glottocode = langs, eta_mean = colMeans(eta), eta_sd = apply(eta, 2, sd), measure = m)
  exposure[[m]] <- e
  sm <- summarise_draws(dr)
  stage1_diag[[m]] <- data.frame(measure = m, level = s$level, n_obs = nrow(dat), n_lang = length(langs),
    max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
    divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
    sd_lang = mean(dr$sd_lang__Intercept),
    sd_pop = if (s$level == "ind") mean(dr$`sd_lang:population__Intercept`) else if (bern) NA else mean(dr$sigma),
    sd_ind = if (s$level == "ind") mean(dr$sigma) else NA)
  message(format(Sys.time(), "%H:%M"), " exposure ", m, " done")
}
expo <- do.call(rbind, exposure)
write.csv(expo, file.path(here, "hier_exposure.csv"), row.names = FALSE)
write.csv(do.call(rbind, stage1_diag), file.path(here, "hier_stage1_diagnostics.csv"), row.names = FALSE)

# --- stage 2: outcome models -----------------------------------------------------------
rhs <- list(N = "", P = " + (1 | gr(lang, cov = A))", S = " + gp(sx, sy, sz)",
            F = " + (1 | gr(lang, cov = A)) + gp(sx, sy, sz)")
make_prior <- function(ctrl, with_x) {
  p <- c(prior(normal(0, 1.5), class = Intercept), prior(gamma(2, 0.1), class = phi))
  if (with_x) p <- c(p, prior(normal(0, 0.5), class = b))
  if (ctrl %in% c("P", "F")) p <- c(p, prior(exponential(1), class = sd))
  if (ctrl %in% c("S", "F")) p <- c(p, prior(exponential(1), class = sdgp))
  p
}

specs <- list()
subsets <- list(all = d$glottocode, fst = exposure$neigh$glottocode)
for (sb in names(subsets)) for (ctrl in names(rhs)) {
  specs[[length(specs) + 1]] <- list(name = paste0("base_", ctrl, "_", sb), measure = "none", ctrl = ctrl,
    subset = sb, data = d[d$glottocode %in% subsets[[sb]], ],
    formula = paste0("n_present | trials(n_coded) ~ 1", rhs[[ctrl]]), with_x = FALSE)
}
for (m in names(exposure_specs)) {
  e <- exposure[[m]]
  dd <- merge(d, e[, c("glottocode", "eta_mean", "eta_sd")], by = "glottocode")
  mu <- mean(dd$eta_mean); s <- sd(dd$eta_mean)
  dd$x <- (dd$eta_mean - mu) / s
  dd$x_sd <- dd$eta_sd / s
  sb <- if (m == "neigh") "fst" else "all"
  for (ctrl in names(rhs)) {
    specs[[length(specs) + 1]] <- list(name = paste0(m, "_", ctrl), measure = m, ctrl = ctrl, subset = sb,
      data = dd, formula = paste0("n_present | trials(n_coded) ~ 1 + me(x, x_sd)", rhs[[ctrl]]), with_x = TRUE)
  }
}

run_spec <- function(s) {
  t0 <- Sys.time()
  fit <- tryCatch(fit_brm(s$name, bf(as.formula(s$formula)), s$data, beta_binomial(),
                          make_prior(s$ctrl, s$with_x), data2 = list(A = A_all[s$data$lang, s$data$lang]),
                          adapt = if (s$ctrl == "F") 0.995 else 0.99),
                  error = function(e) e)
  if (inherits(fit, "error")) return(list(name = s$name, error = conditionMessage(fit)))
  fit <- add_criterion(fit, "loo", file = file.path(fits_dir, s$name))
  dr <- as_draws_df(fit); sm <- summarise_draws(dr)
  out <- list(name = s$name, measure = s$measure, ctrl = s$ctrl, subset = s$subset, n = nrow(s$data),
              max_rhat = max(sm$rhat, na.rm = TRUE), min_ess_bulk = min(sm$ess_bulk, na.rm = TRUE),
              divergences = sum(subset(nuts_params(fit), Parameter == "divergent__")$Value),
              elpd_loo = fit$criteria$loo$estimates["elpd_loo", "Estimate"],
              pareto_k_gt_07 = sum(fit$criteria$loo$diagnostics$pareto_k > 0.7),
              minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  if (s$with_x) {
    b <- dr[[grep("^bsp_me", names(dr), value = TRUE)[1]]]
    out <- c(out, list(b_mean = mean(b), b_q025 = unname(quantile(b, .025)), b_q05 = unname(quantile(b, .05)),
                       b_q95 = unname(quantile(b, .95)), b_q975 = unname(quantile(b, .975)), p_b_neg = mean(b < 0)))
  }
  message(format(Sys.time(), "%H:%M"), " done ", s$name, sprintf(" (%.1f min)", out$minutes))
  out
}
# one model per distinct Stan program first (compilation), then the rest in parallel
key <- vapply(specs, function(s) paste(s$formula, s$with_x), "")
first <- !duplicated(key)
results <- c(lapply(specs[first], run_spec),
             mclapply(specs[!first], run_spec, mc.cores = PAR_FITS, mc.preschedule = FALSE))
for (r in Filter(function(r) !is.null(r$error), results)) message("ERROR ", r$name, ": ", r$error)
ok <- Filter(function(r) is.null(r$error), results)
cols <- unique(unlist(lapply(ok, names)))
tab <- do.call(rbind, lapply(ok, function(r)
  as.data.frame(lapply(setNames(cols, cols), function(k) if (length(r[[k]])) r[[k]] else NA))))
write.csv(tab, file.path(here, "hier_model_results.csv"), row.names = FALSE)

cmp <- do.call(rbind, lapply(which(tab$measure != "none"), function(i) {
  r <- tab[i, ]; base <- paste0("base_", r$ctrl, "_", r$subset)
  l1 <- readRDS(file.path(fits_dir, paste0(r$name, ".rds")))$criteria$loo
  l0 <- readRDS(file.path(fits_dir, paste0(base, ".rds")))$criteria$loo
  data.frame(name = r$name, baseline = base,
             delta_elpd_x = l1$estimates["elpd_loo", "Estimate"] - l0$estimates["elpd_loo", "Estimate"],
             se_delta = loo_compare(l1, l0)[2, "se_diff"])
}))
write.csv(cmp, file.path(here, "hier_loo_x_vs_baseline.csv"), row.names = FALSE)
ctrl_cmp <- do.call(rbind, lapply(names(subsets), function(sb) {
  ls <- lapply(names(rhs), function(c) readRDS(file.path(fits_dir, paste0("base_", c, "_", sb, ".rds")))$criteria$loo)
  names(ls) <- paste0("base_", names(rhs), "_", sb)
  lc <- loo_compare(ls)
  data.frame(subset = sb, model = rownames(lc), elpd_diff = lc[, "elpd_diff"], se_diff = lc[, "se_diff"])
}))
write.csv(ctrl_cmp, file.path(here, "hier_loo_controls.csv"), row.names = FALSE)
message("ALL DONE")
