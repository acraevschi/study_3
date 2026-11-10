# Fit MB_2pl (1-factor 2PL IRT + domain/item varying admixture slopes, P control)
# on the 13 Irregularity features (`irreg13`) across `ent`, `sqrt_het`, and `f3k_est_pos`.
#
# Saves item-specific irregularity parameters to:
#   analyses/grambank/latent_sem/results/item_parameters_irreg13_mb2pl.csv
#
# Run (from repo root or any working directory):
#   LAT_DRY=1 Rscript analyses/grambank/latent_sem/fit_irreg13_mb2pl.R
#   Rscript analyses/grambank/latent_sem/fit_irreg13_mb2pl.R
#
# Environment variables:
#   LAT_DRY=1:    validate Stan syntax and all model data structures; fit nothing
#   LAT_SMOKE=1:  fast smoke run (400 iter, 2 chains) into results_smoke/
#   LAT_CHAINS=k: number of MCMC chains (default: 4; 2 when LAT_SMOKE=1)
#   LAT_CORES=k:  parallel CPU cores for Stan sampling (default: LAT_CHAINS)

Sys.setenv(
  OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES",
  PATH = paste("/usr/bin:/bin:/usr/sbin:/sbin", Sys.getenv("PATH"), sep = ":"),
  MAKE = "/usr/bin/make",
  R_MAKEVARS_USER = "/dev/null"
)
invisible(suppressWarnings(Sys.setlocale("LC_ALL", "en_US.UTF-8")))
suppressPackageStartupMessages({
  library(rstan)
  library(brms)
  library(ape)
  library(posterior)
})

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

dry <- nzchar(Sys.getenv("LAT_DRY"))
smoke <- nzchar(Sys.getenv("LAT_SMOKE"))
CHAINS <- as.integer(Sys.getenv("LAT_CHAINS", if (smoke) "2" else "4"))
CORES <- as.integer(Sys.getenv("LAT_CORES", as.character(CHAINS)))
ITER <- if (smoke) 400L else 2000L
WARMUP <- if (smoke) 200L else 1000L
SEED <- 20261030L
options(mc.cores = CORES)

res_dir <- if (dry) tempdir() else file.path(here, if (smoke) "results_smoke" else "results")
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

langs_df <- read.csv(file.path(data_dir, "language_table.csv"), stringsAsFactors = FALSE)
items_df <- read.csv(file.path(data_dir, "items_metadata.csv"), stringsAsFactors = FALSE)
obs_df <- read.csv(file.path(data_dir, "item_observations.csv"), stringsAsFactors = FALSE)
pop0 <- read.csv(file.path(data_dir, "populations.csv"), stringsAsFactors = FALSE)

k5_file <- file.path(common_data_dir, "f3_scores_k5.csv")
k5 <- read.csv(k5_file, stringsAsFactors = FALSE)
idx_k5 <- match(pop0$population, k5$population)
pop0$f3k_z <- -k5$min_z[idx_k5]
pop0$f3k_est <- -k5$min_est[idx_k5]
pop0$f3k_est_pos <- pmax(0, pop0$f3k_est)

raw_tree <- read.tree(file.path(data_dir, "phylo_tree.nwk"))
d_main_202 <- langs_df[langs_df$in_main == 1 & langs_df$n_coded >= 8, ]
tree <- compute.brlen(collapse.singles(keep.tip(raw_tree, d_main_202$glottocode)), method = "Grafen")
A_all <- vcv(tree, corr = TRUE)

to_rad <- pi / 180
langs_df$sx <- cos(langs_df$lat * to_rad) * cos(langs_df$lon * to_rad)
langs_df$sy <- cos(langs_df$lat * to_rad) * sin(langs_df$lon * to_rad)
langs_df$sz <- sin(langs_df$lat * to_rad)

load_stage1_exposures <- function(set = "main") {
  adm_res <- file.path(adm_dir, "results")
  files <- c(
    file.path(adm_res, paste0("exposure_", set, ".csv")),
    file.path(adm_res, paste0("exposure_", set, "_f3check.csv")),
    file.path(adm_res, paste0("exposure_", set, "_tailcheck.csv"))
  )
  files <- files[file.exists(files)]
  tabs <- lapply(files, read.csv, stringsAsFactors = FALSE)
  expo <- do.call(rbind, tabs)
  expo[!duplicated(expo[, c("measure", "glottocode")], fromLast = TRUE), ]
}

expo_all <- load_stage1_exposures("main")

subset_of <- function(m) {
  switch(m,
    neigh = "neigh",
    het = , sqrt_het = "het",
    f3_z = , f3_est = , f3k_z = , f3k_est = , f3k_est_pos = "f3",
    f3b3_z = , f3b3_est = "f3b3",
    "all"
  )
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
  list(
    Xgp = array(Xgp, dim = c(nrow(Xgp), 3L)),
    shape = nums[1],
    scale = nums[2]
  )
}

build_irreg13_standata <- function(d_lang, x_vec) {
  langs <- d_lang$glottocode
  N_lang <- length(langs)
  lang_map <- setNames(seq_along(langs), langs)

  im <- items_df[items_df$in_irreg13 == 1, , drop = FALSE]
  im <- im[order(im$item_order), , drop = FALSE]
  K_item <- nrow(im)
  stopifnot(K_item == 13L)
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

  A_sub <- A_all[langs, langs, drop = FALSE]
  L_phylo <- t(chol(A_sub))
  gp_in <- build_gp_inputs(d_lang)

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
    with_x = 1L,
    use_2pl = 1L,
    use_var_slopes = 1L,
    use_phylo = 1L,
    use_space = 0L,
    use_testlets = 1L,
    L_phylo = L_phylo,
    Xgp = gp_in$Xgp,
    lscale_prior_shape = gp_in$shape,
    lscale_prior_scale = gp_in$scale,
    .meta = list(items = im$item_id, domains = dom_levels, item_dom_names = im$domain)
  )
}

# Read item_1factor.stan and strip generated quantities block (since beta_item, lambda, alpha, b_x
# are all in transformed parameters/parameters, omitting the 9-pt Gauss-Hermite LOO loop speeds up sampling 10x)
stan_lines <- readLines(file.path(here, "stan", "item_1factor.stan"), warn = FALSE)
gq_idx <- grep("^generated quantities\\s*\\{", stan_lines)[1]
stan_code_fast <- paste(stan_lines[seq_len(gq_idx - 1L)], collapse = "\n")

if (dry) {
  s_chk <- rstan::stanc(model_code = stan_code_fast)
  stopifnot(isTRUE(s_chk$status))
  cat("DRY RUN: Stan syntax check passed for item_1factor.stan (parameter-only build).\n")
  sm <- NULL
} else {
  cat("Compiling item_1factor.stan (fast parameter-only build) via rstan...\n")
  sm <- rstan::stan_model(model_code = stan_code_fast)
}

d_set <- langs_df[langs_df$in_main == 1 & langs_df$n_coded >= 8 & langs_df$irreg13_coded >= 8, , drop = FALSE]
d_set <- d_set[order(d_set$glottocode), , drop = FALSE]
inset <- pop0[pop0$in_main == 1 & pop0$glottocode %in% d_set$glottocode, ]
has_col <- function(col) intersect(d_set$glottocode, unique(inset$glottocode[!is.na(inset[[col]])]))
subs <- list(
  all = d_set$glottocode,
  het = has_col("het"),
  f3 = has_col("f3_z")
)

measures <- c("ent", "sqrt_het", "f3k_est_pos")
all_item_rows <- list()

for (m in measures) {
  t0 <- Sys.time()
  sb <- subset_of(m)
  e_m <- expo_all[expo_all$measure == m & expo_all$glottocode %in% subs[[sb]], , drop = FALSE]
  d_sb <- merge(d_set[d_set$glottocode %in% subs[[sb]], , drop = FALSE],
                e_m[, c("glottocode", "eta_mean", "eta_sd", "y_mean", "y_sd")],
                by = "glottocode")
  d_sb <- d_sb[order(d_sb$glottocode), , drop = FALSE]
  mu <- mean(d_sb$eta_mean)
  s <- sd(d_sb$eta_mean)
  x_std <- (d_sb$eta_mean - mu) / s

  sdata <- build_irreg13_standata(d_sb, x_std)
  meta <- sdata$.meta
  sdata$.meta <- NULL
  nm <- paste0("irreg13__main__MB_2pl__", m, "_P")

  if (dry) {
    cat(sprintf("DRY RUN validated %s (N_lang=%d, K_item=%d, N_obs=%d, chains=%d, cores=%d)\n",
                nm, sdata$N_lang, sdata$K_item, sdata$N_obs, CHAINS, CORES))
    next
  }

  cat(sprintf("Fitting %s (N_lang=%d, K_item=%d, N_obs=%d, chains=%d, cores=%d)...\n",
              nm, sdata$N_lang, sdata$K_item, sdata$N_obs, CHAINS, CORES))
  fit <- rstan::sampling(
    object = sm,
    data = sdata,
    pars = c("alpha", "lambda", "beta_item", "b_x"),
    chains = CHAINS,
    cores = CORES,
    iter = ITER,
    warmup = WARMUP,
    seed = SEED,
    control = list(adapt_delta = 0.98, max_treedepth = 12L),
    refresh = 0
  )

  dr <- as.data.frame(fit)
  s_tab <- summary(fit)$summary
  max_rhat <- max(s_tab[, "Rhat"], na.rm = TRUE)
  min_ess <- min(s_tab[, "n_eff"], na.rm = TRUE)
  bx <- dr$b_x
  cat(sprintf("  Done %s in %.1fs | b_x = %.3f [%.3f, %.3f] | max_rhat = %.3f | min_ess = %.0f\n",
              nm, as.numeric(difftime(Sys.time(), t0, units = "secs")),
              mean(bx), quantile(bx, 0.025), quantile(bx, 0.975), max_rhat, min_ess))

  for (k in seq_len(sdata$K_item)) {
    ak <- dr[[sprintf("alpha[%d]", k)]]
    lk <- dr[[sprintf("lambda[%d]", k)]]
    bk <- dr[[sprintf("beta_item[%d]", k)]]
    all_item_rows[[length(all_item_rows) + 1L]] <- data.frame(
      model = nm,
      set = "main",
      item_set = "irreg13",
      arch = "MB_2pl",
      measure = m,
      ctrl = "phylogeny",
      item_id = meta$items[k],
      domain = meta$item_dom_names[k],
      alpha_mean = mean(ak),
      alpha_q025 = unname(quantile(ak, 0.025)),
      alpha_q975 = unname(quantile(ak, 0.975)),
      lambda_mean = mean(lk),
      lambda_q025 = unname(quantile(lk, 0.025)),
      lambda_q975 = unname(quantile(lk, 0.975)),
      p_lambda_gt_1 = mean(lk > 1.0),
      beta_mean = mean(bk),
      beta_sd = sd(bk),
      beta_q025 = unname(quantile(bk, 0.025)),
      beta_q05 = unname(quantile(bk, 0.05)),
      beta_q10 = unname(quantile(bk, 0.10)),
      beta_q90 = unname(quantile(bk, 0.90)),
      beta_q95 = unname(quantile(bk, 0.95)),
      beta_q975 = unname(quantile(bk, 0.975)),
      p_beta_neg = mean(bk < 0),
      p_beta_pos = mean(bk > 0),
      stringsAsFactors = FALSE
    )
  }
}

if (!dry) {
  out_df <- do.call(rbind, all_item_rows)
  out_csv <- file.path(res_dir, "item_parameters_irreg13_mb2pl.csv")
  write.csv(out_df, out_csv, row.names = FALSE)
  cat("Saved", nrow(out_df), "rows to", out_csv, "\n")
}
