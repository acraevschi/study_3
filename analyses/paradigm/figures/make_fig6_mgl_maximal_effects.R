# Figure 6: Full Anatomy of the Maximal MGL Paradigm-Entropy Model (Verbs & Nouns)
# Extracts exact posterior draws and verified GP coordinate scaling from cached brms
# maximal fits when present, exporting maximal_anatomy_summary.csv for standalone reproduction.
# Outputs: fig6_mgl_maximal_effects.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(grid)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  normalizePath("analyses/paradigm/figures", mustWork = FALSE)
}
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

s2_dir  <- file.path(root_dir, "analyses/paradigm/results/stage2")
fit_v_f <- file.path(s2_dir, "fits/s2_V__main__maximal__h_tree_bits__f3_fixed_PS.rds")
fit_n_f <- file.path(s2_dir, "fits/s2_N__main__maximal__h_tree_bits__f3_fixed_PS.rds")
sum_csv <- file.path(s2_dir, "maximal_anatomy_summary.csv")

summarize_vec <- function(v, section, pos, n_lang, term, label, d_km = NA_real_) {
  v <- as.numeric(v)
  data.frame(
    section    = section,
    pos        = pos,
    n_lang     = as.integer(n_lang),
    term       = term,
    label      = label,
    d_km       = as.numeric(d_km),
    mean       = mean(v),
    sd         = sd(v),
    median     = unname(quantile(v, 0.50)),
    q025       = unname(quantile(v, 0.025)),
    q10        = unname(quantile(v, 0.10)),
    q90        = unname(quantile(v, 0.90)),
    q975       = unname(quantile(v, 0.975)),
    p_gt0      = mean(v > 0),
    p_lt0      = mean(v < 0),
    p_rope_015 = mean(abs(v) <= 0.15),
    stringsAsFactors = FALSE
  )
}

extract_pos_anatomy <- function(fit_path, pos_lbl) {
  suppressPackageStartupMessages({
    library(brms)
    library(posterior)
  })
  fit <- readRDS(fit_path)
  sd_dat <- brms::standata(fit)
  n_lang <- nrow(fit$data)

  # Verify brms gp(sx, sy, sz) isotropic scaling on unit-sphere 3D chord coordinates
  raw_xyz <- as.matrix(fit$data[, c("sx", "sy", "sz")])
  norms <- sqrt(rowSums(raw_xyz^2))
  if (max(abs(norms - 1.0)) > 1e-4) {
    stop(sprintf("[%s] Expected unit-sphere (sx, sy, sz) coordinates.", pos_lbl))
  }
  d_raw <- as.matrix(dist(raw_xyz))
  d_xgp <- as.matrix(dist(sd_dat$Xgp_1))
  dmax_emp <- max(d_raw)
  dmax_stan <- as.numeric(sd_dat$dmax_1)
  scale_ratios <- d_raw[upper.tri(d_raw)] / d_xgp[upper.tri(d_xgp)]
  if (abs(dmax_emp - dmax_stan) > 1e-6 || max(abs(scale_ratios - dmax_stan)) > 1e-6) {
    stop(sprintf("[%s] Mismatch between Xgp_1 distances and raw chord / dmax_1.", pos_lbl))
  }

  # Verify Stan squared-exponential covariance kernel gp_exp_quad_cov(Xgp_1, sdgp, lscale)
  sc_txt <- paste(brms::stancode(fit), collapse = "\n")
  if (!grepl("gp_exp_quad_cov", sc_txt, fixed = TRUE)) {
    stop(sprintf("[%s] Expected gp_exp_quad_cov kernel in Stan code.", pos_lbl))
  }

  dr <- posterior::as_draws_df(fit)
  out <- list()

  fixed_defs <- list(
    list(sec = "fixed_slope", term = "b_x",                   lbl = "Cross-family f3"),
    list(sec = "fixed_slope", term = "b_z_pop",               lbl = "Speaker population (log)"),
    list(sec = "fixed_slope", term = "b_z_area",              lbl = "Geographic area (log)"),
    list(sec = "fixed_slope", term = "b_z_log_C_distinct",    lbl = "Paradigm size (log |C|)"),
    list(sec = "fixed_slope", term = "b_z_log_navail",        lbl = "Available lexicon (log N)"),
    list(sec = "fixed_prov",  term = "b_source_provparalex",  lbl = "Source: Paralex"),
    list(sec = "fixed_prov",  term = "b_source_provunimorph", lbl = "Source: UniMorph")
  )
  for (fd in fixed_defs) {
    out[[length(out) + 1L]] <- summarize_vec(dr[[fd$term]], fd$sec, pos_lbl, n_lang, fd$term, fd$lbl)
  }

  # Variance decomposition per posterior draw: V_tot = sd_lang^2 + sdgp^2 + sigma^2 + mean(se_y^2)
  se2_bar <- mean(fit$data$se_y^2)
  v_phy <- dr$sd_lang__Intercept^2
  v_gp  <- dr$sdgp_gpsxsysz^2
  v_res <- dr$sigma^2
  v_se  <- rep(se2_bar, nrow(dr))
  v_tot <- v_phy + v_gp + v_res + v_se

  var_defs <- list(
    list(term = "var_phylo", lbl = "Phylogeny",         v = 100 * v_phy / v_tot, sd_term = "sd_phylo", sd_v = dr$sd_lang__Intercept),
    list(term = "var_gp",    lbl = "Spatial GP",        v = 100 * v_gp  / v_tot, sd_term = "sd_gp",    sd_v = dr$sdgp_gpsxsysz),
    list(term = "var_resid", lbl = "Language residual", v = 100 * v_res / v_tot, sd_term = "sd_resid", sd_v = dr$sigma),
    list(term = "var_cv_se", lbl = "CV measurement SE", v = 100 * v_se  / v_tot, sd_term = "sd_cv_se", sd_v = sqrt(v_se))
  )
  for (vd in var_defs) {
    out[[length(out) + 1L]] <- summarize_vec(vd$v,    "variance_share", pos_lbl, n_lang, vd$term,    vd$lbl)
    out[[length(out) + 1L]] <- summarize_vec(vd$sd_v, "variance_sd",    pos_lbl, n_lang, vd$sd_term, vd$lbl)
  }

  # Spatial GP correlation decay rho(d_km) = exp(-0.5 * (d_chord(d_km) / (dmax * lscale))^2)
  earth_r_km <- 6371.0
  lsc <- dr$lscale_gpsxsysz
  d_chord_50 <- dmax_stan * lsc * sqrt(2 * log(2))
  d50_km <- 2 * earth_r_km * asin(pmin(1.0, d_chord_50 / 2))

  out[[length(out) + 1L]] <- summarize_vec(rep(dmax_stan, nrow(dr)), "gp_param", pos_lbl, n_lang, "dmax_chord", "Max unit-sphere chord")
  out[[length(out) + 1L]] <- summarize_vec(lsc,                      "gp_param", pos_lbl, n_lang, "lscale_gp",  "GP length-scale (scaled chord)")
  out[[length(out) + 1L]] <- summarize_vec(d50_km,                   "gp_param", pos_lbl, n_lang, "d50_km",     "Half-correlation distance (km)")

  d_grid <- seq(0, 6000, by = 50)
  for (dk in d_grid) {
    d_ch <- 2 * sin(dk / (2 * earth_r_km))
    rho_draws <- exp(-0.5 * (d_ch / (dmax_stan * lsc))^2)
    out[[length(out) + 1L]] <- summarize_vec(rho_draws, "gp_decay", pos_lbl, n_lang, "rho_km", "Spatial correlation", d_km = dk)
  }

  do.call(rbind, out)
}

if (file.exists(fit_v_f) && file.exists(fit_n_f) &&
    !identical(Sys.getenv("FORCE_CSV_FALLBACK"), "1")) {
  anat_df <- rbind(
    extract_pos_anatomy(fit_v_f, "Verbs"),
    extract_pos_anatomy(fit_n_f, "Nouns")
  )
  write.csv(anat_df, sum_csv, row.names = FALSE)
  message(sprintf("Extracted exact maximal posteriors from brms fits -> %s", basename(sum_csv)))
} else if (file.exists(sum_csv)) {
  anat_df <- read.csv(sum_csv, stringsAsFactors = FALSE)
  message(sprintf("Loaded tracked maximal posterior summary -> %s", basename(sum_csv)))
} else {
  stop("Neither cached maximal brms .rds fits nor maximal_anatomy_summary.csv found.")
}

n_v <- unique(anat_df$n_lang[anat_df$pos == "Verbs"])[1]
n_n <- unique(anat_df$n_lang[anat_df$pos == "Nouns"])[1]
lbl_v_leg <- sprintf("Verbs (n = %d)", n_v)
lbl_n_leg <- sprintf("Nouns (n = %d)", n_n)

# --- Panel A: Fixed-Effect Predictors in the Maximal Model (SD of H_tree) ---
dfA <- subset(anat_df, section %in% c("fixed_slope", "fixed_prov"))
row_levels_a <- c(
  "Cross-family f3",
  "Speaker population (log)",
  "Geographic area (log)",
  "Paradigm size (log |C|)",
  "Available lexicon (log N)",
  "Source: Paralex",
  "Source: UniMorph"
)
dfA$label <- factor(dfA$label, levels = rev(row_levels_a))
dfA$grp <- factor(
  ifelse(dfA$section == "fixed_slope",
         "Continuous\nslopes\n(per 1 SD)",
         "Lexicon\nprovenance\n(vs. ref., SD)"),
  levels = c("Continuous\nslopes\n(per 1 SD)", "Lexicon\nprovenance\n(vs. ref., SD)")
)
dfA$pos_plot <- factor(dfA$pos, levels = c("Nouns", "Verbs"))

y_expr_a <- c(
  "Cross-family f3"           = expression("Cross-family " * italic(f)[3]),
  "Speaker population (log)"  = "Speaker population (log)",
  "Geographic area (log)"     = "Geographic area (log)",
  "Paradigm size (log |C|)"   = expression("Paradigm size (log |" * italic(C) * "|)"),
  "Available lexicon (log N)" = expression("Available lexicon (log " * italic(N) * ")"),
  "Source: Paralex"           = "Source: Paralex",
  "Source: UniMorph"          = "Source: UniMorph"
)

v_cdist <- subset(dfA, pos == "Verbs" & term == "b_z_log_C_distinct")
n_cdist <- subset(dfA, pos == "Nouns" & term == "b_z_log_C_distinct")
ann_cdist <- data.frame(
  grp   = factor("Continuous\nslopes\n(per 1 SD)", levels = levels(dfA$grp)),
  label = factor("Paradigm size (log |C|)", levels = levels(dfA$label)),
  x     = max(0.18, n_cdist$q975 + 0.04),
  txt   = sprintf("Paradigm economy:\nVerbs %+.2f, Nouns %+.2f SD", v_cdist$mean, n_cdist$mean),
  stringsAsFactors = FALSE
)

xlim_a <- range(c(dfA$q025, dfA$q975, -0.78, 0.76), na.rm = TRUE)
xlim_a <- xlim_a + c(-0.04, 0.04) * diff(xlim_a)

pA <- ggplot(dfA, aes(x = mean, y = label, colour = pos_plot, shape = pos_plot, linetype = pos_plot)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = q025, xmax = q975), width = 0, orientation = "y",
                position = position_dodge(width = 0.54), linewidth = 0.48, alpha = 0.55,
                linetype = "solid", show.legend = FALSE) +
  geom_errorbar(aes(xmin = q10, xmax = q90), width = 0, orientation = "y",
                position = position_dodge(width = 0.54), linewidth = 1.35, alpha = 1.0,
                linetype = "solid", show.legend = FALSE) +
  geom_point(position = position_dodge(width = 0.54), size = 1.85) +
  # Zero-length segment injects linetype into the shared top legend key for Panel C
  geom_segment(data = subset(dfA, term == "b_x"),
               aes(x = mean, xend = mean, y = label, yend = label),
               linewidth = 0.90) +
  geom_text(data = ann_cdist, aes(x = x, y = label, label = txt),
            inherit.aes = FALSE, hjust = 0, vjust = 0.5, lineheight = 1.04,
            size = 7.1 / .pt, colour = INK_BODY, family = "Helvetica") +
  facet_grid(grp ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(
    values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN),
    breaks = c("Verbs", "Nouns"),
    labels = c(lbl_v_leg, lbl_n_leg),
    name   = NULL
  ) +
  scale_shape_manual(
    values = c("Verbs" = 16, "Nouns" = 17),
    breaks = c("Verbs", "Nouns"),
    labels = c(lbl_v_leg, lbl_n_leg),
    name   = NULL
  ) +
  scale_linetype_manual(
    values = c("Verbs" = "solid", "Nouns" = "22"),
    breaks = c("Verbs", "Nouns"),
    labels = c(lbl_v_leg, lbl_n_leg),
    name   = NULL
  ) +
  scale_y_discrete(labels = y_expr_a) +
  scale_x_continuous(breaks = c(-0.6, -0.3, 0.0, 0.3, 0.6)) +
  coord_cartesian(xlim = xlim_a) +
  labs(
    title = expression(bold("a   Fixed-effect predictors in the maximal ") * bolditalic(H)[bold("tree")] * bold(" model")),
    x = expression("Posterior effect on standardized " * italic(H)[tree] * " (SD; 80% & 95% CrI; shaded: \u00b10.15 SD)"),
    y = NULL
  ) +
  theme_track2() +
  theme(
    strip.placement = "outside",
    strip.text.y.left = element_text(colour = INK_BODY, face = "bold", size = 7.3, angle = 0, hjust = 1, lineheight = 1.04),
    axis.text.y = element_text(colour = INK_BODY, size = 7.5),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing.y = unit(7, "pt"),
    legend.position = "top",
    legend.justification = "right",
    legend.direction = "horizontal",
    legend.key.width = unit(22, "pt"),
    legend.text = element_text(size = 7.5, colour = INK_BODY),
    legend.margin = margin(t = -1, b = -3)
  )

# --- Panel B: Variance Decomposition of Paradigm Entropy (0-100% of total variance) ---
dfB <- subset(anat_df, section == "variance_share")
var_levels <- c("Phylogeny", "Spatial GP", "Language residual", "CV measurement SE")
dfB$label <- factor(dfB$label, levels = rev(var_levels))
dfB$pos_plot <- factor(dfB$pos, levels = c("Nouns", "Verbs"))

pB <- ggplot(dfB, aes(x = mean, y = label, colour = pos_plot, shape = pos_plot)) +
  geom_vline(xintercept = 0, colour = GRID_WARM, linewidth = 0.4) +
  geom_errorbar(aes(xmin = q025, xmax = q975), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 0.48, alpha = 0.55) +
  geom_errorbar(aes(xmin = q10, xmax = q90), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 1.35, alpha = 1.0) +
  geom_point(position = position_dodge(width = 0.52), size = 1.85) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(
    breaks = c(0, 25, 50, 75, 100),
    labels = function(x) paste0(x, "%")
  ) +
  coord_cartesian(xlim = c(-1.5, 102)) +
  labs(
    title = "b   Variance decomposition",
    x = "Share of total variance (80% & 95% CrI)",
    y = NULL
  ) +
  theme_track2() +
  theme(
    axis.text.y = element_text(colour = INK_BODY, size = 7.5),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25)
  )

# --- Panel C: Spatial GP Correlation Decay over Great-Circle Distance (0-6,000 km) ---
dfC <- subset(anat_df, section == "gp_decay")
dfC$pos <- factor(dfC$pos, levels = c("Verbs", "Nouns"))

d50_v <- subset(anat_df, section == "gp_param" & pos == "Verbs" & term == "d50_km")$median
d50_n <- subset(anat_df, section == "gp_param" & pos == "Nouns" & term == "d50_km")$median

pts_d50 <- data.frame(
  pos = factor(c("Verbs", "Nouns"), levels = c("Verbs", "Nouns")),
  d50 = c(d50_v, d50_n),
  y   = c(0.50, 0.50),
  stringsAsFactors = FALSE
)

ann_d50 <- data.frame(
  x   = c(5950, 5950),
  y   = c(0.985, 0.875),
  col = c(COL_VERB, COL_NOUN_TXT),
  lab = c(
    sprintf("'Verbs: '*italic(d)[0.5]*' = %s km'", format(round(d50_v), big.mark = ",")),
    sprintf("'Nouns: '*italic(d)[0.5]*' = %s km'", format(round(d50_n), big.mark = ","))
  ),
  stringsAsFactors = FALSE
)

pC <- ggplot(dfC, aes(x = d_km)) +
  geom_hline(yintercept = 0.50, colour = GREY_MID, linewidth = 0.35, linetype = "dashed") +
  geom_segment(data = pts_d50, aes(x = d50, xend = d50, y = 0, yend = 0.50, colour = pos),
               linewidth = 0.34, linetype = "dotted", alpha = 0.75, inherit.aes = FALSE) +
  geom_ribbon(aes(ymin = q10, ymax = q90, fill = pos), alpha = 0.14, colour = NA) +
  geom_line(aes(y = median, colour = pos, linetype = pos), linewidth = 0.92) +
  geom_point(data = pts_d50, aes(x = d50, y = y, colour = pos, shape = pos),
             size = 1.95, inherit.aes = FALSE) +
  geom_text(data = ann_d50, aes(x = x, y = y, label = lab),
            colour = ann_d50$col, hjust = 1, vjust = 0.5, parse = TRUE,
            size = 7.1 / .pt, family = "Helvetica", inherit.aes = FALSE) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_fill_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_linetype_manual(values = c("Verbs" = "solid", "Nouns" = "22"), guide = "none") +
  scale_x_continuous(
    breaks = c(0, 2000, 4000, 6000),
    labels = function(x) format(x, big.mark = ",", trim = TRUE)
  ) +
  scale_y_continuous(
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = c("0", "0.25", "0.50", "0.75", "1.00")
  ) +
  coord_cartesian(xlim = c(0, 6000), ylim = c(0, 1.05)) +
  labs(
    title = "c   Spatial GP correlation decay",
    x = "Great-circle distance (km)",
    y = expression("Spatial correlation " * rho(italic(d)))
  ) +
  theme_track2() +
  theme(
    plot.margin = margin(5, 6, 5, 10)
  )

row_bc <- wrap_elements(full = (pB | pC) + plot_layout(widths = c(1.04, 1.06)))
fig6 <- pA / row_bc + plot_layout(heights = c(1.36, 0.98))
save_track2_fig(fig6, fig_dir, "fig6_mgl_maximal_effects", width = 6.5, height = 5.65, dpi = 300)
