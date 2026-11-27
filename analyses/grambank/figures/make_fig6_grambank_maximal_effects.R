# Figure 6 (Track 1): Full Anatomy of the Maximal Model Across All 5 Grambank Domains
#   a: Standardized fixed-effect predictors (f3_fixed, z_pop, z_area, z_doc): 2PL IRT-SEM vs. Beta-Binomial
#   b: Latent structural variance decomposition (Phylogeny, Spatial GP, Language residual; 2PL IRT-SEM)
#   c: Spatial GP correlation decay over great-circle distance (0-6,000 km; 2PL IRT-SEM)
# Outputs:
#   - analyses/grambank/latent_sem/results_domain/maximal_anatomy_summary.csv (when .rds caches exist)
#   - analyses/grambank/figures/fig6_grambank_maximal_effects.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(grid)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  normalizePath("analyses/grambank/figures", mustWork = FALSE)
}
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

R_EARTH_KM <- 6371.0
ROPE_HALF  <- 0.15

DOMAIN_META <- list(
  v_breadth = list(coded = "v_breadth_coded", min_coded = 6L, k_item = 9L),
  n_breadth = list(coded = "n_breadth_coded", min_coded = 3L, k_item = 5L),
  v_irreg   = list(coded = "v_irreg_coded",   min_coded = 3L, k_item = 5L),
  n_irreg   = list(coded = "n_irreg_coded",   min_coded = 2L, k_item = 3L),
  np_agr    = list(coded = "np_agr_coded",    min_coded = 5L, k_item = 7L)
)

summary_csv <- file.path(root_dir, "analyses/grambank/latent_sem/results_domain/maximal_anatomy_summary.csv")
sem_rds_paths <- setNames(
  file.path(root_dir, "analyses/grambank/latent_sem/fits/domain",
            sprintf("sem1f__%s__MA_2pl__maximal_main_n3.rds", DOMAIN_ORDER)),
  DOMAIN_ORDER
)
bb_rds_paths <- setNames(
  file.path(root_dir, "analyses/grambank/baseline/fits/bb",
            sprintf("bb__%s__maximal__f3_fixed_main_n3.rds", DOMAIN_ORDER)),
  DOMAIN_ORDER
)

summarise_vec <- function(v, include_moments = TRUE) {
  q <- unname(quantile(v, probs = c(0.025, 0.10, 0.50, 0.90, 0.975), na.rm = TRUE))
  list(
    mean       = if (include_moments && all(is.finite(v))) mean(v) else NA_real_,
    sd         = if (include_moments && all(is.finite(v))) sd(v) else NA_real_,
    q025       = q[1],
    q10        = q[2],
    q50        = q[3],
    q90        = q[4],
    q975       = q[5],
    p_pos      = mean(v > 0, na.rm = TRUE),
    p_neg      = mean(v < 0, na.rm = TRUE),
    p_rope_015 = mean(abs(v) <= ROPE_HALF, na.rm = TRUE)
  )
}

mk_summary_row <- function(dom, n_lang, k_item, d_max, comp_type, estimator, param, label, v,
                           include_moments = TRUE, include_dir = TRUE) {
  st <- summarise_vec(v, include_moments = include_moments)
  data.frame(
    domain         = dom,
    n_lang         = as.integer(n_lang),
    k_item         = as.integer(k_item),
    d_max          = as.numeric(d_max),
    component_type = comp_type,
    estimator      = estimator,
    parameter      = param,
    label          = label,
    mean           = st$mean,
    sd             = st$sd,
    q025           = st$q025,
    q10            = st$q10,
    q50            = st$q50,
    q90            = st$q90,
    q975           = st$q975,
    p_pos          = if (include_dir) st$p_pos else NA_real_,
    p_neg          = if (include_dir) st$p_neg else NA_real_,
    p_rope_015     = if (include_dir) st$p_rope_015 else NA_real_,
    stringsAsFactors = FALSE
  )
}

use_rds <- all(file.exists(c(sem_rds_paths, bb_rds_paths))) &&
           !identical(Sys.getenv("FORCE_CSV_FALLBACK"), "1")

if (use_rds) {
  suppressMessages(suppressWarnings(suppressPackageStartupMessages({
    library(posterior)
    if (requireNamespace("cmdstanr", quietly = TRUE)) library(cmdstanr)
    if (requireNamespace("brms", quietly = TRUE)) library(brms)
  })))

  ld <- read.csv(file.path(root_dir, "analyses/grambank/data/language_data.csv"), stringsAsFactors = FALSE)
  ex <- load_track1_f3_exposure(root_dir, set_name = "main", min_n = 3L)

  to_rad <- pi / 180.0
  ld$sx <- cos(ld$lat * to_rad) * cos(ld$lon * to_rad)
  ld$sy <- cos(ld$lat * to_rad) * sin(ld$lon * to_rad)
  ld$sz <- sin(ld$lat * to_rad)

  sum_rows <- list()

  for (dom in DOMAIN_ORDER) {
    meta <- DOMAIN_META[[dom]]
    sub_lang <- ld[
      ld$in_main == 1L &
        !is.na(ld[[meta$coded]]) & ld[[meta$coded]] >= meta$min_coded &
        !is.na(ld$has_area) & ld$has_area == 1L &
        is.finite(as.numeric(ld$log_pop)) &
        is.finite(as.numeric(ld$log_area)) &
        ld$glottocode %in% ex$glottocode,
      , drop = FALSE
    ]
    n_lang <- nrow(sub_lang)
    coords <- as.matrix(sub_lang[, c("sx", "sy", "sz"), drop = FALSE])
    d_max  <- max(as.matrix(dist(coords)))
    stopifnot(n_lang >= 10L, is.finite(d_max), d_max > 0)

    # 1. Focal 2PL IRT-SEM posterior draws
    fit_sem <- readRDS(sem_rds_paths[[dom]])
    dr_sem <- as_draws_df(fit_sem$draws(
      variables = c("b_x", "b_cov[1]", "b_cov[2]", "b_cov[3]",
                    "sd_phylo[1]", "sd_gp[1]", "sd_lang", "lscale_gp[1]")
    ))

    # Cross-check N_lang against fitted latent vector dimension
    stopifnot(ncol(fit_sem$draws("theta", format = "matrix")) == n_lang)

    sem_fixed <- list(
      b_f3   = list(lbl = "Cross-family f3",       vec = dr_sem[["b_x"]]),
      b_pop  = list(lbl = "Speaker pop. (log)",    vec = dr_sem[["b_cov[1]"]]),
      b_area = list(lbl = "Geographic area (log)", vec = dr_sem[["b_cov[2]"]]),
      b_doc  = list(lbl = "Doc. coverage",         vec = dr_sem[["b_cov[3]"]])
    )
    for (p_nm in names(sem_fixed)) {
      sum_rows[[length(sum_rows) + 1L]] <- mk_summary_row(
        dom, n_lang, meta$k_item, d_max, "fixed_slope", "2PL IRT-SEM",
        p_nm, sem_fixed[[p_nm]]$lbl, sem_fixed[[p_nm]]$vec,
        include_moments = TRUE, include_dir = TRUE
      )
    }

    # 2. Beta-Binomial posterior draws
    fit_bb <- readRDS(bb_rds_paths[[dom]])
    dr_bb <- as_draws_df(fit_bb)
    stopifnot(nrow(fit_bb$data) == n_lang)

    bb_fixed <- list(
      b_f3   = list(lbl = "Cross-family f3",       vec = dr_bb[["b_x"]]),
      b_pop  = list(lbl = "Speaker pop. (log)",    vec = dr_bb[["b_z_pop"]]),
      b_area = list(lbl = "Geographic area (log)", vec = dr_bb[["b_z_area"]]),
      b_doc  = list(lbl = "Doc. coverage",         vec = dr_bb[["b_z_doc"]])
    )
    for (p_nm in names(bb_fixed)) {
      sum_rows[[length(sum_rows) + 1L]] <- mk_summary_row(
        dom, n_lang, meta$k_item, d_max, "fixed_slope", "Beta-Binomial",
        p_nm, bb_fixed[[p_nm]]$lbl, bb_fixed[[p_nm]]$vec,
        include_moments = TRUE, include_dir = TRUE
      )
    }

    # 3. Latent structural variance decomposition (2PL IRT-SEM)
    sd_p <- dr_sem[["sd_phylo[1]"]]
    sd_g <- dr_sem[["sd_gp[1]"]]
    sd_l <- dr_sem[["sd_lang"]]
    v_p  <- sd_p^2
    v_g  <- sd_g^2
    v_l  <- sd_l^2
    v_t  <- v_p + v_g + v_l

    var_shares <- list(
      share_phylo = list(lbl = "Phylogeny",         vec = v_p / v_t),
      share_gp    = list(lbl = "Spatial GP",        vec = v_g / v_t),
      share_lang  = list(lbl = "Language residual", vec = v_l / v_t)
    )
    for (v_nm in names(var_shares)) {
      sum_rows[[length(sum_rows) + 1L]] <- mk_summary_row(
        dom, n_lang, meta$k_item, d_max, "var_share", "2PL IRT-SEM",
        v_nm, var_shares[[v_nm]]$lbl, var_shares[[v_nm]]$vec,
        include_moments = TRUE, include_dir = FALSE
      )
    }

    # 4. Spatial GP lengthscale & exact spherical half-correlation distance d50_km
    ls_gp <- dr_sem[["lscale_gp[1]"]]
    d_chord_50 <- ls_gp * sqrt(2.0 * log(2.0)) * d_max
    arg_asin   <- pmin(1.0, d_chord_50 / 2.0)
    d50_km     <- ifelse(d_chord_50 >= 2.0, Inf, 2.0 * R_EARTH_KM * asin(arg_asin))

    sum_rows[[length(sum_rows) + 1L]] <- mk_summary_row(
      dom, n_lang, meta$k_item, d_max, "gp_param", "2PL IRT-SEM",
      "lscale_gp", "GP lengthscale (scaled chord)", ls_gp,
      include_moments = TRUE, include_dir = FALSE
    )
    sum_rows[[length(sum_rows) + 1L]] <- mk_summary_row(
      dom, n_lang, meta$k_item, d_max, "gp_param", "2PL IRT-SEM",
      "d50_km", "GP half-correlation distance (km)", d50_km,
      include_moments = FALSE, include_dir = FALSE
    )
  }

  sum_df <- do.call(rbind, sum_rows)
  dir.create(dirname(summary_csv), recursive = TRUE, showWarnings = FALSE)
  write.csv(sum_df, summary_csv, row.names = FALSE)
  message("Exported exact posterior anatomy summary to ", summary_csv)
} else {
  if (!file.exists(summary_csv)) {
    stop("Neither cached .rds fits nor fallback summary CSV found at: ", summary_csv)
  }
  sum_df <- read.csv(summary_csv, stringsAsFactors = FALSE)
  message("Loaded fallback posterior anatomy summary from ", summary_csv)
}

# --- Panel A: Standardized Fixed-Effect Predictors (2 x 2 + NP Agreement Grid) ---
pred_order <- c("b_f3", "b_pop", "b_area", "b_doc")
pred_exprs <- c(
  b_f3   = expression("Cross-family " * italic(f)[3]),
  b_pop  = "Speaker pop. (log)",
  b_area = "Geographic area (log)",
  b_doc  = "Doc. coverage"
)

df_fixed <- subset(sum_df, component_type == "fixed_slope" & parameter %in% pred_order)
df_fixed$parameter <- factor(df_fixed$parameter, levels = rev(pred_order))
df_fixed$est_lbl <- factor(
  ifelse(df_fixed$estimator == "2PL IRT-SEM", "2PL IRT-SEM (focal)", "Beta-Binomial"),
  levels = c("Beta-Binomial", "2PL IRT-SEM (focal)")
)

xlim_a <- range(c(df_fixed$q025, df_fixed$q975, -1.05, 1.05), na.rm = TRUE)
xlim_a <- xlim_a + c(-0.04, 0.04) * diff(xlim_a)

make_panel_a_dom <- function(dom_id, show_y = FALSE, show_x = FALSE) {
  d <- subset(df_fixed, domain == dom_id)
  col_d   <- DOMAIN_COLS[[dom_id]]
  shp_sem <- DOMAIN_SHAPES_FILLED[[dom_id]]
  shp_bb  <- DOMAIN_SHAPES_OPEN[[dom_id]]

  n_lang <- d$n_lang[1]
  k_item <- d$k_item[1]
  sub_txt <- sprintf("n = %d languages (%d items)", n_lang, k_item)

  ggplot(d, aes(x = mean, y = parameter, shape = est_lbl, alpha = est_lbl)) +
    annotate("rect", xmin = -ROPE_HALF, xmax = ROPE_HALF, ymin = -Inf, ymax = Inf,
             fill = ROPE_FILL, alpha = 0.78) +
    geom_hline(yintercept = 3.5, colour = GRID_WARM, linewidth = 0.35) +
    geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.38, linetype = "dashed") +
    geom_errorbar(aes(xmin = q025, xmax = q975), width = 0, orientation = "y",
                  colour = col_d, position = position_dodge(width = 0.50), linewidth = 0.42) +
    geom_errorbar(aes(xmin = q10, xmax = q90), width = 0, orientation = "y",
                  colour = col_d, position = position_dodge(width = 0.50), linewidth = 1.15) +
    geom_point(colour = col_d, fill = "white", stroke = 0.60,
               position = position_dodge(width = 0.50), size = 1.65) +
    scale_shape_manual(values = c("2PL IRT-SEM (focal)" = shp_sem, "Beta-Binomial" = shp_bb), guide = "none") +
    scale_alpha_manual(values = c("2PL IRT-SEM (focal)" = 1.0, "Beta-Binomial" = 0.65), guide = "none") +
    scale_y_discrete(labels = pred_exprs) +
    scale_x_continuous(breaks = c(-1.0, -0.5, 0.0, 0.5, 1.0),
                       labels = c("-1.0", "-0.5", "0", "+0.5", "+1.0")) +
    coord_cartesian(xlim = xlim_a) +
    labs(
      title = DOMAIN_SHORT[[dom_id]],
      subtitle = sub_txt,
      x = if (show_x) "Slope per 1 SD" else NULL,
      y = NULL
    ) +
    theme_track1() +
    theme(
      plot.title = element_text(colour = col_d, face = "bold", size = 8.0, margin = margin(b = 1)),
      plot.subtitle = element_text(colour = INK2, size = 7.0, margin = margin(b = 2)),
      axis.text.y = if (show_y) element_text(colour = INK_BODY, size = 7.2) else element_blank(),
      axis.text.x = element_text(colour = INK2, size = 7.0),
      axis.title.x = element_text(colour = INK_BODY, size = 7.6, margin = margin(t = 2)),
      panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
      plot.margin = margin(2, 4, 2, if (show_y) 4 else 2)
    )
}

p_vb  <- make_panel_a_dom("v_breadth", show_y = TRUE,  show_x = FALSE)
p_nb  <- make_panel_a_dom("n_breadth", show_y = FALSE, show_x = FALSE)
p_npa <- make_panel_a_dom("np_agr",    show_y = FALSE, show_x = TRUE)
p_vi  <- make_panel_a_dom("v_irreg",   show_y = TRUE,  show_x = TRUE)
p_ni  <- make_panel_a_dom("n_irreg",   show_y = FALSE, show_x = TRUE)

key_df <- data.frame(
  y   = c(5.70, 4.45, 2.95, 1.80, 0.65),
  lab = c("2PL IRT-SEM (focal)", "Beta-Binomial (simpler)",
          "Verbs (circle)", "Nouns (triangle)", "NP agreement (square)"),
  shp = c(16, 21, 16, 17, 15),
  col = c(INK_BODY, INK2, INK_BODY, INK_BODY, COL_MAG),
  stringsAsFactors = FALSE
)

p_key <- ggplot(key_df, aes(x = 0.10, y = y)) +
  geom_point(aes(shape = I(shp), colour = I(col)), fill = "white", size = 2.0, stroke = 0.65) +
  geom_text(aes(x = 0.22, label = lab), hjust = 0, size = 7.1 / .pt,
            colour = INK_BODY, family = "Helvetica") +
  annotate("text", x = 0.06, y = 7.20, hjust = 0, label = "Estimator & POS key",
           size = 7.4 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  annotate("text", x = 0.06, y = -0.55, hjust = 0, vjust = 1,
           label = "Thick: 80% CrI | Thin: 95% CrI\nShaded: \u00b10.15 SD ROPE",
           size = 7.0 / .pt, colour = INK2, lineheight = 1.18, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1.65), ylim = c(-2.45, 7.65), clip = "off") +
  theme_void()

pA_hdr <- ggplot() +
  annotate("text", x = 0.004, y = 0.5, hjust = 0, vjust = 0.5,
           label = "a   Standardized fixed-effect predictors in the maximal model: 2PL IRT-SEM vs. Beta-Binomial",
           size = 9.0 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  theme_void() +
  theme(plot.margin = margin(1, 0, 1, 0))

pA_grid <- (p_vb | p_nb | p_npa) / (p_vi | p_ni | p_key) +
  plot_layout(heights = c(1.0, 1.12), widths = c(1.0, 1.0, 1.0))

# --- Panel B: Latent Structural Variance Decomposition (0%-100%, 2PL IRT-SEM) ---
df_var <- subset(sum_df, component_type == "var_share")
comp_map <- c(
  share_phylo = "Phylogeny",
  share_gp    = "Spatial GP",
  share_lang  = "Language residual"
)
df_var$comp_lbl <- factor(comp_map[df_var$parameter], levels = unname(comp_map))
df_var$dom_lbl  <- factor(DOMAIN_SHORT[df_var$domain], levels = rev(unname(DOMAIN_SHORT[DOMAIN_ORDER])))
df_var$pct_mean <- 100.0 * df_var$mean
df_var$pct_q025 <- 100.0 * df_var$q025
df_var$pct_q10  <- 100.0 * df_var$q10
df_var$pct_q90  <- 100.0 * df_var$q90
df_var$pct_q975 <- 100.0 * df_var$q975

br_gp_rng <- round(range(subset(df_var, parameter == "share_gp" &
                                domain %in% c("v_breadth", "n_breadth"))$pct_mean))
ir_ph_rng <- round(range(subset(df_var, parameter == "share_phylo" &
                                domain %in% c("v_irreg", "n_irreg", "np_agr"))$pct_mean))
sub_b_txt <- sprintf(
  "Breadth means: %d%%\u2013%d%% spatial GP | Irregularity & NP agreement means: %d%%\u2013%d%% phylogeny",
  br_gp_rng[1], br_gp_rng[2], ir_ph_rng[1], ir_ph_rng[2]
)

pB <- ggplot(df_var, aes(x = pct_mean, y = dom_lbl, colour = domain, shape = domain)) +
  geom_hline(yintercept = 3.5, colour = GRID_WARM, linewidth = 0.40) +
  geom_vline(xintercept = 50, colour = GREY_LT, linewidth = 0.36, linetype = "dashed") +
  geom_errorbar(aes(xmin = pct_q025, xmax = pct_q975), width = 0, orientation = "y",
                linewidth = 0.45, alpha = 0.55) +
  geom_errorbar(aes(xmin = pct_q10, xmax = pct_q90), width = 0, orientation = "y",
                linewidth = 1.25) +
  geom_point(size = 1.85) +
  facet_wrap(~comp_lbl, nrow = 1) +
  scale_colour_manual(values = DOMAIN_COLS, guide = "none") +
  scale_shape_manual(values = DOMAIN_SHAPES_FILLED, guide = "none") +
  scale_x_continuous(breaks = c(0, 25, 50, 75, 100),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  coord_cartesian(xlim = c(-1.5, 102.5)) +
  labs(
    title = "b   Latent structural variance decomposition (2PL IRT-SEM)",
    subtitle = sub_b_txt,
    x = "Share of latent structural variance (%)",
    y = NULL
  ) +
  theme_track1() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.6),
    axis.text.y = element_text(colour = INK_BODY, size = 7.2),
    axis.text.x = element_text(colour = INK2, size = 7.0),
    axis.title.x = element_text(colour = INK_BODY, size = 7.6, margin = margin(t = 2)),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing = unit(7, "pt"),
    plot.margin = margin(3, 4, 2, 4)
  )

# --- Panel C: Spatial GP Correlation Decay Over Great-Circle Distance (0-6,000 km) ---
d_grid_km <- seq(0.0, 6000.0, by = 50.0)
curve_list <- list()
d50_map <- setNames(numeric(length(DOMAIN_ORDER)), DOMAIN_ORDER)

for (dom in DOMAIN_ORDER) {
  r_ls  <- subset(sum_df, domain == dom & component_type == "gp_param" & parameter == "lscale_gp")
  r_d50 <- subset(sum_df, domain == dom & component_type == "gp_param" & parameter == "d50_km")
  stopifnot(nrow(r_ls) == 1L, nrow(r_d50) == 1L)

  d_max <- r_ls$d_max[1]
  u_vec <- (2.0 * sin(d_grid_km / (2.0 * R_EARTH_KM))) / d_max

  # Monotonic transformation of lscale_gp preserves exact order-statistic quantiles
  curve_list[[dom]] <- data.frame(
    domain  = dom,
    d_km    = d_grid_km,
    d_k     = d_grid_km / 1000.0,
    rho_q10 = exp(-0.5 * (u_vec / r_ls$q10[1])^2),
    rho_q50 = exp(-0.5 * (u_vec / r_ls$q50[1])^2),
    rho_q90 = exp(-0.5 * (u_vec / r_ls$q90[1])^2),
    stringsAsFactors = FALSE
  )
  d50_map[[dom]] <- r_d50$q50[1]
}
df_curves <- do.call(rbind, curve_list)

make_panel_c_dom <- function(dom_id, show_y = FALSE) {
  d_foc <- subset(df_curves, domain == dom_id)
  d_bg  <- subset(df_curves, domain != dom_id)
  col_d <- DOMAIN_COLS[[dom_id]]
  lty_d <- DOMAIN_LTY[[dom_id]]
  shp_d <- DOMAIN_SHAPES_FILLED[[dom_id]]
  d50_v <- d50_map[[dom_id]]
  d50_k <- d50_v / 1000.0
  fmt_km <- format(round(d50_v), big.mark = ",", trim = TRUE)
  sub_expr <- bquote(italic(d)[0.5] == .(paste0(fmt_km, " km")))

  ggplot() +
    geom_line(data = d_bg, aes(x = d_k, y = rho_q50, group = domain),
              colour = GREY_LT, linewidth = 0.36, alpha = 0.65) +
    geom_ribbon(data = d_foc, aes(x = d_k, ymin = rho_q10, ymax = rho_q90),
                fill = col_d, alpha = 0.18) +
    geom_hline(yintercept = 0.5, colour = INK2, linewidth = 0.35, linetype = "dashed") +
    { if (is.finite(d50_k) && d50_k <= 6.0)
        annotate("segment", x = d50_k, xend = d50_k, y = 0.0, yend = 0.5,
                 colour = col_d, linewidth = 0.40, linetype = "dotted") } +
    geom_line(data = d_foc, aes(x = d_k, y = rho_q50),
              colour = col_d, linetype = lty_d, linewidth = 0.85) +
    { if (is.finite(d50_k) && d50_k <= 6.0)
        annotate("point", x = d50_k, y = 0.5, colour = col_d, shape = shp_d, size = 1.65) } +
    scale_x_continuous(breaks = c(0, 2, 4, 6), labels = c("0", "2", "4", "6")) +
    scale_y_continuous(breaks = c(0.0, 0.5, 1.0), labels = c("0.0", "0.5", "1.0")) +
    coord_cartesian(xlim = c(0, 6.05), ylim = c(0.0, 1.02)) +
    labs(
      title = DOMAIN_SHORT[[dom_id]],
      subtitle = sub_expr,
      x = NULL,
      y = if (show_y) expression("Spatial corr. " * rho(italic(d))) else NULL
    ) +
    theme_track1() +
    theme(
      panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
      plot.title = element_text(colour = col_d, face = "bold", size = 7.6, margin = margin(b = 1)),
      plot.subtitle = element_text(colour = INK2, size = 7.1, margin = margin(b = 2)),
      axis.text.y = if (show_y) element_text(colour = INK2, size = 7.0) else element_blank(),
      axis.text.x = element_text(colour = INK2, size = 7.0),
      axis.title.y = element_text(colour = INK_BODY, size = 7.6, margin = margin(r = 3)),
      panel.grid.major = element_line(colour = GRID_MAJ, linewidth = 0.25),
      plot.margin = margin(2, 3, 1, if (show_y) 4 else 2)
    )
}

pC_hdr <- ggplot() +
  annotate("text", x = 0.004, y = 0.72, hjust = 0, vjust = 0.5,
           label = "c   Spatial GP correlation decay over great-circle distance (2PL IRT-SEM)",
           size = 9.0 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  annotate("text", x = 0.004, y = 0.24, hjust = 0, vjust = 0.5,
           label = "Posterior median curve & 80% CrI ribbon (grey lines: other domains; dashed line: \u03c1 = 0.5 horizon)",
           size = 7.2 / .pt, colour = INK2, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  theme_void() +
  theme(plot.margin = margin(4, 0, 1, 0))

pC_row <- make_panel_c_dom("v_breadth", show_y = TRUE) |
          make_panel_c_dom("n_breadth", show_y = FALSE) |
          make_panel_c_dom("v_irreg",   show_y = FALSE) |
          make_panel_c_dom("n_irreg",   show_y = FALSE) |
          make_panel_c_dom("np_agr",    show_y = FALSE)

pC_xlab <- ggplot() +
  annotate("text", x = 0.53, y = 0.55, hjust = 0.5, vjust = 0.5,
           label = expression("Great-circle distance (" * 10^3 * " km)"),
           size = 7.6 / .pt, colour = INK_BODY, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
  theme_void() +
  theme(plot.margin = margin(0, 0, 1, 0))

fig6 <- pA_hdr /
        wrap_elements(full = pA_grid) /
        wrap_elements(full = pB) /
        pC_hdr /
        wrap_elements(full = pC_row) /
        pC_xlab +
  plot_layout(heights = c(0.10, 1.40, 0.88, 0.18, 0.72, 0.09))

save_track1_fig(fig6, fig_dir, "fig6_grambank_maximal_effects", width = 6.5, height = 7.65, dpi = 300)
