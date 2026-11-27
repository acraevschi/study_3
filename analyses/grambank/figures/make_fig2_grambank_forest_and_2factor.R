# Figure 2 (Track 1): Symmetrical 2x2 + NP Agreement Forest (2PL IRT-SEM vs. Beta-Binomial)
# and Joint 2-Factor Within-POS Contrast (Verb_2fac & Noun_2fac)
# Outputs: fig2_grambank_forest_and_2factor.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/grambank/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

sem1 <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_1factor_results.csv"), stringsAsFactors = FALSE)
sem2 <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_2factor_pos_results.csv"), stringsAsFactors = FALSE)
bb   <- read.csv(file.path(root_dir, "analyses/grambank/results/stage2_bb_results.csv"), stringsAsFactors = FALSE)

# --- Panel A: 2x2 + NP Agreement Forest Comparing 2PL IRT-SEM vs. Beta-Binomial ---
spec_map <- c(
  maximal            = "Maximal (n \u2265 3)",
  cutoff_n5          = "Cutoff n \u2265 5",
  loo_no_space       = "LOO: no spatial GP",
  loo_no_demog       = "LOO: no demography",
  loo_no_doccov      = "LOO: no doc. control",
  mundlak_wb         = "Within-family (Mundlak)",
  raw_unmasked_zeros = "Raw unmasked zeros"
)
spec_order <- unname(spec_map)

extract_dom_df <- function(dom_id) {
  rows <- list()
  for (sp in names(spec_map)) {
    # Focal 2PL IRT-SEM
    rs <- subset(sem1, domain == dom_id & arch == "MA_2pl" & spec == sp)
    if (nrow(rs) == 1L) {
      q80 <- get_exact_q80(rs, "b")
      rows[[length(rows) + 1L]] <- data.frame(
        domain = dom_id, spec_lbl = spec_map[[sp]],
        estimator = "2PL IRT-SEM (focal)",
        mean = rs$b_mean, lo95 = rs$b_q025, hi95 = rs$b_q975,
        lo80 = q80$lo80, hi80 = q80$hi80,
        stringsAsFactors = FALSE
      )
    }
    # Simpler Beta-Binomial (f3_fixed)
    rb <- subset(bb, outcome == dom_id & spec == sp & measure == "f3_fixed")
    if (nrow(rb) == 1L) {
      q80 <- get_exact_q80(rb, "b")
      rows[[length(rows) + 1L]] <- data.frame(
        domain = dom_id, spec_lbl = spec_map[[sp]],
        estimator = "Beta-Binomial",
        mean = rb$b_mean, lo95 = rb$b_q025, hi95 = rb$b_q975,
        lo80 = q80$lo80, hi80 = q80$hi80,
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  keep_lvls <- spec_order[spec_order %in% out$spec_lbl]
  out$spec_lbl <- factor(out$spec_lbl, levels = rev(keep_lvls))
  out$estimator <- factor(out$estimator, levels = c("Beta-Binomial", "2PL IRT-SEM (focal)"))
  out
}

df_dom_list <- setNames(lapply(DOMAIN_ORDER, extract_dom_df), DOMAIN_ORDER)
all_a <- do.call(rbind, df_dom_list)
xlim_a <- range(c(all_a$lo95, all_a$hi95, -0.55, 0.75), na.rm = TRUE)
xlim_a <- xlim_a + c(-0.04, 0.04) * diff(xlim_a)

make_dom_panel <- function(dom_id, show_y = FALSE, show_x = FALSE) {
  d <- df_dom_list[[dom_id]]
  col_d <- DOMAIN_COLS[[dom_id]]
  shp_sem <- DOMAIN_SHAPES_FILLED[[dom_id]]
  shp_bb  <- DOMAIN_SHAPES_OPEN[[dom_id]]

  # Primary SEM BF_01 / ROPE subtitle
  r_max <- subset(sem1, domain == dom_id & arch == "MA_2pl" & spec == "maximal" & set == "main" & min_n_ind == 3L)
  sub_txt <- sprintf("SEM n=%d | b=%+.2f | ROPE %.0f%%",
                     r_max$n_lang, r_max$b_mean, 100 * r_max$b_p_rope_015)

  ggplot(d, aes(x = mean, y = spec_lbl, shape = estimator, alpha = estimator)) +
    annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
             fill = ROPE_FILL, alpha = 0.78) +
    geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.38, linetype = "dashed") +
    geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                  colour = col_d, position = position_dodge(width = 0.48), linewidth = 0.42) +
    geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                  colour = col_d, position = position_dodge(width = 0.48), linewidth = 1.15) +
    geom_point(colour = col_d, fill = "white", stroke = 0.6,
               position = position_dodge(width = 0.48), size = 1.65) +
    scale_shape_manual(values = c("2PL IRT-SEM (focal)" = shp_sem, "Beta-Binomial" = shp_bb), guide = "none") +
    scale_alpha_manual(values = c("2PL IRT-SEM (focal)" = 1.0, "Beta-Binomial" = 0.65), guide = "none") +
    scale_x_continuous(breaks = c(-0.4, 0.0, 0.4, 0.8)) +
    coord_cartesian(xlim = xlim_a) +
    labs(
      title = DOMAIN_SHORT[[dom_id]],
      subtitle = sub_txt,
      x = if (show_x) "Slope per SD of f3" else NULL,
      y = NULL
    ) +
    theme_track1() +
    theme(
      plot.title = element_text(colour = col_d, face = "bold", size = 8.0, margin = margin(b = 1)),
      plot.subtitle = element_text(colour = INK2, size = 7.0, margin = margin(b = 2)),
      axis.text.y = if (show_y) element_text(colour = INK_BODY, size = 7.1) else element_blank(),
      axis.text.x = element_text(colour = INK2, size = 7.0),
      panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
      plot.margin = margin(3, 4, 3, if (show_y) 4 else 2)
    )
}

p_vb  <- make_dom_panel("v_breadth", show_y = TRUE,  show_x = FALSE)
p_nb  <- make_dom_panel("n_breadth", show_y = FALSE, show_x = FALSE)
p_npa <- make_dom_panel("np_agr",    show_y = FALSE, show_x = TRUE)
p_vi  <- make_dom_panel("v_irreg",   show_y = TRUE,  show_x = TRUE)
p_ni  <- make_dom_panel("n_irreg",   show_y = FALSE, show_x = TRUE)

# Concise unboxed visual key in bottom-right cell (2,3) of the symmetrical 2x2 + NP grid
key_df <- data.frame(
  y = c(5.2, 4.1, 2.7, 1.7, 0.7),
  lab = c("2PL IRT-SEM (focal)", "Beta-Binomial (simpler)",
          "Verbs (circle)", "Nouns (triangle)", "NP agreement (square)"),
  shp = c(16, 21, 16, 17, 15),
  col = c(INK_BODY, INK2, INK_BODY, INK_BODY, COL_MAG),
  stringsAsFactors = FALSE
)
p_key <- ggplot(key_df, aes(x = 0.12, y = y)) +
  geom_point(aes(shape = I(shp), colour = I(col)), fill = "white", size = 2.0, stroke = 0.65) +
  geom_text(aes(x = 0.24, label = lab), hjust = 0, size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  annotate("text", x = 0.08, y = 6.1, hjust = 0, label = "Estimator & POS key",
           size = 7.3 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  annotate("text", x = 0.08, y = -0.60, hjust = 0,
           label = "Thick: 80% CrI | Thin: 95% CrI\nShaded: \u00b10.15 SD ROPE",
           size = 7.0 / .pt, colour = INK2, lineheight = 1.15, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1.65), ylim = c(-1.35, 6.6)) +
  theme_void()

pA_grid <- (p_vb | p_nb | p_npa) / (p_vi | p_ni | p_key) +
  plot_layout(heights = c(1.0, 1.16), widths = c(1.0, 1.0, 1.0))

# --- Panel B: Joint 2-Factor Within-POS 2PL SEM (Verb_2fac & Noun_2fac, n >= 3 & n >= 5) ---
rows_b_slp <- list()
rows_b_rho <- list()

for (i in seq_len(nrow(sem2))) {
  r <- sem2[i, ]
  pos_name <- if (grepl("^Verb", r$pos_pair)) "Verbs" else "Nouns"
  pos_shp  <- if (pos_name == "Verbs") 16 else 17
  cut_lbl  <- sprintf("%s (n \u2265 %d, N=%d)", pos_name, as.integer(r$min_n_ind), as.integer(r$n_lang))

  qb <- get_exact_q80(r, "b_breadth")
  qi <- get_exact_q80(r, "b_irreg")
  qd <- get_exact_q80(r, "b_diff")

  rows_b_slp[[length(rows_b_slp) + 1L]] <- data.frame(
    panel_grp = cut_lbl,
    term = c("Breadth", "Irregularity", "Contrast (\u0394b)"),
    col_key = c("breadth", "irreg", "diff"),
    shp = c(pos_shp, pos_shp, 18),
    mean = c(r$b_breadth_mean, r$b_irreg_mean, r$b_diff_mean),
    lo95 = c(r$b_breadth_q025, r$b_irreg_q025, r$b_diff_q025),
    hi95 = c(r$b_breadth_q975, r$b_irreg_q975, r$b_diff_q975),
    lo80 = c(qb$lo80, qi$lo80, qd$lo80),
    hi80 = c(qb$hi80, qi$hi80, qd$hi80),
    stringsAsFactors = FALSE
  )

  has_rho80 <- all(c("rho_latent_q10", "rho_latent_q90") %in% names(r))
  rows_b_rho[[length(rows_b_rho) + 1L]] <- data.frame(
    panel_grp = cut_lbl,
    pos = pos_name,
    shp = pos_shp,
    mean = r$rho_latent_mean,
    lo95 = r$rho_latent_q025,
    hi95 = r$rho_latent_q975,
    lo80 = if (has_rho80) r$rho_latent_q10 else NA_real_,
    hi80 = if (has_rho80) r$rho_latent_q90 else NA_real_,
    stringsAsFactors = FALSE
  )
}

dfB_slp <- do.call(rbind, rows_b_slp)
dfB_rho <- do.call(rbind, rows_b_rho)

grp_ord <- unique(dfB_slp$panel_grp)
grp_ord <- grp_ord[order(!grepl("^Verbs", grp_ord), grepl("5", grp_ord))]
dfB_slp$panel_grp <- factor(dfB_slp$panel_grp, levels = grp_ord)
dfB_rho$panel_grp <- factor(dfB_rho$panel_grp, levels = rev(grp_ord))
term_ord <- c("Breadth", "Irregularity", "Contrast (\u0394b)")
dfB_slp$term <- factor(dfB_slp$term, levels = rev(term_ord))

xlim_b <- range(c(dfB_slp$lo95, dfB_slp$hi95, -0.55, 0.85), na.rm = TRUE)
xlim_b <- xlim_b + c(-0.05, 0.05) * diff(xlim_b)

pB1 <- ggplot(dfB_slp, aes(x = mean, y = term, colour = col_key)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y", linewidth = 0.45, alpha = 0.55) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y", linewidth = 1.25) +
  geom_point(aes(shape = I(shp)), size = 1.75) +
  facet_wrap(~panel_grp, ncol = 2) +
  scale_colour_manual(values = c("breadth" = COL_BREADTH, "irreg" = COL_IRREG, "diff" = COL_DIFF), guide = "none") +
  scale_x_continuous(breaks = c(-0.5, 0.0, 0.5, 1.0), labels = c("-0.5", "0", "+0.5", "+1.0")) +
  coord_cartesian(xlim = xlim_b) +
  labs(
    title = "b   Joint 2-factor SEM: slopes & contrast",
    x = "Standardized slope / contrast per SD of f3",
    y = NULL
  ) +
  theme_track1() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.2),
    axis.text.y = element_text(colour = INK_BODY, size = 7.2),
    panel.spacing = unit(6, "pt")
  )

pB2 <- ggplot(dfB_rho, aes(x = mean, y = panel_grp)) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                colour = INK2, linewidth = 0.48, alpha = 0.60) +
  { if (all(is.finite(dfB_rho$lo80)))
      geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                    colour = INK_BODY, linewidth = 1.35) } +
  geom_point(aes(shape = I(shp)), colour = INK_BODY, size = 1.9) +
  scale_x_continuous(breaks = c(-0.5, 0.0, 0.5, 1.0)) +
  coord_cartesian(xlim = c(-0.65, 1.0)) +
  labs(
    title = "c   Residual latent \u03c1",
    x = expression("Latent correlation " * rho[latent]),
    y = NULL
  ) +
  theme_track1() +
  theme(
    axis.text.y = element_text(colour = INK_BODY, size = 7.1),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25)
  )

pA_hdr <- ggplot() +
  annotate("text", x = 0.005, y = 0.5, hjust = 0, vjust = 0.5,
           label = "a   Symmetrical 2 \u00d7 2 + NP agreement forest: 2PL IRT-SEM vs. Beta-Binomial",
           size = 9.0 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  theme_void() +
  theme(plot.margin = margin(2, 0, 2, 0))

pB_row <- pB1 + pB2 + plot_layout(widths = c(1.62, 0.92))

fig2 <- pA_hdr / wrap_elements(full = pA_grid) / wrap_elements(full = pB_row) +
  plot_layout(heights = c(0.11, 1.54, 0.96))
save_track1_fig(fig2, fig_dir, "fig2_grambank_forest_and_2factor", width = 6.5, height = 7.0, dpi = 300)
