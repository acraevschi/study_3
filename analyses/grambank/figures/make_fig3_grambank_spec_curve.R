# Figure 3 (Track 1): Full Specification Curves Across All 2PL IRT-SEM & Beta-Binomial Models
# Outputs: fig3_grambank_spec_curve.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/grambank/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

sem1 <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_1factor_results.csv"), stringsAsFactors = FALSE)
bb   <- read.csv(file.path(root_dir, "analyses/grambank/results/stage2_bb_results.csv"), stringsAsFactors = FALSE)

# Combine all fitted 1-factor MA_2pl SEM specs + all Beta-Binomial specs per domain
sem_sub <- subset(sem1, arch == "MA_2pl" & !is.na(b_mean))
q80_s   <- get_exact_q80(sem_sub, "b")
df_s <- data.frame(
  domain = sem_sub$domain,
  estimator = "2PL IRT-SEM",
  spec = sem_sub$spec,
  mean = sem_sub$b_mean,
  lo95 = sem_sub$b_q025,
  hi95 = sem_sub$b_q975,
  lo80 = q80_s$lo80,
  hi80 = q80_s$hi80,
  p_rope = sem_sub$b_p_rope_015,
  stringsAsFactors = FALSE
)

bb_sub <- subset(bb, !is.na(b_mean))
q80_b  <- get_exact_q80(bb_sub, "b")
df_b <- data.frame(
  domain = bb_sub$outcome,
  estimator = "Beta-Binomial",
  spec = bb_sub$spec,
  mean = bb_sub$b_mean,
  lo95 = bb_sub$b_q025,
  hi95 = bb_sub$b_q975,
  lo80 = q80_b$lo80,
  hi80 = q80_b$hi80,
  p_rope = bb_sub$b_p_rope_015,
  stringsAsFactors = FALSE
)

df_all <- rbind(df_s, df_b)
df_all <- subset(df_all, domain %in% DOMAIN_ORDER)

# Rank within each domain
df_all <- do.call(rbind, lapply(DOMAIN_ORDER, function(dom_id) {
  g <- subset(df_all, domain == dom_id)
  g <- g[order(g$mean), ]
  g$rank <- seq_len(nrow(g))
  g$dom_lbl <- DOMAIN_LABELS[[dom_id]]
  g$shp <- ifelse(g$estimator == "2PL IRT-SEM", DOMAIN_SHAPES_FILLED[[dom_id]], DOMAIN_SHAPES_OPEN[[dom_id]])
  g
}))

ylim_all <- range(c(df_all$lo95, df_all$hi95, -0.60, 0.90), na.rm = TRUE)
yspan    <- diff(ylim_all)
y_top    <- ylim_all[2] + 0.30 * yspan
y_bot    <- ylim_all[1] - 0.04 * yspan

make_sc_panel <- function(dom_id, show_y = FALSE) {
  g <- subset(df_all, domain == dom_id)
  n_tot   <- nrow(g)
  n_above <- sum(g$lo95 > 0)
  n_incl  <- sum(g$lo95 <= 0 & g$hi95 >= 0)
  n_below <- sum(g$hi95 < 0)
  col_d   <- DOMAIN_COLS[[dom_id]]

  ann_txt <- sprintf("%d specs (%d SEM, %d BB)\n>0: %d | incl. 0: %d | <0: %d (95%% CrI)",
                     n_tot, sum(g$estimator == "2PL IRT-SEM"), sum(g$estimator == "Beta-Binomial"),
                     n_above, n_incl, n_below)

  ggplot(g, aes(x = rank, y = mean)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.15, ymax = 0.15,
             fill = ROPE_FILL, alpha = 0.78) +
    geom_hline(yintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
    geom_segment(aes(xend = rank, y = lo95, yend = hi95, alpha = estimator),
                 colour = col_d, linewidth = 0.36) +
    geom_segment(aes(xend = rank, y = lo80, yend = hi80, alpha = estimator),
                 colour = col_d, linewidth = 0.88) +
    geom_point(aes(shape = I(shp)), colour = col_d, fill = "white", size = 1.25, stroke = 0.45) +
    annotate("text", x = 1, y = y_top - 0.02 * yspan, hjust = 0, vjust = 1,
             label = ann_txt, size = 7.0 / .pt, lineheight = 1.08, colour = INK_BODY, family = "Helvetica") +
    scale_alpha_manual(values = c("2PL IRT-SEM" = 0.90, "Beta-Binomial" = 0.48), guide = "none") +
    scale_x_continuous(breaks = NULL, expand = expansion(add = 1.2)) +
    scale_y_continuous(breaks = c(-0.5, 0.0, 0.5, 1.0)) +
    coord_cartesian(ylim = c(y_bot, y_top)) +
    labs(
      title = DOMAIN_SHORT[[dom_id]],
      x = NULL,
      y = if (show_y) "Slope per SD" else NULL
    ) +
    theme_track1() +
    theme(
      plot.title = element_text(colour = col_d, face = "bold", size = 8.2, margin = margin(b = 2)),
      panel.grid.major.x = element_blank(),
      axis.text.y = if (show_y) element_text(colour = INK2, size = 7.2) else element_blank(),
      plot.margin = margin(3, 5, 3, if (show_y) 5 else 2)
    )
}

p1 <- make_sc_panel("v_breadth", show_y = TRUE)
p2 <- make_sc_panel("n_breadth", show_y = FALSE)
p3 <- make_sc_panel("np_agr",    show_y = FALSE)
p4 <- make_sc_panel("v_irreg",   show_y = TRUE)
p5 <- make_sc_panel("n_irreg",   show_y = FALSE)

# Unboxed reading guide in bottom-right slot
leg_df <- data.frame(
  y = c(4.0, 3.0, 1.6, 0.7),
  lab = c("Filled: 2PL IRT-SEM (focal)", "Open: Beta-Binomial (SI)",
          "Thick bar: 80% CrI | Thin: 95% CrI", "Shaded band: \u00b10.15 SD ROPE"),
  is_pt = c(TRUE, TRUE, FALSE, FALSE),
  shp = c(16, 21, NA, NA),
  stringsAsFactors = FALSE
)
p_leg <- ggplot(leg_df) +
  geom_point(data = subset(leg_df, is_pt), aes(x = 0.10, y = y, shape = I(shp)),
             colour = INK_BODY, fill = "white", size = 2.0, stroke = 0.6) +
  geom_text(aes(x = ifelse(is_pt, 0.22, 0.06), y = y, label = lab),
            hjust = 0, size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  annotate("text", x = 0.06, y = 5.0, hjust = 0, label = "Reading guide",
           size = 7.5 / .pt, fontface = "bold", colour = INK, family = "Helvetica") +
  coord_cartesian(xlim = c(0, 1.9), ylim = c(0, 5.6)) +
  theme_void()

fig3 <- (p1 | p2 | p3) / (p4 | p5 | p_leg) +
  plot_annotation(
    title = "Specification curves across all fitted Track 1 models (sorted by posterior mean)",
    caption = sprintf("Total: %d 2PL IRT-SEM + %d Beta-Binomial = %d domain specifications across 5 morphological domains.",
                      nrow(df_s), nrow(df_b), nrow(df_all)),
    theme = theme(
      plot.title = element_text(colour = INK, face = "bold", size = 9.0, family = "Helvetica", margin = margin(b = 4)),
      plot.caption = element_text(colour = INK2, size = 7.0, family = "Helvetica", hjust = 0)
    )
  )

save_track1_fig(fig3, fig_dir, "fig3_grambank_spec_curve", width = 6.5, height = 4.9, dpi = 300)
