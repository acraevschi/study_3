# Figure 3: Cross-Linguistic Admixture Slopes & Specification Curve (Verbs vs. Nouns)
# Outputs: fig3_mgl_admixture_synthesis.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

s2_v <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/stage2_results_V.csv"), stringsAsFactors = FALSE)
s2_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/stage2_results_N.csv"), stringsAsFactors = FALSE)
sy   <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results/synthesis_checks_main.csv"), stringsAsFactors = FALSE)

proxies <- c("ent", "sqrt_het", "f3k_est_pos")
proxy_labs <- c(
  ent = "Individual entropy",
  sqrt_het = "Community heterogeneity",
  f3k_est_pos = "Cross-family f3"
)

# --- Panel A: Key Specifications Across 3 Admixture Proxies + Track 1 Reference ---
rows <- list()
add_row <- function(section, row_lbl, series, measure, mean, q025, q05, q95, q975) {
  rows[[length(rows) + 1L]] <<- data.frame(
    section = section,
    row_lbl = row_lbl,
    series  = series,
    measure = measure,
    mean    = mean,
    lo95    = q025,
    hi95    = q975,
    lo80    = mean + (q05 - mean) * K80_FROM_90,
    hi80    = mean + (q95 - mean) * K80_FROM_90,
    stringsAsFactors = FALSE
  )
}

for (m in proxies) {
  r_t1 <- subset(sy, name == sprintf("syn_ref__cat12_irreg13__main__MC_cross__%s__x_P", m))
  stopifnot(nrow(r_t1) == 1L)
  add_row("Track 1\nRef.", "Enumerative breadth", "Track 1: Breadth", m,
          r_t1$b_mean, r_t1$b_q025, r_t1$b_q05, r_t1$b_q95, r_t1$b_q975)
  add_row("Track 1\nRef.", "Grambank irreg13", "Track 1: Irreg13", m,
          r_t1$b_irreg_mean, r_t1$b_irreg_q025, r_t1$b_irreg_q05, r_t1$b_irreg_q95, r_t1$b_irreg_q975)
}

spec_defs <- list(
  list(lbl = "Primary (Phylo)",           set = "main",       sg = "full_covs",             ctrl = "P"),
  list(lbl = "Primary (Phylo + GP)",      set = "main",       sg = "full_covs",             ctrl = "F"),
  list(lbl = "+ Paradigm size (log |C|)", set = "main",       sg = "ctrl_C_distinct",       ctrl = "F"),
  list(lbl = "Demography (pop, area)",    set = "main",       sg = "primary_demog",         ctrl = "F"),
  list(lbl = "Within-family (Mundlak)",   set = "main",       sg = "full_covs_mundlak",     ctrl = "P"),
  list(lbl = "20-draw Berkson check",     set = "main",       sg = "berkson_20draw",        ctrl = "P"),
  list(lbl = "Bouckaert stem tree",       set = "main",       sg = "bouckaert",             ctrl = "P_bouckaert_stem"),
  list(lbl = "Bouckaert crown tree",      set = "main",       sg = "bouckaert",             ctrl = "P_bouckaert_crown"),
  list(lbl = "Uniform-480 (Tier 2)",      set = "main",       sg = "tier2_uniform480",      ctrl = "P"),
  list(lbl = "Top-freq-120",              set = "main",       sg = "topfreq120",            ctrl = "P"),
  list(lbl = "Drop Inner Asia",           set = "main",       sg = "no_inner_asia",         ctrl = "P"),
  list(lbl = "Strict links",              set = "strict",     sg = "link_sensitivity",      ctrl = "F")
)

for (pos_info in list(list(df = s2_v, ser = "Verbs (N <= 45)"),
                      list(df = s2_n, ser = "Nouns (N <= 49)"))) {
  df_p <- pos_info$df
  for (sp in spec_defs) {
    for (m in proxies) {
      r <- subset(df_p, set == sp$set & spec_group == sp$sg & ctrl == sp$ctrl & measure == m)
      if (nrow(r) == 1L) {
        add_row("Track 2\nMGL", sp$lbl, pos_info$ser, m,
                r$b_mean, r$b_q025, r$b_q05, r$b_q95, r$b_q975)
      }
    }
  }
}

dfA <- do.call(rbind, rows)
row_order <- c("Enumerative breadth", "Grambank irreg13", vapply(spec_defs, `[[`, character(1), "lbl"))
dfA$row_lbl <- factor(dfA$row_lbl, levels = rev(row_order))
dfA$section <- factor(dfA$section, levels = c("Track 1\nRef.", "Track 2\nMGL"))
dfA$proxy   <- factor(proxy_labs[dfA$measure], levels = unname(proxy_labs))
series_levels <- c("Verbs (N <= 45)", "Nouns (N <= 49)", "Track 1: Breadth", "Track 1: Irreg13")
dfA$series <- factor(dfA$series, levels = rev(series_levels))

# Track 1 Breadth vertical reference per proxy facet
t1_ref <- subset(dfA, series == "Track 1: Breadth")[, c("proxy", "mean")]

# Key primary annotations on Panel A (ent facet, Primary Phylo row)
v_prim <- subset(s2_v, set == "main" & spec_group == "full_covs" & ctrl == "P" & measure == "ent")
n_prim <- subset(s2_n, set == "main" & spec_group == "full_covs" & ctrl == "P" & measure == "ent")
ann_ent <- data.frame(
  proxy = factor("Individual entropy", levels = unname(proxy_labs)),
  section = factor("Track 2\nMGL", levels = c("Track 1\nRef.", "Track 2\nMGL")),
  x = c(0.68, 0.68),
  y = c("Primary (Phylo)", "Primary (Phylo + GP)"),
  lab = c(
    sprintf("V: BF01=%.1f, P(<.44)=%.2f", v_prim$b_bf01_norm, v_prim$b_p_lt_044),
    sprintf("N: BF01=%.1f, P(<.44)=%.2f", n_prim$b_bf01_norm, n_prim$b_p_lt_044)
  ),
  stringsAsFactors = FALSE
)

col_vals <- c("Verbs (N <= 45)" = COL_VERB, "Nouns (N <= 49)" = COL_NOUN,
              "Track 1: Breadth" = COL_BREADTH, "Track 1: Irreg13" = COL_IRREG)
shp_vals <- c("Verbs (N <= 45)" = 16, "Nouns (N <= 49)" = 17,
              "Track 1: Breadth" = 18, "Track 1: Irreg13" = 5)

pA <- ggplot(dfA, aes(x = mean, y = row_lbl, colour = series, shape = series)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.75) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_vline(data = t1_ref, aes(xintercept = mean), colour = COL_BREADTH,
             linewidth = 0.38, alpha = 0.65, linetype = "dotted") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                position = position_dodge(width = 0.56), linewidth = 0.45, alpha = 0.50) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                position = position_dodge(width = 0.56), linewidth = 1.25, alpha = 1.0) +
  geom_point(position = position_dodge(width = 0.56), size = 1.55) +
  facet_grid(section ~ proxy, scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = col_vals, breaks = series_levels, name = NULL) +
  scale_shape_manual(values = shp_vals, breaks = series_levels, name = NULL) +
  scale_x_continuous(breaks = c(-0.5, 0.0, 0.5)) +
  coord_cartesian(xlim = c(-0.72, 0.72)) +
  labs(
    title = "a   Cross-linguistic admixture slopes (80% & 95% CrI; shaded: ±0.15 SD ROPE)",
    subtitle = sprintf("Primary (Phylo, ent): Verbs BF01 = %.2f [P(β < +0.44) = %.3f]; Nouns BF01 = %.2f [P(β < +0.44) = %.3f]",
                       v_prim$b_bf01_norm, v_prim$b_p_lt_044, n_prim$b_bf01_norm, n_prim$b_p_lt_044),
    x = "Slope per SD of admixture",
    y = NULL
  ) +
  theme_track2() +
  theme(
    strip.placement = "outside",
    strip.text.x = element_text(colour = INK_BODY, face = "bold", size = 7.9, margin = margin(b = 3)),
    strip.text.y.left = element_text(colour = INK_BODY, face = "bold", size = 7.5, angle = 0, hjust = 1),
    axis.text.y = element_text(colour = INK_BODY, size = 7.3),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing.x = unit(10, "pt"),
    panel.spacing.y = unit(6, "pt"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.width = unit(12, "pt"),
    legend.text = element_text(size = 7.2, colour = INK_BODY),
    legend.margin = margin(t = -2, b = 0)
  )

# --- Panel B: Full Specification Curve across all H_tree models ---
h_outcomes <- c("h_tree_bits", "h_tree_480_bits", "h_tree_topfreq_bits", "h_tree_neural_bits")
prep_sc <- function(df, lbl) {
  sub <- subset(df, outcome %in% h_outcomes & !is.na(b_mean))
  sub <- sub[order(sub$b_mean), ]
  sub$rank <- seq_len(nrow(sub))
  sub$pos_grp <- lbl
  sub$lo80 <- sub$b_mean + (sub$b_q05 - sub$b_mean) * K80_FROM_90
  sub$hi80 <- sub$b_mean + (sub$b_q95 - sub$b_mean) * K80_FROM_90
  sub
}
sc_v <- prep_sc(s2_v, "Verbs")
sc_n <- prep_sc(s2_n, "Nouns")
dfB <- rbind(sc_v, sc_n)
dfB$pos_grp <- factor(dfB$pos_grp, levels = c("Verbs", "Nouns"))

mk_sc_lab <- function(sub, grp) {
  n_tot <- nrow(sub)
  n_zero <- sum(sub$b_q025 <= 0 & sub$b_q975 >= 0)
  n_lt <- sum(sub$b_p_lt_044 >= 0.95)
  data.frame(
    pos_grp = grp,
    lab = sprintf("%s (%d specs)\n%d/%d include 0 (95%% CrI)\n%d/%d P(β < +0.44) ≥ 0.95",
                  grp, n_tot, n_zero, n_tot, n_lt, n_tot),
    stringsAsFactors = FALSE
  )
}
sc_labs <- rbind(mk_sc_lab(sc_v, "Verbs"), mk_sc_lab(sc_n, "Nouns"))
sc_labs$pos_grp <- factor(sc_labs$pos_grp, levels = c("Verbs", "Nouns"))

t1_ent_b <- subset(sy, name == "syn_ref__cat12_irreg13__main__MC_cross__ent__x_P")$b_mean

pB <- ggplot(dfB, aes(x = rank, y = b_mean, colour = pos_grp)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.15, ymax = 0.15,
           fill = ROPE_FILL, alpha = 0.75) +
  geom_hline(yintercept = 0, colour = INK_BODY, linewidth = 0.4) +
  geom_hline(yintercept = t1_ent_b, colour = COL_BREADTH, linewidth = 0.4, alpha = 0.65, linetype = "dotted") +
  geom_segment(aes(xend = rank, y = b_q025, yend = b_q975), linewidth = 0.35, alpha = 0.35) +
  geom_segment(aes(xend = rank, y = lo80, yend = hi80), linewidth = 0.85, alpha = 0.85) +
  geom_point(aes(shape = pos_grp), size = 0.95) +
  geom_text(data = sc_labs, aes(x = 1, y = 0.96, label = lab),
            inherit.aes = FALSE, hjust = 0, vjust = 1,
            size = 7.0 / .pt, lineheight = 1.08, colour = INK_BODY, family = "Helvetica") +
  facet_wrap(~pos_grp, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(breaks = NULL, expand = expansion(add = 1.8)) +
  scale_y_continuous(breaks = c(-0.5, 0.0, 0.5)) +
  coord_cartesian(ylim = c(-0.78, 0.98)) +
  labs(
    title = expression(bold("b   Full specification curve across MGL / neural ") * bolditalic(H)[bold("tree")] * bold(" models (dotted: Track 1 +0.44)")),
    x = "Model specifications, sorted by posterior mean (thick: 80% CrI; thin: 95% CrI)",
    y = "Slope per SD"
  ) +
  theme_track2() +
  theme(
    strip.text = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.spacing = unit(12, "pt")
  )

fig3 <- pA / pB + plot_layout(heights = c(1.55, 0.95))
save_track2_fig(fig3, fig_dir, "fig3_mgl_admixture_synthesis", width = 6.5, height = 6.5, dpi = 300)
