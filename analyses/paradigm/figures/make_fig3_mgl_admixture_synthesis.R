# Figure 3: Harmonized f3_fixed Two-Block Synthesis & Specification Curve (Track 2 MGL vs. Track 1)
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
sem1 <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_1factor_results.csv"), stringsAsFactors = FALSE)

# --- Panel A: Track 1 Grambank Reference + Track 2 Two-Block Forest (f3_fixed) ---
rows <- list()
add_row <- function(section, row_lbl, series, mean, lo95, hi95, lo80, hi80) {
  rows[[length(rows) + 1L]] <<- data.frame(
    section = section,
    row_lbl = row_lbl,
    series  = series,
    mean    = mean,
    lo95    = lo95,
    hi95    = hi95,
    lo80    = lo80,
    hi80    = hi80,
    stringsAsFactors = FALSE
  )
}

# Section 1: Track 1 Grambank Reference (2PL IRT-SEM maximal, f3_fixed, n_A >= 3)
t1_defs <- list(
  list(dom = "v_breadth", lbl = "Verb breadth",            ser = "Track 1: Verb breadth"),
  list(dom = "n_breadth", lbl = "Noun breadth",            ser = "Track 1: Noun breadth"),
  list(dom = "v_irreg",   lbl = "Verb irregularity",       ser = "Track 1: Verb irreg."),
  list(dom = "n_irreg",   lbl = "Noun irregularity",       ser = "Track 1: Noun irreg."),
  list(dom = "np_agr",    lbl = "Contextual NP agreement", ser = "Track 1: NP agreement")
)

for (td in t1_defs) {
  r <- subset(sem1, domain == td$dom & arch == "MA_2pl" & spec == "maximal" & set == "main" & min_n_ind == 3L)
  stopifnot(nrow(r) == 1L)
  q80 <- get_exact_q80(r, "b")
  add_row("Track 1\nGrambank\n(2PL SEM)", td$lbl, td$ser,
          r$b_mean, r$b_q025, r$b_q975, q80$lo80, q80$hi80)
}

# Section 2 & 3: Track 2 Paradigm Predictability (Block 1 + Key Block 2)
spec_defs <- list(
  list(sec = "Track 2\nBlock 1", lbl = "Maximal (Phylo + GP)",    sg = "maximal",          meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 1", lbl = "LOO: no spatial GP",      sg = "loo_no_space",     meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 1", lbl = "LOO: no paradigm size",   sg = "loo_no_cdist",     meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 1", lbl = "LOO: no demography",      sg = "loo_no_demog",     meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 1", lbl = "LOO: no lexicon prov.",   sg = "loo_no_lexprov",   meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 1", lbl = "Within-family (Mundlak)", sg = "mundlak_wb",       meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Full sample",             sg = "full_sample_PS",   meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Cutoff n >= 5",           sg = "cutoff_n5",        meas = "f3_fixed", min_n = 5L),
  list(sec = "Track 2\nBlock 2", lbl = "Strict links",            sg = "strict_links",     meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Bouckaert stem",          sg = "bouckaert_stem",   meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Bouckaert crown",         sg = "bouckaert_crown",  meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Drop Inner Asia",         sg = "no_inner_asia",    meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Uniform-480",             sg = "tier2_uniform480", meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Top-freq-120",            sg = "topfreq120",       meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "20-draw Berkson",         sg = "berkson_20draw",   meas = "f3_fixed", min_n = 3L),
  list(sec = "Track 2\nBlock 2", lbl = "Classical ME",            sg = "me_baseline",      meas = "f3_fixed", min_n = 3L)
)

v_max <- subset(s2_v, spec_group == "maximal" & measure == "f3_fixed" & min_n_ind == 3L)
n_max <- subset(s2_n, spec_group == "maximal" & measure == "f3_fixed" & min_n_ind == 3L)
stopifnot(nrow(v_max) == 1L, nrow(n_max) == 1L)

lbl_v_ser <- sprintf("Track 2 Verbs (n = %d)", v_max$n_lang)
lbl_n_ser <- sprintf("Track 2 Nouns (n = %d)", n_max$n_lang)

for (pos_info in list(list(df = s2_v, ser = lbl_v_ser),
                      list(df = s2_n, ser = lbl_n_ser))) {
  df_p <- pos_info$df
  for (sp in spec_defs) {
    r <- subset(df_p, spec_group == sp$sg & measure == sp$meas & min_n_ind == sp$min_n)
    if (nrow(r) == 1L) {
      q80 <- get_exact_q80(r, "b")
      add_row(sp$sec, sp$lbl, pos_info$ser,
              r$b_mean, r$b_q025, r$b_q975, q80$lo80, q80$hi80)
    }
  }
}

dfA <- do.call(rbind, rows)
row_order <- c(vapply(t1_defs, `[[`, character(1), "lbl"),
               vapply(spec_defs, `[[`, character(1), "lbl"))
dfA$row_lbl <- factor(dfA$row_lbl, levels = rev(row_order))
dfA$section <- factor(dfA$section, levels = c("Track 1\nGrambank\n(2PL SEM)", "Track 2\nBlock 1", "Track 2\nBlock 2"))

series_levels <- c(
  lbl_v_ser, lbl_n_ser,
  "Track 1: Verb breadth", "Track 1: Noun breadth",
  "Track 1: Verb irreg.", "Track 1: Noun irreg.", "Track 1: NP agreement"
)
dfA$series <- factor(dfA$series, levels = rev(series_levels))

col_vals <- setNames(
  c(COL_VERB, COL_NOUN, COL_BREADTH, COL_BREADTH, COL_IRREG, COL_IRREG, COL_MAG),
  series_levels
)
shp_vals <- setNames(
  c(16, 17, 16, 17, 16, 17, 15),
  series_levels
)

# Dynamic x-axis limits that never clip any 95% CrI
x_lim_a <- range(c(dfA$lo95, dfA$hi95, -0.60, 0.65), na.rm = TRUE)
x_pad_a <- 0.05 * diff(x_lim_a)
xlim_a  <- c(x_lim_a[1] - x_pad_a, x_lim_a[2] + x_pad_a)

pA <- ggplot(dfA, aes(x = mean, y = row_lbl, colour = series, shape = series)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                position = position_dodge(width = 0.56), linewidth = 0.46, alpha = 0.52) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                position = position_dodge(width = 0.56), linewidth = 1.30, alpha = 1.0) +
  geom_point(position = position_dodge(width = 0.56), size = 1.65) +
  facet_grid(section ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = col_vals, breaks = c(lbl_v_ser, lbl_n_ser), name = NULL) +
  scale_shape_manual(values = shp_vals, breaks = c(lbl_v_ser, lbl_n_ser), name = NULL) +
  scale_x_continuous(breaks = c(-0.6, -0.3, 0.0, 0.3, 0.6)) +
  coord_cartesian(xlim = xlim_a) +
  labs(
    title = "a   Cross-family f3 slopes: Track 1 reference & Track 2 two-block synthesis",
    subtitle = sprintf("Maximal (Phylo + GP): Verbs BF01 = %.2f, P(|b| \u2264 0.15) = %.0f%%; Nouns BF01 = %.2f, P(|b| \u2264 0.15) = %.0f%%",
                       v_max$b_bf01_norm, 100 * v_max$b_p_rope_015,
                       n_max$b_bf01_norm, 100 * n_max$b_p_rope_015),
    x = "Slope per SD of cross-family f3 (80% & 95% CrI; shaded: \u00b10.15 ROPE)",
    y = NULL
  ) +
  theme_track2() +
  theme(
    strip.placement = "outside",
    strip.text.y.left = element_text(colour = INK_BODY, face = "bold", size = 7.4, angle = 0, hjust = 1, lineheight = 1.02),
    axis.text.y = element_text(colour = INK_BODY, size = 7.3),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing.y = unit(6, "pt"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.width = unit(12, "pt"),
    legend.text = element_text(size = 7.2, colour = INK_BODY),
    legend.margin = margin(t = -2, b = 0)
  )

# --- Panel B: Full Specification Curve across all fitted H_tree models (excluding xmode == "none") ---
h_outcomes <- c("h_tree_bits", "h_tree_480_bits", "h_tree_topfreq_bits",
                "h_tree_neural_bits", "h_tree_malouf_bits", "h_tree_phon_bits")

prep_sc <- function(df, lbl) {
  sub <- subset(df, outcome %in% h_outcomes & xmode != "none" & !is.na(b_mean))
  q80 <- get_exact_q80(sub, "b")
  sub$lo80 <- q80$lo80
  sub$hi80 <- q80$hi80
  sub <- sub[order(sub$b_mean), ]
  sub$rank <- seq_len(nrow(sub))
  sub$pos_grp <- lbl
  sub
}
sc_v <- prep_sc(s2_v, "Verbs")
sc_n <- prep_sc(s2_n, "Nouns")
dfB <- rbind(sc_v, sc_n)
dfB$pos_grp <- factor(dfB$pos_grp, levels = c("Verbs", "Nouns"))

y_lim_b <- range(c(dfB$b_q025, dfB$b_q975, -0.55, 0.70), na.rm = TRUE)
y_span  <- diff(y_lim_b)
y_top_b <- y_lim_b[2] + 0.24 * y_span

mk_sc_lab <- function(sub, grp) {
  n_tot <- nrow(sub)
  n_zero <- sum(sub$b_q025 <= 0 & sub$b_q975 >= 0)
  med_rope <- 100 * median(sub$b_p_rope_015, na.rm = TRUE)
  data.frame(
    pos_grp = grp,
    lab = sprintf("%s (%d specs)\n%d/%d include 0 (95%% CrI)\nMedian ROPE share: %.0f%%",
                  grp, n_tot, n_zero, n_tot, med_rope),
    stringsAsFactors = FALSE
  )
}
sc_labs <- rbind(mk_sc_lab(sc_v, "Verbs"), mk_sc_lab(sc_n, "Nouns"))
sc_labs$pos_grp <- factor(sc_labs$pos_grp, levels = c("Verbs", "Nouns"))

pB <- ggplot(dfB, aes(x = rank, y = b_mean, colour = pos_grp)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.15, ymax = 0.15,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_hline(yintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_segment(aes(xend = rank, y = b_q025, yend = b_q975), linewidth = 0.38, alpha = 0.38) +
  geom_segment(aes(xend = rank, y = lo80, yend = hi80), linewidth = 0.90, alpha = 0.88) +
  geom_point(aes(shape = pos_grp), size = 1.05) +
  geom_text(data = sc_labs, aes(x = 1, y = y_top_b - 0.02 * y_span, label = lab),
            inherit.aes = FALSE, hjust = 0, vjust = 1,
            size = 7.0 / .pt, lineheight = 1.08, colour = INK_BODY, family = "Helvetica") +
  facet_wrap(~pos_grp, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(breaks = NULL, expand = expansion(add = 1.5)) +
  scale_y_continuous(breaks = c(-0.5, 0.0, 0.5)) +
  coord_cartesian(ylim = c(y_lim_b[1] - 0.04 * y_span, y_top_b)) +
  labs(
    title = expression(bold("b   Full specification curve across fitted ") * bolditalic(H)[bold("tree")] * bold(" models")),
    x = "Model specifications, sorted by posterior mean (thick: 80% CrI; thin: 95% CrI)",
    y = "Slope per SD"
  ) +
  theme_track2() +
  theme(
    strip.text = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.spacing = unit(12, "pt")
  )

fig3 <- pA / wrap_elements(full = pB) + plot_layout(heights = c(1.85, 0.98))
save_track2_fig(fig3, fig_dir, "fig3_mgl_admixture_synthesis", width = 6.5, height = 7.0, dpi = 300)
