# Figure 2: 4-Panel Methodological & Construct Validation of MGL i-Complexity
# Outputs: fig2_mgl_validation.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

us   <- read.csv(file.path(root_dir, "analyses/paradigm/data/unit_summary.csv"), stringsAsFactors = FALSE)
u120 <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform120.csv"), stringsAsFactors = FALSE)
mal  <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform120_malouf.csv"), stringsAsFactors = FALSE)
u480 <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform480.csv"), stringsAsFactors = FALSE)
tf   <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_topfreq120.csv"), stringsAsFactors = FALSE)
ld_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_N.csv"), stringsAsFactors = FALSE)
ct   <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/cross_track_correlations.csv"), stringsAsFactors = FALSE)

# --- Panel A: Vowel-Tier MGL vs. Malouf (2017) Archiphonemes (n = 33) ---
m_mal <- merge(u120, mal, by = "unit_id", suffixes = c("_mgl", "_mal"))
m_mal <- merge(m_mal, us[, c("unit_id", "language_name")], by = "unit_id")
stopifnot(nrow(m_mal) == 33L)
m_mal$cat_lbl <- ifelse(m_mal$representation_mgl == "phon", "Phonemic (4)",
                 ifelse(m_mal$pos_mgl == "V", "Verbs, orth (12)", "Nouns, orth (17)"))
mal_levels <- c("Verbs, orth (12)", "Nouns, orth (17)", "Phonemic (4)")
m_mal$cat_lbl <- factor(m_mal$cat_lbl, levels = mal_levels)

r_mal_orth <- with(subset(m_mal, representation_mgl == "orth"), cor(h_tree_bits_mgl, h_tree_bits_mal))
r_mal_all  <- with(m_mal, cor(h_tree_bits_mgl, h_tree_bits_mal))

pA <- ggplot(m_mal, aes(x = h_tree_bits_mgl, y = h_tree_bits_mal, colour = cat_lbl, shape = cat_lbl)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = GREY_MID, linewidth = 0.45) +
  geom_point(size = 1.75, alpha = 0.90) +
  annotate("text", x = 0.15, y = 7.1, hjust = 0, vjust = 1,
           label = sprintf("n = 33 (29 orth + 4 phon)\nr = %.3f (orth), %.3f (all)", r_mal_orth, r_mal_all),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.12, family = "Helvetica") +
  scale_colour_manual(values = setNames(c(COL_VERB, COL_NOUN, INK_BODY), mal_levels), name = NULL) +
  scale_shape_manual(values = setNames(c(16, 17, 1), mal_levels), name = NULL) +
  scale_x_continuous(breaks = c(0, 2, 4, 6)) +
  scale_y_continuous(breaks = c(0, 2, 4, 6)) +
  coord_cartesian(xlim = c(-0.1, 7.3), ylim = c(-0.1, 7.3)) +
  labs(
    title = "a   Vowel-tier MGL vs. Malouf harmony",
    x = expression("Vowel-tier MGL " * italic(H)[tree] * " (bits)"),
    y = expression("Malouf-abstracted " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(legend.position = c(0.73, 0.20),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.key.height = unit(8.5, "pt"),
        legend.text = element_text(size = 7.0, colour = INK2))

# --- Panel B: Sample-Size & Lexical-Stratum Stability (uniform480 & topfreq120) ---
m_480 <- merge(u120, u480, by = "unit_id", suffixes = c("_120", "_480"))
m_tf  <- merge(u120, tf,   by = "unit_id", suffixes = c("_120", "_tf"))
stopifnot(nrow(m_480) == 73L, nrow(m_tf) == 74L)

r_480 <- cor(m_480$h_tree_bits_120, m_480$h_tree_bits_480)
pct_480_lower <- 100 * mean(m_480$h_tree_bits_480 <= m_480$h_tree_bits_120)
r_tf  <- cor(m_tf$h_tree_bits_120, m_tf$h_tree_bits_tf)
pct_tf_higher <- 100 * mean(m_tf$h_tree_bits_tf > m_tf$h_tree_bits_120)

dfB <- rbind(
  data.frame(regime = sprintf("Top-freq 120 (n=%d, r=%.3f)", nrow(m_tf), r_tf),
             x = m_tf$h_tree_bits_120, y = m_tf$h_tree_bits_tf, stringsAsFactors = FALSE),
  data.frame(regime = sprintf("Uniform 480 (n=%d, r=%.3f)", nrow(m_480), r_480),
             x = m_480$h_tree_bits_120, y = m_480$h_tree_bits_480, stringsAsFactors = FALSE)
)
reg_levels <- unique(dfB$regime)
dfB$regime <- factor(dfB$regime, levels = reg_levels)

pB <- ggplot(dfB, aes(x = x, y = y, colour = regime, shape = regime)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = GREY_MID, linewidth = 0.45) +
  geom_point(size = 1.5, alpha = 0.78) +
  annotate("text", x = 0.05, y = 6.15, hjust = 0, vjust = 1,
           label = sprintf("Top-freq > 120: %.1f%%\nUniform 480 <= 120: %.1f%%",
                           pct_tf_higher, pct_480_lower),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.12, family = "Helvetica") +
  scale_colour_manual(values = setNames(c(COL_REGIME_TF, COL_REGIME_480), reg_levels), name = NULL) +
  scale_shape_manual(values = setNames(c(18, 15), reg_levels), name = NULL) +
  scale_x_continuous(breaks = c(0, 2, 4, 6)) +
  scale_y_continuous(breaks = c(0, 2, 4, 6)) +
  coord_cartesian(xlim = c(-0.1, 6.2), ylim = c(-0.1, 6.3)) +
  labs(
    title = "b   Training size & frequency stratum",
    x = expression("Uniform-120 " * italic(H)[tree] * " (bits)"),
    y = expression("Comparison regime " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(legend.position = c(0.64, 0.16),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.key.height = unit(9, "pt"),
        legend.text = element_text(size = 7.0, colour = INK2))

# --- Panel C: Paired Orthography vs. Gold Phonemic IPA (12 pairs) ---
p_orth <- merge(
  subset(u120, representation == "orth"),
  subset(u120, representation == "phon"),
  by = c("glottocode", "pos"),
  suffixes = c("_orth", "_phon")
)
p_orth <- merge(p_orth, us[, c("unit_id", "language_name")],
                by.x = "unit_id_orth", by.y = "unit_id")
stopifnot(nrow(p_orth) == 12L)
p_orth$pos_lbl <- ifelse(p_orth$pos == "V", "Verbs (7)", "Nouns (5)")

c_spec <- data.frame(
  unit_id_orth = c("russ1263_N_orth", "esto1258_N_orth", "stan1293_V_orth",
                   "sout1528_N_orth", "esto1258_V_orth"),
  short = c("Russian N\n(vowel red.)", "Estonian N\n(Q2/Q3)", "English V\n(opaque spell.)",
            "Serbo-Cr. N", "Estonian V"),
  dx    = c( 0.08,  0.07,  0.00,  0.14, -0.04),
  dy    = c(-0.04,  0.06,  0.22, -0.10,  0.22),
  hjust = c( 0.00,  0.00,  0.50,  0.00,  0.30),
  stringsAsFactors = FALSE
)
c_labs <- merge(p_orth, c_spec, by = "unit_id_orth")
stopifnot(nrow(c_labs) == nrow(c_spec))
c_labs$lx <- c_labs$h_tree_bits_orth + c_labs$dx
c_labs$ly <- c_labs$h_tree_bits_phon + c_labs$dy
c_labs$txt_col <- ifelse(c_labs$pos == "V", COL_VERB, COL_NOUN_TXT)

pC <- ggplot(p_orth, aes(x = h_tree_bits_orth, y = h_tree_bits_phon,
                         colour = pos_lbl, shape = pos_lbl)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = GREY_MID, linewidth = 0.45) +
  geom_point(size = 2.0, alpha = 0.92) +
  geom_text(data = c_labs, aes(x = lx, y = ly, label = short, hjust = hjust),
            colour = c_labs$txt_col, size = 7.0 / .pt, lineheight = 0.95,
            family = "Helvetica", inherit.aes = FALSE) +
  scale_colour_manual(values = c("Verbs (7)" = COL_VERB, "Nouns (5)" = COL_NOUN), name = NULL) +
  scale_shape_manual(values = c("Verbs (7)" = 16, "Nouns (5)" = 17), name = NULL) +
  scale_x_continuous(breaks = c(0, 0.5, 1.0, 1.5, 2.0, 2.5)) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0, 1.5, 2.0, 2.5)) +
  coord_cartesian(xlim = c(0, 2.65), ylim = c(0, 2.68)) +
  labs(
    title = "c   Orthography vs. Gold IPA (12 pairs)",
    x = expression("Orthographic " * italic(H)[tree] * " (bits)"),
    y = expression("Gold phonemic IPA " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(legend.position = c(0.82, 0.17),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.key.height = unit(9, "pt"),
        legend.text = element_text(size = 7.0, colour = INK2))

# --- Panel D: Cross-Track Construct Validity vs. Track 1 Grambank Irregularity ---
d_prop <- subset(ld_n, !is.na(irreg_noun_prop))
d_theta <- subset(ld_n, !is.na(theta_irreg13_eap))
row_prop <- subset(ct, comparison == "H_tree_N_vs_Track1_irreg_noun_prop")
row_theta <- subset(ct, comparison == "H_tree_N_vs_Track1_theta_irreg13_EAP")
stopifnot(nrow(d_prop) == row_prop$n_overlap, nrow(d_theta) == row_theta$n_overlap)

set.seed(2026)
d_prop$y_jit <- pmin(1.03, pmax(-0.03, d_prop$irreg_noun_prop + runif(nrow(d_prop), -0.022, 0.022)))

prop_spec <- data.frame(
  glottocode = c("mode1248", "iris1253", "russ1263"),
  short = c("Mod. Greek", "Irish", "Russian"),
  dx    = c(-0.08, -0.35,  0.14),
  dy    = c( 0.18, -0.08,  0.00),
  hjust = c( 1.00,  1.00,  0.00),
  stringsAsFactors = FALSE
)
lab_prop <- merge(d_prop, prop_spec, by = "glottocode")
stopifnot(nrow(lab_prop) == nrow(prop_spec))

pD1 <- ggplot(d_prop, aes(x = h_tree_bits, y = y_jit)) +
  geom_smooth(aes(y = irreg_noun_prop), method = "lm", formula = y ~ x,
              colour = COL_NOUN, fill = COL_NOUN, alpha = 0.16, linewidth = 0.7) +
  geom_point(colour = COL_NOUN, shape = 17, size = 1.6, alpha = 0.85) +
  geom_text(data = lab_prop,
            aes(x = h_tree_bits + dx, y = y_jit + dy, label = short, hjust = hjust),
            size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  annotate("text", x = 5.4, y = 0.04, hjust = 1, vjust = 0,
           label = sprintf("n = %d, r = %.3f (p = %.4f)\nSpearman ρ = %.3f",
                           row_prop$n_overlap, row_prop$pearson_r, row_prop$pearson_p, row_prop$spearman_rho),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.08, family = "Helvetica") +
  scale_x_continuous(breaks = c(0, 1, 2, 3, 4, 5)) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0), labels = c("0", "0.5", "1")) +
  coord_cartesian(xlim = c(0, 5.5), ylim = c(-0.08, 1.15)) +
  labs(
    title = "d   Construct validity vs. Track 1 Grambank",
    x = NULL,
    y = "Noun irreg. share"
  ) +
  theme_track2() +
  theme(axis.text.x = element_blank(), plot.margin = margin(5, 6, 1, 5))

pD2 <- ggplot(d_theta, aes(x = h_tree_bits, y = theta_irreg13_eap)) +
  geom_smooth(method = "lm", formula = y ~ x,
              colour = COL_NOUN, fill = COL_NOUN, alpha = 0.16, linewidth = 0.7) +
  geom_point(colour = COL_NOUN, shape = 17, size = 1.6, alpha = 0.85) +
  annotate("text", x = 5.4, y = -1.70, hjust = 1, vjust = 0,
           label = sprintf("n = %d, r = %.3f, ρ = %.3f",
                           row_theta$n_overlap, row_theta$pearson_r, row_theta$spearman_rho),
           size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  scale_x_continuous(breaks = c(0, 1, 2, 3, 4, 5)) +
  scale_y_continuous(breaks = c(-1, 0, 1, 2)) +
  coord_cartesian(xlim = c(0, 5.5), ylim = c(-1.85, 2.75)) +
  labs(
    x = expression("Nominal MGL " * italic(H)[tree] * " (bits)"),
    y = expression("Latent " * theta[irreg13])
  ) +
  theme_track2() +
  theme(plot.margin = margin(1, 6, 5, 5))

pD <- pD1 / pD2

fig2 <- (pA | pB) / (pC | pD)
save_track2_fig(fig2, fig_dir, "fig2_mgl_validation", width = 6.5, height = 5.7, dpi = 300)
