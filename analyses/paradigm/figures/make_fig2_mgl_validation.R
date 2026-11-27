# Figure 2: 4-Panel Methodological & Cross-Track Construct Validation of MGL i-Complexity
# Outputs: fig2_mgl_validation.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

u120 <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform120.csv"), stringsAsFactors = FALSE)
mal  <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform120_malouf.csv"), stringsAsFactors = FALSE)
u480 <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform480.csv"), stringsAsFactors = FALSE)
tf   <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_topfreq120.csv"), stringsAsFactors = FALSE)
ld_v <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_V.csv"), stringsAsFactors = FALSE)
ld_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_N.csv"), stringsAsFactors = FALSE)
ct   <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/cross_track_correlations.csv"), stringsAsFactors = FALSE)

# --- Panel A: Vowel-Tier MGL vs. Malouf (2017) Archiphonemes (n = 33) ---
m_mal <- merge(u120, mal, by = "unit_id", suffixes = c("_mgl", "_mal"))
n_mal_all  <- nrow(m_mal)
n_mal_orth <- sum(m_mal$representation_mgl == "orth")
n_mal_phon <- sum(m_mal$representation_mgl == "phon")
n_mal_v    <- sum(m_mal$representation_mgl == "orth" & m_mal$pos_mgl == "V")
n_mal_n    <- sum(m_mal$representation_mgl == "orth" & m_mal$pos_mgl == "N")

lbl_mv <- sprintf("Verbs, orth (%d)", n_mal_v)
lbl_mn <- sprintf("Nouns, orth (%d)", n_mal_n)
lbl_mp <- sprintf("Phonemic (%d)", n_mal_phon)
mal_levels <- c(lbl_mv, lbl_mn, lbl_mp)

m_mal$cat_lbl <- ifelse(m_mal$representation_mgl == "phon", lbl_mp,
                 ifelse(m_mal$pos_mgl == "V", lbl_mv, lbl_mn))
m_mal$cat_lbl <- factor(m_mal$cat_lbl, levels = mal_levels)

r_mal_orth <- with(subset(m_mal, representation_mgl == "orth"), cor(h_tree_bits_mgl, h_tree_bits_mal))
r_mal_all  <- with(m_mal, cor(h_tree_bits_mgl, h_tree_bits_mal))

pA <- ggplot(m_mal, aes(x = h_tree_bits_mgl, y = h_tree_bits_mal, colour = cat_lbl, shape = cat_lbl)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = GREY_MID, linewidth = 0.45) +
  geom_point(size = 1.75, alpha = 0.90) +
  annotate("text", x = 0.15, y = 7.1, hjust = 0, vjust = 1,
           label = sprintf("n = %d (%d orth + %d phon)\nr = %.3f (orth), %.3f (all)",
                           n_mal_all, n_mal_orth, n_mal_phon, r_mal_orth, r_mal_all),
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
n_pv <- sum(p_orth$pos == "V")
n_pn <- sum(p_orth$pos == "N")
lbl_pv <- sprintf("Verbs (%d)", n_pv)
lbl_pn <- sprintf("Nouns (%d)", n_pn)
p_orth$pos_lbl <- factor(ifelse(p_orth$pos == "V", lbl_pv, lbl_pn), levels = c(lbl_pv, lbl_pn))

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
  scale_colour_manual(values = setNames(c(COL_VERB, COL_NOUN), c(lbl_pv, lbl_pn)), name = NULL) +
  scale_shape_manual(values = setNames(c(16, 17), c(lbl_pv, lbl_pn)), name = NULL) +
  scale_x_continuous(breaks = c(0, 0.5, 1.0, 1.5, 2.0, 2.5)) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0, 1.5, 2.0, 2.5)) +
  coord_cartesian(xlim = c(0, 2.65), ylim = c(0, 2.68)) +
  labs(
    title = sprintf("c   Orthography vs. Gold IPA (%d pairs)", nrow(p_orth)),
    x = expression("Orthographic " * italic(H)[tree] * " (bits)"),
    y = expression("Gold phonemic IPA " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(legend.position = c(0.82, 0.17),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.key.height = unit(9, "pt"),
        legend.text = element_text(size = 7.0, colour = INK2))

# --- Panel D: Cross-Track Construct Validation vs. Track 1 Grambank ---
# Subpanel D1: log_C_distinct vs POS-matched Track-1 Breadth (Verbs & Nouns)
dv_b <- subset(ld_v, !is.na(v_breadth_prop) & is.finite(log_C_distinct))
dn_b <- subset(ld_n, !is.na(n_breadth_prop) & is.finite(log_C_distinct))

row_vb <- subset(ct, comparison == "log_C_distinct_V_vs_Track1_v_breadth_prop")
row_nb <- subset(ct, comparison == "log_C_distinct_N_vs_Track1_n_breadth_prop")
stopifnot(nrow(row_vb) == 1L, nrow(row_nb) == 1L)

df_d1 <- rbind(
  data.frame(pos = "Verbs", x = dv_b$log_C_distinct, y = dv_b$v_breadth_prop, stringsAsFactors = FALSE),
  data.frame(pos = "Nouns", x = dn_b$log_C_distinct, y = dn_b$n_breadth_prop, stringsAsFactors = FALSE)
)
df_d1$pos <- factor(df_d1$pos, levels = c("Verbs", "Nouns"))

pD1 <- ggplot(df_d1, aes(x = x, y = y, colour = pos, fill = pos, shape = pos, linetype = pos)) +
  geom_smooth(method = "lm", formula = y ~ x, alpha = 0.14, linewidth = 0.65) +
  geom_point(size = 1.45, alpha = 0.85) +
  annotate("text", x = 5.65, y = -0.18, hjust = 1, vjust = 0,
           label = sprintf("N (n=%d): r=%.2f, \u03c1=%.2f, PGLS \u03b2=%.2f\nV (n=%d): r=%.2f, \u03c1=%.2f, PGLS \u03b2=%.2f",
                           row_nb$n_overlap, row_nb$pearson_r, row_nb$spearman_rho, row_nb$pgls_std_beta,
                           row_vb$n_overlap, row_vb$pearson_r, row_vb$spearman_rho, row_vb$pgls_std_beta),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.08, family = "Helvetica") +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_fill_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), name = NULL) +
  scale_linetype_manual(values = c("Verbs" = "solid", "Nouns" = "dashed"), name = NULL) +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5)) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0), labels = c("0", "0.5", "1")) +
  coord_cartesian(xlim = c(0.6, 5.75), ylim = c(-0.22, 1.24)) +
  labs(
    title = "d   Cross-track construct validity (Track 1)",
    x = expression("Log distinct paradigm cells (" * log * " |" * italic(C) * "|)"),
    y = "Track-1 breadth"
  ) +
  theme_track2() +
  theme(
    legend.position = c(0.30, 0.93),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = "white", colour = NA),
    legend.key.height = unit(7.5, "pt"),
    legend.text = element_text(size = 7.0, colour = INK_BODY),
    plot.margin = margin(5, 6, 2, 5)
  )

# Subpanel D2: Nominal H_tree vs Track-1 NA-masked nominal irregularity
# Uses observed n_irreg_prop (plus dynamic theta_n_irreg_eap annotation if exported)
d_nirr <- subset(ld_n, !is.na(n_irreg_prop) & is.finite(h_tree_bits))
row_ni <- subset(ct, comparison == "H_tree_N_vs_Track1_n_irreg_prop")
stopifnot(nrow(row_ni) == 1L)

eap_file <- file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_latent_eap.csv")
eap_note <- ""
if (file.exists(eap_file)) {
  eap_df <- read.csv(eap_file, stringsAsFactors = FALSE)
  eap_col <- intersect(c("theta_n_irreg_eap", "n_irreg_eap", "theta_n_irreg"), names(eap_df))
  if (length(eap_col) >= 1L) {
    m_eap <- merge(ld_n[, c("glottocode", "h_tree_bits")],
                   eap_df[, c("glottocode", eap_col[1]), drop = FALSE],
                   by = "glottocode")
    m_eap <- m_eap[is.finite(m_eap$h_tree_bits) & is.finite(m_eap[[eap_col[1]]]), , drop = FALSE]
    if (nrow(m_eap) >= 8L) {
      r_e <- cor(m_eap$h_tree_bits, m_eap[[eap_col[1]]])
      rho_e <- suppressWarnings(cor(m_eap$h_tree_bits, m_eap[[eap_col[1]]], method = "spearman"))
      eap_note <- sprintf("\nLatent \u03b8 (n=%d): r=%.2f, \u03c1=%.2f", nrow(m_eap), r_e, rho_e)
    }
  }
}

set.seed(2026)
d_nirr$y_jit <- pmin(1.03, pmax(-0.03, d_nirr$n_irreg_prop + runif(nrow(d_nirr), -0.022, 0.022)))

nirr_spec <- data.frame(
  glottocode = c("mode1248", "iris1253", "russ1263"),
  short = c("Mod. Greek", "Irish", "Russian"),
  dx    = c(-0.08, -0.25,  0.14),
  dy    = c( 0.16,  0.12,  0.00),
  hjust = c( 1.00,  1.00,  0.00),
  stringsAsFactors = FALSE
)
lab_ni <- merge(d_nirr, nirr_spec, by = "glottocode")

pD2 <- ggplot(d_nirr, aes(x = h_tree_bits, y = y_jit)) +
  geom_smooth(aes(y = n_irreg_prop), method = "lm", formula = y ~ x,
              colour = COL_NOUN, fill = COL_NOUN, alpha = 0.16, linewidth = 0.7) +
  geom_point(colour = COL_NOUN, shape = 17, size = 1.55, alpha = 0.85) +
  geom_text(data = lab_ni,
            aes(x = h_tree_bits + dx, y = y_jit + dy, label = short, hjust = hjust),
            size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  annotate("text", x = 5.45, y = -0.44, hjust = 1, vjust = 0,
           label = sprintf("n=%d: r=%.2f, \u03c1=%.2f, PGLS \u03b2=%.2f%s",
                           row_ni$n_overlap, row_ni$pearson_r, row_ni$spearman_rho,
                           row_ni$pgls_std_beta, eap_note),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.08, family = "Helvetica") +
  scale_x_continuous(breaks = c(0, 1, 2, 3, 4, 5)) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0), labels = c("0", "0.5", "1")) +
  coord_cartesian(xlim = c(0, 5.55), ylim = c(-0.48, 1.15)) +
  labs(
    x = expression("Nominal MGL " * italic(H)[tree] * " (bits)"),
    y = "Noun irreg. share"
  ) +
  theme_track2() +
  theme(plot.margin = margin(2, 6, 5, 5))

pD <- pD1 / pD2

fig2 <- (pA | pB) / (pC | pD)
save_track2_fig(fig2, fig_dir, "fig2_mgl_validation", width = 6.5, height = 5.9, dpi = 300)
