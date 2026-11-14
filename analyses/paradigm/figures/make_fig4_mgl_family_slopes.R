# Figure 4: Within-Family Trajectories (Admixture vs. MGL H_tree)
# Styled directly after Track 1 fig5_family_slopes.png
# Outputs: fig4_mgl_family_slopes.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

ld_v <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_V.csv"), stringsAsFactors = FALSE)
ld_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_N.csv"), stringsAsFactors = FALSE)
s2_v <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/stage2_results_V.csv"), stringsAsFactors = FALSE)
s2_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/stage2_results_N.csv"), stringsAsFactors = FALSE)

# --- Panel A: Global, Mundlak Within/Between-Family, and Leave-Region-Out Slopes (ent) ---
extract_chk <- function(df_p, pos_lbl) {
  r_glob <- subset(df_p, set == "main" & spec_group == "full_covs" & ctrl == "F" & measure == "ent")
  r_mund <- subset(df_p, set == "main" & spec_group == "full_covs_mundlak" & ctrl == "F" & measure == "ent")
  r_ntm  <- subset(df_p, set == "main" & spec_group == "no_turk_mong" & ctrl == "P" & measure == "ent")
  r_nia  <- subset(df_p, set == "main" & spec_group == "no_inner_asia" & ctrl == "P" & measure == "ent")
  stopifnot(nrow(r_glob) == 1L, nrow(r_mund) == 1L, nrow(r_ntm) == 1L, nrow(r_nia) == 1L)

  mk <- function(lbl, mean, q025, q05, q95, q975) {
    data.frame(
      pos = pos_lbl, label = lbl, mean = mean,
      lo95 = q025, hi95 = q975,
      lo80 = mean + (q05 - mean) * K80_FROM_90,
      hi80 = mean + (q95 - mean) * K80_FROM_90,
      stringsAsFactors = FALSE
    )
  }
  rbind(
    mk("Global (Phylo + GP)",       r_glob$b_mean,         r_glob$b_q025,         r_glob$b_q05,         r_glob$b_q95,         r_glob$b_q975),
    mk("Within-family (Mundlak)",   r_mund$b_mean,         r_mund$b_q025,         r_mund$b_q05,         r_mund$b_q95,         r_mund$b_q975),
    mk("Between-family (Mundlak)",  r_mund$b_between_mean, r_mund$b_between_q025, r_mund$b_between_q05, r_mund$b_between_q95, r_mund$b_between_q975),
    mk("Drop Turkic & Mongolic",    r_ntm$b_mean,          r_ntm$b_q025,          r_ntm$b_q05,          r_ntm$b_q95,          r_ntm$b_q975),
    mk("Drop all Inner Asia",       r_nia$b_mean,          r_nia$b_q025,          r_nia$b_q05,          r_nia$b_q95,          r_nia$b_q975)
  )
}

dfA <- rbind(extract_chk(s2_v, "Verbs"), extract_chk(s2_n, "Nouns"))
lbl_levels <- c("Global (Phylo + GP)", "Within-family (Mundlak)", "Between-family (Mundlak)",
                "Drop Turkic & Mongolic", "Drop all Inner Asia")
dfA$label <- factor(dfA$label, levels = rev(lbl_levels))
dfA$pos   <- factor(dfA$pos, levels = c("Nouns", "Verbs"))

pA <- ggplot(dfA, aes(x = mean, y = label, colour = pos, shape = pos)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.75) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 0.48, alpha = 0.55) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 1.35, alpha = 1.0) +
  geom_point(position = position_dodge(width = 0.52), size = 1.75) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(breaks = c(-0.5, 0.0, 0.5)) +
  coord_cartesian(xlim = c(-0.65, 0.70)) +
  labs(
    title = "a   Decomposition & leave-out",
    subtitle = "ent slopes (80% & 95% CrI; ±0.15 ROPE)",
    x = "Slope per SD of entropy",
    y = NULL
  ) +
  theme_track2() +
  theme(
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    axis.text.y = element_text(colour = INK_BODY, size = 7.4),
    legend.position = "none"
  )

# --- Panel B: Small Multiples Across Multi-Language Families (n >= 3 in V and N) ---
fams <- c("Indo-European", "Uralic", "Turkic", "Afro-Asiatic")

prep_fam <- function(df, pos_lbl) {
  sub <- subset(df, family %in% fams)
  sub$pos <- pos_lbl
  sub[, c("glottocode", "name", "family", "pos", "ent", "h_tree_bits", "h_tree_se")]
}
dfB <- rbind(prep_fam(ld_v, "Verbs"), prep_fam(ld_n, "Nouns"))

nv <- table(subset(ld_v, family %in% fams)$family)
nn <- table(subset(ld_n, family %in% fams)$family)
fam_titles <- setNames(
  sprintf("%s (%d V, %d N)", fams, as.integer(nv[fams]), as.integer(nn[fams])),
  fams
)
dfB$fam_panel <- factor(fam_titles[dfB$family], levels = unname(fam_titles))
dfB$pos <- factor(dfB$pos, levels = c("Verbs", "Nouns"))

# Salient family exemplar labels
ex_f <- subset(dfB,
               (glottocode == "moks1248" & pos == "Verbs") |
               (glottocode == "mode1248" & pos == "Nouns") |
               (glottocode == "iris1253" & pos == "Nouns") |
               (glottocode == "gulf1241" & pos == "Verbs") |
               (glottocode == "egyp1253" & pos == "Nouns") |
               (glottocode == "nucl1301" & pos == "Verbs"))
ex_f$short <- c(
  "moks1248" = "Moksha",
  "mode1248" = "Mod. Greek",
  "iris1253" = "Irish",
  "gulf1241" = "Gulf Arabic",
  "egyp1253" = "Egyptian Ar.",
  "nucl1301" = "Turkish"
)[ex_f$glottocode]
ex_f$txt_col <- ifelse(ex_f$pos == "Verbs", COL_VERB, COL_NOUN_TXT)

# Manual offsets per glottocode so zero collisions on sqrt-transformed y-axis
ex_f$dx <- c("moks1248" =  0.06, "mode1248" = -0.05, "iris1253" =  0.06,
             "gulf1241" = -0.06, "egyp1253" =  0.02, "nucl1301" = -0.06)[ex_f$glottocode]
ex_f$dy <- c("moks1248" = -0.35, "mode1248" =  0.45, "iris1253" =  0.22,
             "gulf1241" = -0.15, "egyp1253" =  1.25, "nucl1301" =  0.48)[ex_f$glottocode]
ex_f$hj <- c("moks1248" =  0.00, "mode1248" =  1.00, "iris1253" =  0.00,
             "gulf1241" =  1.00, "egyp1253" =  1.00, "nucl1301" =  1.00)[ex_f$glottocode]

dfB_ribbon <- subset(dfB, family %in% c("Indo-European", "Uralic"))

pB <- ggplot(dfB, aes(x = ent, y = h_tree_bits, colour = pos, fill = pos, shape = pos, linetype = pos)) +
  geom_smooth(data = dfB_ribbon, method = "lm", formula = y ~ x, se = TRUE, alpha = 0.14, linewidth = 0.72) +
  geom_smooth(data = subset(dfB, !(family %in% c("Indo-European", "Uralic"))),
              method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.72) +
  geom_errorbar(aes(ymin = pmax(0, h_tree_bits - h_tree_se), ymax = h_tree_bits + h_tree_se),
                width = 0, linewidth = 0.35, alpha = 0.45, linetype = "solid") +
  geom_point(size = 1.45, alpha = 0.85) +
  geom_text(data = ex_f,
            aes(x = ent + dx, y = pmax(0.02, h_tree_bits + dy), label = short, hjust = hj),
            colour = ex_f$txt_col, size = 7.0 / .pt, family = "Helvetica",
            inherit.aes = FALSE) +
  facet_wrap(~fam_panel, ncol = 2) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_fill_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), name = NULL) +
  scale_linetype_manual(values = c("Verbs" = "solid", "Nouns" = "dashed"), name = NULL) +
  scale_x_continuous(breaks = c(0.5, 1.0, 1.5)) +
  scale_y_sqrt(breaks = c(0, 0.25, 1, 2, 4, 6), labels = c("0", "0.25", "1", "2", "4", "6")) +
  coord_cartesian(xlim = c(0.25, 1.85), ylim = c(0, 7.6)) +
  labs(
    title = "b   Within-family trajectories (4 lineages)",
    subtitle = expression("MGL " * italic(H)[tree] * " (± SE, sqrt scale) vs. entropy (OLS lines)"),
    x = "Mean individual ancestry entropy (nats)",
    y = expression("MGL tree entropy " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.5),
    panel.spacing = unit(5, "pt"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 7.4, colour = INK_BODY),
    legend.margin = margin(t = -2, b = 0)
  )

fig4 <- (pA | pB) + plot_layout(widths = c(0.82, 1.35))
save_track2_fig(fig4, fig_dir, "fig4_mgl_family_slopes", width = 6.5, height = 4.6, dpi = 300)
