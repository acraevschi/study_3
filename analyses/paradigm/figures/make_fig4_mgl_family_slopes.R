# Figure 4: Within-Family Trajectories (Cross-Family f3 vs. MGL H_tree)
# Outputs: fig4_mgl_family_slopes.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

s2_dir   <- file.path(root_dir, "analyses/paradigm/results/stage2")
prep_dir <- file.path(s2_dir, "prepared")

ld_v <- read.csv(file.path(prep_dir, "language_data_V.csv"), stringsAsFactors = FALSE)
ld_n <- read.csv(file.path(prep_dir, "language_data_N.csv"), stringsAsFactors = FALSE)
ex_v <- read.csv(file.path(s2_dir, "exposure_main_V.csv"), stringsAsFactors = FALSE)
ex_n <- read.csv(file.path(s2_dir, "exposure_main_N.csv"), stringsAsFactors = FALSE)
s2_v <- read.csv(file.path(s2_dir, "stage2_results_V.csv"), stringsAsFactors = FALSE)
s2_n <- read.csv(file.path(s2_dir, "stage2_results_N.csv"), stringsAsFactors = FALSE)

# --- Panel A: Maximal, Mundlak Within/Between, Drop Inner Asia, Strict Links, Cutoff n >= 5 ---
extract_chk <- function(df_p, pos_lbl) {
  r_max <- subset(df_p, spec_group == "maximal"       & measure == "f3_fixed" & min_n_ind == 3L)
  r_mun <- subset(df_p, spec_group == "mundlak_wb"    & measure == "f3_fixed" & min_n_ind == 3L)
  r_nia <- subset(df_p, spec_group == "no_inner_asia" & measure == "f3_fixed" & min_n_ind == 3L)
  r_str <- subset(df_p, spec_group == "strict_links"  & measure == "f3_fixed" & min_n_ind == 3L)
  r_n5  <- subset(df_p, spec_group == "cutoff_n5"     & measure == "f3_fixed" & min_n_ind == 5L)
  stopifnot(nrow(r_max) == 1L, nrow(r_mun) == 1L, nrow(r_nia) == 1L, nrow(r_str) == 1L, nrow(r_n5) == 1L)

  mk <- function(lbl, row, pfx = "b") {
    q80 <- get_exact_q80(row, pfx)
    data.frame(
      pos = pos_lbl, label = lbl,
      mean = row[[paste0(pfx, "_mean")]],
      lo95 = row[[paste0(pfx, "_q025")]],
      hi95 = row[[paste0(pfx, "_q975")]],
      lo80 = q80$lo80, hi80 = q80$hi80,
      stringsAsFactors = FALSE
    )
  }
  rbind(
    mk("Maximal (Phylo + GP)",      r_max, "b"),
    mk("Within-family (Mundlak)",   r_mun, "b"),
    mk("Between-family (Mundlak)",  r_mun, "b_between"),
    mk("Drop Inner Asia",           r_nia, "b"),
    mk("Strict links",              r_str, "b"),
    mk("Cutoff n >= 5",             r_n5,  "b")
  )
}

dfA <- rbind(extract_chk(s2_v, "Verbs"), extract_chk(s2_n, "Nouns"))
lbl_levels <- c("Maximal (Phylo + GP)", "Within-family (Mundlak)", "Between-family (Mundlak)",
                "Drop Inner Asia", "Strict links", "Cutoff n >= 5")
dfA$label <- factor(dfA$label, levels = rev(lbl_levels))
dfA$pos   <- factor(dfA$pos, levels = c("Nouns", "Verbs"))

xlim_a <- range(c(dfA$lo95, dfA$hi95, -0.55, 0.55), na.rm = TRUE)
xlim_a <- xlim_a + c(-0.06, 0.06) * diff(xlim_a)

pA <- ggplot(dfA, aes(x = mean, y = label, colour = pos, shape = pos)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 0.48, alpha = 0.55) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y",
                position = position_dodge(width = 0.52), linewidth = 1.35, alpha = 1.0) +
  geom_point(position = position_dodge(width = 0.52), size = 1.75) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(breaks = c(-0.5, 0.0, 0.5)) +
  coord_cartesian(xlim = xlim_a) +
  labs(
    title = "a   Decomposition & sensitivity",
    subtitle = "Cross-family f3 slopes (80% & 95% CrI; \u00b10.15 ROPE)",
    x = "Slope per SD of cross-family f3",
    y = NULL
  ) +
  theme_track2() +
  theme(
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    axis.text.y = element_text(colour = INK_BODY, size = 7.4),
    legend.position = "none"
  )

# --- Panel B: Small Multiples Across Multi-Language Families (f3_fixed from exposure_main) ---
fams <- c("Indo-European", "Uralic", "Turkic", "Afro-Asiatic")

prep_fam <- function(ld_df, ex_df, pos_lbl) {
  ef <- subset(ex_df, set == "main" & measure == "f3_fixed")
  if ("min_n_ind" %in% names(ef)) ef <- subset(ef, min_n_ind == 3L)
  ef$f3_raw <- pmax(0, ef$eta_mean * ef$y_sd + ef$y_mean) * 1e3
  m <- merge(subset(ld_df, family %in% fams), ef[, c("glottocode", "f3_raw")], by = "glottocode")
  m$pos <- pos_lbl
  m[, c("glottocode", "name", "family", "pos", "f3_raw", "h_tree_bits", "h_tree_se")]
}
dfB <- rbind(prep_fam(ld_v, ex_v, "Verbs"), prep_fam(ld_n, ex_n, "Nouns"))

nv <- table(factor(subset(dfB, pos == "Verbs")$family, levels = fams))
nn <- table(factor(subset(dfB, pos == "Nouns")$family, levels = fams))
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
               (glottocode == "amha1245" & pos == "Verbs") |
               (glottocode == "nucl1301" & pos == "Verbs"))
ex_f$short <- c(
  "moks1248" = "Moksha",
  "mode1248" = "Mod. Greek",
  "iris1253" = "Irish",
  "gulf1241" = "Gulf Arabic",
  "amha1245" = "Amharic",
  "nucl1301" = "Turkish"
)[ex_f$glottocode]
ex_f$txt_col <- ifelse(ex_f$pos == "Verbs", COL_VERB, COL_NOUN_TXT)

# Dynamic offsets for sqrt-transformed x and y axes
ex_f$dx <- c("moks1248" =  0.10, "mode1248" =  0.10, "iris1253" =  0.12,
             "gulf1241" =  0.22, "amha1245" =  0.85, "nucl1301" =  0.12)[ex_f$glottocode]
ex_f$dy <- c("moks1248" = -0.45, "mode1248" =  0.45, "iris1253" =  0.22,
             "gulf1241" =  0.30, "amha1245" =  1.25, "nucl1301" =  0.48)[ex_f$glottocode]
ex_f$hj <- c("moks1248" =  0.00, "mode1248" =  0.00, "iris1253" =  0.00,
             "gulf1241" =  0.00, "amha1245" =  1.00, "nucl1301" =  0.00)[ex_f$glottocode]

dfB_ribbon <- subset(dfB, family %in% c("Indo-European", "Uralic"))
xmax_b <- max(dfB$f3_raw, na.rm = TRUE) * 1.15

pB <- ggplot(dfB, aes(x = f3_raw, y = h_tree_bits, colour = pos, fill = pos, shape = pos, linetype = pos)) +
  geom_smooth(data = dfB_ribbon, method = "lm", formula = y ~ x, se = TRUE, alpha = 0.14, linewidth = 0.72) +
  geom_smooth(data = subset(dfB, !(family %in% c("Indo-European", "Uralic"))),
              method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.72) +
  geom_errorbar(aes(ymin = pmax(0, h_tree_bits - h_tree_se), ymax = h_tree_bits + h_tree_se),
                width = 0, linewidth = 0.35, alpha = 0.45, linetype = "solid") +
  geom_point(size = 1.45, alpha = 0.85) +
  geom_text(data = ex_f,
            aes(x = pmax(0.02, f3_raw + dx), y = pmax(0.02, h_tree_bits + dy), label = short, hjust = hj),
            colour = ex_f$txt_col, size = 7.0 / .pt, family = "Helvetica",
            inherit.aes = FALSE) +
  facet_wrap(~fam_panel, ncol = 2) +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_fill_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), name = NULL) +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), name = NULL) +
  scale_linetype_manual(values = c("Verbs" = "solid", "Nouns" = "dashed"), name = NULL) +
  scale_x_sqrt(breaks = c(0, 0.5, 2, 5), labels = c("0", "0.5", "2", "5")) +
  scale_y_sqrt(breaks = c(0, 0.25, 1, 2, 4, 6), labels = c("0", "0.25", "1", "2", "4", "6")) +
  coord_cartesian(xlim = c(0, max(7.6, xmax_b)), ylim = c(0, 7.6)) +
  labs(
    title = "b   Within-family trajectories (4 lineages)",
    subtitle = expression("MGL " * italic(H)[tree] * " (\u00b1 SE, sqrt) vs. cross-family " * italic(f)[3] * " (OLS)"),
    x = expression("Language-level cross-family " * italic(f)[3] * " (" %*% 10^-3 * ", sqrt scale)"),
    y = expression("MGL tree entropy " * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.clip = "off",
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.1),
    panel.spacing = unit(7, "pt"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 7.4, colour = INK_BODY),
    legend.margin = margin(t = -2, b = 0)
  )

fig4 <- (pA | pB) + plot_layout(widths = c(0.86, 1.32))
save_track2_fig(fig4, fig_dir, "fig4_mgl_family_slopes", width = 6.5, height = 4.65, dpi = 300)
