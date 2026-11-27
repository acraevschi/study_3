# Figure 1: Sample Geography & Paradigm Complexity Landscape (Track 2 MGL)
# Outputs: fig1_mgl_map_and_complexity.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(maps)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

ld_v <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_V.csv"), stringsAsFactors = FALSE)
ld_n <- read.csv(file.path(root_dir, "analyses/paradigm/results/stage2/prepared/language_data_N.csv"), stringsAsFactors = FALSE)
us   <- read.csv(file.path(root_dir, "analyses/paradigm/data/unit_summary.csv"), stringsAsFactors = FALSE)
u120 <- read.csv(file.path(root_dir, "analyses/paradigm/results/mgl/unit_outcomes_uniform120.csv"), stringsAsFactors = FALSE)

# --- Panel A: Map of 56 GeLaTo languages ---
all_meta <- unique(rbind(
  ld_v[, c("glottocode", "name", "family", "lat", "lon", "ent")],
  ld_n[, c("glottocode", "name", "family", "lat", "lon", "ent")]
))
stopifnot(nrow(all_meta) == 56L, !anyDuplicated(all_meta$glottocode))

in_v <- all_meta$glottocode %in% ld_v$glottocode
in_n <- all_meta$glottocode %in% ld_n$glottocode
all_meta$coverage <- ifelse(in_v & in_n, "Verbs & Nouns (38)",
                     ifelse(in_v, "Verbs only (7)", "Nouns only (11)"))
all_meta$coverage <- factor(all_meta$coverage,
                            levels = c("Verbs & Nouns (38)", "Nouns only (11)", "Verbs only (7)"))

world_df <- subset(map_data("world"), region != "Antarctica")

pA <- ggplot() +
  geom_polygon(data = world_df, aes(long, lat, group = group),
               fill = MAP_LAND, colour = "white", linewidth = 0.15) +
  geom_point(data = all_meta[order(all_meta$ent), ],
             aes(lon, lat, fill = ent, shape = coverage),
             size = 2.35, colour = "white", stroke = 0.35) +
  scale_fill_gradientn(colours = SEQ_ENTROPY,
                       name = "Ancestry entropy\n(nats)",
                       guide = guide_colorbar(barheight = unit(2.5, "lines"),
                                              barwidth = unit(0.45, "lines"),
                                              order = 2)) +
  scale_shape_manual(values = c("Verbs & Nouns (38)" = 21,
                                "Nouns only (11)" = 22,
                                "Verbs only (7)" = 23),
                     name = "POS coverage",
                     guide = guide_legend(
                       order = 1,
                       override.aes = list(fill = INK_BODY, colour = "white", size = 2.4, stroke = 0.35)
                     )) +
  coord_quickmap(xlim = c(-105, 155), ylim = c(-40, 75), expand = FALSE) +
  labs(
    title = sprintf("a   %d GeLaTo languages (%d orthographic units, %d families)",
                    nrow(all_meta), sum(u120$representation == "orth"), length(unique(all_meta$family)))
  ) +
  theme_track2() +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid.major = element_blank(),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(4, "pt"),
    legend.key.height = unit(9, "pt"),
    legend.title = element_text(size = 7.4, colour = INK_BODY, face = "bold"),
    legend.text = element_text(size = 7.1, colour = INK2),
    plot.margin = margin(4, 4, 2, 4)
  )

# --- Panel B: Distinct paradigm size vs. MGL H_tree (94 orth units) ---
orth <- merge(subset(u120, representation == "orth"),
              us[, c("unit_id", "language_name", "family_name")],
              by = "unit_id")
stopifnot(nrow(orth) == 94L)
orth$pos_lbl <- ifelse(orth$pos == "V", "Verbs (45)", "Nouns (49)")
orth$pos_lbl <- factor(orth$pos_lbl, levels = c("Verbs (45)", "Nouns (49)"))
orth$txt_col <- ifelse(orth$pos == "V", COL_VERB, COL_NOUN_TXT)

# Named lookup for exemplar label positions (explicit lx, ly so zero overlap or clipping)
ex_specs <- data.frame(
  unit_id = c("moks1248_V_orth", "taga1270_V_orth", "mode1248_N_orth",
              "gulf1241_V_orth", "iris1253_N_orth", "stan1293_V_orth",
              "stan1288_V_orth", "finn1318_N_orth", "nucl1301_V_orth",
              "cusc1236_N_orth"),
  short_lab = c("Moksha (V)", "Tagalog (V)", "Modern Greek (N)",
                "Gulf Arabic (V)", "Irish (N)", "English (V)",
                "Spanish (V)", "Finnish (N)", "Turkish (V)",
                "Cusco Quechua (N)"),
  lx    = c( 4.2,  9.5,  5.2, 24.5,  6.8,  8.8,  78.0, 130.0, 210.0, 245.0),
  ly    = c( 7.05, 5.55, 5.65, 4.45, 4.10, 2.80,  1.35,  2.75,  3.60,  2.00),
  hjust = c( 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,   1.0,   1.0,   1.0,   1.0),
  stringsAsFactors = FALSE
)
ex_df <- merge(orth, ex_specs, by = "unit_id")
stopifnot(nrow(ex_df) == nrow(ex_specs))

pB <- ggplot(orth, aes(x = n_cells_distinct, y = h_tree_bits, colour = pos_lbl, shape = pos_lbl)) +
  geom_errorbar(aes(ymin = pmax(0, h_tree_bits - h_tree_se), ymax = h_tree_bits + h_tree_se),
                width = 0, linewidth = 0.35, alpha = 0.45) +
  geom_point(size = 1.60, alpha = 0.88) +
  geom_segment(data = ex_df, aes(x = n_cells_distinct, y = h_tree_bits, xend = lx, yend = ly),
               colour = GREY_LT, linewidth = 0.30, inherit.aes = FALSE) +
  geom_text(data = ex_df, aes(x = lx, y = ly, label = short_lab, hjust = hjust),
            colour = ex_df$txt_col, size = 7.0 / .pt, family = "Helvetica",
            inherit.aes = FALSE) +
  scale_x_log10(breaks = c(2, 5, 10, 20, 50, 100, 250)) +
  scale_y_continuous(breaks = c(0, 2, 4, 6, 8)) +
  coord_cartesian(xlim = c(1.75, 300), ylim = c(-0.2, 8.1)) +
  scale_colour_manual(values = c("Verbs (45)" = COL_VERB, "Nouns (49)" = COL_NOUN), name = NULL) +
  scale_shape_manual(values = c("Verbs (45)" = 16, "Nouns (49)" = 17), name = NULL) +
  labs(
    title = "b   Paradigm size vs. conditional entropy",
    x = "Distinct synthetic cells (log scale)",
    y = expression("MGL tree entropy " * italic(H)[tree] * " (bits ± SE)")
  ) +
  theme_track2() +
  theme(
    legend.position = c(0.80, 0.88),
    legend.background = element_rect(fill = "white", colour = NA),
    legend.key.height = unit(9, "pt"),
    legend.text = element_text(size = 7.2, colour = INK_BODY)
  )

# --- Panel C: PCFP Entropy Drop (H_k1 -> H_k3 -> H_tree) ---
traj <- rbind(
  data.frame(unit_id = orth$unit_id, pos_lbl = orth$pos_lbl, stage = "k = 1", bits = orth$h_k1_bits),
  data.frame(unit_id = orth$unit_id, pos_lbl = orth$pos_lbl, stage = "k = 3", bits = orth$h_k3_bits),
  data.frame(unit_id = orth$unit_id, pos_lbl = orth$pos_lbl, stage = "Tree",  bits = orth$h_tree_bits)
)
traj$stage <- factor(traj$stage, levels = c("k = 1", "k = 3", "Tree"))

sum_pos <- aggregate(bits ~ pos_lbl + stage, data = traj,
                     FUN = function(z) c(med = median(z), q25 = quantile(z, 0.25), q75 = quantile(z, 0.75)))
sum_df <- data.frame(
  pos_lbl = sum_pos$pos_lbl,
  stage = sum_pos$stage,
  med = sum_pos$bits[, "med"],
  q25 = sum_pos$bits[, "q25.25%"],
  q75 = sum_pos$bits[, "q75.75%"]
)
all_med <- aggregate(bits ~ stage, data = traj, FUN = median)

end_labs <- subset(sum_df, stage == "Tree")
end_labs$lab <- sprintf("%.2f", end_labs$med)
end_labs$dy <- ifelse(end_labs$pos_lbl == "Verbs (45)", -0.35, 0.42)
end_labs$col <- ifelse(end_labs$pos_lbl == "Verbs (45)", COL_VERB, COL_NOUN_TXT)

pC <- ggplot() +
  geom_line(data = traj, aes(x = stage, y = bits, group = unit_id),
            colour = GREY_LT, alpha = 0.22, linewidth = 0.28) +
  geom_errorbar(data = sum_df,
                aes(x = stage, ymin = q25, ymax = q75, colour = pos_lbl),
                width = 0, linewidth = 1.6, alpha = 0.35,
                position = position_dodge(width = 0.16)) +
  geom_line(data = sum_df,
            aes(x = stage, y = med, group = pos_lbl, colour = pos_lbl, linetype = pos_lbl),
            linewidth = 0.9, position = position_dodge(width = 0.16)) +
  geom_point(data = sum_df,
             aes(x = stage, y = med, colour = pos_lbl, shape = pos_lbl),
             size = 2.1, position = position_dodge(width = 0.16)) +
  geom_text(data = end_labs,
            aes(x = stage, y = med + dy, label = lab),
            colour = end_labs$col, hjust = -0.18, size = 7.2 / .pt, fontface = "bold",
            family = "Helvetica") +
  annotate("text", x = 1.0, y = 7.9, hjust = 0, vjust = 1,
           label = sprintf("Median (94 units):\nk=1: %.2f | k=3: %.2f\nTree: %.2f bits",
                           all_med$bits[1], all_med$bits[2], all_med$bits[3]),
           size = 7.0 / .pt, colour = INK_BODY, lineheight = 1.12, family = "Helvetica") +
  scale_colour_manual(values = c("Verbs (45)" = COL_VERB, "Nouns (49)" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs (45)" = 16, "Nouns (49)" = 17), guide = "none") +
  scale_linetype_manual(values = c("Verbs (45)" = "solid", "Nouns (49)" = "dashed"), guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.28, 0.58))) +
  scale_y_continuous(breaks = c(0, 2, 4, 6, 8)) +
  coord_cartesian(ylim = c(-0.2, 8.1)) +
  labs(
    title = "c   Low conditional entropy",
    x = "Predictor context",
    y = NULL
  ) +
  theme_track2() +
  theme(axis.text.y = element_blank(), panel.grid.major.x = element_blank())

fig1 <- pA / wrap_elements(full = (pB + pC + plot_layout(widths = c(1.78, 1.0)))) +
  plot_layout(heights = c(0.95, 1.18))

save_track2_fig(fig1, fig_dir, "fig1_mgl_map_and_complexity", width = 6.5, height = 5.4, dpi = 300)
