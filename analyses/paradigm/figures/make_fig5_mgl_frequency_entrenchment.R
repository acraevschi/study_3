# Figure 5: Lexical Token-Frequency Entrenchment vs. Null Contact Erosion (48,034 Lemmas)
# Extracts exact posterior draws from saved entrench_{V,N}__all_lemmas__ent.rds brmsfit objects.
# Outputs: fig5_mgl_frequency_entrenchment.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(brms)
  library(posterior)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/paradigm/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

s2_dir <- file.path(root_dir, "analyses/paradigm/results/stage2")
prep_dir <- file.path(s2_dir, "prepared")

ent_v <- read.csv(file.path(s2_dir, "entrenchment_results_V.csv"), stringsAsFactors = FALSE)
ent_n <- read.csv(file.path(s2_dir, "entrenchment_results_N.csv"), stringsAsFactors = FALSE)
sl_v  <- read.csv(file.path(s2_dir, "language_entrenchment_slopes_V.csv"), stringsAsFactors = FALSE)
sl_n  <- read.csv(file.path(s2_dir, "language_entrenchment_slopes_N.csv"), stringsAsFactors = FALSE)
ld_v  <- read.csv(file.path(prep_dir, "language_data_V.csv"), stringsAsFactors = FALSE)
ld_n  <- read.csv(file.path(prep_dir, "language_data_N.csv"), stringsAsFactors = FALSE)
lf_v  <- read.csv(file.path(prep_dir, "lemma_freq_data_V.csv"), stringsAsFactors = FALSE)
lf_n  <- read.csv(file.path(prep_dir, "lemma_freq_data_N.csv"), stringsAsFactors = FALSE)

fit_v <- readRDS(file.path(s2_dir, "fits/entrench_V__all_lemmas__ent.rds"))
fit_n <- readRDS(file.path(s2_dir, "fits/entrench_N__all_lemmas__ent.rds"))
dr_v <- as_draws_df(fit_v)
dr_n <- as_draws_df(fit_n)

# --- Panel A: Language-Specific Entrenchment Slopes + Hyper-Posterior ---
extract_lang_slopes <- function(sl_df, ld_df, dr, pos_lbl) {
  m <- merge(sl_df, ld_df[, c("glottocode", "name")], by = "glottocode")
  res <- lapply(seq_len(nrow(m)), function(i) {
    gc <- m$glottocode[i]
    rc <- sprintf("r_lang_id[%s,z_freq_within]", gc)
    draws <- dr$b_z_freq_within + dr[["b_x_adm:z_freq_within"]] * m$x_adm[i] + dr[[rc]]
    data.frame(
      pos = pos_lbl,
      glottocode = gc,
      name = m$name[i],
      mean = mean(draws),
      lo95 = unname(quantile(draws, 0.025)),
      lo80 = unname(quantile(draws, 0.10)),
      hi80 = unname(quantile(draws, 0.90)),
      hi95 = unname(quantile(draws, 0.975)),
      is_pop = FALSE,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, res)
  out <- out[order(out$mean), ]
  out$rank <- seq_len(nrow(out))

  # Population hyper-posterior (placed at rank = -2.5 below the language caterpillar)
  pop_draws <- dr$b_z_freq_within
  pop_row <- data.frame(
    pos = pos_lbl, glottocode = "POP", name = "Population mean",
    mean = mean(pop_draws),
    lo95 = unname(quantile(pop_draws, 0.025)),
    lo80 = unname(quantile(pop_draws, 0.10)),
    hi80 = unname(quantile(pop_draws, 0.90)),
    hi95 = unname(quantile(pop_draws, 0.975)),
    is_pop = TRUE, rank = -3.0, stringsAsFactors = FALSE
  )
  rbind(pop_row, out)
}

dfA_v <- extract_lang_slopes(sl_v, ld_v, dr_v, "Verbs (44 langs, 21,078 lemmas)")
dfA_n <- extract_lang_slopes(sl_n, ld_n, dr_n, "Nouns (46 langs, 26,956 lemmas)")
dfA <- rbind(dfA_v, dfA_n)
posA_levels <- c("Verbs (44 langs, 21,078 lemmas)", "Nouns (46 langs, 26,956 lemmas)")
dfA$pos <- factor(dfA$pos, levels = posA_levels)
dfA$pos_short <- ifelse(grepl("^Verbs", dfA$pos), "Verbs", "Nouns")

# Summary annotations & salient landmark labels on Panel A
ev_row <- subset(ent_v, sample == "all_lemmas" & measure == "ent")
en_row <- subset(ent_n, sample == "all_lemmas" & measure == "ent")
bf10_v <- 1 / ev_row$b_freq_bf01_norm

annA <- data.frame(
  pos = factor(posA_levels, levels = posA_levels),
  x = c(0.46, 0.46),
  y = c(1.0, 1.0),
  lab = c(
    sprintf("Pop: +%.3f [%.3f, %.3f] SD\nBF10 = %.0f; %d/44 > 0",
            ev_row$b_freq_mean, ev_row$b_freq_q025, ev_row$b_freq_q975,
            bf10_v, sum(sl_v$slope_freq_mean > 0)),
    sprintf("Pop: +%.3f [%.3f, %.3f] SD\nP(>0) = %.4f; %d/46 > 0",
            en_row$b_freq_mean, en_row$b_freq_q025, en_row$b_freq_q975,
            en_row$b_freq_p_pos, sum(sl_n$slope_freq_mean > 0))
  ),
  stringsAsFactors = FALSE
)

pop_labels <- subset(dfA, is_pop)
land_gcs <- c("stan1293", "russ1263", "finn1318", "stan1290", "mode1248")
land_df <- subset(dfA, glottocode %in% land_gcs)
land_df$short <- c(
  "stan1293" = "English", "russ1263" = "Russian",
  "finn1318" = "Finnish", "stan1290" = "French", "mode1248" = "Mod. Greek"
)[land_df$glottocode]

pA <- ggplot(dfA, aes(x = mean, y = rank, colour = pos_short)) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_hline(yintercept = -0.5, colour = GRID_WARM, linewidth = 0.45) +
  geom_errorbar(data = subset(dfA, !is_pop), aes(xmin = lo95, xmax = hi95),
                width = 0, orientation = "y", linewidth = 0.32, alpha = 0.42) +
  geom_errorbar(data = subset(dfA, !is_pop), aes(xmin = lo80, xmax = hi80),
                width = 0, orientation = "y", linewidth = 0.85, alpha = 0.85) +
  geom_point(data = subset(dfA, !is_pop), aes(shape = pos_short), size = 1.0) +
  # Highlighted population hyper-posterior row at bottom
  geom_errorbar(data = pop_labels, aes(xmin = lo95, xmax = hi95),
                width = 0, orientation = "y", linewidth = 0.70, alpha = 0.65) +
  geom_errorbar(data = pop_labels, aes(xmin = lo80, xmax = hi80),
                width = 0, orientation = "y", linewidth = 1.90, alpha = 1.0) +
  geom_point(data = pop_labels, aes(shape = pos_short), size = 2.2) +
  geom_text(data = pop_labels, aes(x = -0.22, y = rank, label = "Pop. mean"),
            hjust = 0, vjust = 0.5, size = 7.0 / .pt, fontface = "bold",
            colour = INK_BODY, family = "Helvetica") +
  geom_text(data = land_df,
            aes(x = ifelse((pos_short == "Verbs" & glottocode %in% c("mode1248", "stan1290", "stan1293")) |
                           (pos_short == "Nouns" & glottocode %in% c("stan1290")),
                           lo95 - 0.015, hi95 + 0.015),
                y = rank, label = short,
                hjust = ifelse((pos_short == "Verbs" & glottocode %in% c("mode1248", "stan1290", "stan1293")) |
                               (pos_short == "Nouns" & glottocode %in% c("stan1290")), 1, 0)),
            size = 7.0 / .pt, colour = INK2, family = "Helvetica") +
  geom_text(data = annA, aes(x = x, y = y, label = lab),
            inherit.aes = FALSE, hjust = 1, vjust = 0,
            size = 7.0 / .pt, lineheight = 1.08, colour = INK_BODY, family = "Helvetica") +
  facet_wrap(~pos, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_y_continuous(breaks = NULL) +
  scale_x_continuous(breaks = c(-0.2, 0.0, 0.2, 0.4)) +
  coord_cartesian(xlim = c(-0.24, 0.48)) +
  labs(
    title = "a   Language-level frequency slopes",
    x = "Frequency slope on entropy (SD / SD)",
    y = "Languages, sorted by slope"
  ) +
  theme_track2() +
  theme(
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.6),
    panel.grid.major.y = element_blank(),
    panel.spacing = unit(6, "pt")
  )

# --- Panel B1: Within-Language Frequency Quantiles Stratified by Low vs. High Admixture ---
prep_quant_fit <- function(lf_df, sl_df, ent_row, dr, pos_lbl) {
  m <- merge(lf_df, sl_df[, c("glottocode", "x_adm")], by = "glottocode")
  y_sd <- ent_row$y_bits_sd
  med_adm <- median(sl_df$x_adm)
  m$adm_grp <- ifelse(m$x_adm >= med_adm, "High admixture", "Low admixture")
  # Within-language centered H_tree in bits
  lang_means <- ave(m$h_tree_lemma, m$glottocode, FUN = mean)
  m$dh_bits <- m$h_tree_lemma - lang_means
  qs <- quantile(m$z_freq_within, probs = seq(0, 1, length.out = 6))
  m$fbin <- cut(m$z_freq_within, breaks = qs, include.lowest = TRUE, labels = FALSE)

  pts <- aggregate(cbind(z_freq_within, dh_bits) ~ adm_grp + fbin, data = m, FUN = mean)
  ses <- aggregate(dh_bits ~ adm_grp + fbin, data = m,
                   FUN = function(v) sd(v) / sqrt(length(v)))
  pts$se <- ses$dh_bits
  pts$pos <- pos_lbl

  # Model-predicted slopes in bits (b_freq + b_int * x_adm_mean_grp) * z * y_sd
  x_grid <- seq(min(pts$z_freq_within) - 0.15, max(pts$z_freq_within) + 0.15, length.out = 40)
  curves <- do.call(rbind, lapply(c("Low admixture", "High admixture"), function(g) {
    xm <- mean(sl_df$x_adm[ifelse(sl_df$x_adm >= med_adm, "High admixture", "Low admixture") == g])
    sl_draws <- (dr$b_z_freq_within + dr[["b_x_adm:z_freq_within"]] * xm) * y_sd
    do.call(rbind, lapply(x_grid, function(zv) {
      pred <- sl_draws * zv
      data.frame(
        pos = pos_lbl, adm_grp = g, z = zv,
        mean = mean(pred),
        lo95 = unname(quantile(pred, 0.025)),
        hi95 = unname(quantile(pred, 0.975)),
        stringsAsFactors = FALSE
      )
    }))
  }))
  list(pts = pts, curves = curves)
}

qf_v <- prep_quant_fit(lf_v, sl_v, subset(ent_v, sample == "all_lemmas" & measure == "ent"), dr_v, "Verbs")
qf_n <- prep_quant_fit(lf_n, sl_n, subset(ent_n, sample == "all_lemmas" & measure == "ent"), dr_n, "Nouns")

pts_all <- rbind(qf_v$pts, qf_n$pts)
cur_all <- rbind(qf_v$curves, qf_n$curves)
pts_all$pos <- factor(pts_all$pos, levels = c("Verbs", "Nouns"))
cur_all$pos <- factor(cur_all$pos, levels = c("Verbs", "Nouns"))
grp_levels <- c("Low admixture", "High admixture")
pts_all$adm_grp <- factor(pts_all$adm_grp, levels = grp_levels)
cur_all$adm_grp <- factor(cur_all$adm_grp, levels = grp_levels)

pB1 <- ggplot() +
  geom_hline(yintercept = 0, colour = GRID_WARM, linewidth = 0.4) +
  geom_ribbon(data = cur_all, aes(x = z, ymin = lo95, ymax = hi95, fill = adm_grp),
              alpha = 0.15, colour = NA) +
  geom_line(data = cur_all, aes(x = z, y = mean, colour = adm_grp, linetype = adm_grp),
            linewidth = 0.78) +
  geom_errorbar(data = pts_all,
                aes(x = z_freq_within, ymin = dh_bits - se, ymax = dh_bits + se, colour = adm_grp),
                width = 0, linewidth = 0.4, alpha = 0.75) +
  geom_point(data = pts_all,
             aes(x = z_freq_within, y = dh_bits, colour = adm_grp, shape = adm_grp),
             size = 1.7, fill = "white") +
  facet_wrap(~pos, ncol = 1) +
  scale_colour_manual(values = c("Low admixture" = COL_ADM_LOW, "High admixture" = COL_ADM_HIGH), name = NULL) +
  scale_fill_manual(values = c("Low admixture" = COL_ADM_LOW, "High admixture" = COL_ADM_HIGH), name = NULL) +
  scale_shape_manual(values = c("Low admixture" = 21, "High admixture" = 16), name = NULL) +
  scale_linetype_manual(values = c("Low admixture" = "dashed", "High admixture" = "solid"), name = NULL) +
  scale_x_continuous(breaks = c(-1, 0, 1, 2)) +
  scale_y_continuous(breaks = c(-0.2, 0.0, 0.2, 0.4)) +
  coord_cartesian(ylim = c(-0.30, 0.42)) +
  labs(
    title = "b   Parallel entrenchment (empirical ± SE & model)",
    x = "Within-language token frequency (SD)",
    y = expression("Centered " * Delta * italic(H)[tree] * " (bits)")
  ) +
  theme_track2() +
  theme(
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.6),
    panel.spacing = unit(5, "pt"),
    legend.position = "top",
    legend.direction = "horizontal",
    legend.text = element_text(size = 7.2, colour = INK_BODY),
    legend.margin = margin(t = 0, b = -3)
  )

# --- Panel B2: Admixture x Frequency Interaction Posterior inside ROPE ---
mk_dens_df <- function(draws, lbl) {
  d <- density(draws, n = 512)
  df <- data.frame(pos = lbl, x = d$x, y = d$y / max(d$y), stringsAsFactors = FALSE)
  sm <- data.frame(
    pos = lbl,
    mean = mean(draws),
    lo95 = unname(quantile(draws, 0.025)),
    lo80 = unname(quantile(draws, 0.10)),
    hi80 = unname(quantile(draws, 0.90)),
    hi95 = unname(quantile(draws, 0.975)),
    stringsAsFactors = FALSE
  )
  list(dens = df, sm = sm)
}
iv <- mk_dens_df(dr_v[["b_x_adm:z_freq_within"]], "Verbs")
in_ <- mk_dens_df(dr_n[["b_x_adm:z_freq_within"]], "Nouns")

dens_df <- rbind(iv$dens, in_$dens)
sm_df   <- rbind(iv$sm, in_$sm)
dens_df$pos <- factor(dens_df$pos, levels = c("Nouns", "Verbs"))
sm_df$pos   <- factor(sm_df$pos, levels = c("Nouns", "Verbs"))

sm_df$y_off <- ifelse(sm_df$pos == "Verbs", 1.25, 0.0)
dens_df$y0  <- ifelse(dens_df$pos == "Verbs", 1.25, 0.0)
dens_df$y1  <- dens_df$y0 + dens_df$y * 0.88

sm_df$ann_l <- c(
  "Verbs" = sprintf("Verbs: BF01 = %.1f", ev_row$b_int_bf01_norm),
  "Nouns" = sprintf("Nouns: BF01 = %.1f", en_row$b_int_bf01_norm)
)[as.character(sm_df$pos)]
sm_df$ann_r <- "100% in ROPE"

pB2 <- ggplot() +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -0.12, ymax = 2.60,
           fill = ROPE_FILL, alpha = 0.75) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_ribbon(data = dens_df, aes(x = x, ymin = y0, ymax = y1, fill = pos),
              alpha = 0.25, colour = NA) +
  geom_line(data = dens_df, aes(x = x, y = y1, colour = pos), linewidth = 0.65) +
  geom_errorbar(data = sm_df, aes(xmin = lo95, xmax = hi95, y = y_off, colour = pos),
                width = 0, orientation = "y", linewidth = 0.55, alpha = 0.6) +
  geom_errorbar(data = sm_df, aes(xmin = lo80, xmax = hi80, y = y_off, colour = pos),
                width = 0, orientation = "y", linewidth = 1.5) +
  geom_point(data = sm_df, aes(x = mean, y = y_off, colour = pos, shape = pos), size = 1.8) +
  geom_text(data = sm_df, aes(x = -0.012, y = y_off + 0.98, label = ann_l),
            hjust = 1, vjust = 0, size = 7.0 / .pt, colour = INK_BODY, family = "Helvetica") +
  geom_text(data = sm_df, aes(x = 0.012, y = y_off + 0.98, label = ann_r),
            hjust = 0, vjust = 0, size = 7.0 / .pt, colour = INK2, family = "Helvetica") +
  scale_colour_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_fill_manual(values = c("Verbs" = COL_VERB, "Nouns" = COL_NOUN), guide = "none") +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_x_continuous(breaks = c(-0.15, 0.0, 0.15), labels = c("-0.15", "0", "+0.15")) +
  scale_y_continuous(breaks = NULL) +
  coord_cartesian(xlim = c(-0.17, 0.17), ylim = c(-0.12, 2.58)) +
  labs(
    title = "c   Admixture × Frequency (ROPE ±0.15)",
    x = expression("Interaction " * italic(b)[int] * " (SD)"),
    y = NULL
  ) +
  theme_track2() +
  theme(panel.grid.major.y = element_blank())

fig5 <- (pA | (pB1 / pB2 + plot_layout(heights = c(1.55, 0.95)))) +
  plot_layout(widths = c(1.12, 1.0))
save_track2_fig(fig5, fig_dir, "fig5_mgl_frequency_entrenchment", width = 6.5, height = 5.6, dpi = 300)
