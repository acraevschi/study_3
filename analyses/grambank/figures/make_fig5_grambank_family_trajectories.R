# Figure 5 (Track 1): Within-Family Decomposition & Multi-Language Family Trajectories
# Outputs: fig5_grambank_family_trajectories.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/grambank/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

ld   <- read.csv(file.path(root_dir, "analyses/grambank/data/language_data.csv"), stringsAsFactors = FALSE)
sem1 <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_1factor_results.csv"), stringsAsFactors = FALSE)
ex   <- load_track1_f3_exposure(root_dir, set_name = "main", min_n = 3L)

# --- Panel A: Global, Within/Between-Family (Mundlak), Drop Inner Asia, Strict Links (2PL SEM) ---
rows_a <- list()
for (dom_id in DOMAIN_ORDER) {
  sub_d <- subset(sem1, domain == dom_id & arch == "MA_2pl")
  r_max <- subset(sub_d, spec == "maximal"       & set == "main"   & min_n_ind == 3L)
  r_mun <- subset(sub_d, spec == "mundlak_wb"    & set == "main"   & min_n_ind == 3L)
  r_nia <- subset(sub_d, spec == "no_inner_asia" & set == "main"   & min_n_ind == 3L)
  r_str <- subset(sub_d, spec == "strict_links"  & set == "strict" & min_n_ind == 3L)
  stopifnot(nrow(r_max) == 1L, nrow(r_mun) == 1L, nrow(r_nia) == 1L, nrow(r_str) == 1L)

  mk <- function(lbl, row, pfx = "b") {
    q80 <- get_exact_q80(row, pfx)
    data.frame(
      domain = dom_id,
      dom_lbl = DOMAIN_SHORT[[dom_id]],
      spec_lbl = lbl,
      mean = row[[paste0(pfx, "_mean")]],
      lo95 = row[[paste0(pfx, "_q025")]],
      hi95 = row[[paste0(pfx, "_q975")]],
      lo80 = q80$lo80, hi80 = q80$hi80,
      stringsAsFactors = FALSE
    )
  }
  rows_a[[length(rows_a) + 1L]] <- rbind(
    mk("Global (Phylo + GP)",      r_max, "b"),
    mk("Within-family (Mundlak)",  r_mun, "b"),
    mk("Between-family (Mundlak)", r_mun, "b_between"),
    mk("Drop Inner Asia",          r_nia, "b"),
    mk("Strict links",             r_str, "b")
  )
}
dfA <- do.call(rbind, rows_a)
spec_ord_a <- c("Global (Phylo + GP)", "Within-family (Mundlak)", "Between-family (Mundlak)",
                "Drop Inner Asia", "Strict links")
dfA$spec_lbl <- factor(dfA$spec_lbl, levels = rev(spec_ord_a))
dfA$dom_lbl  <- factor(dfA$dom_lbl, levels = unname(DOMAIN_SHORT[DOMAIN_ORDER]))

xlim_a <- range(c(dfA$lo95, dfA$hi95, -0.60, 0.85), na.rm = TRUE)
xlim_a <- xlim_a + c(-0.04, 0.04) * diff(xlim_a)

pA <- ggplot(dfA, aes(x = mean, y = spec_lbl, colour = domain, shape = domain)) +
  annotate("rect", xmin = -0.15, xmax = 0.15, ymin = -Inf, ymax = Inf,
           fill = ROPE_FILL, alpha = 0.78) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.38, linetype = "dashed") +
  geom_errorbar(aes(xmin = lo95, xmax = hi95), width = 0, orientation = "y", linewidth = 0.45, alpha = 0.52) +
  geom_errorbar(aes(xmin = lo80, xmax = hi80), width = 0, orientation = "y", linewidth = 1.25) +
  geom_point(size = 1.65) +
  facet_wrap(~dom_lbl, nrow = 1) +
  scale_colour_manual(values = DOMAIN_COLS, guide = "none") +
  scale_shape_manual(values = DOMAIN_SHAPES_FILLED, guide = "none") +
  scale_x_continuous(breaks = c(-1.0, 0.0, 1.0), labels = c("-1", "0", "+1")) +
  coord_cartesian(xlim = xlim_a) +
  labs(
    title = "a   Within-family decomposition & regional/link sensitivity (2PL IRT-SEM)",
    x = "Slope per SD of cross-family f3 (80% & 95% CrI; shaded: \u00b10.15 ROPE)",
    y = NULL
  ) +
  theme_track1() +
  theme(
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.4),
    axis.text.y = element_text(colour = INK_BODY, size = 7.2),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing = unit(6, "pt")
  )

# --- Panel B: Multi-Language Family Trajectories Across 8 Major Lineages ---
fams <- c(
  "Indo-European", "Sino-Tibetan", "Atlantic-Congo", "Afro-Asiatic",
  "Austronesian", "Uralic", "Turkic", "Mongolic-Khitan"
)

d_fam <- merge(subset(ld, in_main == 1L & family %in% fams),
               ex[, c("glottocode", "f3_raw", "f3_scaled")],
               by = "glottocode")
n_fam <- table(factor(d_fam$family, levels = fams))
fam_titles <- setNames(sprintf("%s (n=%d)", fams, as.integer(n_fam[fams])), fams)

# Observed domain shares for the 4 symmetrical Breadth x Irregularity x POS domains
dom_b <- list(
  list(id = "v_breadth", lbl = "Verb breadth",      pres = "v_breadth_present", cod = "v_breadth_coded", min_c = 6L),
  list(id = "n_breadth", lbl = "Noun breadth",      pres = "n_breadth_present", cod = "n_breadth_coded", min_c = 3L),
  list(id = "v_irreg",   lbl = "Verb irregularity", pres = "v_irreg_present",   cod = "v_irreg_coded",   min_c = 3L),
  list(id = "n_irreg",   lbl = "Noun irregularity", pres = "n_irreg_present",   cod = "n_irreg_coded",   min_c = 2L)
)

rows_b <- list()
for (db in dom_b) {
  ok <- !is.na(d_fam[[db$cod]]) & d_fam[[db$cod]] >= db$min_c
  sub <- d_fam[ok, , drop = FALSE]
  rows_b[[length(rows_b) + 1L]] <- data.frame(
    fam_panel = fam_titles[sub$family],
    domain = db$id,
    dom_lbl = db$lbl,
    f3_scaled = sub$f3_scaled,
    share = sub[[db$pres]] / sub[[db$cod]],
    stringsAsFactors = FALSE
  )
}
dfB <- do.call(rbind, rows_b)
dfB$fam_panel <- factor(dfB$fam_panel, levels = unname(fam_titles))
b_lbls <- vapply(dom_b, `[[`, character(1), "lbl")
dfB$dom_lbl <- factor(dfB$dom_lbl, levels = b_lbls)

col_b <- setNames(c(COL_BREADTH, COL_BREADTH, COL_IRREG, COL_IRREG), b_lbls)
shp_b <- setNames(c(16, 17, 16, 17), b_lbls)
lty_b <- setNames(c("solid", "dashed", "solid", "dashed"), b_lbls)

xmax_b <- max(dfB$f3_scaled, na.rm = TRUE) * 1.06

pB <- ggplot(dfB, aes(x = f3_scaled, y = share, colour = dom_lbl, shape = dom_lbl, linetype = dom_lbl)) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, linewidth = 0.68) +
  geom_point(size = 1.15, alpha = 0.58) +
  facet_wrap(~fam_panel, ncol = 4) +
  scale_colour_manual(values = col_b, name = NULL) +
  scale_shape_manual(values = shp_b, name = NULL) +
  scale_linetype_manual(values = lty_b, name = NULL) +
  scale_x_sqrt(breaks = c(0, 1, 3, 6), labels = c("0", "1", "3", "6")) +
  scale_y_continuous(breaks = c(0, 0.5, 1.0), labels = c("0%", "50%", "100%")) +
  coord_cartesian(xlim = c(0, max(6.8, xmax_b)), ylim = c(-0.04, 1.04)) +
  labs(
    title = "b   Observed domain shares vs. cross-family f3 across 8 major families (OLS lines)",
    x = expression("Language-level cross-family " * italic(f)[3] * " (" %*% 10^-3 * ", sqrt scale)"),
    y = "Coded features present"
  ) +
  theme_track1() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.2),
    panel.spacing = unit(5, "pt"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.key.width = unit(16, "pt"),
    legend.text = element_text(size = 7.3, colour = INK_BODY),
    legend.margin = margin(t = -2, b = 0)
  )

fig5 <- pA / wrap_elements(full = pB) + plot_layout(heights = c(0.88, 1.48))
save_track1_fig(fig5, fig_dir, "fig5_grambank_family_trajectories", width = 6.5, height = 5.9, dpi = 300)
