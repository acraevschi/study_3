# Figure 1 (Track 1): Global Sample Geography (Cross-Family f3) & Clean Domain Distributions
# Outputs: fig1_grambank_map_and_domains.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(maps)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/grambank/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

ld  <- read.csv(file.path(root_dir, "analyses/grambank/data/language_data.csv"), stringsAsFactors = FALSE)
ex  <- load_track1_f3_exposure(root_dir, set_name = "main", min_n = 3L)
eap <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_latent_eap.csv"), stringsAsFactors = FALSE)

# --- Panel A: Global Map of 225 GeLaTo-Linked Grambank Languages ---
d_map <- merge(subset(ld, in_main == 1L), ex[, c("glottocode", "f3_raw", "f3_scaled")],
               by = "glottocode", all.x = TRUE)
d_has <- subset(d_map, !is.na(f3_scaled))
d_no  <- subset(d_map, is.na(f3_scaled))

world_df <- subset(map_data("world"), region != "Antarctica")

pA <- ggplot() +
  geom_polygon(data = world_df, aes(long, lat, group = group),
               fill = MAP_LAND, colour = "white", linewidth = 0.14) +
  geom_point(data = d_no, aes(lon, lat),
             shape = 21, size = 1.65, fill = "white", colour = INK2, stroke = 0.45) +
  geom_point(data = d_has[order(d_has$f3_scaled), ],
             aes(lon, lat, fill = f3_scaled),
             shape = 21, size = 1.95, colour = "white", stroke = 0.30) +
  scale_fill_gradientn(
    colours = SEQ_F3,
    trans = "sqrt",
    limits = c(0, NA),
    breaks = c(0, 0.5, 1.5, 3, 6),
    labels = c("0", "0.5", "1.5", "3", "6"),
    name = expression(atop("Cross-family " * italic(f)[3], "(\u00d7" * 10^-3 * ")")),
    guide = guide_colorbar(barheight = unit(3.2, "lines"), barwidth = unit(0.45, "lines"))
  ) +
  coord_quickmap(xlim = c(-170, 185), ylim = c(-50, 75), expand = FALSE) +
  labs(
    title = sprintf("a   %d GeLaTo-linked Grambank languages (%d families)",
                    nrow(d_map), length(unique(d_map$family))),
    subtitle = sprintf("Filled: %d languages with cross-family f3 (n \u2265 3); open circles: %d without f3 (n < 3 or isolate)",
                       nrow(d_has), nrow(d_no))
  ) +
  theme_track1() +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid.major = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 7.4, colour = INK_BODY, face = "bold"),
    legend.text = element_text(size = 7.1, colour = INK2),
    plot.margin = margin(4, 4, 2, 4)
  )

# --- Panel B1: Empirical Distributions Across the 5 Clean Domains ---
dom_specs <- list(
  list(id = "v_breadth", pres = "v_breadth_present", cod = "v_breadth_coded", min_c = 6L, k = 9L),
  list(id = "n_breadth", pres = "n_breadth_present", cod = "n_breadth_coded", min_c = 3L, k = 5L),
  list(id = "v_irreg",   pres = "v_irreg_present",   cod = "v_irreg_coded",   min_c = 3L, k = 5L),
  list(id = "n_irreg",   pres = "n_irreg_present",   cod = "n_irreg_coded",   min_c = 2L, k = 3L),
  list(id = "np_agr",    pres = "np_agr_present",    cod = "np_agr_coded",    min_c = 5L, k = 7L)
)

dist_rows <- list()
sum_rows  <- list()
set.seed(2026)
for (ds in dom_specs) {
  sub <- subset(d_has, !is.na(d_has[[ds$cod]]) & d_has[[ds$cod]] >= ds$min_c)
  sh  <- sub[[ds$pres]] / sub[[ds$cod]]
  lbl <- sprintf("%s\n(k=%d, n=%d)", DOMAIN_SHORT[[ds$id]], ds$k, nrow(sub))
  dist_rows[[length(dist_rows) + 1L]] <- data.frame(
    domain = ds$id, dom_lbl = lbl, share = sh,
    jit = runif(length(sh), -0.18, 0.18),
    stringsAsFactors = FALSE
  )
  sum_rows[[length(sum_rows) + 1L]] <- data.frame(
    domain = ds$id, dom_lbl = lbl,
    med = median(sh), q25 = unname(quantile(sh, 0.25)), q75 = unname(quantile(sh, 0.75)),
    stringsAsFactors = FALSE
  )
}
df_dist <- do.call(rbind, dist_rows)
df_sum  <- do.call(rbind, sum_rows)

lbl_order <- vapply(dom_specs, function(ds) {
  sub <- subset(d_has, !is.na(d_has[[ds$cod]]) & d_has[[ds$cod]] >= ds$min_c)
  sprintf("%s\n(k=%d, n=%d)", DOMAIN_SHORT[[ds$id]], ds$k, nrow(sub))
}, character(1))

df_dist$dom_lbl <- factor(df_dist$dom_lbl, levels = rev(lbl_order))
df_sum$dom_lbl  <- factor(df_sum$dom_lbl, levels = rev(lbl_order))
df_dist$y_num   <- as.numeric(df_dist$dom_lbl) + df_dist$jit

pB1 <- ggplot() +
  geom_point(data = df_dist, aes(x = share, y = y_num, colour = domain, shape = domain),
             size = 1.05, alpha = 0.26) +
  geom_errorbar(data = df_sum,
                aes(xmin = q25, xmax = q75, y = as.numeric(dom_lbl), colour = domain),
                width = 0, orientation = "y", linewidth = 1.5, alpha = 0.85) +
  geom_point(data = df_sum,
             aes(x = med, y = as.numeric(dom_lbl), colour = domain, shape = domain),
             size = 2.4) +
  scale_colour_manual(values = DOMAIN_COLS, guide = "none") +
  scale_shape_manual(values = DOMAIN_SHAPES_FILLED, guide = "none") +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1.0),
                     labels = c("0%", "25%", "50%", "75%", "100%")) +
  scale_y_continuous(breaks = seq_along(lbl_order), labels = rev(lbl_order)) +
  coord_cartesian(xlim = c(-0.03, 1.03), ylim = c(0.5, length(lbl_order) + 0.5)) +
  labs(
    title = "b   Domain feature shares (median & IQR)",
    x = "Proportion of coded features present",
    y = NULL
  ) +
  theme_track1() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(colour = INK_BODY, size = 7.2, lineheight = 0.98)
  )

# --- Panel B2: Breadth vs. Irregularity Latent Decoupling (2PL SEM EAPs) ---
# Handle either wide (glottocode, theta_v_breadth_eap, ...) or long (glottocode, domain, theta_eap) format
get_eap_wide <- function(df_e) {
  if ("domain" %in% names(df_e)) {
    val_col <- intersect(c("theta_eap", "theta_mean", "eap"), names(df_e))[1]
    stopifnot(!is.na(val_col))
    w <- reshape(df_e[, c("glottocode", "domain", val_col)],
                 idvar = "glottocode", timevar = "domain", direction = "wide")
    names(w) <- sub(paste0("^", val_col, "\\."), "theta_", names(w))
    return(w)
  }
  df_e
}

eap_w <- get_eap_wide(eap)
find_eap_col <- function(dom) {
  cand <- c(paste0("theta_", dom, "_eap"), paste0("theta_", dom), paste0(dom, "_eap"), dom)
  hit <- intersect(cand, names(eap_w))
  stopifnot(length(hit) >= 1L)
  hit[1]
}

c_vb <- find_eap_col("v_breadth")
c_vi <- find_eap_col("v_irreg")
c_nb <- find_eap_col("n_breadth")
c_ni <- find_eap_col("n_irreg")

dv <- eap_w[is.finite(eap_w[[c_vb]]) & is.finite(eap_w[[c_vi]]), c("glottocode", c_vb, c_vi)]
dn <- eap_w[is.finite(eap_w[[c_nb]]) & is.finite(eap_w[[c_ni]]), c("glottocode", c_nb, c_ni)]
names(dv) <- c("glottocode", "breadth", "irreg")
names(dn) <- c("glottocode", "breadth", "irreg")
dv$pos <- "Verbs"
dn$pos <- "Nouns"

rv <- cor(dv$breadth, dv$irreg)
rhov <- suppressWarnings(cor(dv$breadth, dv$irreg, method = "spearman"))
rn <- cor(dn$breadth, dn$irreg)
rhon <- suppressWarnings(cor(dn$breadth, dn$irreg, method = "spearman"))

df_dec <- rbind(dv, dn)
df_dec$pos <- factor(df_dec$pos, levels = c("Verbs", "Nouns"))

lim_y <- range(df_dec$irreg, na.rm = TRUE)
ann_dec <- data.frame(
  pos = factor(c("Verbs", "Nouns"), levels = c("Verbs", "Nouns")),
  lab = c(
    sprintf("n = %d\nr = %.2f, \u03c1 = %.2f", nrow(dv), rv, rhov),
    sprintf("n = %d\nr = %.2f, \u03c1 = %.2f", nrow(dn), rn, rhon)
  ),
  stringsAsFactors = FALSE
)

pB2 <- ggplot(df_dec, aes(x = breadth, y = irreg, shape = pos, linetype = pos)) +
  geom_hline(yintercept = 0, colour = GRID_WARM, linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = GRID_WARM, linewidth = 0.35) +
  geom_smooth(method = "lm", formula = y ~ x, colour = INK_BODY, fill = GREY_LT, alpha = 0.22, linewidth = 0.65) +
  geom_point(colour = INK2, size = 1.15, alpha = 0.60) +
  geom_text(data = ann_dec, aes(x = -Inf, y = Inf, label = lab),
            inherit.aes = FALSE, hjust = -0.08, vjust = 1.15,
            size = 7.0 / .pt, lineheight = 1.08, colour = INK_BODY, family = "Helvetica") +
  facet_wrap(~pos, ncol = 2) +
  scale_shape_manual(values = c("Verbs" = 16, "Nouns" = 17), guide = "none") +
  scale_linetype_manual(values = c("Verbs" = "solid", "Nouns" = "dashed"), guide = "none") +
  scale_x_continuous(breaks = c(-2, 0, 2, 4, 6)) +
  coord_cartesian(ylim = c(lim_y[1] - 0.05 * diff(lim_y), lim_y[2] + 0.30 * diff(lim_y))) +
  labs(
    title = "c   Latent breadth vs. irregularity decoupling",
    x = expression("Latent enumerative breadth (" * theta[breadth] * ")"),
    y = expression("Latent irregularity (" * theta[irreg] * ")")
  ) +
  theme_track1() +
  theme(
    panel.border = element_rect(colour = "#dfded6", fill = NA, linewidth = 0.35),
    strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.5),
    panel.spacing = unit(10, "pt")
  )

fig1 <- pA / wrap_elements(full = (pB1 + pB2 + plot_layout(widths = c(1.02, 1.18)))) +
  plot_layout(heights = c(1.02, 1.05))

save_track1_fig(fig1, fig_dir, "fig1_grambank_map_and_domains", width = 6.5, height = 5.4, dpi = 300)
