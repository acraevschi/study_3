# Figure 4 (Track 1): Item-Level Architecture-B (MB_2pl) f3_fixed Slopes & 2PL Discriminations
# Across all 29 Grambank features grouped into the 5 morphological domains.
# Outputs: fig4_grambank_item_dif.{png,pdf}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
fig_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else normalizePath("analyses/grambank/figures", mustWork = FALSE)
root_dir <- normalizePath(file.path(fig_dir, "../../.."), mustWork = FALSE)
source(file.path(fig_dir, "_style.R"))

items_meta <- read.csv(file.path(root_dir, "analyses/grambank/data/items_metadata.csv"), stringsAsFactors = FALSE)
dif_df     <- read.csv(file.path(root_dir, "analyses/grambank/latent_sem/results_domain/sem_item_dif.csv"), stringsAsFactors = FALSE)

pretty_items <- c(
  present_tense     = "Present tense (GB082)",
  past_tense        = "Past tense (GB083)",
  future_tense      = "Future tense (GB084)",
  aspect            = "Perfective/imperfective aspect (GB086)",
  mood              = "Verbal mood (GB312)",
  subj_indexing     = "Subject indexing on verb (GB089\u2013092)",
  obj_indexing      = "Object indexing on verb (GB093\u2013094)",
  negation          = "Bound verbal negation (GB107)",
  polar_q_bound     = "Bound polar interrogation (GB286)",
  noun_plural       = "Nominal plural (GB044)",
  noun_dual         = "Nominal dual (GB043)",
  noun_case_core    = "Core noun case (GB070)",
  noun_case_oblique = "Oblique noun case (GB072)",
  possessed_affix   = "Possessed-noun affix (GB431/433)",
  conj_classes      = "Conjugation classes (GB111)",
  tam_suppletion    = "Verb TAM suppletion (GB110)",
  pers_stem_alt     = "Person stem alternation (GB099)",
  part_num_suppl    = "Participant-number suppletion (GB109)",
  lex_verb_suppl    = "Suppletive give/see/come (GB300/402/403)",
  noun_num_allom    = "Noun number allomorphy (GB039)",
  noun_num_suppl    = "Suppletive noun plurals (GB041)",
  opaque_gender     = "Unpredictable gender/class (GB321)",
  agr_gender_prop   = "Gender agr.: property word (GB170)",
  agr_gender_dem    = "Gender agr.: demonstrative (GB171)",
  agr_gender_art    = "Gender agr.: article (GB172)",
  agr_gender_num    = "Gender agr.: numeral (GB198)",
  agr_num_prop      = "Number agr.: property word (GB184)",
  agr_num_dem       = "Number agr.: demonstrative (GB185)",
  agr_num_art       = "Number agr.: article (GB186)"
)

df <- merge(dif_df, items_meta[, c("item_id", "item_group", "item_order")], by = "item_id")
stopifnot(nrow(df) == 29L)

q80_beta <- get_exact_q80(df, "beta")
df$beta_lo80 <- q80_beta$lo80
df$beta_hi80 <- q80_beta$hi80

has_lam80 <- all(c("lambda_q10", "lambda_q90") %in% names(df))
df$lam_lo80 <- if (has_lam80) df$lambda_q10 else NA_real_
df$lam_hi80 <- if (has_lam80) df$lambda_q90 else NA_real_

df$item_label <- unname(pretty_items[df$item_id])
stopifnot(all(!is.na(df$item_label)))

facet_lbls <- c(
  v_breadth = "Verb\nbreadth",
  n_breadth = "Noun\nbreadth",
  v_irreg   = "Verb\nirreg.",
  n_irreg   = "Noun\nirreg.",
  np_agr    = "NP\nagreement"
)
df$dom_facet <- factor(facet_lbls[df$domain], levels = unname(facet_lbls[DOMAIN_ORDER]))

# Order items within each domain by beta_mean ascending (highest slope at top of each facet)
df <- df[order(match(df$domain, DOMAIN_ORDER), df$beta_mean), ]
df$item_label <- factor(df$item_label, levels = df$item_label)

xlim_b <- range(c(df$beta_q025, df$beta_q975, -0.8, 1.0), na.rm = TRUE)
xlim_b <- xlim_b + c(-0.04, 0.04) * diff(xlim_b)
xlim_l <- c(0, max(df$lambda_q975, 2.5, na.rm = TRUE) * 1.05)

pA <- ggplot(df, aes(x = beta_mean, y = item_label, colour = domain, shape = domain)) +
  geom_vline(xintercept = 0, colour = INK_BODY, linewidth = 0.4, linetype = "dashed") +
  geom_segment(aes(x = beta_q025, xend = beta_q975, yend = item_label), linewidth = 0.48, alpha = 0.50) +
  geom_segment(aes(x = beta_lo80, xend = beta_hi80, yend = item_label), linewidth = 1.35, alpha = 1.0) +
  geom_point(size = 1.75) +
  facet_grid(dom_facet ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = DOMAIN_COLS, guide = "none") +
  scale_shape_manual(values = DOMAIN_SHAPES_FILLED, guide = "none") +
  scale_x_continuous(breaks = c(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)) +
  coord_cartesian(xlim = xlim_b) +
  labs(
    title = "a   Item-specific cross-family f3 slopes (varying-slope 2PL)",
    x = "Logit slope per SD of f3 (80% & 95% CrI)",
    y = NULL
  ) +
  theme_track1() +
  theme(
    strip.placement = "outside",
    strip.clip = "off",
    strip.text.y.left = element_text(colour = INK_BODY, face = "bold", size = 7.4, angle = 0, hjust = 1, lineheight = 0.96),
    axis.text.y = element_text(colour = INK_BODY, size = 7.2),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing.y = unit(6, "pt"),
    plot.margin = margin(5, 8, 5, 4)
  )

pB <- ggplot(df, aes(x = lambda_mean, y = item_label, colour = domain, shape = domain)) +
  geom_vline(xintercept = 1, colour = GREY_MID, linewidth = 0.4, linetype = "dashed") +
  geom_segment(aes(x = lambda_q025, xend = lambda_q975, yend = item_label), linewidth = 0.48, alpha = 0.50) +
  { if (has_lam80) geom_segment(aes(x = lam_lo80, xend = lam_hi80, yend = item_label), linewidth = 1.35, alpha = 1.0) } +
  geom_point(size = 1.75) +
  facet_grid(dom_facet ~ ., scales = "free_y", space = "free_y") +
  scale_colour_manual(values = DOMAIN_COLS, guide = "none") +
  scale_shape_manual(values = DOMAIN_SHAPES_FILLED, guide = "none") +
  scale_x_continuous(breaks = c(0, 1, 2, 3)) +
  coord_cartesian(xlim = xlim_l) +
  labs(
    title = "b   2PL loading (\u03bb)",
    x = expression("Discrimination " * lambda[k]),
    y = NULL
  ) +
  theme_track1() +
  theme(
    strip.text = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_line(colour = GRID_MIN, linewidth = 0.25),
    panel.spacing.y = unit(6, "pt"),
    plot.margin = margin(5, 6, 5, 2)
  )

fig4 <- pA + pB + plot_layout(widths = c(1.75, 0.72))
save_track1_fig(fig4, fig_dir, "fig4_grambank_item_dif", width = 6.5, height = 7.0, dpi = 300)
