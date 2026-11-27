# Shared visual grammar for Study 3 Track 1 (Grambank 5 Morphological Domains)
# Harmonized with Track 2 (`analyses/paradigm/figures/_style.R`).

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

# Typography & Inks
INK       <- "#0b0b0b"
INK_BODY  <- "#222222"
INK2      <- "#52514e"
GREY_MID  <- "#6b6b6b"
GREY_LT   <- "#a9a9a3"
GRID_MAJ  <- "#ecebe6"
GRID_MIN  <- "#f5f4f0"
GRID_WARM <- "#e1e0d9"
MAP_LAND  <- "#f0efec"
ROPE_FILL <- "#e8e7e1"

# Construct & POS Palette (consistent double-encoding across Track 1 & Track 2)
# Breadth = #1c5cab (blue), Irregularity = #d95f26 (burnt orange), NP Agreement = #8e1f5f (magenta)
# Verbs = circles (16/21) & solid; Nouns = triangles (17/24) & dashed; Agreement = squares (15/22) & dotdash
COL_BREADTH   <- "#1c5cab"
COL_IRREG     <- "#d95f26"
COL_IRREG_TXT <- "#b4481a"
COL_MAG       <- "#8e1f5f"
COL_DIFF      <- "#1e293b"

SEQ_F3 <- c("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

DOMAIN_ORDER <- c("v_breadth", "n_breadth", "v_irreg", "n_irreg", "np_agr")

DOMAIN_LABELS <- c(
  v_breadth = "Verb breadth",
  n_breadth = "Noun breadth",
  v_irreg   = "Verb irregularity",
  n_irreg   = "Noun irregularity",
  np_agr    = "Contextual NP agreement"
)

DOMAIN_SHORT <- c(
  v_breadth = "Verb breadth",
  n_breadth = "Noun breadth",
  v_irreg   = "Verb irreg.",
  n_irreg   = "Noun irreg.",
  np_agr    = "NP agreement"
)

DOMAIN_COLS <- c(
  v_breadth = COL_BREADTH,
  n_breadth = COL_BREADTH,
  v_irreg   = COL_IRREG,
  n_irreg   = COL_IRREG,
  np_agr    = COL_MAG
)

DOMAIN_SHAPES_FILLED <- c(
  v_breadth = 16,
  n_breadth = 17,
  v_irreg   = 16,
  n_irreg   = 17,
  np_agr    = 15
)

DOMAIN_SHAPES_OPEN <- c(
  v_breadth = 21,
  n_breadth = 24,
  v_irreg   = 21,
  n_irreg   = 24,
  np_agr    = 22
)

DOMAIN_LTY <- c(
  v_breadth = "solid",
  n_breadth = "dashed",
  v_irreg   = "solid",
  n_irreg   = "dashed",
  np_agr    = "dotdash"
)

# Extract exact exported posterior [q10, q90] bounds (no normal approximation)
get_exact_q80 <- function(df, pfx = "b") {
  c10 <- paste0(pfx, "_q10")
  c90 <- paste0(pfx, "_q90")
  if (!all(c(c10, c90) %in% names(df))) {
    stop(sprintf("Required exact 80%% CrI columns '%s' and '%s' missing from results CSV.", c10, c90))
  }
  list(lo80 = as.numeric(df[[c10]]), hi80 = as.numeric(df[[c90]]))
}

# Read language-level Stage-1 f3_fixed exposure (prefers SEM export, falls back to BB export)
load_track1_f3_exposure <- function(root_dir, set_name = "main", min_n = 3L) {
  sem_p <- file.path(root_dir, "analyses/grambank/latent_sem/results_domain/exposure_sem_f3_fixed.csv")
  bb_p  <- file.path(root_dir, "analyses/grambank/results/exposure_bb_all.csv")
  if (file.exists(sem_p)) {
    ex <- read.csv(sem_p, stringsAsFactors = FALSE)
  } else if (file.exists(bb_p)) {
    ex <- read.csv(bb_p, stringsAsFactors = FALSE)
  } else {
    stop("Neither exposure_sem_f3_fixed.csv nor exposure_bb_all.csv found.")
  }
  if ("set" %in% names(ex)) ex <- ex[ex$set == set_name, , drop = FALSE]
  if ("measure" %in% names(ex)) ex <- ex[ex$measure == "f3_fixed", , drop = FALSE]
  if ("min_n_ind" %in% names(ex)) ex <- ex[as.integer(ex$min_n_ind) == as.integer(min_n), , drop = FALSE]
  stopifnot(nrow(ex) > 0L, !anyDuplicated(ex$glottocode))
  ex$f3_raw <- pmax(0, ex$eta_mean * ex$y_sd + ex$y_mean)
  ex$f3_scaled <- ex$f3_raw * 1e3
  ex
}

theme_track1 <- function(base_size = 8.5) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = GRID_MAJ, linewidth = 0.3),
      axis.text = element_text(colour = INK2, size = 7.4),
      axis.title = element_text(colour = INK_BODY, size = 8.0),
      plot.title = element_text(colour = INK, face = "bold", size = 9.0, margin = margin(b = 2)),
      plot.subtitle = element_text(colour = INK2, size = 7.3, lineheight = 1.12, margin = margin(b = 4)),
      plot.caption = element_text(colour = INK2, size = 7.0, hjust = 0),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.7),
      plot.title.position = "plot",
      plot.margin = margin(5, 6, 5, 5)
    )
}

save_track1_fig <- function(plot_obj, out_dir, stem, width = 6.5, height = 4.5, dpi = 300) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  dev_png <- if (capabilities("aqua")) "quartz" else "cairo"
  dev_pdf <- if (capabilities("cairo")) cairo_pdf else "pdf"
  ggsave(png_path, plot_obj, width = width, height = height, dpi = dpi, bg = "white", type = dev_png)
  ggsave(pdf_path, plot_obj, width = width, height = height, bg = "white", device = dev_pdf)
  message(sprintf("Saved %s.{png,pdf} (%.2f x %.2f in @ %d dpi)", stem, width, height, dpi))
}
