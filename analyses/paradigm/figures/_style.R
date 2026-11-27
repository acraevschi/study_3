# Shared visual grammar for Study 3 Track 2 (Morphological Paradigm Predictability)
# Harmonized with Track 1 (`analyses/grambank/figures/_style.R`).

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

# Typography & Inks
INK      <- "#0b0b0b"
INK_BODY <- "#222222"
INK2     <- "#52514e"
GREY_MID <- "#6b6b6b"
GREY_LT  <- "#a9a9a3"
GRID_MAJ <- "#ecebe6"
GRID_MIN <- "#f5f4f0"
GRID_WARM <- "#e1e0d9"
MAP_LAND <- "#f0efec"

# Construct Palette (strictly reserved for Track 1 / Cross-Track Complexity Dimensions)
COL_BREADTH   <- "#1c5cab"  # Deep royal blue (Enumerative breadth, Track 1)
COL_IRREG     <- "#d95f26"  # Burnt orange (Paradigmatic irregularity, Track 1)
COL_IRREG_TXT <- "#b4481a"  # Darker burnt orange for >=4.5:1 text contrast on white
COL_MAG       <- "#8e1f5f"  # Magenta accent (Track 1 contextual NP agreement)

# Part-of-Speech Palette (strictly reserved for Verbs vs. Nouns across Track 2 Figures)
# Always double-encoded with shape (circle 16 / triangle 17) and linetype (solid / dashed).
COL_VERB     <- "#0d7a68"   # Deep pine-teal (Verbs; 5.25:1 contrast on white)
COL_NOUN     <- "#9e6b06"   # Warm ochre-bronze (Nouns; 4.61:1 contrast on white)
COL_NOUN_TXT <- "#805504"   # Darker ochre-bronze for >=6.2:1 text contrast on white

# Secondary Non-POS Accents (regime & stratum comparisons that pool or stratify within POS)
COL_REGIME_TF  <- "#6d3b7e" # Muted plum (topfreq120 regime)
COL_REGIME_480 <- "#334155" # Dark slate (uniform480 regime)
COL_ADM_HIGH   <- "#1e293b" # Deep charcoal ink (High-admixture stratum)
COL_ADM_LOW    <- "#76746e" # Muted warm grey (Low-admixture stratum)
COL_REF        <- "#4d4c48" # Neutral dark grey reference

SEQ_ENTROPY <- c("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SEQ_F3      <- SEQ_ENTROPY

ROPE_FILL <- "#e8e7e1"

# Extract exact exported posterior [q10, q90] bounds (no normal approximation)
get_exact_q80 <- function(df, pfx = "b") {
  c10 <- paste0(pfx, "_q10")
  c90 <- paste0(pfx, "_q90")
  if (!all(c(c10, c90) %in% names(df))) {
    stop(sprintf("Required exact 80%% CrI columns '%s' and '%s' missing from results CSV.", c10, c90))
  }
  list(lo80 = as.numeric(df[[c10]]), hi80 = as.numeric(df[[c90]]))
}

theme_track2 <- function(base_size = 8.5) {
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
      strip.text = element_text(colour = INK_BODY, face = "bold", size = 7.8),
      plot.title.position = "plot",
      plot.margin = margin(5, 6, 5, 5)
    )
}

save_track2_fig <- function(plot_obj, out_dir, stem, width = 6.5, height = 4.5, dpi = 300) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  dev_png <- if (capabilities("aqua")) "quartz" else "cairo"
  dev_pdf <- if (capabilities("cairo")) cairo_pdf else "pdf"
  ggsave(png_path, plot_obj, width = width, height = height, dpi = dpi, bg = "white", type = dev_png)
  ggsave(pdf_path, plot_obj, width = width, height = height, bg = "white", device = dev_pdf)
  message(sprintf("Saved %s.{png,pdf} (%.2f x %.2f in @ %d dpi)", stem, width, height, dpi))
}
