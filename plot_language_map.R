#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Language coverage map
# ---------------------------------------------------------------------------
# Eyeball the geographic and genealogical coverage of the study sample: the
# languages that have BOTH morphological-complexity data (MGN) and demographic
# data (the registry), which is what the models are actually fitted on.
#
#   plots/map_mgn64.png   the 64 modelled languages, world + Europe zoom
#
# A map of the whole 7,872-language demographic registry used to be drawn here
# too. It was dropped: registry coverage on its own is not a quantity this study
# reports, only the intersection is.
#
# Points are placed at Glottolog centroids, coloured by language family and
# sized by L1 population (area-proportional, on a log scale -- populations span
# 1,800 to 413 million, so a linear scale would render everything below a few
# million as an invisible dot).
#
# Usage:
#   Rscript plot_language_map.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(maps)
  library(scales)
  library(patchwork)
})

dir.create("plots", showWarnings = FALSE)

REGISTRY <- "global_demographic_registry.csv"
MODELING <- "mgn_modeling_dataset.csv"
stopifnot(file.exists(REGISTRY), file.exists(MODELING))

registry <- read.csv(REGISTRY, stringsAsFactors = FALSE)
modeling <- read.csv(MODELING, stringsAsFactors = FALSE)

world <- map_data("world") %>% filter(region != "Antarctica")

base_map <- function() {
  list(
    geom_polygon(data = world, aes(long, lat, group = group),
                 fill = "grey93", colour = "grey82", linewidth = 0.15),
    coord_fixed(1.35, xlim = c(-170, 180), ylim = c(-56, 78), expand = FALSE),
    theme_minimal(base_size = 11),
    theme(
      panel.grid       = element_blank(),
      axis.text        = element_blank(),
      axis.title       = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      legend.position  = "right",
      legend.key       = element_blank(),
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(colour = "grey30", size = 9.5),
      plot.caption     = element_text(colour = "grey45", size = 8, hjust = 0)
    )
  )
}

# Population -> point area on a log10 scale. Populations span 1,800 to 413
# million, so a linear scale renders everything below a few million invisible.
pop_size_scale <- function(breaks = c(1e4, 1e6, 1e8), max_size = 7) {
  scale_size_area(
    name    = "L1 speakers",
    max_size = max_size,
    breaks  = breaks,
    labels  = label_number(scale_cut = cut_short_scale()),
    trans   = "log10"
  )
}

# The size legend needs an explicit fill: with shape 21 the key is otherwise
# drawn unfilled and the legend looks empty.
size_guide <- function(order = 2) {
  guide_legend(order = order,
               override.aes = list(shape = 21, fill = "grey45", colour = "white"))
}

# ---------------------------------------------------------------------------
# 1. The 64 modelled MGN languages
# ---------------------------------------------------------------------------
mgn <- modeling %>%
  group_by(iso_sanitized) %>%
  summarise(
    lang           = first(lang),
    language_name  = first(language_name),
    family         = first(family),
    population_l1  = first(population_l1),
    lat            = first(lat),
    lon            = first(lon),
    eco_imputed    = first(eco_imputed),
    n_trials       = n(),
    accuracy       = sum(correct) / sum(total),
    .groups = "drop"
  )

cat(sprintf("MGN languages: %d, families: %d\n", nrow(mgn), n_distinct(mgn$family)))

# Families with a single language are lumped so the legend stays readable.
fam_n <- mgn %>% count(family, sort = TRUE)
big_fams <- fam_n$family[fam_n$n >= 2]
mgn <- mgn %>%
  mutate(family_grp = ifelse(family %in% big_fams, family, "Other (single language)"),
         family_grp = factor(family_grp,
                             levels = c(sort(big_fams), "Other (single language)")))

fam_palette <- c(
  "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
  "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
  "#B23A48", "#3C6E71"
)
n_fam <- nlevels(mgn$family_grp)
pal <- rep(fam_palette, length.out = n_fam)
names(pal) <- levels(mgn$family_grp)
pal["Other (single language)"] <- "#9AA0A6"

mgn_points <- function(max_size) {
  geom_point(data = mgn,
             aes(lon, lat, size = population_l1, fill = family_grp),
             shape = 21, colour = "white", stroke = 0.4, alpha = 0.85)
}

p_world <- ggplot() +
  base_map() +
  mgn_points(7) +
  scale_fill_manual(name = "Family", values = pal) +
  pop_size_scale(c(1e4, 1e6, 1e8), max_size = 7) +
  guides(fill = guide_legend(override.aes = list(size = 4.2), order = 1),
         size = size_guide()) +
  labs(
    title    = "Study 3: the 64 modelled MGN languages",
    subtitle = sprintf("%d languages, %d families | point area is log10 L1 population (%s to %s)",
                       nrow(mgn), n_distinct(mgn$family),
                       label_number(scale_cut = cut_short_scale())(min(mgn$population_l1)),
                       label_number(scale_cut = cut_short_scale())(max(mgn$population_l1)))
  )

# 60 of 64 languages sit in Eurasia and overplot badly on a world map, so the
# same data is redrawn zoomed on Europe/the Caucasus.
p_eur <- ggplot() +
  geom_polygon(data = world, aes(long, lat, group = group),
               fill = "grey93", colour = "grey82", linewidth = 0.2) +
  geom_point(data = mgn, aes(lon, lat, size = population_l1, fill = family_grp),
             shape = 21, colour = "white", stroke = 0.4, alpha = 0.85) +
  scale_fill_manual(values = pal, guide = "none") +
  pop_size_scale(c(1e4, 1e6, 1e8), max_size = 9) +
  guides(size = "none") +
  coord_fixed(1.5, xlim = c(-11, 50), ylim = c(35, 66), expand = FALSE) +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank(), axis.text = element_blank(),
        axis.title = element_blank(),
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = "white", colour = NA),
        plot.title = element_text(face = "bold", size = 11)) +
  labs(title = "Europe and the Caucasus (zoom)")

p_mgn <- p_world / p_eur +
  plot_layout(heights = c(1, 1.05)) +
  plot_annotation(
    caption = paste0("Positions are Glottolog centroids. Coverage is heavily Eurasian: ",
                     "the Americas contribute Navajo and two Chatino languages, Africa only Zulu. ",
                     "43 of 64 languages are Indo-European."),
    theme = theme(plot.caption = element_text(colour = "grey45", size = 8.5, hjust = 0),
                  plot.background = element_rect(fill = "white", colour = NA))
  )

ggsave("plots/map_mgn64.png", p_mgn, width = 12, height = 11, dpi = 200)
cat("Wrote plots/map_mgn64.png\n")

# ---------------------------------------------------------------------------
# Coverage summary to stdout
# ---------------------------------------------------------------------------
cat("\n--- MGN-64 coverage by macro-area ---\n")
print(modeling %>% distinct(iso_sanitized, macro_area) %>% count(macro_area, sort = TRUE))
cat("\n--- MGN-64 families (>= 2 languages) ---\n")
print(as.data.frame(fam_n %>% filter(n >= 2)))
cat("\n--- MGN-64 population extremes ---\n")
print(mgn %>% arrange(population_l1) %>%
        select(lang, language_name, population_l1, family) %>%
        slice(c(1:3, (n() - 2):n())))
