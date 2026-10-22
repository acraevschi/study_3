# f3 admixture score per GeLaTo population (EXPLORATORY; README.md section 2).
#
# For each target population A that links to a main-set language:
#   f3(A; B, C)   B = every other population of A's Glottolog family (a proxy for A's own
#                     ancestry), C = every population of a *different* family within
#                     MAX_KM of A (a proxy for contact partners).
# A clearly negative f3 (Z < -3) is evidence that A is a mix of a B-like and a C-like group.
# Per target we keep the triple with the lowest Z, plus the number of triples and the
# number with Z < -3. Populations without a family are never B or C; without coordinates
# never C (coordinate fallbacks: see below; `coord_source` is written to the output).
# Reads the precomputed F2 blocks of Graff et al. 2025 (no genotypes).
# With k (sensitivity check, 2026-10-21): only the k nearest B and the k nearest C are tried, so
# well-sampled populations do not get hundreds of extra chances (README section 4).
#
# Run (from the repo root):
#   Rscript analyses/admixture_links_2026_10_21/f3.R <F2results dir> [max_km] [k]
suppressPackageStartupMessages(library(admixtools))
args <- commandArgs(trailingOnly = TRUE)
f2dir <- args[1]
MAX_KM <- if (length(args) >= 2) as.numeric(args[2]) else 3000
K_NEAR <- if (length(args) >= 3) as.integer(args[3]) else NA   # NA: every B and every C
here <- "analyses/admixture_links_2026_10_21"
src <- "analyses/gelato_feasibility_2026_10_01/sources"

m <- read.csv(file.path(src, "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input/GeLaTo-population-glottocode-mapping.csv"),
              stringsAsFactors = FALSE)
g <- read.csv(file.path(src, "gelato_c625fdc/cldf/populations.csv"), stringsAsFactors = FALSE)
pops_all <- setdiff(sub("\\.DS_Store", "", list.files(f2dir)), c("", grep("^block_lengths", list.files(f2dir), value = TRUE)))
meta <- m[m$PopName %in% pops_all, c("PopName", "lat", "lon", "glottologFamily")]
names(meta) <- c("pop", "lat", "lon", "family")
meta$family[is.na(meta$family)] <- ""

al <- read.csv("outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv", stringsAsFactors = FALSE)
al <- al[al$analysis == "main" & al$in_grambank == "True" & al$meets_coverage_main == "True", ]
targets <- sort(unique(al$population))
stopifnot(all(targets %in% meta$pop))

# coordinates, in this order: the Graff mapping; GeLaTo; the Glottolog point of the population's
# own base language (glottocodeBase); for a target only, the Glottolog point of the language it links to
meta$coord_source <- ifelse(is.na(meta$lat) | is.na(meta$lon), NA_character_, "graff_mapping")
fb <- g[match(meta$pop, g$Name), c("Latitude", "Longitude")]
need <- is.na(meta$coord_source) & !is.na(fb$Latitude) & !is.na(fb$Longitude)
meta$lat[need] <- fb$Latitude[need]; meta$lon[need] <- fb$Longitude[need]; meta$coord_source[need] <- "gelato"
glot <- read.csv("external/glottolog-cldf/cldf/languages.csv", stringsAsFactors = FALSE)
use_glottolog <- function(codes, rows, label) {
  i <- match(codes, glot$ID)
  lat <- suppressWarnings(as.numeric(glot$Latitude[i])); lon <- suppressWarnings(as.numeric(glot$Longitude[i]))
  ok <- rows & !is.na(lat) & !is.na(lon)
  meta$lat[ok] <<- lat[ok]; meta$lon[ok] <<- lon[ok]; meta$coord_source[ok] <<- label
}
use_glottolog(m$glottocodeBase[match(meta$pop, m$PopName)], is.na(meta$coord_source), "glottolog_base_language")
use_glottolog(al$glottocode[match(meta$pop, al$population)], is.na(meta$coord_source) & meta$pop %in% targets,
              "glottolog_linked_language")
message("coordinate sources: ", paste(names(table(meta$coord_source)), table(meta$coord_source), collapse = ", "),
        "; still without: ", sum(is.na(meta$coord_source)), " (targets: ", sum(is.na(meta$coord_source) & meta$pop %in% targets), ")")

hav <- function(lat1, lon1, lat2, lon2) {
  r <- pi / 180; a <- sin((lat2 - lat1) * r / 2)^2 + cos(lat1 * r) * cos(lat2 * r) * sin((lon2 - lon1) * r / 2)^2
  12742 * asin(pmin(1, sqrt(a)))
}

t0 <- Sys.time()
f2 <- f2_from_precomp(f2dir, pops = meta$pop, verbose = FALSE)
message("F2 blocks loaded: ", paste(dim(f2), collapse = " x "), " in ",
        round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), " min")

out <- list()
for (a in targets) {
  ma <- meta[meta$pop == a, ]
  row <- data.frame(population = a, family = ma$family, coord_source = ma$coord_source, n_B_avail = 0L, n_C_avail = 0L,
                    n_B = 0L, n_C = 0L, n_triples = 0L, n_z_lt_m3 = 0L,
                    min_pop2 = NA_character_, min_pop3 = NA_character_, min_est = NA_real_, min_se = NA_real_,
                    min_z = NA_real_, stringsAsFactors = FALSE)
  if (nzchar(ma$family) && !is.na(ma$lat)) {
    B <- meta$pop[meta$family == ma$family & meta$pop != a]
    ok <- nzchar(meta$family) & meta$family != ma$family & !is.na(meta$lat)
    C <- meta$pop[ok & hav(ma$lat, ma$lon, meta$lat, meta$lon) <= MAX_KM]
    row$n_B_avail <- length(B); row$n_C_avail <- length(C)
    if (!is.na(K_NEAR)) {           # the k nearest; a relative without coordinates is never among them
      near <- function(p) { i <- match(p, meta$pop); dk <- hav(ma$lat, ma$lon, meta$lat[i], meta$lon[i])
                            head(p[!is.na(dk)][order(dk[!is.na(dk)])], K_NEAR) }
      B <- near(B); C <- near(C)
    }
    row$n_B <- length(B); row$n_C <- length(C)
    if (length(B) && length(C)) {
      r <- suppressWarnings(f3(f2, pop1 = a, pop2 = B, pop3 = C, verbose = FALSE))
      r <- r[is.finite(r$z), ]
      row$n_triples <- nrow(r); row$n_z_lt_m3 <- sum(r$z < -3)
      if (nrow(r)) { i <- which.min(r$z); row[c("min_pop2", "min_pop3", "min_est", "min_se", "min_z")] <-
        list(r$pop2[i], r$pop3[i], r$est[i], r$se[i], r$z[i]) }
    }
  }
  out[[a]] <- row
  if (length(out) %% 25 == 0) message(format(Sys.time(), "%H:%M"), " ", length(out), "/", length(targets))
}
res <- do.call(rbind, out)
res$max_km <- MAX_KM; res$k_near <- K_NEAR
write.csv(res, file.path(here, if (is.na(K_NEAR)) "f3_scores.csv" else sprintf("f3_scores_k%d.csv", K_NEAR)), row.names = FALSE)
message("done: ", sum(res$n_triples > 0), " of ", nrow(res), " targets have at least one triple")
