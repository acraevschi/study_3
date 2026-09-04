#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phylogenetic covariance matrix for the MGN languages
# ---------------------------------------------------------------------------
# Builds a Brownian-motion phylogenetic correlation matrix A from Glottolog
# classification lineages, for use as `gr(iso_sanitized, cov = A)` in brms.
#
# WHY THIS REPLACES THE PREVIOUS VERSION
# The earlier script set A[i, j] <- shared_levels / max_depth, with the diagonal
# forced to 1. That has three problems:
#   1. `max_depth` is the deepest lineage IN THE SAMPLE, so adding or removing a
#      single language rescaled every off-diagonal entry.
#   2. It is not a Brownian-motion covariance, so the random effect it induces has
#      no clean evolutionary interpretation and the phylogenetic variance parameter
#      is not comparable with anything in the literature.
#   3. The diagonal jumped discontinuously from a maximum off-diagonal of 0.81 to
#      exactly 1.0, silently baking in a large independent per-language variance
#      that could not be separated from the phylogenetic component.
#
# Instead we build an explicit tree from the Glottolog lineages, give every
# classification step unit branch length, and take ape::vcv.phylo(corr = TRUE).
# That is the standard BM correlation matrix: entry (i, j) is shared root-to-MRCA
# path length normalised by sqrt(v_ii * v_jj), which is scale-free and does not
# depend on which other languages happen to be in the sample.
#
# The language inventory and MGN -> ISO mapping are read from mgn_language_map.json
# so this script cannot drift from the Python pipeline.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ape)
  library(jsonlite)
})

PROJECT_ROOT <- getwd()
MAP_JSON     <- file.path(PROJECT_ROOT, "mgn_language_map.json")
GLOTTO_CSV   <- file.path(PROJECT_ROOT, "data_sources", "glottolog_extracted.csv")
OUT_RDS      <- file.path(PROJECT_ROOT, "phylo_cov_matrix.rds")

stopifnot(file.exists(MAP_JSON))
lang_map <- jsonlite::fromJSON(MAP_JSON)

# MGN code -> canonical ISO (same mapping the Python pipeline uses)
canon <- unlist(lang_map$mgn_canonical_isos)
isos  <- sort(unique(unname(canon)))
cat(sprintf("Building phylogeny for %d MGN languages (%d distinct ISO codes)\n",
            length(canon), length(isos)))

# ---------------------------------------------------------------------------
# 1. Glottolog lineages
# ---------------------------------------------------------------------------
get_affiliations <- function(codes) {
  if (file.exists(GLOTTO_CSV)) {
    glt <- utils::read.csv(GLOTTO_CSV, stringsAsFactors = FALSE)
    glt <- glt[glt$level == "language", ]
    aff <- glt$affiliation[match(codes, glt$iso)]
    names(aff) <- codes
    if (!any(is.na(aff))) return(aff)
    cat("Glottolog CSV incomplete; falling back to lingtypology for:",
        paste(codes[is.na(aff)], collapse = ", "), "\n")
  } else {
    aff <- stats::setNames(rep(NA_character_, length(codes)), codes)
  }
  if (!requireNamespace("lingtypology", quietly = TRUE)) {
    stop("Glottolog lineages unavailable and lingtypology is not installed.")
  }
  for (code in codes[is.na(aff)]) {
    nm <- try(lingtypology::lang.iso(code), silent = TRUE)
    if (!inherits(nm, "try-error") && !is.na(nm)) {
      a <- try(lingtypology::aff.lang(nm)[1], silent = TRUE)
      if (!inherits(a, "try-error")) aff[code] <- a
    }
  }
  aff
}

aff <- get_affiliations(isos)
if (any(is.na(aff))) {
  stop("No Glottolog lineage for: ", paste(isos[is.na(aff)], collapse = ", "),
       "\nA language with no lineage would otherwise be silently grouped with ",
       "every other unclassified language under a shared 'Isolate' root.")
}

# Lineage as a character vector, with the language itself as the final tip.
lineages <- lapply(isos, function(i) {
  taxa <- trimws(unlist(strsplit(aff[[i]], ",")))
  taxa <- taxa[nzchar(taxa)]
  c(taxa, i)
})
names(lineages) <- isos

depths <- vapply(lineages, length, integer(1))
cat(sprintf("Lineage depth: min %d, median %.1f, max %d\n",
            min(depths), median(depths), max(depths)))

# ---------------------------------------------------------------------------
# 2. Build a Newick tree, one unit branch per classification step
# ---------------------------------------------------------------------------
# Nodes are keyed by their full path so that identically-named subgroups in
# different families (e.g. two families each with a "Central" branch) never merge.
build_newick <- function(lineages) {
  # children[[path_key]] -> named list of child path_keys
  children <- new.env(parent = emptyenv())
  tips     <- character(0)

  add_child <- function(parent_key, child_key) {
    cur <- if (exists(parent_key, envir = children, inherits = FALSE)) {
      get(parent_key, envir = children, inherits = FALSE)
    } else character(0)
    if (!(child_key %in% cur)) assign(parent_key, c(cur, child_key), envir = children)
  }

  for (i in seq_along(lineages)) {
    path <- lineages[[i]]
    keys <- vapply(seq_along(path), function(k) paste(path[1:k], collapse = "|"),
                   character(1))
    add_child("ROOT", keys[1])
    if (length(keys) > 1) {
      for (k in 2:length(keys)) add_child(keys[k - 1], keys[k])
    }
    tips <- c(tips, keys[length(keys)])
  }

  tip_label <- function(key) sub(".*\\|", "", key)

  render <- function(key) {
    kids <- if (exists(key, envir = children, inherits = FALSE)) {
      get(key, envir = children, inherits = FALSE)
    } else character(0)
    if (length(kids) == 0) return(paste0(tip_label(key), ":1"))
    inner <- paste(vapply(kids, render, character(1)), collapse = ",")
    paste0("(", inner, "):1")
  }

  root_kids <- get("ROOT", envir = children, inherits = FALSE)
  paste0("(", paste(vapply(root_kids, render, character(1)), collapse = ","), ");")
}

newick <- build_newick(lineages)
tree <- ape::read.tree(text = newick)
stopifnot(!is.null(tree))
cat(sprintf("Tree built: %d tips, %d internal nodes\n", length(tree$tip.label), tree$Nnode))

missing_tips <- setdiff(isos, tree$tip.label)
if (length(missing_tips)) stop("Tips missing from tree: ", paste(missing_tips, collapse = ", "))

# ---------------------------------------------------------------------------
# 3. Brownian-motion correlation matrix
# ---------------------------------------------------------------------------
A <- ape::vcv.phylo(tree, corr = TRUE)
A <- A[isos, isos]

# Symmetrise against floating-point asymmetry, then check positive definiteness.
A <- (A + t(A)) / 2
ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values

if (min(ev) < 1e-8) {
  # Languages sharing an entire lineage down to the tip give exactly collinear
  # rows. A small ridge is the standard remedy and is reported, not hidden.
  ridge <- 1e-6 + abs(min(ev))
  cat(sprintf("Min eigenvalue %.3e <= 0; adding ridge %.3e to the diagonal\n",
              min(ev), ridge))
  diag(A) <- diag(A) + ridge
  A <- A / mean(diag(A))
  ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values
}

stopifnot(isSymmetric(A, tol = 1e-8), min(ev) > 0)
cat(sprintf("Covariance matrix: %dx%d | min EV %.6f | max EV %.6f | condition %.2f\n",
            nrow(A), ncol(A), min(ev), max(ev), max(ev) / min(ev)))

# Sanity: within-genus pairs must covary more than cross-family pairs.
show_pairs <- list(c("eng", "nld"), c("eng", "deu"), c("rus", "pol"),
                   c("ctp", "czn"), c("tur", "azj"), c("eng", "tur"),
                   c("nav", "eng"), c("zul", "kat"))
for (p in show_pairs) {
  if (all(p %in% rownames(A))) {
    cat(sprintf("  A[%s, %s] = %.4f\n", p[1], p[2], A[p[1], p[2]]))
  }
}

saveRDS(A, OUT_RDS)
cat(sprintf("Saved %s\n", OUT_RDS))
