# Shared phylogenetic correlation matrix and tree utilities for Grambank and LDL/Wug tracks.
#
# Provides:
#   - build_grafen_corr(tree_or_nwk, glottocodes): prunes a Glottolog topology tree to
#     `glottocodes`, collapses singles, computes Grafen branch lengths, and returns
#     `ape::vcv(tree, corr = TRUE)`.
#   - load_bouckaert_tree(nwk_path, gz_path, apply_split_map): loads the time-calibrated
#     Bouckaert v2.0.0 global language tree (extracting STATE_1 from the Nexus .tree.gz
#     archive if needed) and applies the 4 verified Glottolog 5.3 split mappings.
#   - build_bouckaert_corr(glottocodes, tree, nwk_path): prunes the Bouckaert v2.0.0 tree
#     to `glottocodes`, collapses singles, and returns `ape::vcv(pruned, corr = TRUE)`.

suppressPackageStartupMessages(library(ape))

# Verified 1-to-1 non-colliding mappings from Bouckaert v2.0.0 pre-split/sister tips to Glottolog 5.3:
#   osse1243 (Ossetian)   -> iron1242 (Iron Ossetian)
#   tuuu1240 (Tu/Monguor) -> minh1238 (Mangghuer)
#   ngon1269 (Ngoni)      -> tanz1241 (Tanzanian Ngoni)
#   turo1239 (Turoyo)     -> mlah1239 (Mlahso)
BOUCKAERT_SPLIT_MAP <- c(
  osse1243 = "iron1242",
  tuuu1240 = "minh1238",
  ngon1269 = "tanz1241",
  turo1239 = "mlah1239"
)

build_grafen_corr <- function(tree_or_nwk, glottocodes) {
  raw_tree <- if (inherits(tree_or_nwk, "phylo")) {
    tree_or_nwk
  } else {
    read.tree(as.character(tree_or_nwk))
  }
  pruned <- collapse.singles(keep.tip(raw_tree, glottocodes))
  tree <- compute.brlen(pruned, method = "Grafen")
  A <- vcv(tree, corr = TRUE)
  stopifnot(all(glottocodes %in% rownames(A)))
  A
}

load_bouckaert_tree <- function(nwk_path = NULL, gz_path = NULL, apply_split_map = TRUE) {
  if (is.null(nwk_path)) {
    cand_nwk <- c(
      "analyses/common/data/bouckaert_v2.nwk",
      "../../common/data/bouckaert_v2.nwk",
      "../common/data/bouckaert_v2.nwk",
      "data/bouckaert_v2.nwk"
    )
    found <- cand_nwk[file.exists(cand_nwk)]
    nwk_path <- if (length(found)) found[1] else "analyses/common/data/bouckaert_v2.nwk"
  }
  if (!file.exists(nwk_path)) {
    if (is.null(gz_path)) {
      gz_candidates <- c(
        "analyses/gelato_feasibility_2026_10_01/sources/trees/v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz",
        "../../gelato_feasibility_2026_10_01/sources/trees/v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz",
        "v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz"
      )
      gz_path <- gz_candidates[file.exists(gz_candidates)][1]
    }
    if (is.na(gz_path) || !file.exists(gz_path)) {
      stop("Cannot find Bouckaert v2.0.0 tree at ", nwk_path)
    }
    dir.create(dirname(nwk_path), recursive = TRUE, showWarnings = FALSE)
    con <- gzfile(gz_path, "rt")
    nex_lines <- suppressWarnings(readLines(con))
    close(con)
    tr_start <- grep("^\\s*Translate", nex_lines, ignore.case = TRUE)[1]
    tree_line_idx <- grep("tree STATE_1", nex_lines, fixed = TRUE)[1]
    trans_lines <- trimws(sub(",$", "", nex_lines[(tr_start + 1L):(tree_line_idx - 1L)]))
    trans_lines <- trans_lines[nzchar(trans_lines)]
    parts <- strsplit(trans_lines, "\\s+")
    id_map <- setNames(vapply(parts, function(x) substr(gsub("'", "", x[2]), 1L, 8L), ""),
                       vapply(parts, `[`, "", 1L))
    clean_nwk <- gsub("\\[&[^\\]]*\\]", "", nex_lines[tree_line_idx], perl = TRUE)
    state1_chunk <- strsplit(clean_nwk, ";", fixed = TRUE)[[1]][2]
    state1_nwk <- paste0(sub("^\\s*tree\\s+STATE_1\\s*=\\s*", "", state1_chunk, perl = TRUE), ";")
    m_ids <- gregexpr("(?<=[(,])[0-9]+(?=:)", state1_nwk, perl = TRUE)
    regmatches(state1_nwk, m_ids) <- list(unname(id_map[regmatches(state1_nwk, m_ids)[[1]]]))
    writeLines(state1_nwk, nwk_path)
  }
  tr <- read.tree(nwk_path)
  if (isTRUE(apply_split_map)) {
    for (old_c in names(BOUCKAERT_SPLIT_MAP)) {
      idx <- which(tr$tip.label == old_c)
      if (length(idx) == 1L && !(BOUCKAERT_SPLIT_MAP[old_c] %in% tr$tip.label)) {
        tr$tip.label[idx] <- unname(BOUCKAERT_SPLIT_MAP[old_c])
      }
    }
  }
  tr
}

build_bouckaert_corr <- function(glottocodes, tree = NULL, nwk_path = NULL) {
  if (is.null(tree)) {
    tree <- load_bouckaert_tree(nwk_path = nwk_path, apply_split_map = TRUE)
  }
  pruned <- collapse.singles(keep.tip(tree, glottocodes))
  A <- vcv(pruned, corr = TRUE)
  stopifnot(all(glottocodes %in% rownames(A)))
  A
}
