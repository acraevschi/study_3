# Runner for the demographic extension (log_pop, log_area, and interactions with x)
# across outcomes (default: cat12, flex, irreg13) on the main link set in plain mode.
# Strictly capped at 4 CPU cores (1 fit at a time x 4 chains x 1 thread).
#
# Usage:
#   Rscript analyses/grambank/baseline/run_demog.R [ctrls] [outcomes] [measures] [demog_mode]
# Examples:
#   Rscript analyses/grambank/baseline/run_demog.R N,P cat12,flex,irreg13 ent,het,f3k_est both
#   Rscript analyses/grambank/baseline/run_demog.R F cat12,flex,irreg13 ent,het,f3k_est both

args <- commandArgs(trailingOnly = TRUE)
ctrls <- if (length(args) >= 1L && nzchar(args[1])) args[1] else "N,P"
outcomes <- strsplit(if (length(args) >= 2L && nzchar(args[2])) args[2] else "cat12,flex,irreg13", ",")[[1]]
measures <- if (length(args) >= 3L && nzchar(args[3])) args[3] else "ent,het,f3k_est"
dmode <- if (length(args) >= 4L && nzchar(args[4])) args[4] else "both"

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else "analyses/grambank/baseline"
fit_script <- file.path(here, "fit.R")

for (out in outcomes) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " === Running outcome=", out, " ctrls=", ctrls, " measures=", measures, " demog=", dmode, " ===")
  Sys.setenv(
    ADM_DEMOG = dmode,
    ADM_OUTCOME = out,
    ADM_MIN_COV = "0.6",
    ADM_SAMPLE = "own",
    ADM_XMODE = "plain",
    ADM_MEASURES = measures,
    ADM_CTRLS = ctrls,
    ADM_PAR_FITS = "1",
    ADM_MAX_CORES = "4",
    ADM_REUSE_STAGE1 = "1"
  )
  status <- system2("Rscript", c(fit_script, "main"))
  if (status != 0L) stop("fit.R failed for outcome ", out, " with exit status ", status)
}
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " === ALL OUTCOMES DONE ===")
