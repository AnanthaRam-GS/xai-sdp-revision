# Scott-Knott ESD cross-check (step 8): runs ScottKnottESD::sk_esd on wide CSV files.
# Usage: Rscript scripts/skesd_crosscheck.R <input_dir> <output_dir> <libraries>
# <libraries> is a ":"-separated list of user libraries; the package is loaded from the
# first one. Each input CSV has one column per treatment (syntactic names) and one row per
# block; the output CSV lists the treatment and its group (1 = best).
args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]; out_dir <- args[2]; libs <- strsplit(args[3], ":", fixed = TRUE)[[1]]
.libPaths(c(libs, .libPaths()))  # user libraries hold the package and its dependencies
suppressMessages(library(ScottKnottESD, lib.loc = libs[1]))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
for (f in list.files(in_dir, pattern = "\\.csv$", full.names = TRUE)) {
  x <- read.csv(f, check.names = FALSE)
  sk <- sk_esd(x, alpha = 0.05)
  out <- data.frame(treatment = names(sk$groups), group = as.integer(sk$groups))
  write.csv(out, file.path(out_dir, basename(f)), row.names = FALSE)
}
cat("ScottKnottESD", as.character(packageVersion("ScottKnottESD", lib.loc = libs[1])), "\n")
