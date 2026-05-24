#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(DSS)
})

option_list <- list(
  make_option("--sample-sheet", dest = "sample_sheet", type = "character", help = "TSV: sample, group, counts_path"),
  make_option("--output-dir", dest = "output_dir", type = "character", help = "DMR output directory"),
  make_option("--target-group", dest = "target_group", type = "character", default = "T"),
  make_option("--background-group", dest = "background_group", type = "character", default = "N"),
  make_option("--top-n", dest = "top_n", type = "integer", default = 100),
  make_option("--delta", dest = "delta", type = "double", default = 0.2),
  make_option("--p-threshold", dest = "p_threshold", type = "double", default = 0.05),
  make_option("--min-cpg", dest = "min_cpg", type = "integer", default = 4),
  make_option("--min-len", dest = "min_len", type = "integer", default = 50),
  make_option("--merge-distance", dest = "merge_distance", type = "integer", default = 50),
  make_option("--smoothing", action = "store_true", default = TRUE),
  make_option("--no-smoothing", dest = "no_smoothing", action = "store_true", default = FALSE)
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$sample_sheet) || is.null(opt$output_dir)) {
  stop("--sample-sheet and --output-dir are required")
}
if (isTRUE(opt$no_smoothing)) {
  opt$smoothing <- FALSE
}

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

samples <- read.delim(opt$sample_sheet, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("sample", "group", "counts_path")
missing <- setdiff(required, names(samples))
if (length(missing) > 0) {
  stop("sample sheet missing columns: ", paste(missing, collapse = ", "))
}

samples <- samples[samples$group %in% c(opt$target_group, opt$background_group), ]
if (sum(samples$group == opt$target_group) < 1) {
  stop("no target-group samples found: ", opt$target_group)
}
if (sum(samples$group == opt$background_group) < 1) {
  stop("no background-group samples found: ", opt$background_group)
}

message("Reading ", nrow(samples), " DSS count files")
dat <- lapply(samples$counts_path, function(path) {
  if (!file.exists(path)) {
    stop("missing count file: ", path)
  }
  x <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  missing_cols <- setdiff(c("chr", "pos", "N", "X"), names(x))
  if (length(missing_cols) > 0) {
    stop("count file ", path, " missing columns: ", paste(missing_cols, collapse = ", "))
  }
  x[, c("chr", "pos", "N", "X")]
})

bs <- makeBSseqData(dat, samples$sample)
target_samples <- samples$sample[samples$group == opt$target_group]
background_samples <- samples$sample[samples$group == opt$background_group]

message("Running DSS DMLtest: ", length(target_samples), " target vs ",
        length(background_samples), " background")
dml <- DMLtest(
  BSobj = bs,
  group1 = target_samples,
  group2 = background_samples,
  smoothing = opt$smoothing
)
write.table(
  dml,
  file = file.path(opt$output_dir, "dss_dml_test.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

message("Calling DMRs")
dmr <- callDMR(
  dml,
  p.threshold = opt$p_threshold,
  delta = opt$delta,
  minlen = opt$min_len,
  minCG = opt$min_cpg,
  dis.merge = opt$merge_distance
)

if (nrow(dmr) == 0) {
  stop("DSS returned zero DMRs")
}

if (!"areaStat" %in% names(dmr)) {
  dmr$areaStat <- dmr$stat
}
if (!"diff.Methy" %in% names(dmr)) {
  dmr$diff.Methy <- dmr$meanMethy1 - dmr$meanMethy2
}

dmr$abs_areaStat <- abs(dmr$areaStat)
dmr <- dmr[order(dmr$abs_areaStat, decreasing = TRUE), ]
dmr$ctype <- opt$target_group
dmr$dmr_id <- seq_len(nrow(dmr)) - 1L

all_path <- file.path(opt$output_dir, "dss_dmrs.tsv")
write.table(
  dmr,
  file = all_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

top <- head(dmr, opt$top_n)
top_path <- file.path(opt$output_dir, sprintf("dmrs_top%d.tsv", opt$top_n))
write.table(
  top,
  file = top_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

message("Wrote all DMRs: ", all_path)
message("Wrote top DMRs: ", top_path)
