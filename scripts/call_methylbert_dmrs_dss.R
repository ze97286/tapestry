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
  make_option("--chrom", dest = "chrom", type = "character", default = NULL,
              help = "Optional chromosome to run from a chromosome-level sample sheet"),
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
if (!is.null(opt$chrom)) {
  if (!"chrom" %in% names(samples)) {
    stop("--chrom requires a sample sheet with a chrom column")
  }
  samples <- samples[samples$chrom == opt$chrom, ]
  if (nrow(samples) == 0) {
    stop("no sample rows found for chrom: ", opt$chrom)
  }
}
if (sum(samples$group == opt$target_group) < 1) {
  stop("no target-group samples found: ", opt$target_group)
}
if (sum(samples$group == opt$background_group) < 1) {
  stop("no background-group samples found: ", opt$background_group)
}

read_counts <- function(path) {
  if (!file.exists(path)) {
    stop("missing count file: ", path)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    x <- as.data.frame(data.table::fread(
      path,
      select = c("chr", "pos", "N", "X"),
      showProgress = FALSE
    ))
  } else {
    x <- read.delim(path, stringsAsFactors = FALSE, check.names = FALSE)
  }
  missing_cols <- setdiff(c("chr", "pos", "N", "X"), names(x))
  if (length(missing_cols) > 0) {
    stop("count file ", path, " missing columns: ", paste(missing_cols, collapse = ", "))
  }
  x[, c("chr", "pos", "N", "X")]
}

run_dss <- function(sample_chunk, label, append_dml) {
  message("Reading ", nrow(sample_chunk), " DSS count files", label)
  dat <- lapply(sample_chunk$counts_path, read_counts)
  bs <- makeBSseqData(dat, sample_chunk$sample)
  target_samples <- sample_chunk$sample[sample_chunk$group == opt$target_group]
  background_samples <- sample_chunk$sample[sample_chunk$group == opt$background_group]

  message("Running DSS DMLtest", label, ": ", length(target_samples), " target vs ",
          length(background_samples), " background")
  dml <- DMLtest(
    BSobj = bs,
    group1 = target_samples,
    group2 = background_samples,
    smoothing = opt$smoothing
  )

  dml_path <- file.path(opt$output_dir, "dss_dml_test.tsv")
  write.table(
    dml,
    file = dml_path,
    sep = "\t",
    row.names = FALSE,
    quote = FALSE,
    append = append_dml,
    col.names = !append_dml
  )

  message("Calling DMRs", label)
  dmr <- callDMR(
    dml,
    p.threshold = opt$p_threshold,
    delta = opt$delta,
    minlen = opt$min_len,
    minCG = opt$min_cpg,
    dis.merge = opt$merge_distance
  )
  rm(dat, bs, dml)
  gc()
  dmr
}

if ("chrom" %in% names(samples)) {
  dml_path <- file.path(opt$output_dir, "dss_dml_test.tsv")
  if (file.exists(dml_path)) {
    invisible(file.remove(dml_path))
  }
  dmrs <- list()
  append_dml <- FALSE
  chromosomes <- unique(samples$chrom)
  for (chrom in chromosomes) {
    sample_chunk <- samples[samples$chrom == chrom, ]
    label <- paste0(" for ", chrom)
    if (sum(sample_chunk$group == opt$target_group) < 1 ||
        sum(sample_chunk$group == opt$background_group) < 1) {
      warning("Skipping ", chrom, ": missing target or background samples")
      next
    }
    dmr <- run_dss(sample_chunk, label, append_dml)
    append_dml <- TRUE
    if (!is.null(dmr) && nrow(dmr) > 0) {
      dmrs[[chrom]] <- dmr
    }
  }
  if (length(dmrs) == 0) {
    message("DSS returned zero DMRs")
    empty_dmr <- data.frame(
      chr = character(),
      start = integer(),
      end = integer(),
      areaStat = numeric(),
      stat = numeric(),
      diff.Methy = numeric(),
      abs_areaStat = numeric(),
      ctype = character(),
      dmr_id = integer()
    )
    all_path <- file.path(opt$output_dir, "dss_dmrs.tsv")
    top_path <- file.path(opt$output_dir, sprintf("dmrs_top%d.tsv", opt$top_n))
    write.table(empty_dmr, file = all_path, sep = "\t", row.names = FALSE, quote = FALSE)
    write.table(empty_dmr, file = top_path, sep = "\t", row.names = FALSE, quote = FALSE)
    message("Wrote empty DMR files: ", all_path)
    quit(save = "no", status = 0)
  }
  dmr <- do.call(rbind, dmrs)
} else {
  dmr <- run_dss(samples, "", FALSE)
  if (is.null(dmr) || nrow(dmr) == 0) {
    stop("DSS returned zero DMRs")
  }
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
