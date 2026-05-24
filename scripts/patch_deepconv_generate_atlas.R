#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: patch_deepconv_generate_atlas.R <input-generate_atlas.R> <output-generate_atlas.R>")
}

input <- args[[1]]
output <- args[[2]]

if (!file.exists(input)) {
  stop("input script does not exist: ", input)
}

text <- paste(readLines(input, warn = FALSE), collapse = "\n")

old <- paste(c(
  "  split_regions  <- function(ref_start, ref_end, starts, ends) {",
  "      if (is.na(starts) || is.na(ends) || is.null(starts) || is.null(ends) || ",
  "          length(starts)==0 || length(ends)==0) {",
  "          return(list(start=ref_start, end=ref_end))",
  "      }",
  "      ",
  "      rel_start <- starts-ref_start + 1",
  "      rel_end <- ends-ref_start + 1",
  "      rel_start[rel_start < 1] <- 1",
  "      rel_end[rel_end > ref_end-ref_start + 1] <- ref_end-ref_start + 1",
  "      mask <- rep(TRUE, ref_end-ref_start + 1)",
  "      ",
  "      for (i in 1:length(rel_end)) {",
  "          mask[rel_start[i]:rel_end[i]] <- FALSE",
  "      }",
  "      ",
  "      remaining_indices <- which(mask)",
  "      clusters <- cumsum(c(TRUE, diff(remaining_indices)!=1))",
  "      output <- list(start=c(), end=c())"
), collapse = "\n")

new <- paste(c(
  "  split_regions  <- function(ref_start, ref_end, starts, ends) {",
  "      if (is.null(starts) || is.null(ends) ||",
  "          length(starts)==0 || length(ends)==0) {",
  "          return(list(start=ref_start, end=ref_end))",
  "      }",
  "",
  "      valid <- !is.na(starts) & !is.na(ends)",
  "      starts <- starts[valid]",
  "      ends <- ends[valid]",
  "      if (length(starts)==0 || length(ends)==0) {",
  "          return(list(start=ref_start, end=ref_end))",
  "      }",
  "      ",
  "      rel_start <- starts-ref_start + 1",
  "      rel_end <- ends-ref_start + 1",
  "      rel_start[rel_start < 1] <- 1",
  "      rel_end[rel_end > ref_end-ref_start + 1] <- ref_end-ref_start + 1",
  "      mask <- rep(TRUE, ref_end-ref_start + 1)",
  "      ",
  "      for (i in seq_along(rel_end)) {",
  "          mask[rel_start[i]:rel_end[i]] <- FALSE",
  "      }",
  "      ",
  "      remaining_indices <- which(mask)",
  "      if (length(remaining_indices)==0) {",
  "          return(list(start=integer(), end=integer()))",
  "      }",
  "      clusters <- cumsum(c(TRUE, diff(remaining_indices)!=1))",
  "      output <- list(start=c(), end=c())"
), collapse = "\n")

if (grepl(old, text, fixed = TRUE)) {
  text <- sub(old, new, text, fixed = TRUE)
} else if (!grepl(new, text, fixed = TRUE)) {
  stop("could not find the expected split_regions block to patch in: ", input)
}

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
writeLines(strsplit(text, "\n", fixed = TRUE)[[1]], output, useBytes = TRUE)
Sys.chmod(output, mode = "0755")
message("wrote patched generate_atlas.R: ", output)
