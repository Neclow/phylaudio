library(dplyr)

PER_SENTENCE_DIR <- "data/trees/per_sentence"
OUTPUT_FILE <- "_stats.csv"

source("src/tasks/phylo/stats.R")

get_iqtree_stats <- function(run_dir, pattern = "*.treefile") {
  files <- list.files(path = run_dir, pattern = pattern, full.names = TRUE)

  if (length(files) == 0) {
    stop(
      paste(
        "No files found in ",
        run_dir,
        " (pattern: ",
        pattern,
        ")",
        sep = ""
      ),
      call. = FALSE
    )
  }

  cat(paste("Found", length(files), "files.\n", sep = " "))
  cat("Extracting stats...\n")

  trs <- lapply(files, function(x) read.tree(x))

  trs_meansup <- sapply(trs, function(x) {
    mean(as.numeric(x$node.label), na.rm = TRUE)
  })
  trs_rttcov <- sapply(trs, rtt_cov)
  trs_stemmy <- sapply(trs, function(x) stemmy(x))
  trs_ntip <- sapply(trs, Ntip)

  tree_stats <- cbind(trs_meansup, trs_rttcov, trs_stemmy, trs_ntip)
  colnames(tree_stats) <- c("brsupport", "clock", "stemmy", "Ntips")

  # Use file stem (basename without extension) for rownames
  rownames(tree_stats) <- tools::file_path_sans_ext(basename(files))

  # Sort by clock-likeness (lower = more clock-like)
  result <- as.data.frame(tree_stats) %>% arrange(clock)

  output_file <- file.path(run_dir, OUTPUT_FILE)

  write.csv(result, output_file)

  cat("Done.\n")
  cat(paste("View results at", output_file, "\n", sep = " "))
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check for help flag
if (length(args) > 0 && (args[1] == "-h" || args[1] == "--help")) {
  cat("Usage: Rscript sentence_trees_stats.R <dirname> [pattern]\n\n")
  cat("Arguments:\n")
  cat(
    "  dirname    Name of the subdirectory in ",
    PER_SENTENCE_DIR,
    "\n",
    sep = ""
  )
  cat("  pattern    File extension to match (default: treefile)\n")
  cat("             Will be automatically prefixed with '*.' if not present\n")
  cat("\nOptions:\n")
  cat("  --overwrite  Overwrite existing ", OUTPUT_FILE, " files\n", sep = "")
  cat("\nExample:\n")
  cat("  Rscript sentence_trees_stats.R discrete\n")
  quit(status = 0)
}

if (length(args) < 1) {
  stop(
    "Usage: Rscript sentence_trees_stats.R <dirname> [pattern]\nUse -h or --help for more information",
    call. = FALSE
  )
}

# Parse arguments
overwrite <- "--overwrite" %in% args
positional_args <- args[!grepl("^--", args)]

dirname <- positional_args[1]
pattern <- ifelse(length(positional_args) >= 2, positional_args[2], "treefile")

# If pattern doesn't start with "*.", prepend it
if (!grepl("^\\*\\.", pattern)) {
  pattern <- paste0("*.", pattern)
}

# Resolve dirname to full path under PER_SENTENCE_DIR
input_dir <- file.path(PER_SENTENCE_DIR, dirname)

if (!dir.exists(input_dir)) {
  stop(paste("Directory not found:", input_dir), call. = FALSE)
}

# Find all subdirectories (UUID run dirs)
run_dirs <- list.dirs(input_dir, recursive = FALSE, full.names = TRUE)

if (length(run_dirs) == 0) {
  stop(paste("No subdirectories found in", input_dir), call. = FALSE)
}

cat(paste("Found", length(run_dirs), "run directories in", input_dir, "\n"))

# Run stats on each subdirectory
for (i in seq_along(run_dirs)) {
  run_dir <- run_dirs[i]
  cat(paste0("\n[", i, "/", length(run_dirs), "] ", basename(run_dir), "\n"))

  # Skip if output already exists
  if (!overwrite && file.exists(file.path(run_dir, OUTPUT_FILE))) {
    cat(paste("  ", OUTPUT_FILE, "already exists. Skipping...\n"))
    next
  }

  tryCatch(
    get_iqtree_stats(run_dir, pattern),
    error = function(e) cat(paste("  Skipped:", e$message, "\n"))
  )
}
