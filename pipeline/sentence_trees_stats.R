# Per-sentence tree quality statistics.
#
# Inputs:  a per-sentence tree directory name
# Options: --include/--exclude splits
# Flow:    Load per-sentence .treefile outputs
#                     |
#                     v
#          Compute branch support, clock-likeness, stemminess, n_tips
# Outputs: _stats.csv per run directory

library(dplyr)

PER_SENTENCE_DIR <- "data/trees/per_sentence"
OUTPUT_FILE <- "_stats.csv"

source("src/tasks/phylo/stats.R")

get_iqtree_stats <- function(run_dir, pattern = "*.treefile", splits = NULL) {
  files <- list.files(path = run_dir, pattern = pattern, full.names = TRUE)

  if (!is.null(splits)) {
    split_regex <- paste0("^(", paste(splits, collapse = "|"), ")_")
    files <- files[grepl(split_regex, basename(files))]
  }

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

  if (!is.null(splits)) {
    splits_label <- paste(sort(splits), collapse = "_")
    output_file <- file.path(run_dir, sub("\\.csv$", paste0("_", splits_label, ".csv"), OUTPUT_FILE))
  } else {
    output_file <- file.path(run_dir, OUTPUT_FILE)
  }

  write.csv(result, output_file)

  cat("Done.\n")
  cat(paste("View results at", output_file, "\n", sep = " "))
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check for help flag
if (length(args) > 0 && (args[1] == "-h" || args[1] == "--help")) {
  cat("Usage: pixi run sentence_stats <dirname> [pattern]\n\n")
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
  cat("  --splits s1,s2  Only include files from these splits (comma-separated)\n")
  cat("  --overwrite     Overwrite existing ", OUTPUT_FILE, " files\n", sep = "")
  cat("\nExample:\n")
  cat("  pixi run sentence_stats discrete\n")
  quit(status = 0)
}

if (length(args) < 1) {
  stop(
    "Usage: pixi run sentence_stats <dirname> [pattern]\nUse -h or --help for more information",
    call. = FALSE
  )
}

# Parse arguments
overwrite <- "--overwrite" %in% args

splits <- NULL
splits_idx <- which(args == "--splits")
if (length(splits_idx) > 0) {
  splits <- strsplit(args[splits_idx[1] + 1], ",")[[1]]
  args <- args[-c(splits_idx[1], splits_idx[1] + 1)]
}

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
  if (!is.null(splits)) {
    splits_label <- paste(sort(splits), collapse = "_")
    out_name <- sub("\\.csv$", paste0("_", splits_label, ".csv"), OUTPUT_FILE)
  } else {
    out_name <- OUTPUT_FILE
  }
  if (!overwrite && file.exists(file.path(run_dir, out_name))) {
    cat(paste("  ", out_name, "already exists. Skipping...\n"))
    next
  }

  tryCatch(
    get_iqtree_stats(run_dir, pattern, splits),
    error = function(e) cat(paste("  Skipped:", e$message, "\n"))
  )
}
