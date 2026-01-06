library(dplyr)

PER_SENTENCE_DIR <- "data/trees/per_sentence"

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

  output_file <- file.path(run_dir, "_stats.csv")

  write.csv(result, output_file)

  cat("Done.\n")
  cat(paste("View results at", output_file, "\n", sep = " "))
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check for help flag
if (length(args) > 0 && (args[1] == "-h" || args[1] == "--help")) {
  cat("Usage: Rscript sentence_trees_stats.R <run_dir> [pattern]\n\n")
  cat("Arguments:\n")
  cat("  run_dir    Path to the directory containing tree files\n")
  cat("  pattern    File extension to match (default: treefile)\n")
  cat("             Will be automatically prefixed with '*.' if not present\n")
  quit(status = 0)
}

if (length(args) < 1) {
  stop(
    "Usage: Rscript sentence_trees_stats.R <run_dir> [pattern]\nUse -h or --help for more information",
    call. = FALSE
  )
}

run_dir <- args[1]
pattern <- ifelse(length(args) >= 2, args[2], "treefile")

# If pattern doesn't start with "*.", prepend it
if (!grepl("^\\*\\.", pattern)) {
  pattern <- paste0("*.", pattern)
}

# Resolve run_dir: if not a directory, try to find it in PER_SENTENCE_DIR
if (!dir.exists(run_dir)) {
  potential_dirs <- Sys.glob(file.path(PER_SENTENCE_DIR, "*", run_dir))

  if (length(potential_dirs) == 0) {
    stop(paste("Directory not found:", run_dir), call. = FALSE)
  } else if (length(potential_dirs) > 1) {
    stop(
      paste(
        "Multiple directories found for run_dir '",
        run_dir,
        "':\n  ",
        paste(potential_dirs, collapse = "\n  "),
        sep = ""
      ),
      call. = FALSE
    )
  }

  run_dir <- potential_dirs[1]
  cat(paste("Using run directory:", run_dir, "\n"))
}

# Run the function
get_iqtree_stats(run_dir, pattern)
