#!/usr/bin/env Rscript
#
# nmf_structure.R - sNMF K-sweep and STRUCTURE-style plots
#
# Runs sNMF (LEA) on a BEAST binary alignment, selects K by min cross-entropy,
# and produces STRUCTURE-style admixture bar plots.
#
# Usage:
#   pixi run nmf_structure <run_id> <subdir> [k_min] [k_max] [n_reps] [--plot]
#
# Examples:
#   pixi run nmf_structure ba9f2d2a 0.05
#   pixi run nmf_structure ba9f2d2a 0.05 2 20 10

suppressPackageStartupMessages({
  library(LEA)
})

# --- CLI arguments ---

BEAST_DIR <- "data/trees/beast"
FASTA_FILE <- "__merged_mapped.fa"

all_args <- commandArgs(trailingOnly = TRUE)
do_plot <- "--plot" %in% all_args
args <- all_args[!all_args %in% c("--plot")]

if (length(args) > 0 && (args[1] == "-h" || args[1] == "--help")) {
  cat(
    "Usage: pixi run nmf_structure <run_id> <subdir> [k_min] [k_max] [n_reps] [--plot]\n\n"
  )
  cat("Arguments:\n")
  cat("  run_id    BEAST run UUID, prefix, or full path\n")
  cat("  subdir    Subdirectory name or prefix\n")
  cat("  k_min     Minimum K (default: 2)\n")
  cat("  k_max     Maximum K (default: 30)\n")
  cat("  n_reps    Repetitions per K (default: 20)\n")
  cat("  --plot    Generate diagnostic plots (CE curve, structure bar chart)\n")
  quit(status = 0)
}

if (length(args) < 2) {
  stop(
    "Usage: pixi run nmf_structure <run_id> <subdir> [k_min] [k_max] [n_reps] [--plot]",
    call. = FALSE
  )
}

run_id <- args[1]
subdir <- args[2]
k_min <- if (length(args) >= 3) as.integer(args[3]) else 2L
k_max <- if (length(args) >= 4) as.integer(args[4]) else 30L
n_reps <- if (length(args) >= 5) as.integer(args[5]) else 20L

alpha <- 10
seed <- 42L

# --- Resolve run_id to BEAST directory ---

if (dir.exists(run_id)) {
  beast_dir <- run_id
} else {
  matches <- Sys.glob(file.path(BEAST_DIR, paste0(run_id, "*")))
  matches <- matches[dir.exists(matches)]
  if (length(matches) == 0) {
    stop(
      sprintf("No BEAST run matching '%s' in %s/", run_id, BEAST_DIR),
      call. = FALSE
    )
  }
  if (length(matches) > 1) {
    stop(
      sprintf(
        "Ambiguous run_id '%s': matches %s",
        run_id,
        paste(matches, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  beast_dir <- matches[1]
}

# Find FASTA alignment in matching subdirectory
fa_hits <- Sys.glob(file.path(beast_dir, paste0(subdir, "*"), FASTA_FILE))
if (length(fa_hits) == 0) {
  stop(
    sprintf("No %s found in %s/%s*/", FASTA_FILE, beast_dir, subdir),
    call. = FALSE
  )
}
if (length(fa_hits) > 1) {
  stop(
    sprintf(
      "Ambiguous subdir '%s': matches %s",
      subdir,
      paste(fa_hits, collapse = ", ")
    ),
    call. = FALSE
  )
}
fa_path <- fa_hits[1]
out_dir <- file.path(dirname(fa_path), "nmf")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("Input:  %s\n", fa_path))
cat(sprintf("Output: %s/\n", out_dir))

# --- Parse FASTA to .geno ---

lines <- readLines(fa_path)
header_idx <- grep("^>", lines)
labels <- sub("^>", "", lines[header_idx])
seqs <- lines[header_idx + 1]

geno_path <- file.path(out_dir, "input.geno")
if (!file.exists(geno_path)) {
  char_matrix <- do.call(rbind, strsplit(seqs, ""))
  char_matrix[char_matrix == "?"] <- "9"
  geno_matrix <- t(char_matrix)
  geno_lines <- apply(geno_matrix, 1, paste0, collapse = "")
  writeLines(geno_lines, geno_path)
  cat(sprintf(
    "Wrote %d sites x %d languages to %s\n",
    nrow(geno_matrix),
    ncol(geno_matrix),
    geno_path
  ))
} else {
  cat(sprintf("Using cached %s\n", geno_path))
}

# --- Run sNMF (or load cached project) ---

proj_path <- file.path(out_dir, "input.snmfProject")
if (file.exists(proj_path)) {
  cat("Loading cached sNMF project...\n")
  proj <- load.snmfProject(proj_path)
} else {
  cat(sprintf(
    "Running sNMF: K=%d:%d, %d reps, alpha=%g...\n",
    k_min,
    k_max,
    n_reps,
    alpha
  ))
  proj <- snmf(
    geno_path,
    K = k_min:k_max,
    repetitions = n_reps,
    alpha = alpha,
    entropy = TRUE,
    ploidy = 1L,
    seed = seed,
    iterations = 2000L,
    project = "new",
    CPU = 4L
  )
}

# --- K selection: argmin of min cross-entropy ---

ks <- k_min:k_max
ce_min <- sapply(ks, function(k) min(cross.entropy(proj, K = k)))
k_star <- ks[which.min(ce_min)]
cat(sprintf(
  "\n>>> K* = %d (min cross-entropy = %.4f)\n\n",
  k_star,
  min(ce_min)
))

# --- Save results for downstream (nmf_brms.R) ---

results <- list(
  labels = labels,
  k_star = k_star,
  cross_entropy = data.frame(K = ks, CE = ce_min),
  Q = list()
)

for (k_val in ks) {
  best_run <- which.min(cross.entropy(proj, K = k_val))
  results$Q[[as.character(k_val)]] <- Q(proj, K = k_val, run = best_run)
}

rds_path <- file.path(out_dir, "snmf_results.rds")
saveRDS(results, rds_path)
cat(sprintf("Saved results to %s\n", rds_path))

# CSV for Python plotting scripts
q_star <- results$Q[[as.character(k_star)]]
q_df <- as.data.frame(q_star)
colnames(q_df) <- paste0("C", seq_len(ncol(q_star)))
q_df$language <- labels
q_df <- q_df[, c("language", paste0("C", seq_len(ncol(q_star))))]
csv_path <- file.path(out_dir, sprintf("Q_K%02d.csv", k_star))
write.csv(q_df, csv_path, row.names = FALSE)
cat(sprintf("Saved %s\n", csv_path))

# Cross-entropy table
ce_df <- data.frame(K = ks, CE = ce_min)
ce_csv_path <- file.path(out_dir, "cross_entropy.csv")
write.csv(ce_df, ce_csv_path, row.names = FALSE)
cat(sprintf("Saved %s\n", ce_csv_path))

# --- Diagnostic plots (optional) ---

if (do_plot) {
  suppressPackageStartupMessages({
    library(ggplot2)
    library(tidyr)
  })

  # Cross-entropy plot
  df_ce <- data.frame(K = ks, CE = ce_min)

  p_ce <- ggplot(df_ce, aes(x = K, y = CE)) +
    geom_point(shape = 95, size = 8, colour = "steelblue") +
    scale_x_continuous(breaks = ks[seq(1, length(ks), 2)]) +
    labs(x = "K", y = "Cross-entropy") +
    theme_classic(base_size = 9, base_family = "Helvetica") +
    theme(
      axis.line = element_line(linewidth = 0.3),
      panel.grid.major = element_line(colour = "grey90", linewidth = 0.3)
    )

  ce_path <- file.path(
    out_dir,
    sprintf("cross_entropy_k%d_k%d.png", k_min, k_max)
  )
  ggsave(ce_path, p_ce, width = 5, height = 4, dpi = 200)
  cat(sprintf("Saved %s\n", ce_path))

  # Structure plot for k_star
  tab20 <- c(
    "#4E79A7",
    "#A0CBE8",
    "#F28E2B",
    "#FFBE7D",
    "#59A14F",
    "#8CD17D",
    "#B6992D",
    "#F1CE63",
    "#499894",
    "#86BCB6",
    "#E15759",
    "#FF9D9A",
    "#79706E",
    "#BAB0AC",
    "#D37295",
    "#FABFD2",
    "#B07AA1",
    "#D4A6C8",
    "#9D7660",
    "#D7B5A6",
  )

  theme_paper <- theme_classic(base_size = 9, base_family = "Helvetica") +
    theme(
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank()
    )

  plot_structure <- function(q_mat, labels, k_val, ce_val = NULL, out_dir) {
    n <- nrow(q_mat)
    k <- ncol(q_mat)

    dom <- apply(q_mat, 1, which.max)
    dom_val <- apply(q_mat, 1, max)
    ord <- order(dom, -dom_val)
    q_mat <- q_mat[ord, , drop = FALSE]
    labels_ord <- labels[ord]

    df <- as.data.frame(q_mat)
    colnames(df) <- paste0("C", seq_len(k))
    df$language <- factor(labels_ord, levels = labels_ord)
    df_long <- pivot_longer(
      df,
      -language,
      names_to = "component",
      values_to = "proportion"
    )
    df_long$component <- factor(
      df_long$component,
      levels = paste0("C", seq_len(k))
    )

    title_str <- if (!is.null(ce_val)) {
      sprintf("K = %d  (CE = %.4f)", k_val, ce_val)
    } else {
      sprintf("K = %d", k_val)
    }

    p <- ggplot(df_long, aes(x = proportion, y = language, fill = component)) +
      geom_col(width = 1) +
      scale_fill_manual(values = rep_len(tab20, k)) +
      scale_x_continuous(expand = c(0, 0)) +
      labs(
        x = "Component proportion",
        y = NULL,
        fill = "Component",
        title = title_str
      ) +
      theme_paper +
      theme(legend.position = "bottom", legend.title = element_text(size = 8))

    out_path <- file.path(out_dir, sprintf("structure_K%02d.png", k_val))
    ggsave(out_path, p, width = 7, height = max(8, n * 0.22), dpi = 200)
    cat(sprintf("Saved %s\n", out_path))
  }

  best_run <- which.min(cross.entropy(proj, K = k_star))
  q_mat <- Q(proj, K = k_star, run = best_run)
  ce_val <- cross.entropy(proj, K = k_star)[best_run]
  plot_structure(q_mat, labels, k_star, ce_val, out_dir)
}

# --- Print cluster assignments for all K ---

for (k_val in ks) {
  best_run <- which.min(cross.entropy(proj, K = k_val))
  q_mat <- Q(proj, K = k_val, run = best_run)
  ce_val <- cross.entropy(proj, K = k_val)[best_run]

  cat(sprintf("K=%2d  best_CE=%.4f  ", k_val, ce_val))
  dom <- apply(q_mat, 1, which.max)
  for (comp in sort(unique(dom))) {
    members <- sort(labels[dom == comp])
    cat(sprintf("\n  %2d: %s", comp, paste(members, collapse = ", ")))
  }
  cat("\n")
}

cat(sprintf("\nDone. Plots and results saved to %s/\n", out_dir))
