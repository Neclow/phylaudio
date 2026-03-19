#!/usr/bin/env Rscript
# Compute per-taxon delta scores (Holland et al. 2002) with bootstrap CIs
# from a binary FASTA alignment.
#
# Usage:
#   Rscript pipeline/bootstrap_delta.R <beast_run_dir> [n_boot]
#
# Looks for __merged_mapped.fa inside <beast_run_dir>/0.01_brsupport/,
# saves _delta.csv next to it.
#
# Example:
#   Rscript pipeline/bootstrap_delta.R dd208931-4817-41ad-b18d-aa6a050a3f42

library(parallel)

BEAST_BASE <- "data/trees/beast"
BRSUPPORT  <- "0.01_brsupport"

# ── CLI args ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript pipeline/bootstrap_delta.R <beast_run_dir> [n_boot]")
}

run_dir    <- file.path(BEAST_BASE, args[1], BRSUPPORT)
fasta_file <- file.path(run_dir, "__merged_mapped.fa")
output_csv <- file.path(run_dir, "_delta.csv")
n_boot     <- if (length(args) >= 2) as.integer(args[2]) else 1000L

stopifnot(file.exists(fasta_file))

# ── Read binary FASTA as integer matrix (0/1/NA) ─────────────────────────────
read_fasta_binary <- function(file) {
  lines   <- readLines(file)
  headers <- grep("^>", lines)
  taxa    <- sub("^>\\s*", "", lines[headers])
  starts  <- headers + 1L
  ends    <- c(headers[-1L] - 1L, length(lines))

  seqs <- vapply(seq_along(headers), function(i) {
    paste(lines[starts[i]:ends[i]], collapse = "")
  }, character(1L))

  mat <- do.call(rbind, lapply(strsplit(seqs, ""), function(v) {
    x <- integer(length(v))
    x[v == "1"] <- 1L
    x[v == "?"] <- NA_integer_
    x
  }))
  rownames(mat) <- taxa
  mat
}

# ── Pairwise-complete Hamming distance ────────────────────────────────────────
hamming_dist <- function(mat) {
  obs <- (!is.na(mat)) * 1L
  val <- mat; val[is.na(val)] <- 0L

  valid    <- tcrossprod(obs)
  agree_11 <- tcrossprod(val)
  agree_00 <- tcrossprod(obs - val)
  diff_mat <- valid - agree_11 - agree_00

  D <- diff_mat / valid
  diag(D) <- 0
  rownames(D) <- colnames(D) <- rownames(mat)
  D
}

# ── Per-taxon delta from a distance matrix ────────────────────────────────────
compute_delta <- function(D) {
  n      <- nrow(D)
  quarts <- combn(n, 4L)
  i <- quarts[1L, ]; j <- quarts[2L, ]
  k <- quarts[3L, ]; l <- quarts[4L, ]

  m1 <- D[cbind(i, j)] + D[cbind(k, l)]
  m2 <- D[cbind(i, k)] + D[cbind(j, l)]
  m3 <- D[cbind(i, l)] + D[cbind(j, k)]

  m_max <- pmax(m1, m2, m3)
  m_min <- pmin(m1, m2, m3)
  m_mid <- m1 + m2 + m3 - m_max - m_min

  denom   <- m_max - m_min
  delta_q <- ifelse(denom == 0, 0, (m_max - m_mid) / denom)

  taxon_delta <- vapply(seq_len(n), function(t) {
    in_q <- (i == t) | (j == t) | (k == t) | (l == t)
    mean(delta_q[in_q])
  }, numeric(1L))
  names(taxon_delta) <- rownames(D)

  list(per_taxon = taxon_delta, overall = mean(delta_q), quarts = quarts)
}

# ── Single bootstrap replicate ────────────────────────────────────────────────
boot_delta_one <- function(aln_mat, quarts) {
  L        <- ncol(aln_mat)
  boot_mat <- aln_mat[, sample.int(L, L, replace = TRUE), drop = FALSE]
  D_b      <- hamming_dist(boot_mat)

  n <- nrow(D_b)
  i <- quarts[1L, ]; j <- quarts[2L, ]
  k <- quarts[3L, ]; l <- quarts[4L, ]

  m1 <- D_b[cbind(i, j)] + D_b[cbind(k, l)]
  m2 <- D_b[cbind(i, k)] + D_b[cbind(j, l)]
  m3 <- D_b[cbind(i, l)] + D_b[cbind(j, k)]

  m_max  <- pmax(m1, m2, m3)
  m_min  <- pmin(m1, m2, m3)
  m_mid  <- m1 + m2 + m3 - m_max - m_min
  denom  <- m_max - m_min
  delta_q <- ifelse(denom == 0, 0, (m_max - m_mid) / denom)

  vapply(seq_len(n), function(t) {
    in_q <- (i == t) | (j == t) | (k == t) | (l == t)
    mean(delta_q[in_q])
  }, numeric(1L))
}

# ── Main ──────────────────────────────────────────────────────────────────────
cat("Reading alignment:", fasta_file, "\n")
aln <- read_fasta_binary(fasta_file)
cat(sprintf("Alignment: %d taxa x %d characters\n", nrow(aln), ncol(aln)))

cat("Computing delta scores...\n")
D   <- hamming_dist(aln)
res <- compute_delta(D)
cat(sprintf("Overall mean delta: %.4f\n", res$overall))

n_cores <- min(16L, max(1L, detectCores() - 1L))
cat(sprintf("Running %d bootstrap replicates on %d cores...\n", n_boot, n_cores))

set.seed(42)
t0 <- proc.time()
boot_list <- mclapply(
  seq_len(n_boot),
  function(b) boot_delta_one(aln, res$quarts),
  mc.cores = n_cores
)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Bootstrap done in %.1f s\n", elapsed))

boot_mat <- do.call(rbind, boot_list)
colnames(boot_mat) <- rownames(aln)

ci_lo <- apply(boot_mat, 2, quantile, probs = 0.025)
ci_hi <- apply(boot_mat, 2, quantile, probs = 0.975)

results_df <- data.frame(
  language = rownames(aln),
  delta    = res$per_taxon,
  ci_lo    = ci_lo,
  ci_hi    = ci_hi,
  row.names = NULL
)
results_df <- results_df[order(results_df$delta), ]

cat("\nPer-taxon delta scores (sorted, with 95% bootstrap CIs):\n")
print(results_df, digits = 4, row.names = FALSE)

write.csv(results_df, output_csv, row.names = FALSE)
cat("\nSaved to:", output_csv, "\n")
