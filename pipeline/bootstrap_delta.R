# Compute per-taxon delta scores (Holland et al. 2002) with bootstrap CIs
# from a binary FASTA alignment.
#
# Usage:
#   Rscript pipeline/bootstrap_delta.R <run_id> <subdir> [n_boot] [seed]
#
# Looks for __merged_mapped.fa inside the resolved BEAST directory,
# saves _delta.csv next to it.
#
# Examples:
#   Rscript pipeline/bootstrap_delta.R ba9f2d2a 0.05_brsupport_dev_test
#   Rscript pipeline/bootstrap_delta.R ba9f2d2a 0.01_brsupport 500
#   Rscript pipeline/bootstrap_delta.R ba9f2d2a 0.01_brsupport 1000 123

library(parallel)

BEAST_BASE <- "data/trees/beast"

# ── CLI args ──────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(
    "Usage: Rscript pipeline/bootstrap_delta.R <run_id> <subdir> [n_boot] [seed]"
  )
}

run_id <- args[1]
subdir <- args[2]
n_boot <- if (length(args) >= 3) as.integer(args[3]) else 1000L
seed <- if (length(args) >= 4) as.integer(args[4]) else 42L

# Resolve run_id (prefix match)
run_matches <- Sys.glob(file.path(BEAST_BASE, paste0(run_id, "*")))
run_matches <- run_matches[file.info(run_matches)$isdir]
if (length(run_matches) == 0) {
  stop(sprintf("No BEAST run matching '%s' in %s/", run_id, BEAST_BASE))
}
if (length(run_matches) > 1) {
  stop(sprintf(
    "Ambiguous run_id '%s': %s",
    run_id,
    paste(run_matches, collapse = ", ")
  ))
}
beast_root <- run_matches[1]

# Resolve subdir (prefix match)
subdir_matches <- Sys.glob(file.path(beast_root, paste0(subdir, "*")))
subdir_matches <- subdir_matches[file.info(subdir_matches)$isdir]
if (length(subdir_matches) == 0) {
  stop(sprintf("No subdirectory matching '%s' in %s/", subdir, beast_root))
}
if (length(subdir_matches) > 1) {
  stop(sprintf(
    "Ambiguous subdir '%s': %s",
    subdir,
    paste(subdir_matches, collapse = ", ")
  ))
}
run_dir <- subdir_matches[1]

fasta_file <- file.path(run_dir, "__merged_mapped.fa")
output_csv <- file.path(run_dir, "_delta.csv")

stopifnot(file.exists(fasta_file))

# ── Read binary FASTA as integer matrix (0/1/NA) ─────────────────────────────
read_fasta_binary <- function(file) {
  lines <- readLines(file)
  headers <- grep("^>", lines)
  taxa <- sub("^>\\s*", "", lines[headers])
  starts <- headers + 1L
  ends <- c(headers[-1L] - 1L, length(lines))

  seqs <- vapply(
    seq_along(headers),
    function(i) {
      paste(lines[starts[i]:ends[i]], collapse = "")
    },
    character(1L)
  )

  mat <- do.call(
    rbind,
    lapply(strsplit(seqs, ""), function(v) {
      x <- integer(length(v))
      x[v == "1"] <- 1L
      x[v == "?"] <- NA_integer_
      x
    })
  )
  rownames(mat) <- taxa
  mat
}

# ── Pairwise-complete Hamming distance ────────────────────────────────────────
hamming_dist <- function(mat) {
  obs <- (!is.na(mat)) * 1L
  val <- mat
  val[is.na(val)] <- 0L

  valid <- tcrossprod(obs)
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
  n <- nrow(D)
  quarts <- combn(n, 4L)
  i <- quarts[1L, ]
  j <- quarts[2L, ]
  k <- quarts[3L, ]
  l <- quarts[4L, ]

  m1 <- D[cbind(i, j)] + D[cbind(k, l)]
  m2 <- D[cbind(i, k)] + D[cbind(j, l)]
  m3 <- D[cbind(i, l)] + D[cbind(j, k)]

  m_max <- pmax(m1, m2, m3)
  m_min <- pmin(m1, m2, m3)
  m_mid <- m1 + m2 + m3 - m_max - m_min

  denom <- m_max - m_min
  delta_q <- ifelse(denom == 0, 0, (m_max - m_mid) / denom)

  taxon_delta <- vapply(
    seq_len(n),
    function(t) {
      in_q <- (i == t) | (j == t) | (k == t) | (l == t)
      mean(delta_q[in_q])
    },
    numeric(1L)
  )
  names(taxon_delta) <- rownames(D)

  list(per_taxon = taxon_delta, overall = mean(delta_q), quarts = quarts)
}

# ── Single bootstrap replicate ────────────────────────────────────────────────
boot_delta_one <- function(aln_mat, quarts) {
  L <- ncol(aln_mat)
  boot_mat <- aln_mat[, sample.int(L, L, replace = TRUE), drop = FALSE]
  D_b <- hamming_dist(boot_mat)

  n <- nrow(D_b)
  i <- quarts[1L, ]
  j <- quarts[2L, ]
  k <- quarts[3L, ]
  l <- quarts[4L, ]

  m1 <- D_b[cbind(i, j)] + D_b[cbind(k, l)]
  m2 <- D_b[cbind(i, k)] + D_b[cbind(j, l)]
  m3 <- D_b[cbind(i, l)] + D_b[cbind(j, k)]

  m_max <- pmax(m1, m2, m3)
  m_min <- pmin(m1, m2, m3)
  m_mid <- m1 + m2 + m3 - m_max - m_min
  denom <- m_max - m_min
  delta_q <- ifelse(denom == 0, 0, (m_max - m_mid) / denom)

  vapply(
    seq_len(n),
    function(t) {
      in_q <- (i == t) | (j == t) | (k == t) | (l == t)
      mean(delta_q[in_q])
    },
    numeric(1L)
  )
}

# ── Main ──────────────────────────────────────────────────────────────────────
cat("Reading alignment:", fasta_file, "\n")
aln <- read_fasta_binary(fasta_file)
cat(sprintf("Alignment: %d taxa x %d characters\n", nrow(aln), ncol(aln)))

cat("Computing delta scores...\n")
D <- hamming_dist(aln)
res <- compute_delta(D)
cat(sprintf("Overall mean delta: %.4f\n", res$overall))

n_cores <- min(16L, max(1L, detectCores() - 1L))
cat(sprintf(
  "Running %d bootstrap replicates on %d cores...\n",
  n_boot,
  n_cores
))

set.seed(seed)
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
  delta = res$per_taxon,
  ci_lo = ci_lo,
  ci_hi = ci_hi,
  row.names = NULL
)
results_df <- results_df[order(results_df$delta), ]

cat("\nPer-taxon delta scores (sorted, with 95% bootstrap CIs):\n")
print(results_df, digits = 4, row.names = FALSE)

write.csv(results_df, output_csv, row.names = FALSE)
cat("\nSaved to:", output_csv, "\n")
