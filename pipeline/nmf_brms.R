# nmf_brms.R - Bayesian regression of NMF components on PHOIBLE features
#
# Fits one brms model per NMF component (K separate regressions).
# Predictors: has_{feat} binary indicators + n_phonemes, n_consonants, n_vowels.
# No phylogenetic covariance — just standard Bayesian linear regression.
#
# Usage:
#   pixi run nmf_brms <run_id> <subdir> [K] [dataset]
#
# Examples:
#   pixi run nmf_brms ba9f 0.05
#   pixi run nmf_brms ba9f 0.05 12
#   pixi run nmf_brms ba9f 0.05 12 fleurs-r

# Setup

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

# CLI Arguments

BEAST_DIR <- "data/trees/beast"
METADATA_DIR <- "data/metadata"

args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0 && (args[1] == "-h" || args[1] == "--help")) {
  cat(
    "Usage: pixi run nmf_brms <run_id> <subdir> [K] [dataset]\n\n"
  )
  cat("Arguments:\n")
  cat("  run_id    BEAST run UUID, prefix, or full path\n")
  cat("  subdir    Subdirectory name or prefix (e.g. 0.05)\n")
  cat("  K         Number of components (default: k_star from sweep)\n")
  cat("  dataset   Dataset name (default: fleurs-r)\n")
  quit(status = 0)
}

if (length(args) < 2) {
  stop(
    "Usage: pixi run nmf_brms <run_id> <subdir> [K] [dataset]",
    call. = FALSE
  )
}

run_id <- args[1]
subdir <- args[2]
k_override <- if (length(args) >= 3) as.integer(args[3]) else NULL
dataset <- ifelse(length(args) >= 4, args[4], "fleurs-r")

# Resolve run_id to BEAST directory
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

# Find sNMF results RDS in matching subdirectory
rds_hits <- Sys.glob(file.path(beast_dir, paste0(subdir, "*"), "nmf", "snmf_results.rds"))
if (length(rds_hits) == 0) {
  stop(sprintf("No snmf_results.rds found in %s/%s*/nmf/", beast_dir, subdir), call. = FALSE)
}
if (length(rds_hits) > 1) {
  stop(
    sprintf("Ambiguous subdir '%s': matches %s", subdir, paste(rds_hits, collapse = ", ")),
    call. = FALSE
  )
}

# Load sNMF proportions

rds_path <- rds_hits[1]
cat(sprintf("Loading sNMF results from %s\n", rds_path))
res <- readRDS(rds_path)
K <- if (!is.null(k_override)) k_override else res$k_star
nmf_labels <- res$labels
P <- res$Q[[as.character(K)]]
cat(sprintf("  Using K = %d\n", K))
cat(sprintf("  Proportions: %d languages x %d components\n", nrow(P), ncol(P)))

# Load language metadata

cat("Loading language metadata...\n")
meta_dir <- file.path(METADATA_DIR, dataset)
json_path <- file.path(meta_dir, "languages.json")
json_txt <- gsub("NaN", "null", readLines(json_path, warn = FALSE))
lang_meta <- parse_json(paste(json_txt, collapse = "\n"))

# Map NMF labels (FLEURS names) to directory keys and glottocodes
name_to_dir <- setNames(
  names(lang_meta),
  sapply(lang_meta, function(x) x$fleurs)
)
nmf_dirs <- name_to_dir[nmf_labels]
nmf_glottocodes <- sapply(nmf_dirs, function(d) lang_meta[[d]]$glottolog)

# Load predictor data

cat("Loading PHOIBLE data...\n")
pred_agg <- read.csv(
  file.path(meta_dir, "phoible.csv"),
  stringsAsFactors = FALSE
)
join_col <- "Glottocode"

cat(sprintf(
  "  Raw: %d languages x %d features\n",
  nrow(pred_agg),
  ncol(pred_agg) - 1
))

# Align NMF and predictors

matched_idx <- which(nmf_glottocodes %in% pred_agg[[join_col]])
matched_gc <- nmf_glottocodes[matched_idx]
P_matched <- P[matched_idx, ]

pred_aligned <- pred_agg %>%
  filter(.data[[join_col]] %in% matched_gc) %>%
  arrange(match(.data[[join_col]], matched_gc))

# Select predictor columns

all_cols <- setdiff(colnames(pred_aligned), "Glottocode")
# has_{feat} binary indicators + inventory size counts
feat_names <- c(
  "n_phonemes",
  "n_consonants",
  "n_vowels",
  grep("^has_", all_cols, value = TRUE)
)

# Drop constant features
sds <- sapply(pred_aligned[feat_names], sd, na.rm = TRUE)
variable_feats <- feat_names[sds > 1e-10]

cat(sprintf(
  "  Matched: %d languages, %d variable features (from %d candidates)\n",
  length(matched_idx),
  length(variable_feats),
  length(feat_names)
))

# Standardize predictors
X_df <- pred_aligned[variable_feats]
X_scaled <- as.data.frame(scale(X_df))
# Make column names brms-safe (no spaces, no special chars)
colnames(X_scaled) <- make.names(colnames(X_scaled))
variable_feats_clean <- colnames(X_scaled)


# Output directory

# Output as sibling of nmf/ (e.g. .../0.01_brsupport/brms_phoible/)
output_dir <- file.path(dirname(dirname(rds_path)), "brms_phoible")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Fit one brms model per component

cat(sprintf("\nFitting %d brms models...\n", K))

CHAINS <- 4L
ITER <- 4000L
WARMUP <- 1000L
SEED <- 42L

library(brms)

n_cores <- max(1L, min(CHAINS, parallel::detectCores()))
p <- length(variable_feats_clean)

# Horseshoe prior for handling multicollinearity and potential sparsity in predictors
prior <- set_prior("normal(0,1)", class = "b")

formula_rhs <- paste(variable_feats_clean, collapse = " + ")
formula_obj <- as.formula(paste0("y ~ ", formula_rhs))

all_results <- list()

for (j in seq_len(K)) {
  cat(sprintf("\n══════ Component %d / %d ══════\n", j, K))

  model_df <- X_scaled
  model_df$y <- P_matched[, j]

  fit <- brm(
    formula = formula_obj,
    data = model_df,
    family = gaussian(),
    prior = prior,
    chains = CHAINS,
    iter = ITER,
    warmup = WARMUP,
    seed = SEED,
    cores = n_cores,
    refresh = 0,
    silent = 2
  )

  # Diagnostics
  summ <- summary(fit)
  cat(sprintf(
    "  Max Rhat: %.4f\n",
    max(summ$fixed$Rhat, na.rm = TRUE)
  ))
  cat(sprintf(
    "  Min Bulk ESS: %.0f\n",
    min(summ$fixed$Bulk_ESS, na.rm = TRUE)
  ))

  # Extract fixed effects
  fe <- as.data.frame(fixef(fit, probs = c(0.025, 0.975)))
  colnames(fe) <- c("estimate", "se", "ci_lower", "ci_upper")
  fe$feature <- rownames(fe)
  fe$component <- j

  # R²
  r2 <- bayes_R2(fit)
  cat(sprintf(
    "  R²: %.3f [%.3f, %.3f]\n",
    mean(r2),
    quantile(r2, 0.025),
    quantile(r2, 0.975)
  ))
  fe$r2_mean <- mean(r2)

  all_results[[j]] <- fe

  # Save individual model
  model_path <- file.path(
    output_dir,
    sprintf("component_%02d.rds", j)
  )
  saveRDS(fit, file = model_path)
  cat(sprintf("  Saved: %s\n", model_path))
}

# Combine and save results

results_df <- bind_rows(all_results) %>%
  filter(feature != "Intercept") %>%
  select(component, feature, estimate, se, ci_lower, ci_upper, r2_mean)

csv_path <- file.path(output_dir, "nmf_phoible_brms.csv")
write.csv(results_df, csv_path, row.names = FALSE)
cat(sprintf("\nSaved combined results: %s\n", csv_path))

# Summary: features with 95% CI excluding zero
cat("\nFeatures per component (95% CI excludes zero):\n")
sig <- results_df %>% filter(ci_lower > 0 | ci_upper < 0)
for (j in seq_len(K)) {
  comp_sig <- sig %>% filter(component == j)
  if (nrow(comp_sig) == 0) {
    cat(sprintf("  Comp %2d: (none)\n", j))
  } else {
    parts <- sprintf(
      "%s (%.4f [%.4f, %.4f])",
      comp_sig$feature,
      comp_sig$estimate,
      comp_sig$ci_lower,
      comp_sig$ci_upper
    )
    cat(sprintf(
      "  Comp %2d: %s\n",
      j,
      paste(parts, collapse = ", ")
    ))
  }
}

cat(sprintf(
  "\nTotal: %d / %d coefficients with CI excluding zero\n",
  nrow(sig),
  nrow(results_df)
))

# LOO-CV (optional, slow)
# Uncomment to compare ridge vs horseshoe or check predictive fit.
# Requires re-fitting with horseshoe for comparison.
#
# loo_results <- list()
# for (j in seq_len(K)) {
#   fit <- readRDS(file.path(output_dir, sprintf("component_%02d.rds", j)))
#   loo_results[[j]] <- loo(fit)
#   cat(sprintf("  Comp %d ELPD: %.1f (SE %.1f)\n",
#     j, loo_results[[j]]$estimates["elpd_loo", "Estimate"],
#     loo_results[[j]]$estimates["elpd_loo", "SE"]))
# }

cat("Done.\n")
