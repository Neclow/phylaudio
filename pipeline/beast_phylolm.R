#!/usr/bin/env Rscript
#
# beast_phylolm.R - Phylogenetic regression on BEAST tree metadata
#
# Fits Bayesian phylogenetic regression models using brms to analyze
# the relationship between speech rate and geographic/demographic factors
# while accounting for phylogenetic covariance.
#
# Usage:
#   Rscript pipeline/beast_phylolm.R [options]
#
# Examples:
#   Rscript pipeline/beast_phylolm.R --tree iecor/raw_mcc.nex
#   Rscript pipeline/beast_phylolm.R --coord-method log_plus_10

# ─── Setup ───────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ape)
  library(brms)
  library(dplyr)
  library(tibble)
  library(jsonlite)
  library(purrr)
  library(argparse)
})

# Source helper functions
source("src/tasks/phylo/beast.R")
source("src/tasks/phylo/regression.R")

# ─── CLI Arguments ───────────────────────────────────────────────────────────

parser <- ArgumentParser(description = "Run phylogenetic regression on BEAST trees")

parser$add_argument(
  "--tree",
  type = "character",
  nargs = "+",
  default = c("iecor/raw_mcc.nex"),
  help = "Tree path(s) relative to beast_dir (default: iecor/raw_mcc.nex)"
)

parser$add_argument(
  "--dataset",
  type = "character",
  default = "fleurs-r",
  help = "Dataset name (default: fleurs-r)"
)

parser$add_argument(
  "--min-speakers",
  type = "double",
  default = 1.0,
  dest = "min_speakers",
  help = "Minimum speakers in millions (default: 1.0)"
)

parser$add_argument(
  "--coord-method",
  type = "character",
  nargs = "+",
  default = c("standard"),
  dest = "coord_methods",
  choices = c("standard", "sqrt_plus_10", "log_plus_10"),
  help = "Coordinate scaling method(s) (default: standard)"
)

parser$add_argument(
  "--chains",
  type = "integer",
  default = 4L,
  help = "Number of MCMC chains (default: 4)"
)

parser$add_argument(
  "--iter",
  type = "integer",
  default = 4000L,
  help = "Number of iterations per chain (default: 4000)"
)

parser$add_argument(
  "--warmup",
  type = "integer",
  default = 1000L,
  help = "Number of warmup iterations (default: 1000)"
)

parser$add_argument(
  "--adapt-delta",
  type = "double",
  default = 0.99,
  dest = "adapt_delta",
  help = "Adaptation delta for NUTS (default: 0.99)"
)

parser$add_argument(
  "--seed",
  type = "integer",
  default = 20231103L,
  help = "Random seed (default: 20231103)"
)

args <- parser$parse_args()

# ─── Configuration ───────────────────────────────────────────────────────────

meta_dir <- "data/metadata"
beast_dir <- "data/trees/beast"

tree_names <- args$tree
dataset <- args$dataset
min_speakers <- args$min_speakers
coord_methods <- args$coord_methods

cat("Configuration:\n")
cat(sprintf("  Trees: %s\n", paste(tree_names, collapse = ", ")))
cat(sprintf("  Dataset: %s\n", dataset))
cat(sprintf("  Min speakers: %.1f million\n", min_speakers))
cat(sprintf("  Coord methods: %s\n", paste(coord_methods, collapse = ", ")))
cat(sprintf("  Chains: %d, Iter: %d, Warmup: %d\n", args$chains, args$iter, args$warmup))

# ─── Load Language Metadata ──────────────────────────────────────────────────

cat("\nLoading language metadata...\n")

json_path <- file.path(meta_dir, dataset, "languages.json")
mapping <- read_json(json_path)

# Build glottocode to IECOR name mapping
glottocode_to_iecor <- tibble(
  glottocode = map_chr(mapping, "glottolog"),
  iecor = map_chr(mapping, ~ .x$iecor %||% NA_character_)
)

glotto_path <- file.path(meta_dir, dataset, "glottolog.csv")
lang_metadata <- read.csv(glotto_path, stringsAsFactors = FALSE) %>%
  left_join(glottocode_to_iecor, by = "glottocode") %>%
  rename(language = iecor) %>%
  filter(!is.na(language)) %>%
  filter(n_speakers >= min_speakers) %>%
  group_by(language) %>%
  slice(1L) %>%
  ungroup() %>%
  select(-matches("^H\\d+$"))

cat(sprintf("  Loaded %d languages\n", nrow(lang_metadata)))

# ─── Main Loop ───────────────────────────────────────────────────────────────

for (tree_name in tree_names) {
  cat("\n")
  cat("======================================================================\n")
  cat(sprintf("Processing tree: %s\n", tree_name))
  cat("======================================================================\n")

  # ─── Read Tree ─────────────────────────────────────────────────────────────

  tree_path <- file.path(beast_dir, tree_name)
  if (!file.exists(tree_path)) {
    cat(sprintf("  Warning: Tree file not found: %s, skipping...\n", tree_path))
    next
  }

  tr <- read.annot.beast(tree_path)
  tree_dir <- dirname(tree_path)
  stem <- tools::file_path_sans_ext(basename(tree_path))

  # ─── Extract & Merge Metadata ──────────────────────────────────────────────

  beast_metadata <- extract_beast_metadata(tree = tr)
  df <- inner_join(beast_metadata, lang_metadata, by = "language") %>%
    column_to_rownames(var = "language")

  cat(sprintf("  Merged metadata: %d languages\n", nrow(df)))

  # ─── Write Metadata CSV ────────────────────────────────────────────────────

  metadata_path <- file.path(tree_dir, paste0(stem, "_metadata.csv"))
  write.csv(
    df %>% rownames_to_column("language"),
    file = metadata_path,
    row.names = FALSE,
    quote = TRUE
  )
  cat(sprintf("  Wrote metadata: %s\n", metadata_path))

  # ─── Prune Tree & Compute Covariance ───────────────────────────────────────

  to_drop <- setdiff(tr$tip.label, rownames(df))
  tr <- drop.tip(tr, to_drop)
  V_raw <- vcv(unroot(tr))
  V_raw <- V_raw[rownames(df), rownames(df)]

  # Write covariance matrix
  vcv_path <- file.path(tree_dir, paste0(stem, "_vcv.csv"))
  write.csv(V_raw, file = vcv_path, row.names = TRUE, quote = FALSE)
  cat(sprintf("  Wrote VCV matrix: %s\n", vcv_path))

  # ─── Speaker Count Transform ───────────────────────────────────────────────

  log_n_scaled <- scale(log(df$n_speakers))
  df$log_n_speakers_norm <- log_n_scaled[, 1]

  base_df <- df

  # ─── Fit Models ────────────────────────────────────────────────────────────

  for (coord_method in coord_methods) {
    cat(sprintf("\n--- Coordinate method: %s ---\n", coord_method))

    # Prepare data
    data <- prepare_brms_data(
      base_df = base_df,
      V_raw = V_raw,
      coord_method = coord_method
    )

    model_df <- data$model_df
    V <- data$V

    # Build formula
    formula_str <- paste(
      "log_rate_median ~",
      "longitude_norm + latitude_norm + log_n_speakers_norm +",
      "longitude_norm:latitude_norm +",
      "latitude_norm:log_n_speakers_norm +",
      "longitude_norm:log_n_speakers_norm +",
      "(1|gr(language_factor, cov = V))"
    )
    formula_obj <- as.formula(formula_str)

    # Model arguments
    brm_args <- list(
      formula = formula_obj,
      data = model_df,
      family = gaussian(),
      refresh = 0,
      chains = args$chains,
      iter = args$iter,
      warmup = args$warmup,
      seed = args$seed,
      cores = max(1L, min(args$chains, parallel::detectCores())),
      control = list(adapt_delta = args$adapt_delta, max_treedepth = 15),
      data2 = list(V = V)
    )

    cat(sprintf("Fitting phylogenetic regression...\n"))
    model <- do.call(brm, brm_args)
    model_type <- "brms_phylo"

    # Save model
    model_path <- file.path(tree_dir, paste0(stem, "_phylolm_", coord_method, ".rds"))
    saveRDS(model, file = model_path)
    cat(sprintf("  Model saved: %s\n", model_path))

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    diagnostics <- diagnose_brms(model, brm_args)

    # ─── Variance Decomposition ──────────────────────────────────────────────

    posterior <- as.data.frame(model)

    # Build design matrix
    X <- model.matrix(
      ~ longitude_norm +
        latitude_norm +
        log_n_speakers_norm +
        longitude_norm:latitude_norm +
        latitude_norm:log_n_speakers_norm +
        longitude_norm:log_n_speakers_norm,
      data = model_df
    )

    # Map brms coefficient names
    brms_names <- c(
      "(Intercept)" = "b_Intercept",
      "longitude_norm" = "b_longitude_norm",
      "latitude_norm" = "b_latitude_norm",
      "log_n_speakers_norm" = "b_log_n_speakers_norm",
      "longitude_norm:latitude_norm" = "b_longitude_norm:latitude_norm",
      "latitude_norm:log_n_speakers_norm" = "b_latitude_norm:log_n_speakers_norm",
      "longitude_norm:log_n_speakers_norm" = "b_longitude_norm:log_n_speakers_norm"
    )

    beta_samples <- as.matrix(posterior[, brms_names[colnames(X)]])
    colnames(beta_samples) <- colnames(X)

    variance_decomposition <- decompose_variance(
      model = model,
      posterior = posterior,
      V = V,
      X = X,
      beta_samples = beta_samples
    )

    # Print summary
    print_variance_decomposition(variance_decomposition)

    # ─── Save Results ────────────────────────────────────────────────────────

    csv_path <- file.path(tree_dir, paste0(stem, "_phylolm_", coord_method, ".csv"))
    save_phylolm_results(
      output_path = csv_path,
      model_type = model_type,
      tree_name = tree_name,
      coord_method = coord_method,
      coef_df = variance_decomposition$coef_df,
      variance_decomposition = variance_decomposition,
      diagnostics = diagnostics
    )
  }
}

cat("\nDone.\n")
