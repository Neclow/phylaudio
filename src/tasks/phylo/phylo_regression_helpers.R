# phylo_regression_helpers.R
#
# Shared utilities for phylogenetic regression scripts
# (run_phylo_regression.R and run_phylo_regression_nonlinear.R).

suppressPackageStartupMessages({
    library(ape)
})

apply_coord_scaling <- function(vec) {
    scaled <- scale(vec)
    list(
        values = as.numeric(scaled),
        center = as.numeric(attr(scaled, "scaled:center")),
        scale  = as.numeric(attr(scaled, "scaled:scale"))
    )
}

summarize_posterior <- function(x) {
    list(
        mean  = mean(x, na.rm = TRUE),
        sd    = sd(x, na.rm = TRUE),
        q2.5  = unname(quantile(x, 0.025, na.rm = TRUE)),
        q50   = unname(quantile(x, 0.50, na.rm = TRUE)),
        q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
    )
}

shapley_one_sample <- function(components) {
    D <- length(components)
    comp_names <- names(components)
    comp_mat <- do.call(cbind, components)

    V_fixed <- var(rowSums(comp_mat))

    n_coal <- 2^D
    v <- numeric(n_coal)
    for (bits in 1:(n_coal - 1)) {
        members <- which(as.logical(intToBits(bits)[1:D]))
        v[bits + 1] <- var(rowSums(comp_mat[, members, drop = FALSE]))
    }

    shapley_raw <- numeric(D)
    names(shapley_raw) <- comp_names

    for (j in 1:D) {
        j_bit <- 2^(j - 1)
        phi <- 0
        for (bits in 0:(n_coal - 1)) {
            if (bitwAnd(bits, j_bit) != 0) next
            s_size <- sum(as.logical(intToBits(bits)[1:D]))
            weight <- factorial(s_size) * factorial(D - s_size - 1) / factorial(D)
            phi <- phi + weight * (v[bitwOr(bits, j_bit) + 1] - v[bits + 1])
        }
        shapley_raw[j] <- phi
    }

    marginal_raw <- apply(comp_mat, 2, var)
    names(marginal_raw) <- comp_names

    list(
        shapley  = shapley_raw / V_fixed,
        marginal = marginal_raw / V_fixed,
        V_fixed  = V_fixed
    )
}

find_tree_file <- function(beast_dir, tree_file = NULL) {
    if (!is.null(tree_file)) {
        if (!file.exists(tree_file))
            stop(sprintf("Specified tree file does not exist: %s", tree_file))
        return(tree_file)
    }
    for (ext in c("*.mcc", "*.nex")) {
        hits <- Sys.glob(file.path(beast_dir, ext))
        if (length(hits) == 1) return(hits[1])
        if (length(hits) > 1) {
            stop(sprintf(
                "Multiple %s files in %s:\n  %s\nUse --tree_file to specify which one.",
                ext, beast_dir, paste(hits, collapse = "\n  ")))
        }
    }
    stop(sprintf("No .mcc or .nex tree file in %s", beast_dir))
}

load_regression_data <- function(beast_dir, variant, tree_file = NULL) {
    source("src/tasks/phylo/beast.R")
    tree_file <- find_tree_file(beast_dir, tree_file)
    cat(sprintf("  tree_file: %s\n", tree_file))
    tr <- read.annot.beast(tree_file)

    meta_csv <- if (variant == "with_inventory") {
        file.path(beast_dir, "metadata_with_inventory.csv")
    } else {
        file.path(beast_dir, "metadata.csv")
    }
    df <- read.csv(meta_csv, stringsAsFactors = FALSE)
    rownames(df) <- df$language

    tr <- drop.tip(tr, setdiff(tr$tip.label, rownames(df)))
    V_raw <- vcv(tr)
    V_raw <- V_raw[rownames(df), rownames(df)]

    list(df = df, tr = tr, V_raw = V_raw)
}

normalize_predictors <- function(df, variant) {
    df$log_n_speakers_norm <- as.numeric(scale(log(df$n_speakers)))

    lon_stats <- apply_coord_scaling(df$longitude)
    lat_stats <- apply_coord_scaling(df$latitude)
    df$longitude_norm <- lon_stats$values
    df$latitude_norm  <- lat_stats$values

    if (variant == "with_inventory") {
        df$n_phonemes_norm <- as.numeric(scale(df$n_phonemes))
    }
    df$delta_norm <- as.numeric(scale(df$delta))

    df
}

collect_shapley_results <- function(results_shapley, comp_names) {
    shap_mat <- do.call(rbind, lapply(results_shapley, function(r) r$shapley))
    marg_mat <- do.call(rbind, lapply(results_shapley, function(r) r$marginal))
    vfix_vec <- sapply(results_shapley, function(r) r$V_fixed)

    term_props <- list()
    marg_props <- list()
    for (j in seq_along(comp_names)) {
        term_props[[comp_names[j]]] <- summarize_posterior(shap_mat[, j])
        marg_props[[comp_names[j]]] <- summarize_posterior(marg_mat[, j])
    }

    list(
        shap_mat     = shap_mat,
        marg_mat     = marg_mat,
        vfix_vec     = vfix_vec,
        term_props   = term_props,
        marg_props   = marg_props,
        vfix_summary = summarize_posterior(vfix_vec)
    )
}

save_variance_samples <- function(out_dir, file_prefix, beast_dir, model_type,
                                   vardecomp, shap, comp_names) {
    S <- length(vardecomp$R2_full)
    samples_df <- data.frame(
        sample_id        = 1:S,
        beast_dir        = beast_dir,
        model_type       = model_type,
        R2_full          = vardecomp$R2_full,
        prop_fixed       = vardecomp$prop_fixed,
        prop_linear      = vardecomp$prop_linear,
        prop_phylo       = vardecomp$prop_phylo,
        prop_residual    = vardecomp$prop_residual,
        prop_cov_fix_phy = vardecomp$prop_cov_fix_phy,
        V_total          = vardecomp$V_total,
        V_fixed          = shap$vfix_vec
    )

    for (term in comp_names) {
        tc <- gsub(":", "_", term)
        samples_df[[paste0("shapley_", tc)]]  <- shap$shap_mat[, term]
        samples_df[[paste0("marginal_", tc)]] <- shap$marg_mat[, term]
    }

    path <- file.path(out_dir, paste0("variance_samples_", file_prefix, ".csv"))
    write.csv(samples_df, path, row.names = FALSE)
    cat("Posterior samples written to", path, "\n")
}

add_decomp_columns <- function(csv_results, vardecomp, shap) {
    for (comp in c("R2_full", "prop_fixed", "prop_linear", "prop_phylo",
                   "prop_cov_fix_phy", "prop_residual")) {
        s <- summarize_posterior(vardecomp[[comp]])
        csv_results[[paste0(comp, "_mean")]]  <- s$mean
        csv_results[[paste0(comp, "_q2_5")]]  <- s$q2.5
        csv_results[[paste0(comp, "_q97_5")]] <- s$q97.5
    }

    csv_results$V_fixed_mean  <- shap$vfix_summary$mean
    csv_results$V_fixed_q2_5  <- shap$vfix_summary$q2.5
    csv_results$V_fixed_q97_5 <- shap$vfix_summary$q97.5

    for (term in names(shap$term_props)) {
        tc <- gsub("\\(|\\)", "", gsub(":", "_", term))
        csv_results[[paste0("shapley_", tc, "_mean")]]  <- shap$term_props[[term]]$mean
        csv_results[[paste0("shapley_", tc, "_q2_5")]]  <- shap$term_props[[term]]$q2.5
        csv_results[[paste0("shapley_", tc, "_q97_5")]] <- shap$term_props[[term]]$q97.5
    }

    for (term in names(shap$marg_props)) {
        tc <- gsub("\\(|\\)", "", gsub(":", "_", term))
        csv_results[[paste0("marginal_", tc, "_mean")]]  <- shap$marg_props[[term]]$mean
        csv_results[[paste0("marginal_", tc, "_q2_5")]]  <- shap$marg_props[[term]]$q2.5
        csv_results[[paste0("marginal_", tc, "_q97_5")]] <- shap$marg_props[[term]]$q97.5
    }

    csv_results
}

print_variance_summary <- function(model_type, beast_dir, vardecomp, shap) {
    cat("\n======================================================================\n")
    cat(sprintf("COMPLETED: %s x %s\n", model_type, beast_dir))
    cat("======================================================================\n")
    cat(sprintf("  R2: %.3f (95%% CI: %.3f - %.3f)\n",
                summarize_posterior(vardecomp$R2_full)$mean,
                summarize_posterior(vardecomp$R2_full)$q2.5,
                summarize_posterior(vardecomp$R2_full)$q97.5))

    cat("\n  Variance Components (Proportion of V_total):\n")
    for (row in list(
        list("Fixed Effects", "prop_fixed"),
        list("Phylogenetic",  "prop_phylo"),
        list("Cov(Fix,Phy)",  "prop_cov_fix_phy"),
        list("Residual",      "prop_residual")
    )) {
        s <- summarize_posterior(vardecomp[[row[[2]]]])
        cat(sprintf("    %-17s %.1f%% (95%% CI: %.1f - %.1f)\n",
                    paste0(row[[1]], ":"),
                    s$mean * 100, s$q2.5 * 100, s$q97.5 * 100))
    }

    cat(sprintf("\n  V_fixed: %.4f (95%% CI: %.4f - %.4f)\n",
                shap$vfix_summary$mean,
                shap$vfix_summary$q2.5,
                shap$vfix_summary$q97.5))

    cat("\n  Per-term Shapley (proportion of V_fixed, sum to 1):\n")
    for (term in names(shap$term_props)) {
        cat(sprintf("    %-40s: %.1f%% (95%% CI: %.1f - %.1f)\n", term,
                    shap$term_props[[term]]$mean * 100,
                    shap$term_props[[term]]$q2.5 * 100,
                    shap$term_props[[term]]$q97.5 * 100))
    }

    cat("\n  Per-term Marginal Variance (proportion of V_fixed, do NOT sum to 1):\n")
    for (term in names(shap$marg_props)) {
        cat(sprintf("    %-40s: %.1f%% (95%% CI: %.1f - %.1f)\n", term,
                    shap$marg_props[[term]]$mean * 100,
                    shap$marg_props[[term]]$q2.5 * 100,
                    shap$marg_props[[term]]$q97.5 * 100))
    }
    cat("======================================================================\n")
}
