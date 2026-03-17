#!/usr/bin/env Rscript
#
# beast_phylolm.R - Phylogenetic regression pipeline dispatcher
#
# Dispatches to the appropriate regression script based on --model_type:
#   linear_geo  -> src/tasks/phylo/run_phylo_regression.R
#   gp_geo      -> src/tasks/phylo/run_phylo_regression_nonlinear.R
#
# Usage:
#   Rscript pipeline/beast_phylolm.R --model_type linear_geo --tree <tree_name> [options]
#   Rscript pipeline/beast_phylolm.R --model_type gp_geo --tree <tree_name> [options]
#
# Linear model options:
#   --variant with_inventory|no_inventory  (default: with_inventory)
#   --iter 100000       Total iterations per chain
#   --warmup 20000      Warmup iterations
#   --adapt_delta 0.9999
#   --thin 10
#   --seed 20231103
#
# GP model options:
#   --variant with_inventory|no_inventory  (default: with_inventory)
#   --iter_sampling 1000
#   --iter_warmup 1000
#   --adapt_delta 0.95
#   --max_treedepth 12
#   --seed 20231103
#
# Examples:
#   Rscript pipeline/beast_phylolm.R --model_type linear_geo --tree heggarty2024_raw
#   Rscript pipeline/beast_phylolm.R --model_type gp_geo --tree input_v12_combined_resampled

args <- commandArgs(trailingOnly = TRUE)
args <- args[args != "--"]

# Extract --model_type from args
model_type <- NULL
i <- 1
while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (key == "model_type" && i + 1 <= length(args)) {
        model_type <- args[i + 1]
        break
    }
    i <- i + 2
}

if (is.null(model_type)) {
    stop("--model_type is required (linear_geo or gp_geo)")
}

# Dispatch to the appropriate task script, forwarding all CLI args
if (model_type == "linear_geo") {
    cat("Dispatching to linear regression (brms)...\n\n")
    script <- "src/tasks/phylo/run_phylo_regression.R"
} else if (model_type == "gp_geo") {
    cat("Dispatching to GP regression (cmdstanr)...\n\n")
    # GP script doesn't use --model_type; strip it from forwarded args
    filtered_args <- character(0)
    i <- 1
    while (i <= length(args)) {
        key <- sub("^--", "", args[i])
        if (key == "model_type") {
            i <- i + 2
            next
        }
        filtered_args <- c(filtered_args, args[i])
        i <- i + 1
    }
    script <- "src/tasks/phylo/run_phylo_regression_nonlinear.R"
    # Re-invoke with filtered args
    cmd <- paste(
        "Rscript", shQuote(script),
        paste(shQuote(filtered_args), collapse = " ")
    )
    status <- system(cmd)
    quit(status = status)
} else {
    stop(sprintf(
        "Unknown model_type '%s'. Must be one of: linear_geo, gp_geo",
        model_type
    ))
}

# For linear_geo, forward all args directly
cmd <- paste(
    "Rscript", shQuote(script),
    paste(shQuote(args), collapse = " ")
)
cat(sprintf("Running: %s\n\n", cmd))
status <- system(cmd)
quit(status = status)
