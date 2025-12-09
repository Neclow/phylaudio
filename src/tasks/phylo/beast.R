# -----------------------------------------------------------------------------
# Portions of this code are adapted from rBt
# Original repository: https://github.com/santiagosnchez/rBt
# Copyright (c) Heinrich Dinkel, 2018
# Licensed under the GPL-3 license
# -----------------------------------------------------------------------------

library(ape)
library(doParallel)
library(dplyr)
library(tidyr)
library(TreeDist)

cldws <- function(b, lst) {
  if (b == "(") {
    lst$currnode <- lst$currnode + 1
    if (
      length(grep(paste(lst$currnode), lst$opened)) == 1 |
        length(grep(paste(lst$currnode), lst$closed)) == 1
    ) {
      return(cldws(b, lst))
    } else {
      lst$opened <- c(lst$opened, lst$currnode)
      return(lst)
    }
  }
  if (b == ")") {
    if (length(grep(paste(lst$currnode), lst$closed)) == 1) {
      lst$currnode <- lst$currnode - 1
      return(cldws(b, lst))
    } else {
      lst$closed <- c(lst$closed, lst$currnode)
      return(lst)
    }
  }
}

process_annot <- function(an) {
  if (is.na(an)) {
    return(NA)
  } else {
    res <- list()
    currlybr <- gregexpr(pattern = "\\{|\\}", an[[1]])[[1]]
    if (currlybr[1] == -1) {
      sep_annot <- strsplit(an, ",")[[1]]
    } else {
      commas <- vector()
      sep_annot <- vector()
      ctmp <- gregexpr(pattern = ",", an[[1]])[[1]]
      currlybr <- data.frame(
        st = currlybr[seq(1, length(currlybr), 2)],
        en = currlybr[seq(2, length(currlybr), 2)]
      )
      check_comma <- function(b, cm) {
        return(any(apply(currlybr, 1, function(x, y = cm) {
          return(x[1] < y & x[2] > y)
        })))
      }
      for (cm in ctmp) {
        if (!check_comma(currlybr, cm)) {
          commas <- c(commas, cm)
        }
      }
      if (length(commas) == 0) {
        sep_annot <- an
      } else if (length(commas) == 1) {
        sep_annot <- c(sep_annot, substr(an, 1, commas - 1))
        sep_annot <- c(sep_annot, substr(an, commas + 1, nchar(an)))
      } else {
        for (i in 1:length(commas)) {
          if (i == 1) {
            sep_annot <- c(sep_annot, substr(an, 1, commas[i] - 1))
            sep_annot <- c(
              sep_annot,
              substr(an, commas[i] + 1, commas[i + 1] - 1)
            )
          } else if (i == length(commas)) {
            sep_annot <- c(sep_annot, substr(an, commas[i] + 1, nchar(an)))
          } else {
            sep_annot <- c(
              sep_annot,
              substr(an, commas[i] + 1, commas[i + 1] - 1)
            )
          }
        }
      }
    }
    sep_annot <- gsub("\\[|\\[\\&|\\]|\\{|\\}", "", sep_annot)
    for (i in sep_annot) {
      tmp <- strsplit(i, "=")[[1]]
      res[[paste(tmp[1])]] <- tmp[2]
    }
    return(res)
  }
}


#' read.annot.beast
#'
#' @description
#' This function reads a nexus formated beast file and
#' appends all meta-data and annotations.
#'
#' @details
#' Node ordering is the same as in the \code{read.tree}/\code{read.nexus}
#' functions from ape (cladewise). The metadata is stored as a
#' data.frame named 'metadata'. Only node posterior
#' probabilites are passed as an additional numeric vector
#' named 'posterior'. Note that 'metadata' includes tip
#' annotations as well. The code might be a bit slow for large trees.
#'
#' @param file The path to the MCC file from BEAST
#' @return annotated \code{phylo} object
#' @seealso \code{\link{read.nexus}}
#' @export
#' @examples
#' file <- system.file("data/mcc.tre", package = "rBt")
#' tr <- read.annot.beast(file)
#' class(tr)
#' # [1] "phylo"
#' names(tr)
#' # [1] "edge"        "edge.length" "Nnode"       "root.edge"   "tip.label"
#' # [6] "metadata"    "posterior"
#' # for add pp to nodes you could try:
#' plot(tr)
#' nodelabels(text = round(tr$posterior, 2), cex = 0.5)
#' # for edges, try:
#' tr$ordered.edges <- order.edges(tr)
#' plot(tr)
#' edgelabels(edge = tr$ordered.edges, text = round(tr$posterior, 2), cex = 0.5)
read.annot.beast <- function(file) {
  tree <- scan(file = file, what = character(), sep = "\n", quiet = TRUE)
  tree <- tree[grep("^[[:space:]]*tree", tree)]
  if (length(tree) > 1) {
    trs <- read.nexus(file)
    for (itr in seq_along(tree)) {
      # if ((itr - 1) %% 37 != 0) {
      #   next
      # }
      tr <- tree[[itr]]
      tr <- sub("tree.* \\(", "\\(", tr)
      edges <- list()
      annot <- list()
      for (i in strsplit(tr, ":")[[1]]) {
        if (
          substr(i, nchar(i), nchar(i)) == "]" |
            substr(i, nchar(i) - 1, nchar(i) - 1) == "]"
        ) {
          brack <- gregexpr(pattern = "\\[|\\]", i)[[1]]
          start <- rev(brack)[2]
          end <- rev(brack)[1]
          annot <- c(annot, list(substr(i, start, end)))
          end <- start - 1
          if (substr(i, 1, 1) == "[") {
            start <- brack[2] + 1
          } else {
            start <- 1
          }
          edges <- c(edges, list(substr(i, start, end)))
        }
      }
      nodes <- lapply(edges, function(x) {
        if (substr(x, nchar(x), nchar(x)) == ")") {
          return("node")
        } else {
          start <- gregexpr(pattern = "[\\(|,][[:alnum:]]+", x)[[1]][1]
          return(substr(x, start + 1, nchar(x)))
        }
      })
      backbone <- strsplit(tr, "")[[1]][grep("\\(|\\)", strsplit(tr, "")[[1]])]
      tipsidx <- grep("node", nodes, invert = T)
      nodesidx <- grep("node", nodes)
      Ntips <- length(tipsidx)
      Nnodes <- length(nodesidx)
      lst <- list(opened = vector(), closed = vector(), currnode = Ntips)
      for (b in backbone) {
        lst <- cldws(b, lst)
      }
      tips <- vector()
      tip_num <- 1
      for (i in 1:length(nodes)) {
        if (nodes[[i]] != "node") {
          tips <- c(tips, nodes[[i]])
          nodes[[i]] <- tip_num
          tip_num <- tip_num + 1
        }
      }
      nodes[nodesidx] <- lst$closed
      nodes <- sapply(nodes, as.numeric)
      annot <- lapply(annot, process_annot)
      annot_names <- sort(unlist(lapply(annot, function(x) names(x))))
      annot_names <- annot_names[!duplicated(annot_names)]
      annot_mat <- data.frame(matrix(
        NA,
        ncol = length(annot_names) + 1,
        nrow = length(nodes)
      ))
      colnames(annot_mat) <- c("node", annot_names)
      annot_mat$node <- nodes
      for (i in annot_names) {
        tmp <- lapply(annot, function(x, y = i) x[[paste(y)]])
        tmp <- lapply(tmp, function(x) {
          if (is.null(x)) {
            return(NA)
          } else {
            return(x)
          }
        })
        annot_mat[, paste(i)] <- unlist(tmp)
      }
      annot_mat <- annot_mat[order(annot_mat[, 1]), ]
      rownames(annot_mat) <- 1:dim(annot_mat)[1]
      trs[[itr]]$metadata <- annot_mat
      trs[[itr]]$posterior <- as.numeric(annot_mat$posterior[
        lst$currnode:length(nodes)
      ])
      cat("tree #", itr, "\r", sep = "")
    }
    message(paste("\nRead ", length(trs), " trees"))
    return(trs)
  } else {
    tr <- tree[[1]]
    tr <- sub("^[[:space:]]*tree.* \\(", "\\(", tr)
    edges <- list()
    annot <- list()
    for (i in strsplit(tr, ":")[[1]]) {
      if (
        substr(i, nchar(i), nchar(i)) == "]" |
          substr(i, nchar(i) - 1, nchar(i) - 1) == "]"
      ) {
        brack <- gregexpr(pattern = "\\[|\\]", i)[[1]]
        start <- rev(brack)[2]
        end <- rev(brack)[1]
        annot <- c(annot, list(substr(i, start, end)))
        end <- start - 1
        if (substr(i, 1, 1) == "[") {
          start <- brack[2] + 1
        } else {
          start <- 1
        }
        edges <- c(edges, list(substr(i, start, end)))
      }
    }
    nodes <- lapply(edges, function(x) {
      if (substr(x, nchar(x), nchar(x)) == ")") {
        return("node")
      } else {
        start <- gregexpr(pattern = "[\\(|,][[:alnum:]]+", x)[[1]][1]
        return(substr(x, start + 1, nchar(x)))
      }
    })
    backbone <- strsplit(tr, "")[[1]][grep("\\(|\\)", strsplit(tr, "")[[1]])]
    tipsidx <- grep("node", nodes, invert = T)
    nodesidx <- grep("node", nodes)
    Ntips <- length(tipsidx)
    Nnodes <- length(nodesidx)
    lst <- list(opened = vector(), closed = vector(), currnode = Ntips)
    for (b in backbone) {
      lst <- cldws(b, lst)
    }
    tips <- vector()
    tip_num <- 1
    for (i in 1:length(nodes)) {
      if (nodes[[i]] != "node") {
        tips <- c(tips, nodes[[i]])
        nodes[[i]] <- tip_num
        tip_num <- tip_num + 1
      }
    }
    nodes[nodesidx] <- lst$closed
    nodes <- sapply(nodes, as.numeric)
    annot <- lapply(annot, process_annot)
    annot_names <- sort(unlist(lapply(annot, function(x) names(x))))
    annot_names <- annot_names[!duplicated(annot_names)]
    annot_mat <- data.frame(matrix(
      NA,
      ncol = length(annot_names) + 1,
      nrow = length(nodes)
    ))
    colnames(annot_mat) <- c("node", annot_names)
    annot_mat$node <- nodes
    for (i in annot_names) {
      tmp <- lapply(annot, function(x, y = i) x[[paste(y)]])
      tmp <- lapply(tmp, function(x) {
        if (is.null(x)) {
          return(NA)
        } else {
          return(x)
        }
      })
      annot_mat[, paste(i)] <- unlist(tmp)
    }
    annot_mat <- annot_mat[order(annot_mat[, 1]), ]
    rownames(annot_mat) <- 1:dim(annot_mat)[1]
    tr <- read.nexus(file)
    tr$metadata <- annot_mat
    tr$posterior <- as.numeric(annot_mat$posterior[lst$currnode:length(nodes)])
    return(tr)
  }
}


extract_beast_metadata <- function(file = "", tree = NULL) {
  if (!is.null(tree)) {
    if (!inherits(tree, "phylo")) {
      stop("argument 'tree' must be of mode 'phylo'")
    }
  } else if (file != "") {
    tree <- read.annot.beast(file)
  } else {
    stop("argument 'file' or 'tree' must be provided")
  }

  is_tip <- tree$edge[, 2] <= length(tree$tip.label)
  ordered_tips <- tree$edge[is_tip, 2]
  df <- tree$metadata |>
    mutate(language = tree$tip.label[ordered_tips[node]]) |>
    filter(!is.na(language)) |>
    mutate(across(everything(), ~ type.convert(., as.is = TRUE)))

  df
}

#' @examples
#' input_dir <- "data/beast/eab44e7f-54cc-4469-87d1-282cc81e02c2/sentence"
#' pattern <- "*\\_treeannotator_CCD.nex"
#' cores <- 16
#' extract_beast_metadata(input_dir, pattern, cores)
extract_beast_metadata2 <- function(
  files = "",
  trees = "",
  output_file = "",
  cores = 1
) {
  if (!is.null(trees)) {
    if (!inherits(trees, "multiPhylo")) {
      stop("argument 'trees' must be of mode 'multiPhylo'")
    }
    obj <- trees
  } else if (files != "") {
    obj <- files
  } else {
    stop("argument 'files' or 'trees' must be provided")
  }

  stopifnot(length(obj) > 0)

  registerDoParallel(cores = cores)
  print(paste("Using", cores, "core(s)", sep = " "))

  result <- foreach(i = seq_along(obj), .combine = rbind) %dopar%
    {
      if (inherits(obj[[i]], "phylo")) {
        extract_beast_metadata(tree = obj[[i]])
      } else {
        extract_beast_metadata(file = obj[[i]])
      }
    }

  if (output_file != "") {
    write.csv(result, output_file)
  } else {
    result
  }
}

extract_beast_dists <- function(input_dir, pattern, cores = 1) {
  files <- list.files(
    input_dir,
    pattern = pattern,
    full.names = TRUE
  )

  stopifnot(length(files) > 0)
  print(paste("Found", length(files), "files", sep = " "))

  registerDoParallel(cores = cores)
  print(paste("Using", cores, "core(s)", sep = " "))

  print("Reading trees...")
  trees <- foreach(i = seq_along(files)) %dopar%
    {
      read.annot.beast(files[i])
    }

  dists <- TreeDistance(trees)

  output_file <- file.path(input_dir, "_dists.Rdata")

  save(dists, file = output_file)

  print(paste("Done. View results at", output_file, sep = " "))
}

extract_beast_heights <- function(file = "", tree = NULL, cores = 1) {
  if (!is.null(tree)) {
    if (!inherits(tree, "phylo")) {
      stop("argument 'tree' must be of mode 'phylo'")
    }
  } else if (file != "") {
    tree <- read.annot.beast(file)
  } else {
    stop("argument 'file' or 'tree' must be provided")
  }

  if (length(tree) > 1) {
    registerDoParallel(cores = cores)
    print(paste("Using", cores, "core(s)", sep = " "))

    heights <- foreach(i = seq_along(tree), .combine = rbind) %dopar%
      {
        max(branching.times(tree[[i]]))
      }
    heights
  } else {
    max(branching.times(tree))
  }
}
