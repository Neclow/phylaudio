library(phangorn)

pathnode <- function(phylo, tipsonly = TRUE) {
  dist_nodes <- ape::dist.nodes(phylo)
  root <- phylo$edge[, 1][!(phylo$edge[, 1] %in% phylo$edge[, 2])][1]

  if (tipsonly == TRUE) {
    roottotippath <- dist_nodes[
      as.numeric(rownames(dist_nodes)) == root,
      seq_along(phylo$tip.label)
    ]
    nodesinpath <- sapply(seq_along(phylo$tip.label), function(x) {
      length(Ancestors(
        phylo,
        x
      ))
    })
  } else {
    roottotippath <- dist_nodes[as.numeric(rownames(dist_nodes)) == root, ]
    nodesinpath <- sapply(
      1:(length(phylo$tip.label) + phylo$Nnode),
      function(x) length(Ancestors(phylo, x))
    )
  }

  return(list(rtt = roottotippath, nip = nodesinpath))
}


stemmy <- function(tre) {
  total_internal <- sum(
    tre$edge.length[which(tre$edge[, 2] > Ntip(tre))],
    na.rm = TRUE
  )
  total_edge_length <- sum(tre$edge.length, na.rm = TRUE)
  return(total_internal / total_edge_length)
}


rtt_cov <- function(tre) {
  if (!is.rooted(tre)) tre <- midpoint(tre)
  rtts <- pathnode(tre)[[1]]
  rttcov <- sd(rtts, na.rm = TRUE) / mean(rtts, na.rm = TRUE)
  return(rttcov)
}
