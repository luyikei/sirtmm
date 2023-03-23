sirtmm <- function(data,
                   k = NULL,
                   itemtype = "SIRM-MMe",
                   quadpts = 15,
                   tol = 0.0001,
                   max_emsteps = 10,
                   max_nriter = 50,
                   lambda1 = 1,
                   lamnda2 = 2,
                   ngs = 1,
                   verbose = FALSE) {
  rule <- fastGHQuad::gaussHermiteData(quadpts)
  points <- rule$x
  points <- points * sqrt(2)
  weights <- rule$w
  weights <- weights / sqrt(pi)

  if (is.null(k)) {
    k <- max(data)
    if (verbose) message("Using the maximum of the response matrix: ", k)
  }

  res <- EMSteps(data, k, points, weights, lambda1, lamnda2, ngs, max_emsteps, max_nriter, verbose)
  se <- SE(data, res[["b"]],  res[["a"]], res[["c"]], res[["g"]], k, points, weights)
  return(list(
    itempar = as.data.frame(res),
    se = as.data.frame(se)
  ))
}
