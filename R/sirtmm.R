sirtmm <- function(data,
                   k = NULL,
                   itemtype = "SIRT-MMe",
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
  
  extended <- ifelse(itemtype == "SIRT-MMe", TRUE, FALSE)

  res <- EMSteps(data, k, points, weights, lambda1, lamnda2, ngs, max_emsteps, max_nriter, verbose, extended)
  cs <- if (itemtype == "SIRT-MMe") res[["c"]] else rep(1/k, ncol(data))
  se <- SE(data, res[["b"]],  res[["a"]], cs, res[["g"]], k, points, weights, extended)
  return(list(
    itempar = as.data.frame(res),
    se = as.data.frame(se)
  ))
}
