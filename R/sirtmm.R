#' Estimation 
#' 
#' Test
#'
#' @param data The reponse matrix
#' @param k The number of attempts
#' @export
sirtmm <- function(data,
                   k = NULL,
                   itemtype = "SIRT-MMe",
                   quadpts = 15,
                   tol = 0.0001,
                   max_emsteps = 10,
                   max_nriter = 50,
                   lambda1 = 1,
                   lambda2 = 50,
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
  } else {
    if (k < max(data)) {
      k <- max(data)
      warning("k is smaller than the maximum value in the response matrix. Use ", k, " instead.")
    }
  }

  extended <- ifelse(itemtype == "SIRT-MMe", TRUE, FALSE)
  nog <- ngs == 0

  res <- EMSteps(data, k, points, weights, lambda1, lambda2, ngs, max_emsteps, max_nriter, verbose, extended)
  if (itemtype == "SIRT-MM") res[["c"]] <-  rep(1/k, ncol(data))
  se <- SE(data, res[["b"]],  res[["a"]], res[["c"]], res[["g"]], k, points, weights, extended, nog)

  # if (ngs == 0) {
  #   res[["g"]] = NULL
  #   se[["g"]] = NULL
  # }
  
  mod <- list(
    data = X,
    options = list(
      k = k,
      itemtype = itemtype,
      quadpts = quadpts,
      tol = tol,
      max_emsteps = max_emsteps,
      max_nriter = max_nriter,
      lambda1 = lambda1,
      lambda2 = lambda2,
      ngs = ngs
    ),
    quadpts = list(
      weights = weights,
      points = points
    ),
    itempar = as.data.frame(res),
    g.mat = res[["g"]],
    se = as.data.frame(se)
  )
  class(mod) <- "sirtmmModel"
  return(mod)
}

#' ANOVA
#'
#' Test
#'
#' @param ... SIRT-MM models.
#' @export
#' @examples
#' 
#' mod1 = sirtmm(X, k = K, itemtype="SIRT-MM", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod2 = sirtmm(X, k = K, itemtype="SIRT-MM", ngs = 1, lambda1 = 1, lambda2 = 50)
#' mod3 = sirtmm(X, k = K, itemtype="SIRT-MMe", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod4 = sirtmm(X, k = K, itemtype="SIRT-MMe", ngs = 1, lambda1 = 1, lambda2 = 50)
#' anova(mod1, mod2, mod3, mod4)
anova.sirtmmModel <- function(...) {
  ret <- data.frame()
  models <- list(...)
  for (mod in models) {
    eret <- EStep(mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, mod$g.mat, mod$options$k,
                  mod$quadpts$points, mod$quadpts$weights)
    logLik <- LogLikliTotal(mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, mod$g.mat, mod$options$k,
                            eret[["rhat"]], mod$quadpts$points)
    npar <- dim(mod$itempar)[1] * dim(mod$itempar)[2]
    if (mod$options$itemtype != "SIRT-MMe") npar <- npar - dim(mod$itempar)[1]
    if (mod$options$ngs == 0) npar <- npar - dim(mod$itempar)[1]
    ret <- rbind(ret, data.frame(
      AIC=c(2 * npar - 2 * logLik),
      BIC=c(npar * log(nrow(mod$data)) - 2 * logLik),
      logLik=c(logLik),
      df=npar)
    )
  }
  return(ret)
}
