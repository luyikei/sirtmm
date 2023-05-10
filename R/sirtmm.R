pad_mat <- function(mat, M) {
  if (!length(mat)) {
    nmat <- matrix(0, nrow = M, ncol = 1)
  } else {
    nmat <- cbind(0, mat)
  }
  nmat
}

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
                   nds = 0,
                   verbose = FALSE) {
  rule <- fastGHQuad::gaussHermiteData(quadpts)
  points <- rule$x
  points <- points * sqrt(2)
  weights <- rule$w
  weights <- weights / sqrt(pi)

  data[is.na(data)] <- -999

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

  res <- EMSteps(data, k, points, weights, lambda1, lambda2, ngs, nds, max_emsteps, max_nriter, verbose, extended)
  if (itemtype == "SIRT-MM") res[["c"]] <- rep(1 / k, ncol(data))
  se <- SE(data, res[["b"]], res[["a"]], res[["c"]], res[["g"]], res[["d"]], k, points, weights, extended)

  res[["g"]] <- res[["g"]][, -1]
  res[["d"]] <- res[["d"]][, -1]

  data[data == -999] <- NA

  mod <- list(
    data = data,
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
    d.mat = res[["d"]],
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
#' mod1 <- sirtmm(X, k = K, itemtype = "SIRT-MM", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod2 <- sirtmm(X, k = K, itemtype = "SIRT-MM", ngs = 1, lambda1 = 1, lambda2 = 50)
#' mod3 <- sirtmm(X, k = K, itemtype = "SIRT-MMe", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod4 <- sirtmm(X, k = K, itemtype = "SIRT-MMe", ngs = 1, lambda1 = 1, lambda2 = 50)
#' anova(mod1, mod2, mod3, mod4)
anova.sirtmmModel <- function(...) {
  ret <- data.frame()
  models <- list(...)
  for (mod in models) {
    eret <- EStep(
      mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, mod$g.mat, mod$d.mat, mod$options$k,
      mod$quadpts$points, mod$quadpts$weights
    )
    logLik <- LogLikliTotal(
      mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, mod$g.mat, mod$d.mat, mod$options$k,
      eret[["rhat"]], mod$quadpts$points
    )
    npar <- dim(mod$itempar)[1] * dim(mod$itempar)[2]
    if (mod$options$itemtype != "SIRT-MMe") npar <- npar - dim(mod$itempar)[1]
    if (mod$options$ngs == 0) npar <- npar - dim(mod$itempar)[1]
    ret <- rbind(ret, data.frame(
      AIC = c(2 * npar - 2 * logLik),
      BIC = c(npar * log(dim(mod$itempar)[1]) - 2 * logLik),
      logLik = c(logLik),
      napr = npar
    ))
  }
  return(ret)
}

# retuns prob for all possible responses
item_single_all <- function(theta, b, a, c, g, d, K) {
  ret <- rep(0, K)
  for (k in 1:K) {
    ret[k] <- item_single(k, theta, b, a, c, g, d, K)
  }
  ret
}

#' Simulate SIRT-MM models
#'
#' Test
#'
#' @param N number of subjects
#' @param M number of items
#' @param thetadist A function returns a M x 1 vector of theta
#' @param bdist A function returns a M x 1 vector of b parameter
#' @param adist A function returns a M x 1 vector of a parameter
#' @param cdist A function returns a M x 1 vector of c parameter
#' @param gdist A function returns a M x V matrix of g parameter where V is the number of effective g parameters.
#' @export
simsirt <- function(N, M, K, theta, b, a, c, g, d) {
  X <- matrix(0, nrow = N, ncol = M)
  gp <- pad_mat(g, M)
  dp <- pad_mat(d, M)
  for (i in 1:N) {
    for (j in 1:M) {
      probs <- item_single_all(theta[i], b[j], a[j], c[j], gp[j, ], dp[j, ], K)
      X[i, j] <- sample(K, 1, prob = probs, replace = TRUE)
    }
  }
  return(list(
    data = X,
    theta = theta,
    itempar = data.frame(b = b, a = a, c = c, g = g, d = d),
    g.mat = g,
    d.mat = d
  ))
}
