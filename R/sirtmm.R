pad_mat <- function(mat, M) {
  if (!length(mat)) {
    nmat <- matrix(0, nrow = M, ncol = 1)
  } else {
    nmat <- cbind(0, mat)
  }
  nmat
}

item_single_all <- function(theta, b, a, c, g, d, K, MK) {
  ret <- rep(0, K)
  for (k in 1:K) {
    ret[k] <- item_single(k, theta, b, a, c, g, d, K, MK)
  }
  ret
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
                   mk = -1,
                   itemtype = "SIRT-MMe",
                   quadpts = 15,
                   tol = 0.0001,
                   max_emsteps = 10,
                   max_nriter = 50,
                   lambda1 = 1,
                   lambda2 = 1,
                   lambda3 = 50,
                   ngs = 1,
                   nds = 0,
                   verbose = FALSE,
                   lasso = FALSE) {
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

  res <- EMSteps(data, k, mk, points, weights, lambda1, lambda2, lambda3, ngs, nds, max_emsteps, max_nriter, verbose, extended, lasso)
  if (itemtype == "SIRT-MM") res[["c"]] <- rep(1 / k, ncol(data))
  se <- SE(data, res[["b"]], res[["a"]], res[["c"]], res[["g"]], res[["d"]], k, mk, points, weights, extended)

  conv <- res[["conv"]]
  res[["conv"]] <- NULL
  res[["g"]] <- res[["g"]][, -1]
  res[["d"]] <- res[["d"]][, -1]
  se[["g"]] <- se[["g"]][, -1]
  se[["d"]] <- se[["d"]][, -1]

  data[data == -999] <- NA

  mod <- list(
    data = data,
    options = list(
      k = k,
      mk = mk,
      itemtype = itemtype,
      quadpts = quadpts,
      tol = tol,
      max_emsteps = max_emsteps,
      max_nriter = max_nriter,
      lambda1 = lambda1,
      lambda2 = lambda2,
      ngs = ngs,
      nds = nds
    ),
    quadpts = list(
      weights = weights,
      points = points
    ),
    conv = conv,
    itempar = as.data.frame(res),
    g.mat = as.matrix(res[["g"]]),
    d.mat = as.matrix(res[["d"]]),
    se = as.data.frame(se),
    se.g.mat = (if (!is.null(se[["g"]])) as.matrix(se[["g"]]) else NULL),
    se.d.mat = (if (!is.null(se[["d"]])) as.matrix(se[["d"]]) else NULL)
  )
  class(mod) <- "sirtmmModel"
  return(mod)
}

#' ANOVA
#'
#' Test
#'
#' @param ... SIRT-MM models.
#' @examples
#'
#' mod1 <- sirtmm(X, k = K, itemtype = "SIRT-MM", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod2 <- sirtmm(X, k = K, itemtype = "SIRT-MM", ngs = 1, lambda1 = 1, lambda2 = 50)
#' mod3 <- sirtmm(X, k = K, itemtype = "SIRT-MMe", ngs = 0, lambda1 = 1, lambda2 = 50)
#' mod4 <- sirtmm(X, k = K, itemtype = "SIRT-MMe", ngs = 1, lambda1 = 1, lambda2 = 50)
#' anova(mod1, mod2, mod3, mod4)
#'
#' @export
anova.sirtmmModel <- function(...) {
  ret <- data.frame()
  models <- list(...)
  for (mod in models) {
    M <- nrow(mod$itempar)
    eret <- EStep(
      mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk,
      mod$quadpts$points, mod$quadpts$weights
    )
    logLik <- LogLikliTotal(
      mod$data, mod$itempar$b, mod$itempar$a, mod$itempar$c, pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk,
      eret[["rhat"]], mod$quadpts$points
    )
    npar <- dim(mod$itempar)[1] * dim(mod$itempar)[2]
    if (mod$options$itemtype != "SIRT-MMe") npar <- npar - dim(mod$itempar)[1] # c is not estimated
    ret <- rbind(ret, data.frame(
      AIC = c(2 * npar - 2 * logLik),
      BIC = c(npar * log(nrow(mod$itempar)) - 2 * logLik),
      logLik = c(logLik),
      npar = npar
    ))
  }
  return(ret)
}

#' Person Estimation
#'
#' Person parameters
#'
#' @param mod The reponse matrix
#' @param data data
#' @export
estimate <- function(mod, data = NULL) {
  if (is.null(data)) {
    data <- mod$data
  }

  rule <- fastGHQuad::gaussHermiteData(61)
  points <- rule$x
  points <- points * sqrt(2)
  weights <- rule$w
  weights <- weights / sqrt(pi)
  
  M <- nrow(mod$itempar)
  thetas.est <- eap_theta(
    data, mod$itempar$b, mod$itempar$a, mod$itempar$c,
    pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk, points, weights
  )
  theta.fi <- FI(
    thetas.est, mod$itempar$b, mod$itempar$a, mod$itempar$c,
    pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk
  )
  se.theta <- 1 / sqrt(theta.fi)
  return(list(
    thetas = thetas.est,
    fi = theta.fi,
    se = se.theta
  ))
}


#' scoring using a scheme
#'
#' scoring using a scheme
#'
#' @param X The reponse matrix
#' @param scheme scoring scheme
#' @export
score_scheme <- function(X, scheme) {
  return(
    ScoreScheme(X, scheme)
  )
}

#' Simulate SIRT-MM models
#'
#' Test
#'
#' @param N number of subjects
#' @param M number of items
#' @param thetadist A N x 1 vector of theta
#' @param bdist A M x 1 vector of b parameter
#' @param adist A M x 1 vector of a parameter
#' @param cdist A M x 1 vector of c parameter
#' @param gdist A M x V matrix of g parameter where V is the number of effective g parameters.
#' @param ddist A M x W matrix of d parameter where W is the number of effective d parameters.
#' @export
simsirt <- function(N, M, K, theta, b, a, c, g, d, MK = -1) {
  X <- matrix(0, nrow = N, ncol = M)
  if (MK == -1) MK <- K
  gp <- pad_mat(g, M)
  dp <- pad_mat(d, M)
  for (i in 1:N) {
    for (j in 1:M) {
      probs <- item_single_all(theta[i], b[j], a[j], c[j], gp[j, ], dp[j, ], K, MK)
      X[i, j] <- sample(MK, 1, prob = probs, replace = TRUE)
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
