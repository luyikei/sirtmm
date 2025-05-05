#' pad_mat (internal function)
#'
#' @export
pad_mat <- function(mat, M) {
  if (!length(mat) || (!is.matrix(mat) && length(mat) == 1 && all(!mat))) {
    nmat <- matrix(0, nrow = M, ncol = 1)
  } else if ((is.matrix(mat) && nrow(mat) == M) || length(mat) == M) {
    nmat <- cbind(0, mat)
  } else {
    stop("mat, a vector or a matrix, needs to have M items!")
  }
  nmat <- unname(nmat)
  nmat
}

item_single_all <- function(theta, b, a, c, g, d, K, MK = -1) {
  if (MK == -1) MK <- K
  ret <- rep(0, MK)
  for (k in 1:MK) {
    ret[k] <- item_single(k, theta, b, a, c, g, d, K, MK)
  }
  ret
}

#' Item Parameter Estimation of Sequential Item Response Models for Multiple-Choice, Multiple-Attempt Test Items
#'
#' sirtmm fits a SIRT-MM model to number of attempts data using marginal maximum likelihood estimation with an EM algorithm.
#'
#' @param data A reponse matrix (number of attempts data).
#' @param k a number of answer options
#' @param mk a number of response categories (the max number of attempts + 1). -1 if answer-until-correct.
#' @param itemtype specify "SIRT-MMe" for an extended SIRT-MM model.
#' @param quadpts a number of quadrature points in the EM algorithm.
#' @param max_emsteps a maximum number of EM steps.
#' @param max_nriter a maximum number of iterations in the Newton's method.
#' @param lambda1 a regularization penalty for gammma.
#' @param lambda2 a regularization penalty for delta.
#' @param lambda3 a regularization penalty for c (only used when SIRT-MMe is specified).
#' @param ngs a number of gammma parameters.
#' @param nds a number of delta parameters.
#' @param verbose output debug information during estimation.
#' @param penalty Specify penalty. "Ridge", "LASSO", or "SCAD"
#' @example man/examples/sirtmm.R
#' @export
sirtmm <- function(data,
                   k = NULL,
                   mk = -1,
                   itemtype = "SIRT-MM",
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
                   penalty = "LASSO") {
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

  if (mk == -1) {
    mk <- k
  }

  extended <- ifelse(itemtype == "SIRT-MMe", TRUE, FALSE)
  
  penaltyint <- switch(penalty,
         Ridge = 0,
         LASSO = 1,
         SCAD = 2,
         AdaptiveLASSO = 3,
         999) 

  res <- EMSteps(data, k, mk, points, weights, lambda1, lambda2, lambda3,
                 ngs, nds, max_emsteps, max_nriter, verbose, extended, penaltyint)
  if (itemtype == "SIRT-MM") res[["c"]] <- rep(1 / k, ncol(data))
  se <- SE(data, res[["b"]], res[["a"]], res[["c"]], res[["g"]], res[["d"]],
           k, mk, points, weights, extended)

  conv <- res[["conv"]]
  res[["conv"]] <- NULL
  res[["g"]] <- res[["g"]][, -1]
  res[["d"]] <- res[["d"]][, -1]
  se[["g"]] <- se[["g"]][, -1]
  se[["d"]] <- se[["d"]][, -1]

  data[data == -999] <- NA

  mod <- list(
    N = nrow(data),
    M = ncol(data),
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


#' Create a model with fixed parameters
#'
#' Create a model with provided parameters
#'
#' @param mod A SIRT-MM model
#' @param data A reponse matrix used to estimate thetas
#' @param method The estimation method (EAP, ML, or WLE)
#' @param max_value The maximum absolute value for theta used for MLE estimation
#' @example man/examples/sirtmm.R
#' @export
createSirtmmModel <- function(b, a, k, g.mat = 0, d.mat = 0, c = NULL, mk = -1) {
  M <- length(b)
  if (is.null(c)) {
    c <- rep(1 / k, M)
    itemtype <- "SIRT-MM"
  } else {
    itemtype <- "SIRT-MMe"
  }
  ngs <- ifelse(all(!g.mat), 0, ncol(g.mat))
  nds <- ifelse(all(!d.mat), 0, ncol(d.mat))

  if (ngs == 0) {
    g.mat <- matrix(0, nrow = M, ncol = 1)
  }
  if (nds == 0) {
    d.mat <- matrix(0, nrow = M, ncol = 1)
  }

  mod <- list(
    data = NULL,
    options = list(
      k = k,
      mk = mk,
      itemtype = itemtype,
      ngs = ngs,
      nds = nds
    ),
    itempar = data.frame(b = b, a = a, c = c, g = g.mat, d = d.mat),
    g.mat = g.mat,
    d.mat = d.mat
  )
  class(mod) <- "sirtmmModel"
  return(mod)
}


#' ANOVA
#'
#' Test
#'
#' @param ... SIRT-MM models.
#' @example man/examples/sirtmm.R
#'
#' @export
anova.sirtmmModel <- function(...) {
  ret <- data.frame()
  models <- list(...)
  for (mod in models) {
    M <- nrow(mod$itempar)
    N <- nrow(mod$data)
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
      BIC = c(npar * log(N) - 2 * logLik),
      logLik = logLik,
      npar = npar
    ))
  }
  return(ret)
}


#' Test Information
#'
#' Test Information
#'
#' @param thetas A vector of thetas
#' @param mod A sirtmmmodel object
#' @export
testinfoSirtmm <- function(thetas, mod) {
  M <- nrow(mod$itempar)
  theta.fi <- FI(
    thetas, mod$itempar$b, mod$itempar$a, mod$itempar$c,
    pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk
  )
  return(theta.fi)
}

#' Person Estimation
#'
#' Person parameters
#'
#' @param mod A SIRT-MM model
#' @param data A reponse matrix used to estimate thetas
#' @param method The estimation method (EAP, MLE, or WLE)
#' @param max_value The maximum absolute value for theta used for MLE estimation
#' @example man/examples/sirtmm.R
#' @export
estimate <- function(mod, data = NULL, method = "EAP", max_value = 6) {
  if (is.null(data)) {
    data <- mod$data
  }

  if (method == "EAP") {
    if (is.null(mod$options$quadpts)) {
      rule <- fastGHQuad::gaussHermiteData(61)
      points <- rule$x
      weights <- rule$w
      points <- points * sqrt(2)
      weights <- weights / sqrt(pi)
    } else {
      points <- mod$options$quadpts$points
      weights <- mod$options$quadpts$weights
    }

    M <- nrow(mod$itempar)
    thetas.est <- eap_theta(
      data, mod$itempar$b, mod$itempar$a, mod$itempar$c,
      pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk, points, weights
    )
    theta.psd <- PSD(
      data, thetas.est, mod$itempar$b, mod$itempar$a, mod$itempar$c,
      pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk, points, weights
    )
    return(list(
      thetas = thetas.est,
      se = theta.psd
    ))
  } else if (method == "MLE") {
    M <- nrow(mod$itempar)
    thetas.est <- mle_theta(
      data, mod$itempar$b, mod$itempar$a, mod$itempar$c,
      pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk, max_value
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
  } else if (method == "WLE") {
    M <- nrow(mod$itempar)
    thetas.est <- wle_theta(
      data, mod$itempar$b, mod$itempar$a, mod$itempar$c,
      pad_mat(mod$g.mat, M), pad_mat(mod$d.mat, M), mod$options$k, mod$options$mk
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
  } else {
    stop("The provided method is not supported!")
  }
}


#' scoring using a scheme
#'
#' @param data A reponse matrix (the number of attempts)
#' @param scheme A scoring scheme vector
#' @export
score_scheme <- function(X, scheme) {
  return(apply(data, 1, function(x) sum(sapply(x, function(y) scheme[y]))))
}

#' Simulate SIRT-MM models
#'
#' Test
#'
#' @param N a number of subjects
#' @param M a number of items
#' @param K a number of answer options
#' @param thetas a N x 1 vector of thetas
#' @param b a M x 1 vector of b parameter
#' @param a a M x 1 vector of a parameter
#' @param ct a M x 1 vector of c parameter
#' @param g a M x V matrix of g parameter where V is the number of g parameters.
#' @param d a M x W matrix of d parameter where W is the number of d parameters.
#' @param MK a number of response categories (the number of max attempt + 1). -1 if answer-until-correct.
#' @export
simsirt <- function(N, M, K, thetas, b, a, c, g, d, MK = -1) {
  X <- matrix(0, nrow = N, ncol = M)
  if (MK == -1) MK <- K
  gp <- pad_mat(g, M)
  dp <- pad_mat(d, M)
  for (i in 1:N) {
    for (j in 1:M) {
      probs <- item_single_all(thetas[i], b[j], a[j], c[j], gp[j, ], dp[j, ], K, MK)
      X[i, j] <- sample(MK, 1, prob = probs, replace = TRUE)
    }
  }
  return(list(
    data = X,
    theta = thetas,
    itempar = data.frame(b = b, a = a, c = c, g = g, d = d),
    g.mat = g,
    d.mat = d
  ))
}
