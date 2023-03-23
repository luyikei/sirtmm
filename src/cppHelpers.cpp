// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

double Z(unsigned int y, double theta, double b, double a, const arma::rowvec& g) {
  double r;
  if (y == 1 || y - 2 >= g.n_elem)
    r = a*(theta - b);
  else
    r = a*(theta - b + g(y - 2) );
  if (r > 700) return 700;
  return r;
}

void z_d(arma::mat& zd1, unsigned int y, double theta, double b, double a, const arma::rowvec& g , int K) {
  zd1.fill(0);
  zd1(0) = -a;
  zd1(1) = theta - b;
  if (1 < y && y - 2 < g.n_elem) {
    zd1(1) += g(y - 2);
    zd1(y) = a;
  }
}

void z_Hess(arma::mat& zH, unsigned int y, double theta, double b, double a, const arma::rowvec& g , int K) {
  zH.fill(0);
  zH(0, 1) = -1;
  zH(1, 0) = -1;
  if (1 < y && y - 2 < g.n_elem) {
    zH(1, y) = 1;
    zH(y, 1) = 1;
  }
}


std::pair<arma::vec, arma::mat> mam_deriv(int y, double theta, double b, double a, double c, const arma::rowvec& g , int K) {
  arma::vec d;
  arma::mat V;
  arma::vec zd1(g.n_elem + 3);
  arma::mat zH(g.n_elem + 3, g.n_elem + 3);
  arma::mat H(g.n_elem + 3, g.n_elem + 3);
  z_d(zd1, y, theta, b, a, g, K);
  z_Hess(H, y, theta, b, a, g, K);
  d.zeros(3 + g.n_elem);
  V.zeros(3 + g.n_elem, 3 + g.n_elem);

  double e = exp( Z(y, theta, b, a, g) );
  double A = e / (c + e);
  d = zd1 * A;
  V = H * A + zd1 * zd1.t() * A * (1 - A);

  d(2 + g.n_elem) = 1 / (c + e) - (y - 1) / (1 - c);
  V(2 + g.n_elem, 2 + g.n_elem) = - 1 /  (c + e) /  (c + e) - (y - 1) / (1 - c) / (1 - c);
  for (unsigned int i = 0; i < 2 + g.n_elem; ++i) {
    V(2 + g.n_elem, i) = V(i, 2 + g.n_elem) = - zd1(i) * e /  (c + e) /  (c + e);
  }

  for (int k = 1; k <= y; ++k) {
    e =  exp( Z(k, theta, b, a, g) );
    double Cd = (( 1.0 - c) * (K - k) / (K - 1.0) + c + e);
    double B = e / Cd;
    z_d(zd1, k, theta, b, a, g, K);
    z_Hess(H, k, theta, b, a, g, K);
    d -= zd1 * B;
    V -= H * B + zd1 * zd1.t() * B * (1 - B);

    double C = (-(K - k) / (K - 1.0) + 1.0) / Cd;
    d(2 + g.n_elem) -= C;
    V(2 + g.n_elem, 2 + g.n_elem) += C * C;
    for (unsigned int i = 0; i < 2 + g.n_elem; ++i) {
      V(i, 2 + g.n_elem) += zd1(i) * (-(K - k) / (K - 1.0) + 1.0) * e / Cd / Cd;
      V(2 + g.n_elem, i) = V(i, 2 + g.n_elem);
    }
  }

  return std::make_pair(d, V);
}

// [[Rcpp::export]]
double mam_single(unsigned int y, double theta, double b, double a, double c, const arma::rowvec& g, int K) {
  double num = (c + exp( Z(y, theta, b, a, g) )) ;
  double den = 1;
  for (unsigned int k = 1; k <= y; ++k) {
    if (k > 1)
      num = num * (K - k + 1) * (1 - c) / (K - 1);
    den = den * ((K - k) * (1 - c) / (K - 1) + c + exp( Z(k, theta, b, a, g) ));
  }
  return num / den;
}

// [[Rcpp::export]]
arma::vec mam_d(int y, double theta, double b, double a, double c, const arma::rowvec& g , int K) {
  return mam_deriv(y, theta, b, a, c, g, K).first;
}

// [[Rcpp::export]]
arma::mat mam_Hess(int y, double theta, double b, double a, double c, const arma::rowvec& g, int K) {
  return mam_deriv(y, theta, b, a, c, g, K).second;
}


// [[Rcpp::export]]
double mam_vec(const arma::rowvec& ys, double theta,
              const arma::vec& bs, const arma::vec& as, const arma::vec& cs, const arma::mat& gs, int K) {
  double s = 1;
  for (unsigned int i = 0; i < ys.n_elem; ++i) {
    double num = 1;
    double den = 1;
    for (int k = 1; k <= ys[i]; ++k) {
      if (k > 1)
        num = num * (K - k + 1.0) * (1.0 - cs(i)) / (K - 1.0);
      den = den * ((K - k) * (1.0 - cs(i)) / (K - 1) + cs(i) + exp( Z(k, theta, bs(i), as(i), gs.row(i))  ));
    }
    s *= ((cs(i) +  exp( Z(ys(i), theta, bs(i), as(i), gs.row(i))  )) *  num / den);
  }
  return s;
}

// [[Rcpp::export]]
List EStep(const NumericMatrix& X, const NumericVector& bs, const NumericVector& as, const NumericVector& cs, const NumericMatrix& gs, int K,
          const NumericVector& points, const NumericVector& weights) {
  int M = X.cols();
  int PN = points.length();
  auto rMat = arma::vec(K * PN * M);
  auto nMat = arma::vec(PN);
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto tmp = arma::vec(K * PN * M);
  auto tmp2 = arma::vec(PN);
  
  for (int i = 0; i < X.rows(); ++i) {
    tmp.fill(0);
    tmp2.fill(0);
    double den = 0;
    for (int f = 0; f < PN; ++f) {
      double l = mam_vec(X(i, _), points[f], bs , as, cs, gMat, K);
      
      den += l * weights[f];
      for (int h = 0; h < M; ++h) {
        int k = X(i,h) - 1;
        tmp(k + f * K + h * K * PN) +=  l * weights[f];
      }
      tmp2(f) +=  l * weights[f];
    }
    
    rMat += tmp / den;
    nMat += tmp2 / den;
  }

  NumericVector rhat = wrap(rMat);
  rhat.attr("dim") = IntegerVector{K, PN, M};
  return List::create(_["rhat"] = rhat, _["nhat"] = nMat);
}

// [[Rcpp::export]]
List fisherScoring(const NumericVector& bs, const NumericVector& as, const NumericVector& cs, const NumericMatrix& gs, int K ,
                  const NumericVector& rhat, const NumericVector& nhat, const NumericVector& points, double lambda, double lambda2, int maxIter = 50) {
  int M = bs.length();
  int PN = points.length();
  NumericVector newB(M);
  NumericVector newA(M);
  NumericVector newC(M);
  NumericMatrix newG(gs.rows(), gs.cols());
  arma::vec t;
  arma::mat V;
  auto gMat = Rcpp::as<arma::mat>(gs);
  for (int j = 0; j < M; ++j) {
    arma::vec v(3 + gMat.n_cols);
    v(0) = bs[j];
    v(1) = as[j];
    for (unsigned int k = 0; k < gMat.n_cols; ++k) {
      v(k + 2) = gMat(j, k);
    }
    v(2 + gMat.n_cols) = cs[j];
    for (int i = 0; i < maxIter; ++i) {
      arma::rowvec gRow(gMat.n_cols);
      for (unsigned int k = 0; k < gMat.n_cols; ++k) {
        gRow(k) = v(k + 2);
      }
      t.zeros(3 + gMat.n_cols);
      V.zeros(3 + gMat.n_cols, 3 + gMat.n_cols);
      for (int f = 0; f < PN; ++f) {
        for (int k = 0; k < K; ++k) {
          auto d = mam_d(k + 1, points[f], v(0), v(1), v(2 + gMat.n_cols), gRow, K);
          auto H = mam_Hess(k + 1, points[f], v(0), v(1), v(2 + gMat.n_cols), gRow, K);
          t += rhat[k + f * K + j * K * PN] * d;
          V += -nhat[f] * mam_single(k + 1, points[f], v(0), v(1), v(2 + gMat.n_cols), gRow, K) * H;
        }
      }

      
      // Newton's method hack
      if (arma::trace(V) < 3 + gMat.n_cols) {
        V.diag() += 1.0;
      }
      
      // Regularization
      for (unsigned int k = 0; k < gMat.n_cols; ++k) {
        t(k + 2) += - lambda * v(k + 2);
        V(k + 2, k + 2) += lambda;
      }

       t(2 + gMat.n_cols) += - lambda2 * (v(2 + gMat.n_cols) - 1.0 / K );
       V(2 + gMat.n_cols, 2 + gMat.n_cols) += lambda2;

      //if (arma::det(cov) < 0.0001) break;
      if (t.has_nan()) break;
      if (V.has_nan()) break;
      auto x_tmp = arma::solve(V, t, arma::solve_opts::likely_sympd);
      auto x = x_tmp.eval();
      // Convergence
      if (arma::norm(x, 2) < 0.000001) break;
      /*
      if (std::abs(x(2 + gMat.n_cols)) > 0.1) {
        if (x(2 + gMat.n_cols) > 0)
          x(2 + gMat.n_cols) = 0.1;
        else
          x(2 + gMat.n_cols) = -0.1;
      }*/
      auto x2 = arma::clamp(x, -0.3, 0.3); // Avoid moving too much
      v = v + x2;
    }
    newB(j) = v(0);
    newA(j) = v(1);
    newC(j) = v(2 + gMat.n_cols);
    for (unsigned int k = 0; k < gMat.n_cols; ++k) {
      newG(j, k) = v(k + 2);
    }
  }
  return List::create(_["b"] = newB,
                             _["a"] = newA,
                             _["c"] = newC,
                             _["g"] = newG);
}

// [[Rcpp::export]]
List EMSteps(const NumericMatrix& X, int K , const NumericVector& points, const NumericVector& weights,
              double lambda, double lambda2, double ngs, int maxEMIter = 10, int maxNRIter = 50, bool verbose = false) {
  auto M = X.cols();
  NumericVector bs(M);
  bs.fill(0);
  NumericVector as(M);
  as.fill(0.1);
  NumericVector cs(M);
  cs.fill(1.0 / K);
  NumericMatrix gs(M, ngs);
  gs.fill(0);

  for (int iter = 0; iter < maxEMIter; ++iter) {
    auto E = EStep(X, bs, as, cs, gs, K, points, weights);
    auto rhat = E["rhat"];
    auto nhat = E["nhat"];

    auto ret = fisherScoring(bs, as, cs, gs, K, rhat, nhat, points, lambda, lambda2, maxNRIter);
    bs = ret["b"];
    as = ret["a"];
    cs = ret["c"];
    gs = Rcpp::as<NumericMatrix>(ret["g"]);
    if (verbose) Rcout << "EM Step" << " " << iter + 1 << std::endl;
  } 
  return List::create(_["b"] = bs,
                             _["a"] = as,
                             _["c"] = cs,
                             _["g"] = gs);
}

// [[Rcpp::export]]
double LogLikliTotal(const NumericMatrix& X, const NumericVector& bs, const NumericVector& as, const NumericVector& cs, const NumericMatrix& gs, int K,
          const NumericVector& rhat, const NumericVector& points) {
  auto gMat = Rcpp::as<arma::mat>(gs);
  int PN = points.length();
  double ll = 0;
  for (int j = 0; j < X.cols(); ++j) {
    for (int f = 0; f < PN; ++f) {
      for (int k = 0; k < K; ++k) {
        ll += rhat[k + f * K + j * K * PN] * log(mam_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), K));
      }
    }
  }
  return ll;
}

/*
// [[Rcpp::export]]
double LogLikli(double j, double B, double A, arma::rowvec gs, int K,
          const NumericVector& rhat, const NumericVector& points, double lambda) {
  int PN = points.length();
  double t = 0;
  for (int f = 0; f < PN; ++f) {
    for (int k = 0; k < K; ++k) {
      t += rhat[k + f * K + j * K * PN] * log(mam_single(k + 1, points[f], B, A, gs, K));
    }
  }
  return t;
}

// [[Rcpp::export]]
arma::vec LogLikliGrad(double j, double B, double A, arma::rowvec gs, int K,
          const NumericVector& rhat, const NumericVector& points, double lambda) {
  int PN = points.length();
  arma::vec t;
  t.zeros(2 + gs.n_elem);
  for (int f = 0; f < PN; ++f) {
    for (int k = 0; k < K; ++k) {
      auto d = mam_d(k + 1, points[f], B, A, gs, K);
      t += rhat[k + f * K + j * K * PN] * d;
    }
  }
  return t;
}
*/

// [[Rcpp::export]]
List SE(const NumericMatrix& X, NumericVector bs, NumericVector as, NumericVector cs,
        const NumericMatrix& gs, int K, NumericVector points, const NumericVector& weights) {

  int M = bs.length();
  int PN = points.length();
  NumericVector sdB(M);
  NumericVector sdA(M);
  NumericVector sdC(M);
  NumericMatrix sdG(gs.rows(), gs.cols());
  arma::mat V;
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto E = EStep(X, bs, as, cs, gs, K, points, weights);
  NumericVector rhat = E["rhat"];
  NumericVector nhat = E["nhat"];
  for (int j = 0; j < M; ++j) {
    arma::rowvec gRow = gMat.row(j);
    V.zeros(3 + gMat.n_cols, 3 + gMat.n_cols);
    for (int f = 0; f < PN; ++f) {
      for (int k = 0; k < K; ++k) {
        auto H = mam_Hess(k + 1, points[f], bs[j], as[j], cs[j], gRow, K);
        V += -nhat(f) * mam_single(k + 1, points[f], bs[j], as[j], cs[j], gRow, K) * H;
      }
    }
    //Rcout << "Determinant of Fisher Information Matrix is " << arma::det(V) << std::endl;
    arma::mat cov = arma::pinv(V);
    arma::vec dg = cov.diag();
    arma::vec sd = sqrt(dg);
    sdB(j) = sd(0);
    sdA(j) = sd(1);
    sdC(j) = sd(gMat.n_cols + 2);
    for (unsigned int k = 0; k < gMat.n_cols; ++k) {
      sdG(j, k) = sd(k + 2);
    }
  }
  return List::create(_["b"] = sdB,
                             _["a"] = sdA,
                             _["c"] = sdC,
                             _["g"] = sdG);
}


// [[Rcpp::export]]
NumericVector eap_theta(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix& gs, int K, 
                        NumericVector points, NumericVector weights){
  int N = X.rows();
  int PN = points.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  NumericVector eap(N);
  for (int i = 0; i < N; ++i) {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f) {
      double l = mam_vec(X(i, _), points[f], bs , as, cs, gMat, K);
      num += points[f] * l * weights[f];
      den += l * weights[f];
    }
    eap[i] = num / den;
  }
  return eap;
}
/*
// [[Rcpp::export]]
NumericVector mle_theta(NumericMatrix X, NumericVector bs, NumericVector as, const NumericMatrix& gs, int K, 
                        NumericVector points, NumericVector weights){
  int N = X.rows();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  NumericVector est(N);
  est.fill(0);
  for (int i = 0; i < N; ++i) {
    double theta = 0;
    for (int iter = 0; iter < 50; ++iter) {
      double fi = 0;
      double grad = 0;
      for (int j = 0; j < M; ++j) {
        auto g = gMat.row(j);
        double a = as(j), b = bs(j);
        for (int y = 1; y <= K; ++y) {
          double d2 = 0;
          double e = K * exp( Z(y, theta, b, a, g) );
          double A = e / (1 + e);
          d2 = a * a * A * (1 - A);
          if (y == X(i,j))
            grad += a * A;

          for (int k = 1; k <= y; ++k) {
            e = K * exp( Z(k, theta, b, a, g) );
            double B = e / (K - k + 1 + e);
            d2 -= a * a * B * (1 - B);
            if (y == X(i,j))
              grad -= a*B;
          }
          fi += - mam_single(y, theta, b, a, g, K) * d2;
        }
      }
      double x = grad / fi;
      if (std::abs(x) > 0.1)
        x = 0.1 * (grad / fi) / std::abs(x);
      theta = theta + x;
    }
    est(i) = theta;
  }
  return est;
}*/

// [[Rcpp::export]]
NumericVector PSD(NumericMatrix X, NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix& gs, int K, 
                        NumericVector points, NumericVector weights){
  int N = X.rows();
  int PN = points.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  NumericVector psd(N);
  for (int i = 0; i < N; ++i) {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f) {
      double l = mam_vec(X(i, _), points[f], bs , as, cs, gMat, K);
      num += (points[f] - thetas(i)) * (points[f] - thetas(i)) * l * weights[f];
      den += l * weights[f];
    }
    psd[i] = std::sqrt(num / den);
  }
  return psd;
}



// [[Rcpp::export]]
NumericVector eap_theta_irt(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, int K, 
                        NumericVector points, NumericVector weights){
  int N = X.rows();
  int M = bs.length();
  int PN = points.length();
  NumericVector eap(N);
  for (int i = 0; i < N; ++i) {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f) {
      double l = 1;
      for (int j = 0; j < M; ++j) {
        if (X(i,j))
          l *= (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j)))) );
        else
          l *= (1 - (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j)))) ));
      }
      num += points[f] * l * weights[f];
      den += l * weights[f];
    }
    eap[i] = num / den;
  }
  return eap;
}


// [[Rcpp::export]]
NumericVector PSD3PL(NumericMatrix X, NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs, 
                        NumericVector points, NumericVector weights){
  int N = X.rows();
  int M = bs.length();
  int PN = points.length();
  NumericVector psd(N);
  for (int i = 0; i < N; ++i) {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f) {
      double l = 1;
      for (int j = 0; j < M; ++j) {
        if (X(i,j))
          l *= (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j)))) );
        else
          l *= (1 - (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j)))) ));
      }
      num += (points[f] - thetas(i)) * (points[f] - thetas(i)) * l * weights[f];
      den += l * weights[f];
    }
    psd[i] = std::sqrt(num / den);
  }
  return psd;
}

// [[Rcpp::export]]
NumericVector Score(NumericMatrix X, NumericVector scheme){
  int N = X.rows();
  int M = X.cols();
  NumericVector scores(N);
  scores.fill(0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      unsigned int d = X(i, j) - 1;
      scores[i] += scheme[d];
    }
  }
  return scores;
}


// [[Rcpp::export]]
NumericVector FI(NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix& gs, int K){
  int N = thetas.length();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  NumericVector fi(N);
  fi.fill(0);
  for (int i = 0; i < N; ++i) {
    double theta = thetas(i);
    for (int j = 0; j < M; ++j) {
      auto g = gMat.row(j);
      double a = as(j), b = bs(j), c = cs(j), tempFI = 0;
      for (int y = 1; y <= K; ++y) {
        double d2 = 0;
        double e = exp( Z(y, theta, b, a, g) );
        double A = e / (c + e);
        d2 = a * a * A * (1 - A);

        for (int k = 1; k <= y; ++k) {
          e = exp( Z(k, theta, b, a, g) );
          double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
          d2 -= a * a * B * (1 - B);
        }
        tempFI += - mam_single(y, theta, b, a, c, g, K) * d2;
      }
      fi(i) += tempFI;
    }
  }
  return fi;
}


// [[Rcpp::export]]
NumericVector FI3PL(NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs){
  int N = thetas.length();
  int M = bs.length();
  NumericVector fi(N);
  fi.fill(0);
  for (int i = 0; i < N; ++i) {
    double theta = thetas(i);
    for (int j = 0; j < M; ++j) {
      double a = as(j), b = bs(j), c = cs(j);
      double p = c + (1 - c) / (1 + std::exp(-a*(theta - b)));
      double q = 1 - p;

      fi(i) += std::pow(a * (p - c) / (1 - c), 2) * q / p;
    }
  }
  return fi;
}
