//' @useDynLib sirtmm, .registration = TRUE
//' @importFrom Rcpp evalCpp

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;

const int MISS_VAL = -999;

double Z(unsigned int y, double theta, double b, double a, const arma::rowvec &g, const arma::rowvec &d)
{
  double r;
  if (y == 1 || y - 1 >= g.n_elem)
  {
    r = a * (theta - b);
    if (y - 1 < d.n_elem)
      r += d(y - 1) * (theta - b);
  }
  else
  {
    r = a * (theta - b + g(y - 1));
    if (y - 1 < d.n_elem)
      r += d(y - 1) * (theta - b + g(y - 1));
  }
  if (r > 700)
    return 700;
  return r;
}

void z_d(arma::mat &zd1, unsigned int y, double theta, double b, double a, const arma::rowvec &g, const arma::rowvec &d)
{
  zd1.fill(0);
  zd1(0) = -a;        // b
  zd1(1) = theta - b; // a
  if (1 < y && y - 1 < g.n_elem)
  {
    zd1(1) += g(y - 1); // a
    zd1(y) = a;         // g
  }
  const int d_ind = g.n_elem + y - 1;
  if (1 < y && y - 1 < d.n_elem)
  {
    zd1(0) += -d(y - 1); // b
    zd1(y) += d(y - 1);  // g
    zd1(d_ind) = zd1(1); // d
  }
}

void z_Hess(arma::mat &zH, unsigned int y, double theta, double b, double a, const arma::rowvec &g, const arma::rowvec &d)
{
  zH.fill(0);
  zH(1, 0) = -1;
  zH(0, 1) = -1;
  if (1 < y && y - 1 < g.n_elem)
  {
    zH(1, y) = 1;
    zH(y, 1) = 1;
  }
  const int d_ind = g.n_elem + y - 1;
  if (1 < y && y - 1 < d.n_elem)
  {
    zH(d_ind, 0) = -1;
    zH(0, d_ind) = -1;
    zH(d_ind, y) = 1;
    zH(y, d_ind) = 1;
  }
}

std::pair<arma::vec, arma::mat> item_deriv(int y, double theta, double b, double a, double c, const arma::rowvec &g, const arma::rowvec &dp, int K, int MK, bool extended)
{
  const int size = extended ? g.n_elem + dp.n_elem + 1 : g.n_elem + dp.n_elem;
  const unsigned int C_IDX = g.n_elem + dp.n_elem;

  arma::vec d;
  arma::mat V;
  arma::vec zd1;
  arma::mat H;
  zd1.zeros(size);
  H.zeros(size, size);
  d.zeros(size); // In this order: b a g d c
  V.zeros(size, size);

  if (MK == -1)
    MK = K;

  if (MK < K && MK == y)
  {
    if (extended)
    {
      d(C_IDX) += (1.0 - y) / (1.0 - c);
      V(C_IDX, C_IDX) += (1.0 - y) / (1.0 - c) / (1.0 - c);
    }
    for (int k = 1; k <= y - 1; ++k)
    {
      z_d(zd1, k, theta, b, a, g, dp);
      z_Hess(H, k, theta, b, a, g, dp);
      double e = exp(Z(k, theta, b, a, g, dp));
      double C = ((double)K - 1.0) / ((K - k) * (1.0 - c) + (K - 1.0) * (c + e));
      double D = e * C;
      d -= zd1 * D;
      V -= H * D + (zd1 * zd1.t()) * D * (1.0 - D);

      if (extended)
      {
        double Cd = (((double)K - k) * (1.0 - c) + (K - 1.0) * (c + e));
        double C2 = (-((double)K - k) + (K - 1.0)) / Cd;
        d(C_IDX) += -C2;
        V(C_IDX, C_IDX) += C2 * C2;
        for (unsigned int i = 0; i < C_IDX; ++i)
        {
          V(i, C_IDX) += zd1(i) * C2 * (K - 1) * e / Cd;
          V(C_IDX, i) = V(i, C_IDX);
        }
      }
    }

    return std::make_pair(d, V);
  }

  z_d(zd1, y, theta, b, a, g, dp);
  z_Hess(H, y, theta, b, a, g, dp);

  double e = exp(Z(y, theta, b, a, g, dp));
  double A = e / (c + e);
  d = zd1 * A;
  V = H * A + zd1 * zd1.t() * A * (1 - A);

  if (extended)
  {
    d(C_IDX) = 1 / (c + e) - (y - 1) / (1 - c);
    V(C_IDX, C_IDX) = -1 / (c + e) / (c + e) - (y - 1) / (1 - c) / (1 - c);
    for (unsigned int i = 0; i < C_IDX; ++i)
    {
      V(C_IDX, i) = V(i, C_IDX) = -zd1(i) * e / (c + e) / (c + e);
    }
  }

  for (int k = 1; k <= y; ++k)
  {
    e = exp(Z(k, theta, b, a, g, dp));
    double Cd = ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
    double B = e / Cd;
    z_d(zd1, k, theta, b, a, g, dp);
    z_Hess(H, k, theta, b, a, g, dp);
    d -= zd1 * B;
    V -= H * B + zd1 * zd1.t() * B * (1 - B);

    if (extended)
    {
      double C = (-(K - k) / (K - 1.0) + 1.0) / Cd;
      d(C_IDX) -= C;
      V(C_IDX, C_IDX) += C * C;
      for (unsigned int i = 0; i < C_IDX; ++i)
      {
        V(i, C_IDX) += zd1(i) * (-(K - k) / (K - 1.0) + 1.0) * e / Cd / Cd;
        V(C_IDX, i) = V(i, C_IDX);
      }
    }
  }

  return std::make_pair(d, V);
}

//' Item Response Function
//'
//' @export
// [[Rcpp::export]]
double item_single(unsigned int y, double theta, double b, double a, double c, const arma::rowvec &g, const arma::rowvec &d, int K, unsigned int MK)
{
  double num = 1;
  double den = 1;
  if (MK == -1)
    MK = K;

  // If there is maximum number of attempts.
  if (MK < K && y == MK)
  {
    for (unsigned int k = 1; k < y; ++k)
    {
      num *= (K - k) * (1.0 - c);
      den *= (K - k) * (1.0 - c) + (K - 1) * (c + exp(Z(k, theta, b, a, g, d)));
    }
    return num / den;
  }

  num = (c + exp(Z(y, theta, b, a, g, d)));
  for (unsigned int k = 1; k <= y; ++k)
  {
    if (k > 1)
      num = num * (K - k + 1.0) * (1.0 - c) / (K - 1.0);
    den = den * ((K - k) * (1.0 - c) / (K - 1.0) + c + exp(Z(k, theta, b, a, g, d)));
  }
  return num / den;
}

// [[Rcpp::export]]
arma::vec item_d(int y, double theta, double b, double a, double c, const arma::rowvec &g, const arma::rowvec &d, int K, int MK, bool extended)
{
  return item_deriv(y, theta, b, a, c, g, d, K, MK, extended).first;
}

// [[Rcpp::export]]
arma::mat item_Hess(int y, double theta, double b, double a, double c, const arma::rowvec &g, const arma::rowvec &d, int K, int MK, bool extended)
{
  return item_deriv(y, theta, b, a, c, g, d, K, MK, extended).second;
}

// [[Rcpp::export]]
double item_vec(const arma::rowvec &ys, double theta,
                const arma::vec &bs, const arma::vec &as, const arma::vec &cs, const arma::mat &gs, const arma::mat &ds, int K, int MK)
{
  double s = 1;
  for (unsigned int i = 0; i < ys.n_elem; ++i)
  {
    if (ys[i] == MISS_VAL)
    {
      continue;
    }

    s *= item_single(ys(i), theta, bs(i), as(i), cs(i), gs.row(i), ds.row(i), K, MK);
  }
  return s;
}

// [[Rcpp::export]]
List EStep(const NumericMatrix &X, const NumericVector &bs, const NumericVector &as, const NumericVector &cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
           const NumericVector &points, const NumericVector &weights)
{
  int M = X.cols();
  int PN = points.length();
  auto rMat = arma::vec(MK * PN * M);
  auto nMat = arma::vec(PN);
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  auto tmp = arma::vec(MK * PN * M);
  auto tmp2 = arma::vec(PN);

  for (int i = 0; i < X.rows(); ++i)
  {
    tmp.fill(0);
    tmp2.fill(0);
    double den = 0;
    for (int f = 0; f < PN; ++f)
    {
      double l = item_vec(X(i, _), points[f], bs, as, cs, gMat, dMat, K, MK);

      den += l * weights[f];
      for (int h = 0; h < M; ++h)
      {
        if (X(i, h) == MISS_VAL)
        {
          continue;
        }
        int k = X(i, h) - 1;
        tmp(k + f * MK + h * MK * PN) += l * weights[f];
      }
      tmp2(f) += l * weights[f];
    }

    rMat += tmp / den;
    nMat += tmp2 / den;
  }

  NumericVector rhat = wrap(rMat);
  rhat.attr("dim") = IntegerVector{MK, PN, M};
  return List::create(_["rhat"] = rhat, _["nhat"] = nMat);
}

// [[Rcpp::export]]
List fisherScoring(const NumericVector &bs, const NumericVector &as, const NumericVector &cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                   const NumericVector &rhat, const NumericVector &nhat, const NumericVector &points, bool extended,
                   double lambda, double lambda2, double lambda3, int maxIter = 50, bool LASSO = false)
{
  int M = bs.length();
  int PN = points.length();
  NumericVector newB(M);
  NumericVector newA(M);
  NumericVector newC(M, 1.0 / K);
  LogicalVector conv(M);
  arma::vec t;
  arma::mat V;
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  const unsigned int size = extended ? gMat.n_cols + dMat.n_cols + 1 : gMat.n_cols + dMat.n_cols;

  const bool nog = gMat.n_cols == 1;
  const unsigned int ng = gMat.n_cols - 1;
  const bool nod = dMat.n_cols == 1;
  const unsigned int nd = dMat.n_cols - 1;
  const unsigned int C_IDX = dMat.n_cols + gMat.n_cols;

  NumericMatrix newG(gMat.n_rows, gMat.n_cols);
  NumericMatrix newD(dMat.n_rows, dMat.n_cols);

  if (MK == -1)
    MK = K;

  for (int j = 0; j < M; ++j)
  {
    arma::vec v(size);
    v(0) = bs[j];
    v(1) = as[j];

    for (unsigned int k = 0; k < ng; ++k)
    {
      v(k + 2) = gMat(j, k + 1);
    }
    for (unsigned int k = 0; k < nd; ++k)
    {
      v(k + 2 + ng) = dMat(j, k + 1);
    }
    if (extended)
      v(C_IDX) = cs[j];
    for (int i = 0; i < maxIter; ++i)
    {
      arma::rowvec gRow(gMat.n_cols);
      arma::rowvec dRow(dMat.n_cols);
      for (unsigned int k = 0; k < ng; ++k)
      {
        gRow(k + 1) = v(k + 2);
      }
      for (unsigned int k = 0; k < nd; ++k)
      {
        dRow(k + 1) = v(k + 2 + ng);
      }
      t.zeros(size);
      V.zeros(size, size);
      for (int f = 0; f < PN; ++f)
      {
        for (int k = 0; k < MK; ++k)
        {
          auto deriv = item_deriv(k + 1, points[f], v(0), v(1), extended ? v(C_IDX) : 1.0 / K, gRow, dRow, K, MK, extended);
          auto d = deriv.first;
          auto H = deriv.second;
          t += rhat[k + f * MK + j * MK * PN] * d;
          V += -nhat[f] * item_single(k + 1, points[f], v(0), v(1), extended ? v(C_IDX) : 1.0 / K, gRow, dRow, K, MK) * H;
        }
      }

      // Newton's method hack
      //if (arma::trace(V) < (double) size / 5)
      //{
      //  V.diag() += 0.2;
      //}

      // Regularization
      if (!nog)
      {
        for (unsigned int k = 0; k < ng; ++k)
        {
          if (LASSO)
          {
            t(k + 2) += -lambda * R::sign(v(k + 2));
          }
          else
          {
            t(k + 2) += -lambda * v(k + 2);
            V(k + 2, k + 2) += lambda;
          }
        }
      }
      if (!nod)
      {
        for (unsigned int k = 0; k < nd; ++k)
        {
          if (LASSO)
          {
            t(k + 2 + ng) += -lambda2 * R::sign(v(k + 2 + ng));
          }
          else
          {
            t(k + 2 + ng) += -lambda2 * v(k + 2 + ng);
            V(k + 2 + ng, k + 2 + ng) += lambda2;
          }
        }
      }
      if (extended)
      {
        if (LASSO)
        {
          t(C_IDX) += -lambda3 * R::sign(v(C_IDX) - 1.0 / K);
        }
        else
        {
          t(C_IDX) += -lambda3 * (v(C_IDX) - 1.0 / K);
          V(C_IDX, C_IDX) += lambda3;
        }
      }

      // if (arma::det(cov) < 0.0001) break;
      if (t.has_nan()) {
        Rcout << "break: t has NaN" << std::endl;
        conv(j) = false;
        break;
      }
      if (V.has_nan()) {
        Rcout << "break: V has NaN" << std::endl;
        conv(j) = false;
        break;
      }
      auto x_tmp = arma::solve(V, t, arma::solve_opts::likely_sympd);
      auto x = x_tmp.eval();
      // Convergence
      if (arma::norm(x, 2) < 0.000001) {
        conv(j) = true;
        break;
      }

      // Rcout << x << std::endl;
      /*
      if (std::abs(x(2 + gMat.n_cols)) > 0.1) {
        if (x(2 + gMat.n_cols) > 0)
          x(2 + gMat.n_cols) = 0.1;
        else
          x(2 + gMat.n_cols) = -0.1;
      }*/
      auto x2 = arma::clamp(x, -0.1, 0.1); // Avoid moving too much
      v = v + x2;
      conv(j) = false;
    }
    newB(j) = v(0);
    newA(j) = v(1);
    if (extended)
      newC(j) = v(C_IDX);
    for (unsigned int k = 0; k < ng; ++k)
    {
      newG(j, k + 1) = v(k + 2);
    }
    for (unsigned int k = 0; k < nd; ++k)
    {
      newD(j, k + 1) = v(k + 2 + ng);
    }
  }
  if (extended)
    return List::create(_["b"] = newB,
                        _["a"] = newA,
                        _["c"] = newC,
                        _["g"] = newG,
                        _["d"] = newD,
                        _["conv"] = conv);
  else
    return List::create(_["b"] = newB,
                        _["a"] = newA,
                        _["g"] = newG,
                        _["d"] = newD,
                        _["conv"] = conv);
}

// [[Rcpp::export]]
List EMSteps(const NumericMatrix &X, int K, int MK, const NumericVector &points, const NumericVector &weights,
             double lambda, double lambda2, double lambda3, int ngs, int nds, int maxEMIter = 10, int maxNRIter = 50, bool verbose = false, bool exntended = true, bool LASSO = false)
{
  auto M = X.cols();
  NumericVector bs(M);
  bs.fill(0);
  NumericVector as(M);
  as.fill(0.1);
  NumericVector cs(M);
  cs.fill(1.0 / K);
  LogicalVector conv(M);

  // if no g parameter. Still we provide the g matrix for simplicity.
  NumericMatrix gs(M, ngs + 1);
  gs.fill(0);
  NumericMatrix ds(M, nds + 1);
  ds.fill(0);

  if (MK == -1)
    MK = K;

  for (int iter = 0; iter < maxEMIter; ++iter)
  {
    auto E = EStep(X, bs, as, cs, gs, ds, K, MK, points, weights);
    auto rhat = E["rhat"];
    auto nhat = E["nhat"];

    auto ret = fisherScoring(bs, as, cs, gs, ds, K, MK, rhat, nhat, points, exntended, lambda, lambda2, lambda3, maxNRIter, LASSO);
    bs = ret["b"];
    as = ret["a"];
    if (exntended)
      cs = ret["c"];
    conv = ret["conv"];
    gs = Rcpp::as<NumericMatrix>(ret["g"]);
    ds = Rcpp::as<NumericMatrix>(ret["d"]);
    if (verbose)
      Rcout << "EM Step"
            << " " << iter + 1 << std::endl;
  }
  if (exntended)
  {
    return List::create(_["b"] = bs,
                        _["a"] = as,
                        _["c"] = cs,
                        _["g"] = gs,
                        _["d"] = ds,
                        _["conv"] = conv);
  }
  else
  {
    return List::create(_["b"] = bs,
                        _["a"] = as,
                        _["g"] = gs,
                        _["d"] = ds,
                        _["conv"] = conv);
  }
}

// [[Rcpp::export]]
double LogLikliTotal(const NumericMatrix &X, const NumericVector &bs, const NumericVector &as, const NumericVector &cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                     const NumericVector &rhat, const NumericVector &points)
{
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  int PN = points.length();
  double ll = 0;
  if (MK == -1)
    MK = K;
  for (int j = 0; j < X.cols(); ++j)
  {
    for (int f = 0; f < PN; ++f)
    {
      for (int k = 0; k < MK; ++k)
      {
        ll += rhat[k + f * MK + j * MK * PN] * log(item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK));
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
      t += rhat[k + f * K + j * K * PN] * log(item_single(k + 1, points[f], B, A, gs, K));
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
      auto d = item_d(k + 1, points[f], B, A, gs, K);
      t += rhat[k + f * K + j * K * PN] * d;
    }
  }
  return t;
}
*/

// [[Rcpp::export]]
List SE(const NumericMatrix &X, NumericVector bs, NumericVector as, NumericVector cs,
        const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK, NumericVector points, const NumericVector &weights, bool extended)
{

  int M = bs.length();
  int PN = points.length();
  NumericVector sdB(M);
  NumericVector sdA(M);
  NumericVector sdC(M);
  NumericMatrix sdG(gs.rows(), gs.cols());
  NumericMatrix sdD(ds.rows(), ds.cols());

  arma::mat V;
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  const bool nog = gMat.n_cols == 1;
  const unsigned int ng = gMat.n_cols - 1;
  const bool nod = dMat.n_cols == 1;
  const unsigned int nd = dMat.n_cols - 1;
  auto E = EStep(X, bs, as, cs, gs, ds, K, MK, points, weights);
  NumericVector rhat = E["rhat"];
  NumericVector nhat = E["nhat"];

  const int size = extended ? gMat.n_cols + dMat.n_cols + 1 : gMat.n_cols + dMat.n_cols;

  for (int j = 0; j < M; ++j)
  {
    V.zeros(size, size);
    for (int f = 0; f < PN; ++f)
    {
      for (int k = 0; k < MK; ++k)
      {
        auto H = item_Hess(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK, extended);
        V += -nhat(f) * item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK) * H;
      }
    }
    // Rcout << "Determinant of Fisher Information Matrix is " << arma::det(V) << std::endl;
    arma::mat cov = arma::pinv(V);
    arma::vec dg = cov.diag();
    arma::vec sd = sqrt(dg);
    sdB(j) = sd(0);
    sdA(j) = sd(1);
    if (extended)
      sdC(j) = sd(cov.n_cols - 1);
    if (!nog)
    {
      for (unsigned int k = 0; k < ng; ++k)
      {
        sdG(j, k + 1) = sd(k + 2);
      }
    }
    if (!nod)
    {
      for (unsigned int k = 0; k < nd; ++k)
      {
        sdD(j, k + 1) = sd(k + 2 + ng);
      }
    }
  }

  if (extended)
  {
    if (nog)
    {
      if (nod)
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["c"] = sdC);
      }
      else
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["c"] = sdC,
                            _["d"] = sdD);
      }
    }
    else
    {
      if (nod)
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["c"] = sdC,
                            _["g"] = sdG);
      }
      else
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["c"] = sdC,
                            _["g"] = sdG,
                            _["d"] = sdD);
      }
    }
  }
  else
  {
    if (nog)
    {
      if (nod)
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA);
      }
      else
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["d"] = sdD);
      }
    }
    else
    {
      if (nod)
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["g"] = sdG);
      }
      else
      {
        return List::create(_["b"] = sdB,
                            _["a"] = sdA,
                            _["g"] = sdG,
                            _["d"] = sdD);
      }
    }
  }
}

// [[Rcpp::export]]
NumericVector eap_theta(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                        NumericVector points, NumericVector weights)
{
  int N = X.rows();
  int PN = points.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector eap(N);
  for (int i = 0; i < N; ++i)
  {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f)
    {
      double l = item_vec(X(i, _), points[f], bs, as, cs, gMat, dMat, K, MK);
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
          fi += - item_single(y, theta, b, a, g, K) * d2;
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
NumericVector PSD(NumericMatrix X, NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                  NumericVector points, NumericVector weights)
{
  int N = X.rows();
  int PN = points.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector psd(N);
  for (int i = 0; i < N; ++i)
  {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f)
    {
      double l = item_vec(X(i, _), points[f], bs, as, cs, gMat, dMat, K, MK);
      num += (points[f] - thetas(i)) * (points[f] - thetas(i)) * l * weights[f];
      den += l * weights[f];
    }
    psd[i] = std::sqrt(num / den);
  }
  return psd;
}

// [[Rcpp::export]]
NumericVector eap_theta_irt(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, int K,
                            NumericVector points, NumericVector weights)
{
  int N = X.rows();
  int M = bs.length();
  int PN = points.length();
  NumericVector eap(N);
  for (int i = 0; i < N; ++i)
  {
    double den = 0;
    double num = 0;
    for (int f = 0; f < PN; ++f)
    {
      double l = 1;
      for (int j = 0; j < M; ++j)
      {
        if (X(i, j))
          l *= (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j)))));
        else
          l *= (1 - (cs(j) + (1 - cs(j)) / (1 + exp(-as(j) * (points[f] - bs(j))))));
      }
      num += points[f] * l * weights[f];
      den += l * weights[f];
    }
    eap[i] = num / den;
  }
  return eap;
}

// [[Rcpp::export]]
NumericVector ScoreScheme(NumericMatrix X, NumericVector scheme)
{
  int N = X.rows();
  int M = X.cols();
  NumericVector scores(N);
  scores.fill(0);
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      unsigned int d = X(i, j) - 1;
      scores[i] += scheme[d];
    }
  }
  return scores;
}

// [[Rcpp::export]]
NumericVector FI(NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK)
{
  int N = thetas.length();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector fi(N);
  fi.fill(0);
  if (MK == -1)
    MK = K;
  for (int i = 0; i < N; ++i)
  {
    double theta = thetas(i);
    for (int j = 0; j < M; ++j)
    {
      auto g = gMat.row(j);
      auto d = dMat.row(j);
      double a = as(j), b = bs(j), c = cs(j), tempFI = 0;
      for (int y = 1; y <= MK; ++y)
      {
        double d2 = 0;
        double e = exp(Z(y, theta, b, a, g, d));
        double A = e / (c + e);
        double delta = 0;
        if (y - 1 < d.n_elem)
          delta = d(y - 1);
        d2 = (a + delta) * (a + delta) * A * (1 - A);

        for (int k = 1; k <= y; ++k)
        {
          e = exp(Z(k, theta, b, a, g, d));
          double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
          d2 -= (a + delta) * (a + delta) * B * (1 - B);
        }
        tempFI += -item_single(y, theta, b, a, c, g, d, K, MK) * d2;
      }
      fi(i) += tempFI;
    }
  }
  return fi;
}

// [[Rcpp::export]]
NumericVector FI3PL(NumericVector thetas, NumericVector bs, NumericVector as, NumericVector cs)
{
  int N = thetas.length();
  int M = bs.length();
  NumericVector fi(N);
  fi.fill(0);
  for (int i = 0; i < N; ++i)
  {
    double theta = thetas(i);
    for (int j = 0; j < M; ++j)
    {
      double a = as(j), b = bs(j), c = cs(j);
      double p = c + (1 - c) / (1 + std::exp(-a * (theta - b)));
      double q = 1 - p;

      fi(i) += std::pow(a * (p - c) / (1 - c), 2) * q / p;
    }
  }
  return fi;
}
