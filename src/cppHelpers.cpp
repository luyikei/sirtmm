//' @useDynLib sirtmm, .registration = TRUE
//' @importFrom Rcpp evalCpp

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>
using namespace Rcpp;

const int MISS_VAL = -999;
const double sml = 1e-6;
const double lsml = 1e-3;

enum class RegType
{
  Ridge,
  LASSO,
  SCAD,
  AdaptiveLASSO
};

double LSE(double a, double b)
{
  double m = std::max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

double Z(unsigned int y, double theta, double b, double a, const arma::rowvec &g, const arma::rowvec &d, bool nobround = false)
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
  // if (!nobround && r > 700)
  //   return 700;
  return r;
}

void z_d(arma::mat &zd1, unsigned int y, double theta, double b, double a, const arma::rowvec &g, const arma::rowvec &d)
{
  zd1.fill(0.0);
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
  zH.fill(0.0);
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

  // double e  = exp(z);
  // double A  = e / (c + e);

  double z = Z(y, theta, b, a, g, dp);
  double A = 1.0 / (1.0 + c * exp(-z));
  d = zd1 * A;
  V = H * A + zd1 * zd1.t() * A * (1 - A);

  if (extended)
  {
    double logDen = LSE(log(c), z);           // log(c + e^z)
    double invDen = exp(-logDen);             // 1 / (c + e)
    double e_over_den2 = exp(z - 2 * logDen); // e / (c + e)^2

    // d(C_IDX) = 1.0 / (c + e) - (y - 1) / (1 - c);
    d(C_IDX) = invDen - ((double)y - 1.0) / (1.0 - c);

    // V(C_IDX, C_IDX) = -1.0 / (c + e) / (c + e) - (y - 1) / (1 - c) / (1 - c);
    V(C_IDX, C_IDX) = -invDen * invDen - ((double)y - 1.0) / ((1.0 - c) * (1.0 - c));

    for (unsigned int i = 0; i < C_IDX; ++i)
    {
      // V(C_IDX, i) = V(i, C_IDX) = -zd1(i) * e / (c + e) / (c + e);
      V(C_IDX, i) = V(i, C_IDX) = -zd1(i) * e_over_den2;
    }
  }

  for (int k = 1; k <= y; ++k)
  {
    // e = exp(Z(k, theta, b, a, g, dp));
    //   double Cd = ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
    //   double B = e / Cd;
    double z = Z(k, theta, b, a, g, dp);
    double B = 1.0 / (((1.0 - c) * (K - k) / (K - 1.0) + c) * exp(-z) + 1.0);

    z_d(zd1, k, theta, b, a, g, dp);
    z_Hess(H, k, theta, b, a, g, dp);
    d -= zd1 * B;
    V -= H * B + zd1 * zd1.t() * B * (1 - B);

    if (extended)
    {
      double coef = (-((double)K - k) / (K - 1.0) + 1.0);
      double logCd = LSE(log((1.0 - c) * (K - k) / (K - 1.0) + c), z); // log( (1‑c)(K‑k)/(K‑1)+c ,  e )
      double invCd = exp(-logCd);                                      // 1 / Cd
      double C = coef * invCd;                                         // old C
      // double C = (-((double)K - k) / (K - 1.0) + 1.0) / Cd;

      // if (d.has_nan())
      // {
      //   Rcout << "!--" << std::endl;
      //   Rcout << c << std::endl;
      //   Rcout << k << std::endl;
      //   Rcout << z << std::endl;
      //   Rcout << logCd << std::endl;
      //   Rcout << C << std::endl;
      //   Rcout << "--!" << std::endl;
      // }

      d(C_IDX) -= C;
      V(C_IDX, C_IDX) += C * C;
      for (unsigned int i = 0; i < C_IDX; ++i)
      {
        // V(i, C_IDX) += zd1(i) * (-((double)K - k) / (K - 1.0) + 1.0) * e / Cd / Cd;
        double e_over_Cd2 = exp(z - 2 * logCd); // e / Cd²
        V(i, C_IDX) += zd1(i) * coef * e_over_Cd2;
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
double item_single(
    unsigned int y, double theta, double b, double a, double c,
    const arma::rowvec &g, const arma::rowvec &d, int K, unsigned int MK)
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

//' Item Response Function
//'
//' @export
// [[Rcpp::export]]
double item_single_log(
    unsigned int y, double theta, double b, double a, double c,
    const arma::rowvec &g, const arma::rowvec &d, int K, unsigned int MK)
{
  double num = 0;
  double den = 0;
  if (MK == -1)
    MK = K;

  // If there is maximum number of attempts.
  if (MK < K && y == MK)
  {
    for (unsigned int k = 1; k < y; ++k)
    {
      num += log(K - k) + log(1.0 - c);
      den += log(K - 1.0) + LSE(log(((K - k) * (1.0 - c)) / (K - 1.0) + c), Z(k, theta, b, a, g, d, true));
    }
    return num - den;
  }

  num = LSE(log(c), Z(y, theta, b, a, g, d, true));
  for (unsigned int k = 1; k <= y; ++k)
  {
    if (k > 1)
      num += log(K - k + 1.0) + log(1.0 - c) - log(K - 1.0);
    den += LSE(log((K - k) * (1.0 - c) / (K - 1.0) + c), Z(k, theta, b, a, g, d, true));
  }
  return num - den;
}

// [[Rcpp::export]]
arma::vec item_d(int y, double theta, double b, double a, double c,
                 const arma::rowvec &g, const arma::rowvec &d, int K, int MK, bool extended)
{
  return item_deriv(y, theta, b, a, c, g, d, K, MK, extended).first;
}

// [[Rcpp::export]]
arma::mat item_Hess(int y, double theta, double b, double a, double c, const arma::rowvec &g, const arma::rowvec &d, int K, int MK, bool extended)
{
  return item_deriv(y, theta, b, a, c, g, d, K, MK, extended).second;
}

// [[Rcpp::export]]
double item_vec_log(const arma::rowvec &ys, double theta,
                    const arma::vec &bs, const arma::vec &as, const arma::vec &cs, const arma::mat &gs, const arma::mat &ds, int K, int MK)
{
  double s = 0;
  for (unsigned int i = 0; i < ys.n_elem; ++i)
  {
    if (ys[i] == MISS_VAL)
    {
      continue;
    }

    s += item_single_log(ys(i), theta, bs(i), as(i), cs(i), gs.row(i), ds.row(i), K, MK);
  }
  return s;
}

// [[Rcpp::export]]
double item_vec(const arma::rowvec &ys, double theta,
                const arma::vec &bs, const arma::vec &as, const arma::vec &cs, const arma::mat &gs, const arma::mat &ds, int K, int MK)
{
  return exp(item_vec_log(ys, theta, bs, as, cs, gs, ds, K, MK));
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

double SCAD1d(double alpha, double beta, double L)
{
  double abeta = std::abs(beta);
  if (abeta <= L)
  {
    return L;
  }
  else if (L < abeta && abeta <= alpha * L)
  {
    return (alpha * L - abeta) / (alpha - 1);
  }
  return 0.0;
};

// [[Rcpp::export]]
List iterativeEstimation(int N, const NumericVector &bs, const NumericVector &as, const NumericVector &cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                         const NumericVector &rhat, const NumericVector &nhat, const NumericVector &points, const arma::mat &pweights,
                         bool extended, double lambda, double lambda2, double lambda3, int maxIter = 50, unsigned int penalty = 0, bool scoring = true)
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
  // auto SCAD2d = [](double alpha, double beta, double lambda)
  // {
  //   double abeta = std::abs(beta);
  //   if (lambda < abeta && abeta <= alpha * lambda)
  //   {
  //     return -1.0 / (alpha - 1);
  //   }
  //   return 0.0;
  // };

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

      // arma::rowvec NK(MK);
      // for (int k = 0; k < MK; ++k)
      //   NK(k) = N;
      //  Rcout << NK << std::endl;

      for (int f = 0; f < PN; ++f)
      {
        for (int k = 0; k < MK; ++k)
        {
          auto deriv = item_deriv(k + 1, points[f], v(0), v(1), extended ? v(C_IDX) : 1.0 / K, gRow, dRow, K, MK, extended);
          auto d = deriv.first;
          auto H = deriv.second;
          t += rhat[k + f * MK + j * MK * PN] * d;
          if (scoring)
          {
            V += -nhat[f] * item_single(k + 1, points[f], v(0), v(1), extended ? v(C_IDX) : 1.0 / K, gRow, dRow, K, MK) * H;
          }
          else
          {
            V += -rhat[k + f * MK + j * MK * PN] * H;
          }
          // for (int kk = k + 1; kk < MK; ++kk)
          //   NK(kk) -= rhat[k + f * MK + j * MK * PN];
        }
      }

      // Rcout << NK << std::endl;

      // Newton's method hack
      // if (arma::trace(V) < (double)size / 5)
      //{
      //  V.diag() += 0.01;
      //}

      // Regularization
      if (!nog)
      {
        for (unsigned int k = 0; k < ng; ++k)
        {
          if (RegType(penalty) == RegType::LASSO || (i < 5 && RegType(penalty) == RegType::SCAD))
          {
            t(k + 2) += -lambda * N * (v(k + 2)) / (std::abs(v(k + 2)) + sml);
            V(k + 2, k + 2) += lambda * N / (std::abs(v(k + 2)) + sml);
          }
          else if (RegType(penalty) == RegType::Ridge)
          {
            t(k + 2) += -lambda * N * v(k + 2);
            V(k + 2, k + 2) += lambda * N;
          }
          else if (RegType(penalty) == RegType::SCAD)
          {
            double beta = v(k + 2);
            double pen = SCAD1d(3.7, beta, lambda);

            t(k + 2) += -N * pen * beta / std::abs(beta + sml);
            V(k + 2, k + 2) += N * pen / std::abs(beta + sml);
          }
          else if (RegType(penalty) == RegType::AdaptiveLASSO)
          {
            t(k + 2) += -lambda / (abs(pweights(j, k)) + sml) * N * (v(k + 2)) / (std::abs(v(k + 2)) + sml);
            V(k + 2, k + 2) += lambda / (abs(pweights(j, k)) + sml) * N / (std::abs(v(k + 2)) + sml);
          }
        }
      }
      if (!nod)
      {
        for (unsigned int k = 0; k < nd; ++k)
        {
          if (RegType(penalty) == RegType::LASSO || (i < 5 && RegType(penalty) == RegType::SCAD))
          {
            t(k + 2 + ng) += -lambda2 * N * (v(k + 2 + ng)) / (std::abs(v(k + 2 + ng)) + sml);
            V(k + 2 + ng, k + 2 + ng) += lambda2 * N / (std::abs(v(k + 2 + ng)) + sml);
          }
          else if (RegType(penalty) == RegType::Ridge)
          {
            t(k + 2 + ng) += -lambda2 * N * v(k + 2 + ng);
            V(k + 2 + ng, k + 2 + ng) += lambda2 * N;
          }
          else if (RegType(penalty) == RegType::SCAD)
          {
            double beta = v(k + 2 + ng);
            double pen = SCAD1d(3.7, beta, lambda2);

            t(k + 2 + ng) += -N * pen * beta / std::abs(beta + sml);
            V(k + 2 + ng, k + 2 + ng) += N * pen / std::abs(beta + sml);
          }
          else if (RegType(penalty) == RegType::AdaptiveLASSO)
          {
            t(k + 2 + ng) += -lambda2 / (abs(pweights(j, k + ng)) + sml) * N * (v(k + 2 + ng)) / (std::abs(v(k + 2 + ng)) + sml);
            V(k + 2 + ng, k + 2 + ng) += lambda2 / (abs(pweights(j, k + ng)) + sml) * N / (std::abs(v(k + 2 + ng)) + sml);
          }
        }
      }
      if (extended)
      {
        if (RegType(penalty) == RegType::LASSO || (i < 5 && RegType(penalty) == RegType::SCAD))
        {
          t(C_IDX) += -lambda3 * N * (v(C_IDX) - 1.0 / K) / (std::abs(v(C_IDX) - 1.0 / K) + sml);
          V(C_IDX, C_IDX) += lambda3 * N / (std::abs(v(C_IDX) - 1.0 / K) + sml);
        }
        else if (RegType(penalty) == RegType::Ridge)
        {
          t(C_IDX) += -lambda3 * N * (v(C_IDX) - 1.0 / K);
          V(C_IDX, C_IDX) += lambda3 * N;
        }
        else if (RegType(penalty) == RegType::SCAD)
        {
          double beta = (v(C_IDX) - 1.0 / K);
          double pen = SCAD1d(3.7, beta, lambda3);

          t(C_IDX) += -N * pen * beta / std::abs(beta + sml);
          V(C_IDX, C_IDX) += N * pen / std::abs(beta + sml);
        }
        else if (RegType(penalty) == RegType::AdaptiveLASSO)
        {
          t(C_IDX) += -lambda3 / (abs(pweights(j, ng + nd)) + sml) * N * (v(C_IDX) - 1.0 / K) / (std::abs(v(C_IDX) - 1.0 / K) + sml);
          V(C_IDX, C_IDX) += lambda3 / (abs(pweights(j, ng + nd)) + sml) * N / (std::abs(v(C_IDX) - 1.0 / K) + sml);
        }
      }

      // if (arma::det(cov) < 0.0001) break;
      if (t.has_nan())
      {
        Rcout << "break: t has NaN" << std::endl;
        conv(j) = false;
        break;
      }
      if (V.has_nan())
      {
        Rcout << "break: V has NaN" << std::endl;
        conv(j) = false;
        break;
      }
      auto x_tmp = arma::solve(V, t, arma::solve_opts::likely_sympd);
      auto x = x_tmp.eval();
      // Convergence
      if (arma::norm(x, 2) < 0.000001)
      {
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
      if (extended)
      {
        double c = v(C_IDX);
        if (c < 0.0)
          c = 0.0;
        else if (c > 1.0)
          c = 1.0;
        v(C_IDX) = c;
      }
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
List SE(const NumericMatrix &X, NumericVector bs, NumericVector as, NumericVector cs,
        const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK, NumericVector points, const NumericVector &weights,
        bool extended, const arma::mat &pweights, int penalty = 0, bool sandwitch = true, double lambda = 0, double lambda2 = 0, double lambda3 = 0)
{
  int N = X.nrow();
  int M = bs.length();
  int PN = points.length();
  NumericVector sdB(M);
  NumericVector sdA(M);
  NumericVector sdC(M);
  NumericMatrix sdG(gs.rows(), gs.cols());
  NumericMatrix sdD(ds.rows(), ds.cols());

  arma::mat J;
  arma::mat V;
  // arma::vec El;
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
    J.zeros(size, size);
    if (sandwitch)
    {
      V.zeros(size, size);
    }

    for (int f = 0; f < PN; ++f)
    {
      for (int k = 0; k < MK; ++k)
      {
        auto deriv = item_deriv(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK, extended);
        auto d = deriv.first;
        auto H = deriv.second;
        if (sandwitch)
        {
          // V_n(θ) = var_g{∇ ln ℓ(θ)}
          // J_n(θ) = −E_g{∇² ln ℓ(θ)}
          // J_n(θ)^-1 V_n(θ) J_n(θ)^-1
          J += -nhat(f) * item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK) * H;
          V += nhat(f) * item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK) * d * d.t();
          // El += nhat(f) * item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK) * d;

          if (!nog)
          {
            for (unsigned int k = 0; k < ng; ++k)
            {
              double g = gMat(j, 1 + k);
              if (RegType(penalty) == RegType::LASSO)
                J(k + 2, k + 2) += lambda * N / (std::abs(g) + sml);
              else if (RegType(penalty) == RegType::Ridge)
                J(k + 2, k + 2) += lambda * N;
              else if (RegType(penalty) == RegType::SCAD)
              {
                double pen = SCAD1d(3.7, g, lambda);
                J(k + 2, k + 2) += N * pen / std::abs(g + sml);
              }
              else if (RegType(penalty) == RegType::AdaptiveLASSO)
                J(k + 2, k + 2) += lambda / (abs(pweights(j, k)) + sml) * N / (std::abs(g) + sml);
            }
          }
          if (!nod)
          {
            for (unsigned int k = 0; k < nd; ++k)
            {
              double d = dMat(j, 1 + k);
              if (RegType(penalty) == RegType::LASSO)
                J(k + 2 + ng, k + 2 + ng) += lambda2 * N / (std::abs(d) + sml);
              else if (RegType(penalty) == RegType::Ridge)
                J(k + 2 + ng, k + 2 + ng) += lambda2 * N;
              else if (RegType(penalty) == RegType::SCAD)
              {
                double pen = SCAD1d(3.7, d, lambda2);
                J(k + 2 + ng, k + 2 + ng) += N * pen / std::abs(d + sml);
              }
              else if (RegType(penalty) == RegType::AdaptiveLASSO)
                J(k + 2 + ng, k + 2 + ng) += lambda2 / (abs(pweights(j, k + ng)) + sml) * N / (std::abs(d) + sml);
            }
          }
          if (extended)
          {
            unsigned C_IDX = size - 1;
            if (RegType(penalty) == RegType::LASSO)
              J(C_IDX, C_IDX) += lambda3 * N / (std::abs(cs[j] - 1.0 / K) + sml);
            else if (RegType(penalty) == RegType::Ridge)
              J(C_IDX, C_IDX) += lambda3 * N;
            else if (RegType(penalty) == RegType::SCAD)
            {
              double beta = (cs[j] - 1.0 / K);
              double pen = SCAD1d(3.7, beta, lambda3);
              J(C_IDX, C_IDX) += N * pen / std::abs(beta + sml);
            }
            else if (RegType(penalty) == RegType::AdaptiveLASSO)
              J(C_IDX, C_IDX) += lambda3 / (abs(pweights(j, ng + nd)) + sml) * N / (std::abs(cs[j] - 1.0 / K) + sml);
          }
        }
        else
        {
          J += -nhat(f) * item_single(k + 1, points[f], bs[j], as[j], cs[j], gMat.row(j), dMat.row(j), K, MK) * H;
        }
      }
    }

    arma::mat cov;
    arma::vec sd(size);

    if (sandwitch)
    {
      // Rcout << V << std::endl;
      // V -= El * El.t();

      // first solve  J * X = V      → X = J^{-1} V
      bool ok = arma::solve(cov, J, V, arma::solve_opts::likely_sympd);

      if (ok)
      {

        // second solve J * Sᵗ = Xᵗ    → S = (J^{-1} Xᵗ)ᵗ = J^{-1} V J^{-1}
        arma::mat S = arma::solve(J, cov.t(), arma::solve_opts::likely_sympd).t();
        arma::vec dg = S.diag();
        sd = sqrt(dg);
      }
      else
      {
        sd.fill(1e9);
      }
    }
    else
    {
      // Rcout << "Determinant of Fisher Information Matrix is " << arma::det(J) << std::endl;
      bool ok = arma::pinv(cov, J);
      arma::vec dg = cov.diag();
      if (ok)
      {
        sd = sqrt(dg);
      }
      else
      {
        sd.fill(1e9);
      }
    }
    sdB(j) = sd(0);
    sdA(j) = sd(1);
    if (extended)
      sdC(j) = sd(size - 1);
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

  // Always-present components
  List res = List::create(
      _["b"] = sdB,
      _["a"] = sdA);

  // Add conditionally
  if (extended)
    res["c"] = sdC; // include c when extended
  if (!nog)
    res["g"] = sdG; // include g unless nog is true
  if (!nod)
    res["d"] = sdD; // include d unless nod is true

  return res;
}

// [[Rcpp::export]]
List EMSteps(const NumericMatrix &X, int K, int MK, const NumericVector &points, const NumericVector &weights,
             double lambda, double lambda2, double lambda3, int ngs, int nds, int maxEMIter = 10, int maxNRIter = 50, bool verbose = false, bool extended = true,
             unsigned int penalty = 0, bool sandwitch = true)
{
  auto M = X.cols();
  auto N = X.rows();
  NumericVector bs(M);
  bs.fill(0);
  NumericVector as(M);
  as.fill(0.15);
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

  bool AL = RegType(penalty) == RegType::AdaptiveLASSO;

  arma::mat pweights(M, ngs + nds + (unsigned)extended);
  pweights.fill(1);

  for (int iter = 0; iter < maxEMIter; ++iter)
  {
    auto E = EStep(X, bs, as, cs, gs, ds, K, MK, points, weights);
    auto rhat = E["rhat"];
    auto nhat = E["nhat"];

    auto ret = iterativeEstimation(N, bs, as, cs, gs, ds, K, MK, rhat, nhat, points, pweights,
                                   extended, AL ? 1e-5 : lambda, AL ? 1e-5 : lambda2, AL ? 1e-5 : lambda3, maxNRIter, AL ? 0 : penalty);

    bs = ret["b"];
    as = ret["a"];
    if (extended)
      cs = ret["c"];
    conv = ret["conv"];
    gs = Rcpp::as<NumericMatrix>(ret["g"]);
    ds = Rcpp::as<NumericMatrix>(ret["d"]);
    if (verbose)
      Rcout << "EM Step"
            << " " << iter + 1 << std::endl;
  }

  if (AL)
  {
    for (int i = 0; i < M; ++i)
    {
      for (int j = 0; j < ngs; ++j)
      {
        pweights(i, j) = gs(i, j + 1);
      }
      for (int j = 0; j < nds; ++j)
      {
        pweights(i, ngs + j) = ds(i, j + 1);
      }
      if (extended)
      {
        pweights(i, ngs + nds) = cs(i);
      }
    }

    for (int iter = 0; iter < 3; ++iter)
    {
      auto E = EStep(X, bs, as, cs, gs, ds, K, MK, points, weights);
      auto rhat = E["rhat"];
      auto nhat = E["nhat"];

      auto ret = iterativeEstimation(N, bs, as, cs, gs, ds, K, MK, rhat, nhat, points, pweights,
                                     extended, lambda, lambda2, lambda3, maxNRIter, penalty);
      bs = ret["b"];
      as = ret["a"];
      if (extended)
        cs = ret["c"];
      conv = ret["conv"];
      gs = Rcpp::as<NumericMatrix>(ret["g"]);
      ds = Rcpp::as<NumericMatrix>(ret["d"]);
      if (verbose)
        Rcout << "EM Step"
              << " " << iter + 1 << std::endl;
    }
  }

  // auto E = EStep(X, bs, as, cs, gs, ds, K, MK, points, weights);

  auto se = SE(X, bs, as, cs, gs, ds, K, MK, points, weights, extended, pweights, penalty, sandwitch, lambda, lambda2, lambda3);

  if (extended)
  {
    return List::create(_["b"] = bs,
                        _["a"] = as,
                        _["c"] = cs,
                        _["g"] = gs,
                        _["d"] = ds,
                        _["se"] = se,
                        _["conv"] = conv);
  }
  else
  {
    return List::create(_["b"] = bs,
                        _["a"] = as,
                        _["g"] = gs,
                        _["d"] = ds,
                        _["se"] = se,
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
NumericVector eap_theta_log(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK,
                            NumericVector points, NumericVector weights)
{
  int N = X.rows();
  int PN = points.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector eap(N);
  std::vector<double> expLW(PN);
  std::vector<double> LW(PN);
  for (int i = 0; i < N; ++i)
  {
    double mlw = 0;
    for (int f = 0; f < PN; ++f)
    {
      double l = item_vec_log(X(i, _), points[f], bs, as, cs, gMat, dMat, K, MK);
      expLW[f] = exp(l) * weights[f];
      LW[f] = l + log(weights[f]);
      if (f == 0)
      {
        mlw = LW[f];
      }
      else
      {
        mlw = std::max(mlw, LW[f]);
      }
    }
    double num = 0;
    double den = 0;
    for (int f = 0; f < PN; ++f)
    {
      num += points[f] * expLW[f];
      den += exp(LW[f] - mlw);
    }
    den = exp(mlw) * den;
    eap[i] = num / den;
  }
  return eap;
}

//' EAP for SIRT-MM
//'
//' @export
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

//' MLE for SIRT-MM using the second derivative (recommended)
//'
//' @export
// [[Rcpp::export]]
NumericVector mle_theta(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs,
                        const NumericMatrix &ds, int K, int MK, double max_value = 6)
{
  int N = X.rows();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector est(N);
  est.fill(0);
  for (int i = 0; i < N; ++i)
  {
    double theta = 0;
    for (int iter = 0; iter < 50; ++iter)
    {
      double fi = 0;
      double grad = 0;
      for (int j = 0; j < M; ++j)
      {
        if (X(i, j) == MISS_VAL)
        {
          continue;
        }
        auto g = gMat.row(j);
        auto d = dMat.row(j);
        double a = as(j), b = bs(j), c = cs(j);
        for (int y = 1; y <= K; ++y)
        {
          double d2 = 0;
          double e = exp(Z(y, theta, b, a, g, d));
          double A = e / (c + e);
          double delta = 0;
          if (y - 1 < d.n_elem)
          {
            delta = d(y - 1);
          }

          if (y == X(i, j))
            grad += (a + delta) * A;

          d2 = (a + delta) * (a + delta) * A * (1 - A);
          for (int k = 1; k <= y; ++k)
          {
            e = exp(Z(y, theta, b, a, g, d));
            double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
            d2 -= (a + delta) * (a + delta) * B * (1 - B);
            if (y == X(i, j))
              grad -= (a + delta) * B;
          }
          fi += -item_single(y, theta, b, a, c, g, d, K, MK) * d2;
        }
      }
      double x = grad / fi;
      if (std::abs(x) > 1)
        x = x / std::abs(x); // clamp [-1, 1]
      else if (std::abs(x) < 0.005)
        break;

      double new_theta = theta + x;
      if (std::abs(new_theta) > max_value)
        new_theta = new_theta / std::abs(new_theta) * max_value; // clamp [-6, 6]
      if (std::abs(theta - new_theta) < 0.0001)
        break;
      theta = new_theta;
    }
    est(i) = theta;
  }
  return est;
}

//' MLE for 3PL
//'
//' @export
// [[Rcpp::export]]
NumericVector mle_theta_3PL(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, double max_value = 6)
{
  int N = X.rows();
  int M = bs.length();
  NumericVector est(N);
  est.fill(0);
  for (int i = 0; i < N; ++i)
  {
    double theta = 0;
    for (int iter = 0; iter < 50; ++iter)
    {
      double fi = 0;
      double grad = 0;

      for (int j = 0; j < M; ++j)
      {
        if (X(i, j) == MISS_VAL)
        {
          continue;
        }

        // # prod (p^x)(q^(1-x))
        // # sum x log p + (1-x) log (1-p)
        // # sum x (p'/p) - (1-x) (p'/(1-p))

        double x = X(i, j);
        double a = as(j), b = bs(j), c = cs(j);
        double p = c + (1 - c) / (1 + exp(-a * (theta - b)));
        double q = 1 - p;
        double p1 = a * (p - c) / (1 - c) * q;
        grad += x * p1 / p - (1.0 - x) * p1 / q;
        fi += std::pow(a * (p - c) / (1 - c), 2) * q / p;
      }

      double x = grad / fi;
      if (std::abs(x) > 1)
        x = x / std::abs(x); // clamp [-1, 1]
      else if (std::abs(x) < 0.005)
        break;

      double new_theta = theta + x;
      if (std::abs(new_theta) > max_value)
        new_theta = new_theta / std::abs(new_theta) * max_value; // clamp [-6, 6]
      if (std::abs(theta - new_theta) < 0.0001)
        break;
      theta = new_theta;
    }
    est(i) = theta;
  }
  return est;
}

//' MLE for SIRT-MM using the original Fisher Info definiton
//'
//' @export
// [[Rcpp::export]]
NumericVector mle2_theta(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs,
                         const NumericMatrix &ds, int K, int MK, double max_value = 6)
{
  int N = X.rows();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector est(N);
  est.fill(0);
  for (int i = 0; i < N; ++i)
  {
    double theta = 0;
    for (int iter = 0; iter < 50; ++iter)
    {
      double fi = 0;
      double grad = 0;
      for (int j = 0; j < M; ++j)
      {
        if (X(i, j) == MISS_VAL)
        {
          continue;
        }
        auto g = gMat.row(j);
        auto d = dMat.row(j);
        double a = as(j), b = bs(j), c = cs(j);
        for (int y = 1; y <= K; ++y)
        {
          double d1 = 0;
          double d2 = 0;
          double e = exp(Z(y, theta, b, a, g, d));
          double A = e / (c + e);
          double delta = 0;
          if (y - 1 < d.n_elem)
          {
            delta = d(y - 1);
          }

          d1 += (a + delta) * A;

          // d2 = (a + delta) * (a + delta) * A * (1 - A);
          for (int k = 1; k <= y; ++k)
          {
            e = exp(Z(y, theta, b, a, g, d));
            double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
            // d2 -= (a + delta) * (a + delta) * B * (1 - B);
            d1 -= (a + delta) * B;
          }

          if (y == X(i, j))
            grad += d1;
          fi += item_single(y, theta, b, a, c, g, d, K, MK) * d1 * d1;
        }
      }
      double x = grad / fi;
      if (std::abs(x) > 1)
        x = x / std::abs(x); // clamp [-1, 1]
      else if (std::abs(x) < 0.005)
        break;

      double new_theta = theta + x;
      if (std::abs(new_theta) > max_value)
        new_theta = new_theta / std::abs(new_theta) * max_value; // clamp [-6, 6]
      if (std::abs(theta - new_theta) < 0.0001)
        break;
      theta = new_theta;
    }
    est(i) = theta;
  }
  return est;
}

//' WLE for SIRT-MM
//'
//' @export
// [[Rcpp::export]]
NumericVector wle_theta(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs, const NumericMatrix &gs,
                        const NumericMatrix &ds, int K, int MK)
{
  int N = X.rows();
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector est(N);
  est.fill(0);
  for (int i = 0; i < N; ++i)
  {
    double theta = 0;
    double x_old = 1;
    for (int iter = 0; iter < 50; ++iter)
    {
      double fi = 0;
      double J = 0;
      double grad = 0;
      for (int j = 0; j < M; ++j)
      {
        if (X(i, j) == MISS_VAL)
        {
          continue;
        }
        auto g = gMat.row(j);
        auto d = dMat.row(j);
        double a = as(j), b = bs(j), c = cs(j);
        for (int y = 1; y <= K; ++y)
        {
          double d1 = 0;
          double d2 = 0;
          double e = exp(Z(y, theta, b, a, g, d));
          double A = e / (c + e);
          double delta = 0;
          if (y - 1 < d.n_elem)
          {
            delta = d(y - 1);
          }

          d1 += (a + delta) * A;

          d2 = (a + delta) * (a + delta) * A * (1 - A);
          for (int k = 1; k <= y; ++k)
          {
            e = exp(Z(y, theta, b, a, g, d));
            double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
            d2 -= (a + delta) * (a + delta) * B * (1 - B);
            d1 -= (a + delta) * B;
          }

          if (y == X(i, j))
            grad += d1;
          double p = item_single(y, theta, b, a, c, g, d, K, MK);
          fi += p * d1 * d1;
          J += p * d1 * (d2 + d1 * d1);
        }
      }
      // Rcout << iter << ": " << theta << " + " << grad << " + " << J / fi / 2 << std::endl;
      double x = grad + J / fi / 2;
      if (std::abs(x) > 1 && iter < 6)
        x = x / std::abs(x); // clamp [-1, 1]
      else if (std::abs(x) < 0.005)
        break;
      else
        x = x / std::abs(x) * std::abs(x_old) / 2;
      x_old = x;
      // Rcout << iter << ": " << x << std::endl;

      theta = theta + x;
    }
    est(i) = theta;
  }
  return est;
}

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

//' EAP for 3PL
//'
//' @export
// [[Rcpp::export]]
NumericVector eap_theta_3PL(NumericMatrix X, NumericVector bs, NumericVector as, NumericVector cs,
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

//' Fisher Information for SIRT-MM
//'
//' @export
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

        if (y == MK && MK < K)
        {
          for (int k = 1; k <= y - 1; ++k)
          {
            auto zd1 = (a + delta);
            double e = exp(Z(k, theta, b, a, g, d));
            double C = ((double)K - 1.0) / ((K - k) * (1.0 - c) + (K - 1.0) * (c + e));
            double D = e * C;
            d2 -= (zd1 * zd1) * D * (1.0 - D);
          }
        }
        else
        {
          d2 = (a + delta) * (a + delta) * A * (1 - A);

          for (int k = 1; k <= y; ++k)
          {
            e = exp(Z(k, theta, b, a, g, d));
            double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
            d2 -= (a + delta) * (a + delta) * B * (1 - B);
          }
        }
        tempFI += -item_single(y, theta, b, a, c, g, d, K, MK) * d2;
      }
      fi(i) += tempFI;
    }
  }
  return fi;
}

//' Item Information for SIRT-MM
//'
//' @export
// [[Rcpp::export]]
NumericVector ItemFI(double theta,
                     NumericVector bs, NumericVector as, NumericVector cs,
                     const NumericMatrix &gs, const NumericMatrix &ds, int K, int MK)
{
  int M = bs.length();
  auto gMat = Rcpp::as<arma::mat>(gs);
  auto dMat = Rcpp::as<arma::mat>(ds);
  NumericVector fi(M);
  fi.fill(0);
  if (MK == -1)
    MK = K;
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

      if (y == MK && MK < K)
      {
        for (int k = 1; k <= y - 1; ++k)
        {
          auto zd1 = (a + delta);
          double e = exp(Z(k, theta, b, a, g, d));
          double C = ((double)K - 1.0) / ((K - k) * (1.0 - c) + (K - 1.0) * (c + e));
          double D = e * C;
          d2 -= (zd1 * zd1) * D * (1.0 - D);
        }
      }
      else
      {
        d2 = (a + delta) * (a + delta) * A * (1 - A);

        for (int k = 1; k <= y; ++k)
        {
          e = exp(Z(k, theta, b, a, g, d));
          double B = e / ((1.0 - c) * (K - k) / (K - 1.0) + c + e);
          d2 -= (a + delta) * (a + delta) * B * (1 - B);
        }
      }
      tempFI += -item_single(y, theta, b, a, c, g, d, K, MK) * d2;
    }
    fi(j) = tempFI;
  }
  return fi;
}

//' Test Information for 3PL
//'
//' @export
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
      double p = c + (1 - c) / (1 + exp(-a * (theta - b)));
      double q = 1 - p;

      fi(i) += std::pow(a * (p - c) / (1 - c), 2) * q / p;
    }
  }
  return fi;
}

//' Item Information of the 3PL model
//'
//' @export
// [[Rcpp::export]]
NumericVector ItemFI3PL(double theta, NumericVector bs, NumericVector as, NumericVector cs)
{
  int M = bs.length();
  NumericVector fi(M);
  fi.fill(0);
  for (int j = 0; j < M; ++j)
  {
    double a = as(j), b = bs(j), c = cs(j);
    double p = c + (1 - c) / (1 + exp(-a * (theta - b)));
    double q = 1 - p;
    fi(j) = std::pow(a * (p - c) / (1 - c), 2) * q / p;
  }
  return fi;
}