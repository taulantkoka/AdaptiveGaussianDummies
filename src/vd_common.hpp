// vd_common.hpp
// Shared types, options, and helpers for the VD-* family of solvers.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <optional>
#include <utility>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>

#if !defined(_WIN32)
  #include <sys/mman.h>
  #include <unistd.h>
#endif

using Vec     = Eigen::VectorXd;
using MatC    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatR    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapMatC = Eigen::Map<const MatC>;
using MapVec  = Eigen::Map<const Vec>;

class SharedRealCache;
class SharedSphericalControls;

enum class VDDummyLaw : uint8_t {
  Spherical = 0,
  Gaussian = 1,
  AdaptiveGaussian = 2,
  AdaptiveBeta = 3,
  SubGaussian = 4,
  SphericalAdaptiveGaussian = 5
};
enum class DummyRadiusCovarianceModel : uint8_t {
  None = 0, CompoundSymmetric = 1, AR1 = 2
};

struct VDOptions {
  int    T_stop       = 100;
  // OMP/NNOMP only: stop after this many consecutive path-local stopping
  // dummy entries.  A real, persistent, or calibration entry resets the
  // streak.  Zero retains the usual cumulative-T stopping rule.
  int    consecutive_dummy_stop = 0;
  int    max_vd_proj = 100;
  double eps         = 1e-12;
  bool   standardize = false;
  bool   debug       = false;
  bool   nonnegative = false;   // positive-correlation selection + NNLS refit (OMP)
  unsigned long long seed = 0ULL;
  VDDummyLaw dummy_law = VDDummyLaw::Spherical;
  double rho = 1.0;
  double trim_q = 1.0;          // quantile trim on |corr| for AdaptiveGaussian (1.0 = no trim)
  int ag_tail_m = 0;            // >0: fit Gaussian scale from the largest m
                                // absolute conditional residuals using the
                                // normalized two-sided truncated likelihood
  double sigma_reg = 1e-8;      // regularization for covariance inversion
  double sg_beta = 4.0;         // SubGaussian shape at step 0 (2 = Gaussian)
  bool sg_adaptive = false;     // fit one beta per conditional projection row
  double sg_fit_q_lo = 0.00;    // lower absolute-residual quantile used by fit
  double sg_fit_q_hi = 0.80;    // retain central 80%; larger residuals are trimmed
  double sg_beta_min = 1.0;
  double sg_beta_max = 6.0;
  double sg_beta_smooth = 0.5;  // weight of the current fit in beta_k
  double vd_seed_corr = -1.0;   // >=0: seed the basis with a RANDOM direction of
                                //   this correlation with yhat, putting y SECOND.
                                //   -1 = classic (basis starts at yhat).
  double sg_tau  = 0.0;         // >0: beta_k decays to 2 with step k (contamination
                                //     by unselected signals shrinks as they enter)
  bool   trim_by_entry = false; // trim on LARS entry order (step size) not |corr|
  // Group labels (length p) for AdaptiveGaussian.  Real designs are GROUPED
  // (LD blocks, co-expression modules) with heterogeneous within-group
  // correlation.  Fitting ONE pooled Gaussian over all predictors then yields
  // an over-dispersed mixture, the dummies become too competitive and the
  // early-stopping rule fires immediately.  With groups supplied, the dummy law
  // is fitted PER GROUP and each dummy is drawn from its own group's law.
  // Empty => single pooled group (previous behaviour).
  std::vector<int> groups;
  int mmap_fd         = -1;
  int mmap_block_cols = 0;
  // Internal shared cache supplied by TRexSelector.  Standalone VD solvers
  // leave this null and retain the original direct observation-space engine.
  std::shared_ptr<SharedRealCache> shared_real_cache;
  // Optional fixed pool shared by all K OMP experiments.  These are candidate
  // controls, not the path-local stopping dummies counted by T.
  std::shared_ptr<SharedSphericalControls> shared_spherical_controls;
  // Path-local calibration dummies.  Their spherical geometry is freshly
  // sampled in every random experiment, but their labels are aligned across
  // experiments so their entry indicators can be aggregated.  They are
  // separate from the final ``num_dummies`` local columns, which alone count
  // toward the T stopping depth.
  int fresh_calibration_dummies = 0;
  // Optional label-specific priority multipliers shared across experiments.
  // A null pointer (or an all-ones vector) gives the fully fresh event-mass
  // construction; nonconstant marks give fresh geometry with persistent
  // priority labels.
  std::shared_ptr<const Vec> fresh_calibration_priorities;
  double dummy_priority_radius = 1.0;
  bool adaptive_dummy_radius = false;
  // Fit zero-mean Gaussian pseudo-scales to the largest absolute real and
  // dummy scores using the normalized two-sided truncated likelihood.  Their
  // ratio is the adaptive priority radius.  tail_m takes precedence when
  // positive; otherwise the same retained fraction is used in both pools.
  int dummy_radius_tail_m = 0;
  double dummy_radius_tail_fraction = 0.95;
  // If strictly between zero and one, replace the tail fit by the
  // history-conditioned spherical maximum calibration.  Each control's
  // revealed coordinates and remaining stick are used before its next
  // projection is drawn.
  double dummy_radius_target_win_prob = -1.0;
  // Replace the fixed target above by
  // M_k/(M_k+(p_k-s_bar)_+), where M_k and p_k are the remaining persistent
  // controls and real candidates and s_bar is an upper bound on the number
  // of remaining active variables.  Setting s_bar=0 recovers M_k/(M_k+p_k).
  bool dummy_radius_urn_target = false;
  // For persistent controls, calibrate their conditional pool maximum
  // against both the real-score reference and the current unscaled stopping
  // maximum.  The urn target becomes
  // M_k/(M_k+(p_k-s_bar)_+ + L_k).
  bool dummy_radius_account_for_stopping = false;
  int dummy_radius_active_upper_bound = 0;
  // Empirical quantile of the inactive-real absolute scores used as the
  // reference threshold.  One uses the maximum.
  double dummy_radius_real_score_quantile = 1.0;
  // Positive values override the quantile with a fixed descending rank:
  // one is the maximum, two the second largest, and so forth.
  int dummy_radius_real_score_rank = 0;
  // ERC-style upper envelope for the strongest null score:
  // min(A_real, theta*A_real + (1+theta)*noise_bound).
  double dummy_radius_envelope_theta = 1.0;
  double dummy_radius_noise_score_bound = 0.0;
  // Optional population-covariance bound replacing the manual theta.  For
  // compound symmetry the bound is updated with the number of selected real
  // columns; for Gaussian AR(1), the worst-case conditional l1 regression
  // coefficient is 2*rho/(1+rho^2).
  DummyRadiusCovarianceModel dummy_radius_covariance_model =
      DummyRadiusCovarianceModel::None;
  double dummy_radius_covariance_rho = 0.0;
  // Apply an AR(1) recursion across persistent-dummy priority scores and
  // rescale the result so its maximum over the remaining pool is unchanged.
  // This alters winner identity persistence without changing the realized
  // one-step persistent-pool maximum.
  bool persistent_score_ar1 = false;
  // Persistent controls and stopping dummies have separate roles.  By default
  // only persistent controls receive the adaptive priority radius.
  bool dummy_radius_scale_stopping = false;
  // Carry-over weight used only when the required radius decreases.  Radius
  // increases are immediate.
  double dummy_radius_smooth = 0.80;
  double dummy_radius_min = 1.0;
  double dummy_radius_max = 5.0;
};

static constexpr int VD_Y_SENTINEL     = -1;
static constexpr int VD_DUMMY_SENTINEL = -2;

namespace vd_detail {

inline double covariance_envelope_theta(
    DummyRadiusCovarianceModel model, double rho,
    int remaining_active_upper_bound, int selected_real) {
  const double a = std::abs(rho);
  if (!(a < 1.0)) return 1.0;
  const int s = std::max(remaining_active_upper_bound, 0);
  const int q = std::max(selected_real, 0);
  double theta = 1.0;
  if (model == DummyRadiusCovarianceModel::CompoundSymmetric) {
    if (s == 0) return 0.0;
    const double denominator = 1.0 - a + double(q + s) * a;
    if (denominator > 0.0)
      theta = double(s) * a / denominator;
  } else if (model == DummyRadiusCovarianceModel::AR1) {
    theta = 2.0 * a / (1.0 + a * a);
  }
  return std::min(std::max(theta, 0.0), 1.0);
}

}  // namespace vd_detail

struct ActiveFeature {
  enum class Kind : uint8_t { Real, Dummy, Persistent, Calibration };
  Kind kind;
  int  index;
  // For a realized calibration dummy, ``index`` is its materialized-column
  // slot and ``label`` is its path-local label shared across experiments.
  // It remains -1 for the original feature kinds.
  int  label = -1;
};

namespace vd_detail {

inline int colblock() {
  static int B = [](){
    const char* s = std::getenv("VD_COLBLOCK");
    long v = s ? std::strtol(s, nullptr, 10) : 0;
    if (v <= 0) v = 8192;
    return int(v);
  }();
  return B;
}

#if !defined(_WIN32)
inline bool pread_full(int fd, void* buf, std::size_t len, off_t offset) {
  char* p = static_cast<char*>(buf);
  std::size_t remaining = len;
  while (remaining > 0) {
    ssize_t r = ::pread(fd, p, remaining, offset);
    if (r <= 0) return false;
    p += r; offset += r; remaining -= static_cast<std::size_t>(r);
  }
  return true;
}
#endif

inline void advise_sequential(const void* ptr, std::size_t len) {
#if !defined(_WIN32)
  if (ptr && len > 0)
    ::madvise(const_cast<void*>(ptr), len, MADV_SEQUENTIAL);
#else
  (void)ptr; (void)len;
#endif
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  out.setZero(p);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += x[i]*v[i];
      out[j0+j] = s;
    }
  }
}

inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y,
                    int mmap_fd, int block_cols, double* scratch)
{
  const int n = (int)X.rows(), p = (int)X.cols();
  y.setZero(n);
#if !defined(_WIN32)
  const bool use_pread = (mmap_fd >= 0 && scratch && block_cols > 0);
#else
  const bool use_pread = false;
  (void)mmap_fd; (void)scratch;
#endif
  const int B = (block_cols > 0) ? block_cols : colblock();
  for (int j0 = 0; j0 < p; j0 += B) {
    const int jb = std::min(B, p - j0);
    const double* Xblk;
#if !defined(_WIN32)
    if (use_pread) {
      pread_full(mmap_fd, scratch, std::size_t(jb)*n*sizeof(double),
                 off_t(j0)*n*sizeof(double));
      Xblk = scratch;
    } else
#endif
    { Xblk = X.col(j0).data(); }
    for (int j = 0; j < jb; ++j) {
      const double* x = Xblk + std::ptrdiff_t(j)*n;
      const double w = v[j0+j];
      if (w == 0.0) continue;
      for (int i = 0; i < n; ++i) y[i] += x[i]*w;
    }
  }
}

inline void gemv_Xt(const MapMatC& X, const Vec& v, Vec& out) {
  gemv_Xt(X, v, out, -1, 0, nullptr);
}
inline void gemv_Xv(const MapMatC& X, const Vec& v, Vec& y) {
  gemv_Xv(X, v, y, -1, 0, nullptr);
}

// Beta(a,b) via two Gammas.  Used by the AdaptiveBeta dummy law, whose
// stick-breaking increments are Beta-distributed (the spherical law is the
// special case Beta(1/2, (n-k-2)/2)).
inline double beta_rv(double a, double b, std::mt19937_64& rng);
inline double gamma_mt(double m, std::mt19937_64& rng);

// log density of Beta(a,b) at u in (0,1).
// Standardized generalized-normal (exponential power) draw.  beta = 2 is the
// Gaussian, beta < 2 is heavier-tailed, and beta > 2 is lighter-tailed.  It is
// scaled to unit variance as a drop-in AdaptiveGaussian innovation.
inline double subgauss_rv(double beta, std::mt19937_64& rng) {
  if (beta <= 0.0) beta = 2.0;
  const double g = gamma_mt(1.0 / beta, rng);
  const double mag = std::pow(std::max(g, 0.0), 1.0 / beta);
  // Var[GN(alpha=1,beta)] = Gamma(3/b)/Gamma(1/b)  ->  divide by its sqrt
  const double lv = std::lgamma(3.0 / beta) - std::lgamma(1.0 / beta);
  const double sd = std::exp(0.5 * lv);
  const double sgn = (rng() & 1ULL) ? 1.0 : -1.0;
  return sgn * mag / (sd > 0.0 ? sd : 1.0);
}

// Regularized lower incomplete gamma P(a,x).  The series is stable for
// x < a+1 and the continued fraction evaluates Q(a,x) otherwise.  This small
// implementation avoids introducing a new dependency solely for the
// truncated generalized-normal shape fit.
inline double gamma_p(double a, double x) {
  if (!(a > 0.0) || !(x > 0.0)) return 0.0;
  constexpr int max_iter = 200;
  constexpr double tol = 1e-13;
  constexpr double fpmin = 1e-300;
  const double log_prefactor = -x + a * std::log(x) - std::lgamma(a);

  if (x < a + 1.0) {
    double ap = a;
    double term = 1.0 / a;
    double sum = term;
    for (int iteration = 1; iteration <= max_iter; ++iteration) {
      ap += 1.0;
      term *= x / ap;
      sum += term;
      if (std::abs(term) <= std::abs(sum) * tol) break;
    }
    return std::min(1.0, std::max(0.0, sum * std::exp(log_prefactor)));
  }

  double b = x + 1.0 - a;
  double c = 1.0 / fpmin;
  double d = 1.0 / std::max(std::abs(b), fpmin);
  double h = d;
  for (int iteration = 1; iteration <= max_iter; ++iteration) {
    const double an = -double(iteration) * (double(iteration) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < fpmin) d = fpmin;
    c = b + an / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) <= tol) break;
  }
  const double q = std::exp(log_prefactor) * h;
  return std::min(1.0, std::max(0.0, 1.0 - q));
}

inline double beta_logpdf(double u, double a, double b) {
  u = std::min(std::max(u, 1e-12), 1.0 - 1e-12);
  const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  return (a - 1.0) * std::log(u) + (b - 1.0) * std::log1p(-u) - lbeta;
}

inline double beta_continued_fraction(double a, double b, double x) {
  constexpr int max_iter = 256;
  constexpr double tol = 3e-14;
  constexpr double fpmin = 1e-300;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < fpmin) d = fpmin;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= max_iter; ++m) {
    const int m2 = 2 * m;
    double aa = double(m) * (b - double(m)) * x
              / ((qam + double(m2)) * (a + double(m2)));
    d = 1.0 + aa * d;
    if (std::abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + double(m)) * (qab + double(m)) * x
       / ((a + double(m2)) * (qap + double(m2)));
    d = 1.0 + aa * d;
    if (std::abs(d) < fpmin) d = fpmin;
    c = 1.0 + aa / c;
    if (std::abs(c) < fpmin) c = fpmin;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) <= tol) break;
  }
  return h;
}

inline double beta_cdf(double x, double a, double b) {
  if (!(x > 0.0)) return 0.0;
  if (!(x < 1.0)) return 1.0;
  const double log_bt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
      + a * std::log(x) + b * std::log1p(-x);
  const double bt = std::exp(log_bt);
  double value;
  if (x < (a + 1.0) / (a + b + 2.0)) {
    value = bt * beta_continued_fraction(a, b, x) / a;
  } else {
    value = 1.0
        - bt * beta_continued_fraction(b, a, 1.0 - x) / b;
  }
  return std::min(1.0, std::max(0.0, value));
}

inline double beta_quantile(double probability, double a, double b) {
  if (!(probability > 0.0)) return 0.0;
  if (!(probability < 1.0)) return 1.0;
  double lower = 0.0, upper = 1.0;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double midpoint = 0.5 * (lower + upper);
    if (beta_cdf(midpoint, a, b) < probability)
      lower = midpoint;
    else
      upper = midpoint;
  }
  return 0.5 * (lower + upper);
}

// CDF of the first coordinate of a vector drawn uniformly from the unit
// sphere in R^dimension.  For dimension > 1 the squared coordinate is
// Beta(1/2, (dimension-1)/2); dimension == 1 is the two-point sphere {-1,+1}.
inline double spherical_coordinate_cdf(double x, int dimension) {
  if (dimension <= 0) return (x >= 0.0 ? 1.0 : 0.0);
  if (dimension == 1) {
    if (x < -1.0) return 0.0;
    if (x < 1.0) return 0.5;
    return 1.0;
  }
  if (x <= -1.0) return 0.0;
  if (x >= 1.0) return 1.0;
  if (x == 0.0) return 0.5;
  const double central = beta_cdf(
      x * x, 0.5, 0.5 * double(dimension - 1));
  return x > 0.0 ? 0.5 * (1.0 + central)
                 : 0.5 * (1.0 - central);
}

// Conditional probability that |mean + scale*U| is at most threshold, where
// U is the spherical coordinate above.  The separate dimension-one branch
// handles the atoms at {-1,+1} without relying on endpoint conventions.
inline double shifted_spherical_abs_cdf(
    double threshold, double mean, double scale, int dimension)
{
  if (!(threshold >= 0.0)) return 0.0;
  scale = std::abs(scale);
  if (!(scale > 1e-15) || dimension <= 0)
    return std::abs(mean) <= threshold ? 1.0 : 0.0;
  if (dimension == 1) {
    const double first = std::abs(mean - scale) <= threshold ? 0.5 : 0.0;
    const double second = std::abs(mean + scale) <= threshold ? 0.5 : 0.0;
    return first + second;
  }
  const double lower = (-threshold - mean) / scale;
  const double upper = ( threshold - mean) / scale;
  if (upper < lower) return 0.0;
  const double value = spherical_coordinate_cdf(upper, dimension)
                     - spherical_coordinate_cdf(lower, dimension);
  return std::min(1.0, std::max(0.0, value));
}

// Let S_j = means[j] + scales[j] U_j, with conditionally independent
// spherical coordinates U_j.  Return the common score multiplier R for which
// P(max_j R |S_j| > threshold) equals target_probability.  Heterogeneous
// remaining sticks enter through scales[j], while already-revealed components
// enter through means[j].  With discrete/deterministic components, the exact
// probability need not be attainable; the generalized (1-target) quantile is
// used.
inline double conditional_spherical_pool_radius(
    double threshold,
    double target_probability,
    const std::vector<double>& means,
    const std::vector<double>& scales,
    int remaining_dimension)
{
  if (means.empty() || means.size() != scales.size())
    return 1.0;
  if (!(target_probability > 0.0 && target_probability < 1.0))
    return 1.0;
  if (!(threshold > 0.0)) return 0.0;

  double support_upper = 0.0;
  double deterministic_upper = 0.0;
  for (std::size_t j = 0; j < means.size(); ++j) {
    const double scale = std::abs(scales[j]);
    support_upper =
        std::max(support_upper, std::abs(means[j]) + scale);
    deterministic_upper =
        std::max(deterministic_upper, std::abs(means[j]));
  }
  bool conditionally_deterministic = remaining_dimension <= 0;
  if (!conditionally_deterministic) {
    conditionally_deterministic = true;
    for (double scale : scales) {
      if (std::abs(scale) > 1e-15) {
        conditionally_deterministic = false;
        break;
      }
    }
  }
  if (conditionally_deterministic) {
    if (!(deterministic_upper > 0.0))
      return std::numeric_limits<double>::infinity();
    return threshold / deterministic_upper;
  }

  const double log_target = std::log1p(-target_probability);
  auto log_pool_cdf = [&](double value) {
    double total = 0.0;
    for (std::size_t j = 0; j < means.size(); ++j) {
      const double probability = shifted_spherical_abs_cdf(
          value, means[j], scales[j], remaining_dimension);
      if (!(probability > 0.0))
        return -std::numeric_limits<double>::infinity();
      total += std::log(probability);
    }
    return total;
  };

  double upper = support_upper;
  if (!(upper > 0.0))
    return std::numeric_limits<double>::infinity();

  // Generalized quantile q = inf{t: P(max_j |S_j| <= t) >= 1-target}.
  double lower = 0.0;
  // Forty bisection steps resolve the bounded support to better than 1e-12
  // relative precision, already below the precision relevant to candidate
  // ranking and the radius bounds.
  for (int iteration = 0; iteration < 40; ++iteration) {
    const double midpoint = 0.5 * (lower + upper);
    if (log_pool_cdf(midpoint) < log_target)
      lower = midpoint;
    else
      upper = midpoint;
  }
  const double quantile = upper;
  if (!(quantile > 0.0))
    return std::numeric_limits<double>::infinity();
  return threshold / quantile;
}

inline double gamma_mt(double m, std::mt19937_64& rng) {
    auto uniform01 = [&](){
      constexpr double scale = 1.0/(1ULL<<53);
      return (rng()>>(64-53))*scale;
    };
    std::normal_distribution<double> normal(0.0,1.0);
    if (m < 1.0) {
      double g = gamma_mt(m+1.0, rng);
      double u = std::max(uniform01(), std::numeric_limits<double>::min());
      return g*std::pow(u, 1.0/m);
    }
    const double d=m-1.0/3.0, c=1.0/std::sqrt(9.0*d);
    for (;;) {
      double z=normal(rng), v=1.0+c*z;
      if (v<=0.0) continue;
      double v3=v*v*v;
      if (uniform01()<1.0-0.0331*(z*z)*(z*z)) return d*v3;
      double u=std::max(uniform01(), std::numeric_limits<double>::min());
      if (std::log(u)<0.5*z*z+d*(1.0-v3+std::log(v3))) return d*v3;
    }
}

inline double beta_rv(double a, double b, std::mt19937_64& rng) {
  const double x = gamma_mt(std::max(a, 1e-6), rng);
  const double y = gamma_mt(std::max(b, 1e-6), rng);
  const double d = x + y;
  return (d > 0.0) ? x / d : 0.5;
}

} // namespace vd_detail
