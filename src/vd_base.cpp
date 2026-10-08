// vd_base.cpp
#include "vd_base.hpp"
#include "shared_real_cache.hpp"
#include <algorithm>
using DummyLaw = VDDummyLaw;

// ---------- Constructors ----------
VD_Base::VD_Base(const double* Xptr, int n, int p,
                 const double* yptr, int ny,
                 int num_dummies, const VDOptions& o)
  : n_(n), p_(p),
    L_(num_dummies + std::max(o.fresh_calibration_dummies, 0)),
    calibration_dummies_(std::max(o.fresh_calibration_dummies, 0)),
    stopping_dummies_(num_dummies),
    opt_(o), rng_(opt_.seed),
    X_(Xptr, n, p),
    y_(yptr, ny)
{
  init_common_state_();
}

VD_Base::VD_Base(const Eigen::Ref<const MatC>& X_in,
                 const Eigen::Ref<const Vec>&  y_in,
                 int num_dummies, const VDOptions& o)
  : VD_Base(X_in.data(), (int)X_in.rows(), (int)X_in.cols(),
            y_in.data(), (int)y_in.size(), num_dummies, o)
{}

// ---------- Common initialization ----------
void VD_Base::init_common_state_() {
  if (stopping_dummies_ < 0)
    throw std::invalid_argument("num_dummies must be nonnegative");
  if (opt_.fresh_calibration_priorities &&
      opt_.fresh_calibration_priorities->size() != calibration_dummies_) {
    throw std::invalid_argument(
        "fresh calibration priority vector has the wrong length");
  }
  persistent_controls_ = opt_.shared_spherical_controls
      ? opt_.shared_spherical_controls->size() : 0;
  const int maximum_realized =
      std::max(opt_.T_stop, 0) + calibration_dummies_;
  const int maximum_active = std::min(
      n_, p_ + persistent_controls_ + maximum_realized);

  standardized_ = false;
  y_norm_       = y_.norm();
  normx_        = Vec::Ones(p_);

  beta_       = Vec::Zero(p_);
  beta_dummy_ = Vec::Zero(maximum_realized);
  beta_persistent_ = Vec::Zero(persistent_controls_);
  mu_         = Vec::Zero(n_);
  residuals_  = y_;

  corr_.resize(p_);

  // pread scratch
  if (opt_.mmap_fd >= 0 && opt_.mmap_block_cols > 0) {
    gemv_scratch_.resize(std::size_t(opt_.mmap_block_cols) * n_);
    vd_detail::advise_sequential(X_.data(),
        std::size_t(n_) * p_ * sizeof(double));
  }

  // Initial correlations.  A TRexSelector may provide the product once for
  // all K random experiments; standalone solvers retain the direct path.
  if (shared_products_())
    corr_ = opt_.shared_real_cache->Xty();
  else
    vd_detail::gemv_Xt(X_, residuals_, corr_,
        opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());

  // Active set
  is_active_.assign(p_, 0);
  actives_.clear();
  actives_.reserve(std::min(p_, maximum_active));
  active_features_.clear();
  active_features_.reserve(maximum_active);
  persistent_is_active_.assign(persistent_controls_, 0);

  // Basis
  basis_.resize(n_, maximum_active);
  basis_indices_.clear();
  basis_indices_.reserve(maximum_active);

  // Reusable buffers
  proj_coeffs_.resize(std::max(1, maximum_active));
  ortho_buffer_.resize(n_);

  // Active columns & Cholesky
  X_active_.resize(n_, maximum_active);
  chol_factor_.resize(0, 0);

  // Realized dummies
  X_realized_.setZero(n_, maximum_realized);
  corr_realized_.setZero(maximum_realized);
  Xt_realized_.clear();
  Xt_realized_.reserve(maximum_realized);
  T_realized_ = 0;
  T_stopping_realized_ = 0;
  T_calibration_realized_ = 0;

  // Virtual-dummy pool
  vd_stick_ = Vec::Ones(L_);
  vd_corr_  = Vec::Zero(L_);
  vd_is_realized_.assign(L_, 0);
  vd_unrealized_idx_.clear();
  vd_unrealized_idx_.reserve(L_);
  vd_rows_cap_ = std::min(opt_.max_vd_proj, std::max(8, 128));
  vd_proj_.resize(vd_rows_cap_, L_);
  vd_rows_filled_ = 0;

  persistent_corr_ = Vec::Zero(persistent_controls_);
  current_dummy_radius_ = std::max(opt_.dummy_priority_radius, 0.0);
  conditional_dummy_radius_ = current_dummy_radius_;
  conditional_dummy_radius_available_ = false;

  // Real-variable projections for AdaptiveGaussian
  Z_proj_.setZero(p_, std::max(1, opt_.max_vd_proj));
  real_cross_local_.clear();

  init_groups_();

  step_ = 0;

  // Seed basis with y, init VD
  initialize_basis_();
  initialize_virtual_dummies_();
  initialize_persistent_controls_();
}

std::shared_ptr<const Vec> VD_Base::real_cross_(int j) {
  auto found = real_cross_local_.find(j);
  if (found != real_cross_local_.end()) return found->second;
  if (!shared_products_()) return {};
  auto column = opt_.shared_real_cache->gram_column(j);
  real_cross_local_.emplace(j, column);
  return column;
}

// ---------- Shared persistent spherical controls ----------
void VD_Base::initialize_persistent_controls_() {
  if (persistent_controls_ <= 0) return;
  refresh_persistent_correlations_();
}

void VD_Base::update_persistent_controls_() {
  // Persistent correlations are requested against the current residual, not
  // stored separately in every stream-local basis.  The shared pool expresses
  // that residual in its master basis and reveals only genuinely new
  // directions.
  refresh_persistent_correlations_();
}

void VD_Base::refresh_persistent_correlations_() {
  if (persistent_controls_ <= 0) return;
  conditional_dummy_radius_available_ = false;
  double target_probability = opt_.dummy_radius_target_win_prob;
  int remaining_real = 0;
  for (int j = 0; j < p_; ++j)
    if (!is_active_[j]) ++remaining_real;
  int remaining_controls = 0;
  for (int j = 0; j < persistent_controls_; ++j)
    if (!persistent_is_active_[j]) ++remaining_controls;
  int remaining_stopping = 0;
  double stopping_maximum = 0.0;
  if (opt_.dummy_radius_account_for_stopping) {
    for (int d = 0; d < L_; ++d) {
      if (vd_is_realized_[d]) continue;
      ++remaining_stopping;
      stopping_maximum =
          std::max(stopping_maximum, std::abs(vd_corr_(d)));
    }
  }
  const int active_upper_bound =
      std::max(opt_.dummy_radius_active_upper_bound, 0);
  const int null_lower_bound =
      std::max(remaining_real - active_upper_bound, 0);
  if (opt_.dummy_radius_urn_target && remaining_controls > 0
      && null_lower_bound > 0) {
    const int stopping_denominator =
        opt_.dummy_radius_account_for_stopping
            ? remaining_stopping : 0;
    target_probability =
        double(remaining_controls)
        / double(
            remaining_controls + null_lower_bound
            + stopping_denominator);
  }
  if (opt_.adaptive_dummy_radius
      && target_probability > 0.0 && target_probability < 1.0) {
    std::vector<double> inactive_real_scores;
    inactive_real_scores.reserve(remaining_real);
    for (int j = 0; j < p_; ++j)
      if (!is_active_[j])
        inactive_real_scores.push_back(std::abs(corr_(j)));
    double real_reference = 0.0;
    if (!inactive_real_scores.empty()) {
      std::size_t index = 0;
      if (opt_.dummy_radius_real_score_rank > 0) {
        const std::size_t descending_rank = std::min(
            static_cast<std::size_t>(
                opt_.dummy_radius_real_score_rank),
            inactive_real_scores.size());
        index = inactive_real_scores.size() - descending_rank;
      } else {
        const double quantile = std::min(
            std::max(
                opt_.dummy_radius_real_score_quantile, 0.0), 1.0);
        const std::size_t rank = std::min(
            static_cast<std::size_t>(
                std::ceil(quantile * inactive_real_scores.size())),
            inactive_real_scores.size());
        index = rank > 0 ? rank - 1 : 0;
      }
      std::nth_element(
          inactive_real_scores.begin(),
          inactive_real_scores.begin() + index,
          inactive_real_scores.end());
      real_reference = inactive_real_scores[index];
    }
    if (remaining_controls > 0) {
      double theta = std::min(
          std::max(opt_.dummy_radius_envelope_theta, 0.0), 1.0);
      if (opt_.dummy_radius_covariance_model
          != DummyRadiusCovarianceModel::None) {
        const int selected_real = p_ - remaining_real;
        const int remaining_active_upper_bound =
            std::min(active_upper_bound, remaining_real);
        theta = vd_detail::covariance_envelope_theta(
            opt_.dummy_radius_covariance_model,
            opt_.dummy_radius_covariance_rho,
            remaining_active_upper_bound, selected_real);
      }
      const double noise_bound = std::max(
          opt_.dummy_radius_noise_score_bound, 0.0);
      const double null_envelope = std::min(
          real_reference,
          theta * real_reference + (1.0 + theta) * noise_bound);
      const double competition_threshold =
          opt_.dummy_radius_account_for_stopping
              ? std::max(null_envelope, stopping_maximum)
              : null_envelope;
      persistent_corr_ =
          opt_.shared_spherical_controls
              ->projections_with_conditional_radius(
                  residuals_, persistent_is_active_, competition_threshold,
                  target_probability, conditional_dummy_radius_);
      conditional_dummy_radius_available_ = true;
    } else {
      persistent_corr_ =
          opt_.shared_spherical_controls->projections(residuals_);
    }
  } else {
    persistent_corr_ =
        opt_.shared_spherical_controls->projections(residuals_);
  }
  apply_persistent_score_ar1_();
  for (int j = 0; j < persistent_controls_; ++j)
    if (persistent_is_active_[j]) persistent_corr_(j) = 0.0;
}

void VD_Base::apply_persistent_score_ar1_() {
  if (!opt_.persistent_score_ar1 || persistent_controls_ <= 1) return;

  const double rho = std::min(
      std::max(opt_.dummy_radius_covariance_rho, -0.999999), 0.999999);
  const double innovation_scale =
      std::sqrt(std::max(1.0 - rho * rho, 0.0));
  const Vec raw = persistent_corr_;
  Vec coupled = Vec::Zero(persistent_controls_);
  double previous = 0.0;
  double raw_maximum = 0.0;
  double coupled_maximum = 0.0;

  for (int j = 0; j < persistent_controls_; ++j) {
    if (persistent_is_active_[j]) {
      previous = 0.0;
      continue;
    }
    const double value = (
        j == 0 || persistent_is_active_[j - 1])
        ? raw(j)
        : rho * previous + innovation_scale * raw(j);
    coupled(j) = value;
    previous = value;
    raw_maximum = std::max(raw_maximum, std::abs(raw(j)));
    coupled_maximum = std::max(coupled_maximum, std::abs(value));
  }

  // Preserve the realized pool maximum exactly. Consequently the current
  // depth-wise competition and its conditional radius are unchanged; only
  // the identity of the winning persistent dummy and the distribution of
  // occurrences across fixed labels can change.
  if (coupled_maximum > opt_.eps) {
    coupled *= raw_maximum / coupled_maximum;
    persistent_corr_.swap(coupled);
  }
}

std::shared_ptr<const Vec> VD_Base::persistent_column_(int j) {
  if (!opt_.shared_spherical_controls)
    throw std::runtime_error("persistent control pool is unavailable");
  return opt_.shared_spherical_controls->materialize(j);
}

std::shared_ptr<const Vec> VD_Base::persistent_real_cross_(int j) {
  if (!opt_.shared_spherical_controls)
    throw std::runtime_error("persistent control pool is unavailable");
  return opt_.shared_spherical_controls->real_cross(j);
}

double VD_Base::choose_dummy_radius_() {
  const double fixed = std::max(opt_.dummy_priority_radius, 0.0);
  if (!opt_.adaptive_dummy_radius) {
    current_dummy_radius_ = fixed;
    dummy_radius_path_.push_back(current_dummy_radius_);
    return current_dummy_radius_;
  }

  std::vector<double> real_scores;
  real_scores.reserve(p_);
  for (int j = 0; j < p_; ++j)
    if (!is_active_[j]) real_scores.push_back(corr_(j));

  std::vector<double> dummy_scores;
  dummy_scores.reserve(L_ + persistent_controls_);
  if (calibration_dummies_ > 0) {
    for (int d = 0; d < calibration_dummies_; ++d) {
      if (vd_is_realized_[d]) continue;
      double priority = 1.0;
      if (opt_.fresh_calibration_priorities)
        priority = (*opt_.fresh_calibration_priorities)(d);
      dummy_scores.push_back(priority * vd_corr_(d));
    }
  } else {
    for (int d = 0; d < L_; ++d)
      if (!vd_is_realized_[d])
        dummy_scores.push_back(vd_corr_(d));
  }
  for (int j = 0; j < persistent_controls_; ++j)
    if (!persistent_is_active_[j])
      dummy_scores.push_back(persistent_corr_(j));

  if (real_scores.empty() || dummy_scores.empty()) {
    current_dummy_radius_ = std::min(
        std::max(fixed, opt_.dummy_radius_min), opt_.dummy_radius_max);
    dummy_radius_path_.push_back(current_dummy_radius_);
    return current_dummy_radius_;
  }

  double required = fixed;
  if (conditional_dummy_radius_available_) {
    required = conditional_dummy_radius_;
  } else {
    auto zero_mean_scale = [this](const std::vector<double>& values) {
      double sumsq = 0.0;
      for (double value : values) sumsq += value * value;
      return std::max(
          std::sqrt(sumsq / double(std::max<std::size_t>(
              values.size(), 1))),
          std::sqrt(opt_.sigma_reg));
    };

    double tail_fraction = std::min(
        std::max(opt_.dummy_radius_tail_fraction, 0.0), 1.0);
    int real_tail_m = opt_.dummy_radius_tail_m;
    if (real_tail_m <= 0) {
      real_tail_m = static_cast<int>(
          std::round(tail_fraction * double(real_scores.size())));
    } else {
      tail_fraction =
          double(real_tail_m) / double(std::max<std::size_t>(
              real_scores.size(), 1));
    }
    real_tail_m = std::min(
        std::max(real_tail_m, 1), static_cast<int>(real_scores.size()));
    int dummy_tail_m = static_cast<int>(
        std::round(tail_fraction * double(dummy_scores.size())));
    dummy_tail_m = std::min(
        std::max(dummy_tail_m, 1), static_cast<int>(dummy_scores.size()));

    const double real_scale = fit_gaussian_scale_top_tail_(
        real_scores, zero_mean_scale(real_scores), real_tail_m);
    const double dummy_scale = fit_gaussian_scale_top_tail_(
        dummy_scores, zero_mean_scale(dummy_scores), dummy_tail_m);
    if (dummy_scale > opt_.eps)
      required *= real_scale / dummy_scale;
    else if (real_scale > opt_.eps)
      required = std::numeric_limits<double>::infinity();
  }
  const double lower = std::max(
      std::max(opt_.dummy_radius_min, 0.0), fixed);
  const double upper = std::max(opt_.dummy_radius_max, lower);
  required = std::min(std::max(required, lower), upper);

  // A required increase is immediate.  On a decrease, retain a fraction of
  // the previous radius.  This stabilizes the path while keeping the applied
  // radius at least as large as the current one-sided requirement.
  const double smooth = std::min(
      std::max(opt_.dummy_radius_smooth, 0.0), 1.0);
  const bool exact_probability_mode =
      conditional_dummy_radius_available_;
  if (exact_probability_mode || required >= current_dummy_radius_) {
    current_dummy_radius_ = required;
  } else {
    current_dummy_radius_ =
        smooth * current_dummy_radius_ + (1.0 - smooth) * required;
  }
  current_dummy_radius_ =
      std::min(std::max(current_dummy_radius_, lower), upper);
  dummy_radius_path_.push_back(current_dummy_radius_);
  return current_dummy_radius_;
}

// ---------- Basis ----------
void VD_Base::initialize_basis_() {
  Eigen::Map<const Vec> yv(y_.data(), n_);
  const double yn = yv.norm();
  if (!(yn > opt_.eps * std::sqrt(double(n_)))) return;
  Vec yhat = yv / yn;

  if (opt_.vd_seed_corr < 0.0) {           // classic: basis starts at yhat
    basis_.col(0) = yhat;
    basis_indices_.push_back(VD_Y_SENTINEL);
    return;
  }

  // Seed the basis with a RANDOM direction v of correlation c with yhat, and
  // put yhat SECOND (orthonormalized against v).  The whole dummy law is
  // decided by the k = 0 fit, and with basis(0) = yhat that fit sits on the one
  // direction the signals are maximally aligned with.  Seeding with a random v
  // moves the contaminated direction to k = 1, where it enters as a CONDITIONAL
  // draw rather than the marginal.
  const double c = std::min(std::max(opt_.vd_seed_corr, 0.0), 0.999);
  std::normal_distribution<double> N01(0.0, 1.0);
  Vec g(n_);
  for (int i = 0; i < n_; ++i) g(i) = N01(rng_);
  g.array() -= g.mean();                   // stay in the centered subspace
  g -= g.dot(yhat) * yhat;                 // and orthogonal to yhat
  const double gn = g.norm();
  if (!(gn > opt_.eps)) {
    basis_.col(0) = yhat;
    basis_indices_.push_back(VD_Y_SENTINEL);
    return;
  }
  g /= gn;
  Vec v = c * yhat + std::sqrt(std::max(1.0 - c * c, 0.0)) * g;
  const double vn = v.norm();
  if (!(vn > opt_.eps)) {
    basis_.col(0) = yhat;
    basis_indices_.push_back(VD_Y_SENTINEL);
    return;
  }
  v /= vn;
  basis_.col(0) = v;
  basis_indices_.push_back(VD_DUMMY_SENTINEL);   // v never enters the active set

  Vec y2 = yhat - yhat.dot(v) * v;
  const double y2n = y2.norm();
  if (y2n > opt_.eps) {
    basis_.col(1) = y2 / y2n;
    basis_indices_.push_back(VD_Y_SENTINEL);
  }
}

Eigen::VectorXi VD_Base::active_indices() const {
  Eigen::VectorXi out(int(actives_.size()));
  for (int i = 0; i < (int)actives_.size(); ++i) out(i) = actives_[i];
  return out;
}

std::optional<Vec> VD_Base::orthonormalize_(const Vec& v) const {
  const int nb = basis_size_();
  if (nb == 0) {
    const double nv = v.norm();
    if (nv <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
    return Vec(v / nv);
  }
  proj_coeffs_.head(nb).noalias() = basis_.leftCols(nb).transpose() * v;
  ortho_buffer_.noalias() = v - basis_.leftCols(nb) * proj_coeffs_.head(nb);
  const double nu = ortho_buffer_.norm();
  if (nu <= opt_.eps * std::sqrt(double(n_))) return std::nullopt;
  return Vec(ortho_buffer_ / nu);
}

Vec VD_Base::project_to_Vperp_(const Vec& z) const {
  const int nb = basis_size_();
  if (nb == 0) return z;
  Vec coeff = basis_.leftCols(nb).transpose() * z;
  return z - basis_.leftCols(nb) * coeff;
}

// ---------- VD pool ----------
void VD_Base::ensure_vd_rows_capacity_(int need) {
  if (need <= vd_rows_cap_) return;
  int new_cap = std::min(opt_.max_vd_proj,
      std::max(vd_rows_cap_ > 0 ? vd_rows_cap_ * 2 : 128, need));
  MatR tmp(new_cap, L_);
  if (vd_rows_filled_ > 0)
    tmp.topRows(vd_rows_filled_) = vd_proj_.topRows(vd_rows_filled_);
  vd_proj_.swap(tmp);
  vd_rows_cap_ = new_cap;
}

void VD_Base::prepare_stopping_conditional_radius_() {
  const bool fresh_calibration = calibration_dummies_ > 0;
  if (!opt_.adaptive_dummy_radius
      || (!fresh_calibration && !opt_.dummy_radius_scale_stopping)
      || opt_.dummy_law != DummyLaw::Spherical
      || L_ <= 0) {
    return;
  }

  int remaining_real = 0;
  for (int j = 0; j < p_; ++j)
    if (!is_active_[j]) ++remaining_real;
  const int pool_begin = 0;
  const int pool_end =
      fresh_calibration ? calibration_dummies_ : L_;
  int remaining_dummies = 0;
  for (int d = pool_begin; d < pool_end; ++d)
    if (!vd_is_realized_[d]) ++remaining_dummies;
  if (remaining_real <= 0 || remaining_dummies <= 0) return;

  int remaining_stopping = 0;
  double stopping_maximum = 0.0;
  if (fresh_calibration && opt_.dummy_radius_account_for_stopping) {
    for (int d = calibration_dummies_; d < L_; ++d) {
      if (vd_is_realized_[d]) continue;
      ++remaining_stopping;
      stopping_maximum =
          std::max(stopping_maximum, std::abs(vd_corr_(d)));
    }
  }

  const int active_upper_bound =
      std::max(opt_.dummy_radius_active_upper_bound, 0);
  const int null_lower_bound =
      std::max(remaining_real - active_upper_bound, 0);
  double target_probability = opt_.dummy_radius_target_win_prob;
  if (opt_.dummy_radius_urn_target && null_lower_bound > 0) {
    target_probability =
        double(remaining_dummies)
        / double(
            remaining_dummies + null_lower_bound
            + (opt_.dummy_radius_account_for_stopping
                   ? remaining_stopping : 0));
  }
  if (!(target_probability > 0.0 && target_probability < 1.0)) return;

  std::vector<double> inactive_real_scores;
  inactive_real_scores.reserve(remaining_real);
  for (int j = 0; j < p_; ++j)
    if (!is_active_[j])
      inactive_real_scores.push_back(std::abs(corr_(j)));
  if (inactive_real_scores.empty()) return;

  std::size_t index = 0;
  if (opt_.dummy_radius_real_score_rank > 0) {
    const std::size_t descending_rank = std::min(
        static_cast<std::size_t>(opt_.dummy_radius_real_score_rank),
        inactive_real_scores.size());
    index = inactive_real_scores.size() - descending_rank;
  } else {
    const double quantile = std::min(
        std::max(opt_.dummy_radius_real_score_quantile, 0.0), 1.0);
    const std::size_t rank = std::min(
        static_cast<std::size_t>(
            std::ceil(quantile * inactive_real_scores.size())),
        inactive_real_scores.size());
    index = rank > 0 ? rank - 1 : 0;
  }
  std::nth_element(
      inactive_real_scores.begin(),
      inactive_real_scores.begin() + index,
      inactive_real_scores.end());
  const double real_reference = inactive_real_scores[index];

  double theta = std::min(
      std::max(opt_.dummy_radius_envelope_theta, 0.0), 1.0);
  if (opt_.dummy_radius_covariance_model
      != DummyRadiusCovarianceModel::None) {
    const int selected_real = p_ - remaining_real;
    const int remaining_active_upper_bound =
        std::min(active_upper_bound, remaining_real);
    theta = vd_detail::covariance_envelope_theta(
        opt_.dummy_radius_covariance_model,
        opt_.dummy_radius_covariance_rho,
        remaining_active_upper_bound, selected_real);
  }
  const double noise_bound =
      std::max(opt_.dummy_radius_noise_score_bound, 0.0);
  double threshold = std::min(
      real_reference,
      theta * real_reference + (1.0 + theta) * noise_bound);
  if (fresh_calibration && opt_.dummy_radius_account_for_stopping)
    threshold = std::max(threshold, stopping_maximum);

  Vec centered = residuals_;
  centered.array() -= centered.mean();
  const int filled = std::min(vd_rows_filled_, basis_size_());
  Vec coefficients;
  Vec unexplained = centered;
  if (filled > 0) {
    coefficients =
        basis_.leftCols(filled).transpose() * centered;
    unexplained.noalias() -=
        basis_.leftCols(filled) * coefficients;
    Vec correction =
        basis_.leftCols(filled).transpose() * unexplained;
    unexplained.noalias() -=
        basis_.leftCols(filled) * correction;
    coefficients += correction;
  }
  const double unexplained_norm = unexplained.norm();
  const int remaining_dimension =
      std::max((n_ - 1) - filled, 1);

  std::vector<double> means;
  std::vector<double> scales;
  means.reserve(remaining_dummies);
  scales.reserve(remaining_dummies);
  for (int d = pool_begin; d < pool_end; ++d) {
    if (vd_is_realized_[d]) continue;
    double priority = 1.0;
    if (fresh_calibration &&
        opt_.fresh_calibration_priorities) {
      priority = std::max(
          (*opt_.fresh_calibration_priorities)(d), 0.0);
    }
    double known = 0.0;
    if (filled > 0)
      known = vd_proj_.topRows(filled).col(d).dot(coefficients);
    means.push_back(priority * known);
    scales.push_back(
        priority * unexplained_norm
        * std::sqrt(std::max(vd_stick_(d), 0.0)));
  }
  conditional_dummy_radius_ =
      vd_detail::conditional_spherical_pool_radius(
          threshold, target_probability, means, scales,
          remaining_dimension);
  conditional_dummy_radius_available_ = true;
}

void VD_Base::initialize_virtual_dummies_() {
  ensure_vd_rows_capacity_(1);
  std::fill(vd_is_realized_.begin(), vd_is_realized_.end(), 0);
  vd_unrealized_idx_.clear();
  for (int d = 0; d < L_; ++d) vd_unrealized_idx_.push_back(d);
  vd_rows_filled_ = 0;

  // Compute Z_proj_(:, 0) = X^T e_0 for all laws (needed by AdaptiveGaussian,
  // cheap for others).
  if (basis_size_() > 0) {
    if (shared_products_() && basis_indices_[0] == VD_Y_SENTINEL) {
      Z_proj_.col(0) = opt_.shared_real_cache->Xty() / y_norm_;
    } else {
      Vec tmp(p_);
      vd_detail::gemv_Xt(X_, basis_.col(0), tmp,
          opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
      Z_proj_.col(0) = tmp;
    }
  }

  prepare_stopping_conditional_radius_();

  if (opt_.dummy_law == DummyLaw::Spherical || opt_.vd_seed_corr >= 0.0) {
    // With a seeded basis, row 0 is the projection onto the RANDOM direction v.
    // We do not FIT it -- we impose the clean spherical null law.  The dummy's
    // correlation with y is then ||y|| (c*z0 + sqrt(1-c^2)*z1): a fraction c of
    // it comes from an UNCONTAMINATED source, and the contaminated fitted part
    // (z1, learned on the completing direction) is scaled down by sqrt(1-c^2).
    const double b = std::max(0.5 * double(n_ - 2), 1e-8);
    std::normal_distribution<double> N01(0.0, 1.0);
    double* row0 = vd_proj_.row(0).data();
    for (int d = 0; d < L_; ++d) {
      double z = N01(rng_);
      double X = 0.5*z*z;
      double Y = vd_detail::gamma_mt(b, rng_);
      double U = X/(X+Y);
      double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
      row0[d] = sgn * std::sqrt(U);
      vd_stick_(d) = 1.0 - row0[d]*row0[d];
    }
  } else if (opt_.dummy_law == DummyLaw::AdaptiveGaussian ||
      opt_.dummy_law == DummyLaw::SubGaussian ||
      opt_.dummy_law == DummyLaw::SphericalAdaptiveGaussian) {
    // Marginal fit from Z_proj_(:, 0), trimmed by |corr_|.
    adaptive_gaussian_update_(0);
  } else if (opt_.dummy_law == DummyLaw::AdaptiveBeta) {
    adaptive_beta_update_(0);
  } else {
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row0 = vd_proj_.row(0).data();
    for (int d = 0; d < L_; ++d) {
      row0[d] = scale * N01(rng_);
      vd_stick_(d) = 1.0;
    }
  }
  vd_rows_filled_ = 1;
  // With a seeded basis, yhat is basis column 1, so its dummy projections must
  // be drawn too before the correlations mean anything.
  const int nb0 = basis_size_();
  if (nb0 > 1) {
    ensure_vd_rows_capacity_(nb0);
    for (int k = 1; k < nb0; ++k) {
      if (k < Z_proj_.cols()) {
        Vec tmp(p_);
        vd_detail::gemv_Xt(X_, basis_.col(k), tmp,
            opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
        Z_proj_.col(k) = tmp;
      }
      if (opt_.dummy_law == DummyLaw::AdaptiveGaussian ||
          opt_.dummy_law == DummyLaw::SubGaussian ||
          opt_.dummy_law == DummyLaw::SphericalAdaptiveGaussian)
        adaptive_gaussian_update_(k);
      else if (opt_.dummy_law == DummyLaw::AdaptiveBeta)
        adaptive_beta_update_(k);
    }
    vd_rows_filled_ = nb0;
  }

  // vd_corr = A_k^T (E_k^T r) with r = y at initialization.  Reduces to
  // y_norm * row0 in the classic case where basis(0) = yhat.
  {
    const int nb = basis_size_();
    const int mr = std::min(nb, vd_rows_filled_);
    if (mr > 0 && L_ > 0) {
      Eigen::Map<const Vec> yv2(y_.data(), n_);
      Vec bp = basis_.leftCols(nb).transpose() * yv2;
      vd_corr_.noalias() =
          vd_proj_.topRows(mr).transpose() * bp.head(mr);
    }
  }
}

void VD_Base::update_virtual_dummies_(const Vec* source,
                                      const Vec* Xt_source) {
  const int m = basis_size_() - 1;
  if (m < 0 || m >= opt_.max_vd_proj) return;
  ensure_vd_rows_capacity_(m + 1);
  prepare_stopping_conditional_radius_();

  // Compute Z_proj_(:, m) = X^T e_m for all laws.
  if (m < Z_proj_.cols()) {
    bool used_gram = false;
    if (shared_products_() && source && Xt_source &&
        source->size() == n_ && Xt_source->size() == p_) {
      const int prev = m;
      const double eta = basis_.col(m).dot(*source);
      if (std::abs(eta) > opt_.eps) {
        if (prev > 0) {
          Vec coeff = basis_.leftCols(prev).transpose() * (*source);
          Z_proj_.col(m).noalias() =
              ((*Xt_source) - Z_proj_.leftCols(prev) * coeff) / eta;
        } else {
          Z_proj_.col(m) = (*Xt_source) / eta;
        }
        used_gram = true;
      }
    }
    if (!used_gram) {
      Vec tmp(p_);
      vd_detail::gemv_Xt(X_, basis_.col(m), tmp,
          opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
      Z_proj_.col(m) = tmp;
    }
  }

  if (opt_.dummy_law == DummyLaw::AdaptiveGaussian ||
      opt_.dummy_law == DummyLaw::SubGaussian ||
      opt_.dummy_law == DummyLaw::SphericalAdaptiveGaussian) {
    adaptive_gaussian_update_(m);
    if (vd_rows_filled_ < m+1) vd_rows_filled_ = m+1;
    return;
  }

  if (opt_.dummy_law == DummyLaw::AdaptiveBeta) {
    adaptive_beta_update_(m);
    if (vd_rows_filled_ < m+1) vd_rows_filled_ = m+1;
    return;
  }

  if (opt_.dummy_law == DummyLaw::Gaussian) {
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row_m = vd_proj_.row(m).data();
    for (int d = 0; d < L_; ++d) {
      if (!vd_is_realized_[d]) row_m[d] = scale * N01(rng_);
    }
    if (vd_rows_filled_ < m+1) vd_rows_filled_ = m+1;
    return;
  }

  const double b = std::max(0.5 * double(n_ - m - 2), 1e-8);
  if ((int)vd_unrealized_idx_.size() == 0 || vd_rows_filled_ == 0) {
    vd_unrealized_idx_.clear();
    for (int d = 0; d < L_; ++d)
      if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
        vd_unrealized_idx_.push_back(d);
  }
  if (vd_unrealized_idx_.empty()) {
    if (vd_rows_filled_ < m+1) vd_rows_filled_ = m+1;
    return;
  }

  std::normal_distribution<double> N01(0.0, 1.0);
  double* row_m = vd_proj_.row(m).data();
  int keep = 0;
  for (int i = 0; i < (int)vd_unrealized_idx_.size(); ++i) {
    const int d = vd_unrealized_idx_[i];
    double z   = N01(rng_);
    double X   = 0.5*z*z;
    double Y   = vd_detail::gamma_mt(b, rng_);
    double U   = X/(X+Y);
    double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
    double st  = vd_stick_(d);
    double a   = sgn * std::sqrt(st * U);
    row_m[d] = a;
    double st_new = st - a*a;
    vd_stick_(d) = (st_new > 0.0 ? st_new : 0.0);
    if (!vd_is_realized_[d] && vd_stick_(d) > opt_.eps)
      vd_unrealized_idx_[keep++] = d;
  }
  if (keep != (int)vd_unrealized_idx_.size())
    vd_unrealized_idx_.resize(keep);
  if (vd_rows_filled_ < m+1) vd_rows_filled_ = m+1;
}

// ---------- Adaptive Gaussian conditional sampling ----------
// Partition predictors and dummies into groups.  Real designs are grouped (LD
// blocks, co-expression modules); a single pooled Gaussian over heterogeneous
// groups is an over-dispersed mixture, which makes the dummies too competitive.
void VD_Base::init_groups_() {
  const int G = opt_.groups.empty()
      ? 1
      : (1 + *std::max_element(opt_.groups.begin(), opt_.groups.end()));
  group_members_.assign(std::max(G, 1), {});
  group_dummies_.assign(std::max(G, 1), {});
  sg_beta_state_by_group_.assign(
      std::max(G, 1), std::numeric_limits<double>::quiet_NaN());
  sg_beta_path_.clear();

  for (int j = 0; j < p_; ++j) {
    int g = opt_.groups.empty() ? 0 : opt_.groups[j];
    if (g < 0 || g >= (int)group_members_.size()) g = 0;
    group_members_[g].push_back(j);
  }
  // Assign dummies to groups in proportion to group size, so the dummy
  // population mirrors the real population's group composition.
  if (group_members_.size() == 1) {
    for (int d = 0; d < L_; ++d) group_dummies_[0].push_back(d);
    return;
  }
  std::vector<double> cum;
  cum.reserve(group_members_.size());
  double tot = 0.0;
  for (const auto& m : group_members_) { tot += (double)m.size(); cum.push_back(tot); }
  for (int d = 0; d < L_; ++d) {
    // deterministic proportional allocation (stable across runs)
    double u = (tot * (d + 0.5)) / (double)L_;
    int g = 0;
    while (g + 1 < (int)cum.size() && u > cum[g]) ++g;
    group_dummies_[g].push_back(d);
  }
}


// ---------------------------------------------------------------------------
// AdaptiveBeta dummy law.
//
// The real columns are unit-norm, so for the orthonormal active basis E_k the
// squared projections plus the leftover "stick" lie on the simplex:
//     z_0^2 + ... + z_k^2 + (1 - ||z||^2) = 1.
// The natural law on a simplex is Dirichlet, whose sequential form is
// stick-breaking with Beta increments.  The spherical law is exactly this with
// the fixed ISOTROPIC parameters Beta(1/2, (n-k-2)/2).  AdaptiveBeta keeps that
// geometry (so the dummy norms are automatically admissible, unlike the
// Gaussian law) but FITS the increment law to the design.
//
// Under dependence the empirical increments are a mixture: most inactive reals
// stay near-isotropic, while a minority are dragged into alignment with the
// active subspace ("proxies") and carry a heavy tail that no single Gaussian
// (and no single Beta) can represent without destroying the bulk.  We therefore
// fit a two-component mixture by EM:
//     component 1 : Beta(1/2, (n-k-2)/2)   -- the exact isotropic null (fixed)
//     component 2 : Beta(a2, b2)           -- the fitted heavy proxy component
//     weight w    : the proxy fraction
// which degenerates to the spherical law as w -> 0 (orthogonal design).
void VD_Base::adaptive_beta_update_(int k) {
  const int min_samples = std::max(8, k + 2);

  // Inactive reals, trimmed exactly as in AdaptiveGaussian.
  std::vector<int> keep_idx;
  keep_idx.reserve(p_);
  for (int j = 0; j < p_; ++j)
    if (!is_active_[j]) keep_idx.push_back(j);

  if (opt_.trim_q < 1.0 && (int)keep_idx.size() > min_samples) {
    std::vector<double> ac(keep_idx.size());
    for (size_t i = 0; i < keep_idx.size(); ++i)
      ac[i] = std::abs(corr_(keep_idx[i]));
    std::vector<double> srt = ac;
    std::sort(srt.begin(), srt.end());
    int qi = std::min((int)(opt_.trim_q * srt.size()), (int)srt.size() - 1);
    const double thr = srt[qi];
    std::vector<int> trimmed;
    trimmed.reserve(keep_idx.size());
    for (size_t i = 0; i < keep_idx.size(); ++i)
      if (ac[i] <= thr) trimmed.push_back(keep_idx[i]);
    if ((int)trimmed.size() >= min_samples) keep_idx.swap(trimmed);
  }

  // Isotropic null parameters for this step (the spherical law).
  const double a1 = 0.5;
  const double b1 = std::max(0.5 * double(n_ - k - 2), 0.5);

  // Empirical stick fractions u_j = z_{j,k}^2 / s_j, with the real column's
  // running stick s_j = 1 - sum_{l<k} z_{j,l}^2 (columns have unit norm).
  std::vector<double> u;
  u.reserve(keep_idx.size());
  for (int j : keep_idx) {
    double used = 0.0;
    for (int l = 0; l < k; ++l) {
      const double z = Z_proj_(j, l);
      used += z * z;
    }
    const double s = 1.0 - used;
    if (s <= 1e-12) continue;
    const double zk = Z_proj_(j, k);
    double uj = (zk * zk) / s;
    if (!(uj > 0.0) || !(uj < 1.0)) uj = std::min(std::max(uj, 1e-12), 1.0 - 1e-12);
    u.push_back(uj);
  }

  double a2 = a1, b2 = b1, w = 0.0;
  if ((int)u.size() >= min_samples) {
    // init the heavy component from the upper half of the increments
    std::vector<double> srt = u;
    std::sort(srt.begin(), srt.end());
    const double med = srt[srt.size() / 2];
    double m0 = 0.0, v0 = 0.0; int c0 = 0;
    for (double x : u) if (x >= med) { m0 += x; ++c0; }
    if (c0 > 1) {
      m0 /= c0;
      for (double x : u) if (x >= med) v0 += (x - m0) * (x - m0);
      v0 /= (c0 - 1);
      if (v0 > 1e-12 && v0 < m0 * (1.0 - m0)) {
        const double c = m0 * (1.0 - m0) / v0 - 1.0;
        a2 = std::max(m0 * c, 1e-3);
        b2 = std::max((1.0 - m0) * c, 1e-3);
      }
    }
    w = 0.1;

    std::vector<double> r(u.size(), 0.0);
    for (int it = 0; it < 40; ++it) {
      // E-step: responsibility of the heavy component
      double rsum = 0.0;
      for (size_t i = 0; i < u.size(); ++i) {
        const double l1 = std::log(std::max(1.0 - w, 1e-12))
                        + vd_detail::beta_logpdf(u[i], a1, b1);
        const double l2 = std::log(std::max(w, 1e-12))
                        + vd_detail::beta_logpdf(u[i], a2, b2);
        const double mx = std::max(l1, l2);
        const double e1 = std::exp(l1 - mx), e2 = std::exp(l2 - mx);
        r[i] = e2 / (e1 + e2);
        rsum += r[i];
      }
      w = std::min(std::max(rsum / double(u.size()), 1e-4), 0.9);
      if (rsum < 5.0) break;
      // M-step: weighted method-of-moments for the heavy component
      double m = 0.0;
      for (size_t i = 0; i < u.size(); ++i) m += r[i] * u[i];
      m /= rsum;
      double v = 0.0;
      for (size_t i = 0; i < u.size(); ++i) v += r[i] * (u[i] - m) * (u[i] - m);
      v /= rsum;
      if (v > 1e-12 && v < m * (1.0 - m)) {
        const double c = m * (1.0 - m) / v - 1.0;
        a2 = std::max(m * c, 1e-3);
        b2 = std::max((1.0 - m) * c, 1e-3);
      }
    }

    // Model selection.  A two-component Beta mixture can always "explain" a
    // single Beta by splitting it, so EM alone never drives w to 0 -- even
    // under isotropy, where the heavy component must NOT exist.  Keep it only
    // if it pays for its 3 extra parameters (w, a2, b2) in BIC; otherwise set
    // w = 0, which makes the law reduce EXACTLY to the spherical null.
    double ll_mix = 0.0, ll_iso = 0.0;
    for (size_t i = 0; i < u.size(); ++i) {
      const double lp1 = vd_detail::beta_logpdf(u[i], a1, b1);
      const double lp2 = vd_detail::beta_logpdf(u[i], a2, b2);
      const double l1 = std::log(std::max(1.0 - w, 1e-12)) + lp1;
      const double l2 = std::log(std::max(w, 1e-12)) + lp2;
      const double mx = std::max(l1, l2);
      ll_mix += mx + std::log(std::exp(l1 - mx) + std::exp(l2 - mx));
      ll_iso += lp1;
    }
    const double bic_gain =
        2.0 * (ll_mix - ll_iso) - 3.0 * std::log(double(u.size()));
    if (!(bic_gain > 0.0)) w = 0.0;
  }

  // Draw each unrealised dummy's increment from the fitted mixture and break
  // its stick.  This keeps ||z_d||^2 <= 1 by construction.
  auto uniform01 = [&]() {
    constexpr double scale = 1.0 / (1ULL << 53);
    return (rng_() >> (64 - 53)) * scale;
  };
  // Assign the proxy label ONCE; it is a property of the dummy column, so that
  // a "proxy" dummy is heavy at every step, exactly like a proxy real null.
  if (!vd_heavy_assigned_) {
    vd_heavy_.assign(L_, 0);
    for (int d = 0; d < L_; ++d)
      vd_heavy_[d] = (uniform01() < w) ? 1 : 0;
    vd_heavy_assigned_ = true;
  }
  double* row_k = vd_proj_.row(k).data();
  for (int d = 0; d < L_; ++d) {
    if (vd_is_realized_[d]) continue;
    const double st = vd_stick_(d);
    if (st <= opt_.eps) { row_k[d] = 0.0; continue; }
    const bool heavy = (vd_heavy_[d] != 0);
    double ud = heavy ? vd_detail::beta_rv(a2, b2, rng_)
                      : vd_detail::beta_rv(a1, b1, rng_);
    ud = std::min(std::max(ud, 0.0), 1.0);
    const double sgn = (rng_() & 1ULL) ? 1.0 : -1.0;
    row_k[d] = sgn * std::sqrt(std::max(st * ud, 0.0));
    const double st_new = st * (1.0 - ud);
    vd_stick_(d) = (st_new > 0.0 ? st_new : 0.0);
  }
  if (vd_rows_filled_ < k + 1) vd_rows_filled_ = k + 1;
}

void VD_Base::adaptive_gaussian_update_(int k) {
  // Fit and sample the AdaptiveGaussian law SEPARATELY within each group.
  for (size_t g = 0; g < group_members_.size(); ++g) {
    if (group_dummies_[g].empty()) continue;
    std::vector<int> keep_idx;
    keep_idx.reserve(group_members_[g].size());
    for (int j : group_members_[g])
      if (!is_active_[j]) keep_idx.push_back(j);
    ag_fit_and_sample_(k, keep_idx, group_dummies_[g], static_cast<int>(g));
  }
  if (vd_rows_filled_ < k + 1) vd_rows_filled_ = k + 1;
}

// Estimate the zero-mean Gaussian scale from only the largest m absolute
// residuals.  The likelihood is normalized by P(|N(0,sigma^2)| >= u), where
// u is the smallest retained magnitude.  Consequently, ordinary Gaussian
// data recover their original scale instead of the inflated variance of a
// naive tail-only sample variance.  If m covers the full pool, pooled AG is
// returned exactly.
double VD_Base::fit_gaussian_scale_top_tail_(
    const std::vector<double>& residuals, double fallback,
    int tail_m) const {
  const int n = static_cast<int>(residuals.size());
  const int requested_m = (tail_m >= 0) ? tail_m : opt_.ag_tail_m;
  const int m = std::min(requested_m, n);
  if (m <= 0 || m >= n || n < 8)
    return std::max(fallback, std::sqrt(opt_.sigma_reg));

  std::vector<double> absolute;
  absolute.reserve(n);
  for (double value : residuals)
    if (std::isfinite(value)) absolute.push_back(std::abs(value));
  if (static_cast<int>(absolute.size()) < m)
    return std::max(fallback, std::sqrt(opt_.sigma_reg));
  std::sort(absolute.begin(), absolute.end());
  const int first = static_cast<int>(absolute.size()) - m;
  const double threshold = absolute[first];
  double sumsq = 0.0;
  for (int i = first; i < static_cast<int>(absolute.size()); ++i)
    sumsq += absolute[i] * absolute[i];

  const double safe_fallback = std::max(
      fallback, std::sqrt(opt_.sigma_reg));
  const double tail_rms = std::sqrt(sumsq / double(m));
  const double lo = std::log(std::max(0.1 * safe_fallback, 1e-12));
  const double hi = std::log(std::max(
      10.0 * safe_fallback, 5.0 * tail_rms));

  auto objective = [&](double log_scale) {
    const double scale = std::exp(log_scale);
    const double a = threshold / scale;
    const double survival = std::erfc(a / std::sqrt(2.0));
    const double log_survival = std::log(std::max(survival, 1e-300));
    return -double(m) * log_scale
         - 0.5 * sumsq / (scale * scale)
         - double(m) * log_survival;
  };

  // Golden-section maximization on log(sigma).  A one-dimensional bounded
  // search is deterministic and avoids introducing a numerical dependency.
  constexpr double ratio = 0.6180339887498948482;
  double left = lo, right = hi;
  double x1 = right - ratio * (right - left);
  double x2 = left + ratio * (right - left);
  double f1 = objective(x1), f2 = objective(x2);
  for (int iteration = 0; iteration < 70; ++iteration) {
    if (f1 > f2) {
      right = x2;
      x2 = x1;
      f2 = f1;
      x1 = right - ratio * (right - left);
      f1 = objective(x1);
    } else {
      left = x1;
      x1 = x2;
      f1 = f2;
      x2 = left + ratio * (right - left);
      f2 = objective(x2);
    }
  }
  const double fitted = std::exp(f1 > f2 ? x1 : x2);
  return std::isfinite(fitted) ? fitted : safe_fallback;
}

// Fit the generalized-normal shape to an absolute-residual interval.  The
// likelihood is normalized over the retained interval, so discarding the most
// signal-contaminated residuals does not masquerade as evidence for light
// tails.  Scale is profiled out numerically for each candidate shape.
double VD_Base::fit_subbotin_beta_truncated_(
    const std::vector<double>& residuals, double fallback) const {
  std::vector<double> abs_residuals;
  abs_residuals.reserve(residuals.size());
  double sumsq = 0.0;
  for (double value : residuals) {
    if (!std::isfinite(value)) continue;
    const double absolute = std::abs(value);
    abs_residuals.push_back(absolute);
    sumsq += absolute * absolute;
  }
  if (abs_residuals.size() < 30) return fallback;
  std::sort(abs_residuals.begin(), abs_residuals.end());

  const double q_lo = std::min(std::max(opt_.sg_fit_q_lo, 0.0), 0.94);
  const double q_hi = std::min(
      std::max(opt_.sg_fit_q_hi, q_lo + 0.05), 1.0);
  const int n = static_cast<int>(abs_residuals.size());
  const int i_lo = std::min(
      n - 1, std::max(0, static_cast<int>(std::floor(q_lo * (n - 1)))));
  const int i_hi = std::min(
      n - 1, std::max(i_lo + 1,
                      static_cast<int>(std::ceil(q_hi * (n - 1)))));
  const double c_lo = (q_lo <= 0.0) ? 0.0 : abs_residuals[i_lo];
  const double c_hi = abs_residuals[i_hi];
  if (!(c_hi > c_lo)) return fallback;

  std::vector<double> retained;
  retained.reserve(i_hi - i_lo + 1);
  for (double absolute : abs_residuals)
    if (absolute >= c_lo && absolute <= c_hi) retained.push_back(absolute);
  if (retained.size() < 20) return fallback;

  const double scale_reference = std::max(
      std::sqrt(sumsq / static_cast<double>(abs_residuals.size())), 1e-6);
  const double log_scale_lo = std::log(scale_reference) - 3.5;
  const double log_scale_hi = std::log(scale_reference) + 3.5;

  auto log_likelihood = [&](double beta, double log_scale) {
    const double scale = std::exp(log_scale);
    const double inv_beta = 1.0 / beta;
    const double x_lo = (c_lo > 0.0)
        ? std::pow(c_lo / scale, beta) : 0.0;
    const double x_hi = std::pow(c_hi / scale, beta);
    const double p_lo = (c_lo > 0.0)
        ? vd_detail::gamma_p(inv_beta, x_lo) : 0.0;
    const double p_hi = vd_detail::gamma_p(inv_beta, x_hi);
    const double normalizer = p_hi - p_lo;
    if (!(normalizer > 1e-14) || !std::isfinite(normalizer))
      return -std::numeric_limits<double>::infinity();

    double power_sum = 0.0;
    for (double absolute : retained)
      power_sum += std::pow(absolute / scale, beta);
    const double count = static_cast<double>(retained.size());
    return count * (std::log(beta) - log_scale
                    - std::lgamma(inv_beta) - std::log(normalizer))
           - power_sum;
  };

  auto profile_scale = [&](double beta) {
    // Golden-section maximization on log(scale).
    constexpr double ratio = 0.6180339887498948482;
    double left = log_scale_lo;
    double right = log_scale_hi;
    double x1 = right - ratio * (right - left);
    double x2 = left + ratio * (right - left);
    double f1 = log_likelihood(beta, x1);
    double f2 = log_likelihood(beta, x2);
    for (int iteration = 0; iteration < 28; ++iteration) {
      if (f1 > f2) {
        right = x2;
        x2 = x1;
        f2 = f1;
        x1 = right - ratio * (right - left);
        f1 = log_likelihood(beta, x1);
      } else {
        left = x1;
        x1 = x2;
        f1 = f2;
        x2 = left + ratio * (right - left);
        f2 = log_likelihood(beta, x2);
      }
    }
    return std::max(f1, f2);
  };

  const double beta_min = std::max(opt_.sg_beta_min, 0.25);
  const double beta_max = std::max(opt_.sg_beta_max, beta_min + 0.05);
  double best_beta = std::min(std::max(fallback, beta_min), beta_max);
  double best_value = -std::numeric_limits<double>::infinity();

  // A coarse scan prevents a local numerical feature in the truncated
  // likelihood from determining the answer.  The final short golden search
  // only refines the best coarse interval.
  constexpr double coarse_step = 0.25;
  for (double beta = beta_min; beta < beta_max; beta += coarse_step) {
    const double value = profile_scale(beta);
    if (value > best_value) {
      best_value = value;
      best_beta = beta;
    }
  }
  const double endpoint_value = profile_scale(beta_max);
  if (endpoint_value > best_value) {
    best_value = endpoint_value;
    best_beta = beta_max;
  }
  if (!std::isfinite(best_value)) return fallback;

  double left = std::max(beta_min, best_beta - coarse_step);
  double right = std::min(beta_max, best_beta + coarse_step);
  if (right > left + 1e-8) {
    constexpr double ratio = 0.6180339887498948482;
    double x1 = right - ratio * (right - left);
    double x2 = left + ratio * (right - left);
    double f1 = profile_scale(x1);
    double f2 = profile_scale(x2);
    for (int iteration = 0; iteration < 18; ++iteration) {
      if (f1 > f2) {
        right = x2;
        x2 = x1;
        f2 = f1;
        x1 = right - ratio * (right - left);
        f1 = profile_scale(x1);
      } else {
        left = x1;
        x1 = x2;
        f1 = f2;
        x2 = left + ratio * (right - left);
        f2 = profile_scale(x2);
      }
    }
    if (f1 > best_value) { best_value = f1; best_beta = x1; }
    if (f2 > best_value) { best_value = f2; best_beta = x2; }
  }
  return std::min(std::max(best_beta, beta_min), beta_max);
}

void VD_Base::ag_fit_and_sample_(int k, const std::vector<int>& keep_in,
                                 const std::vector<int>& dummies,
                                 int group_index) {
  const double sreg = opt_.sigma_reg;
  const int min_samples = std::max(5, k + 2);
  std::vector<int> keep_idx = keep_in;

  // Quantile trim on |corr_|
  if (opt_.trim_q < 1.0 && (int)keep_idx.size() > min_samples) {
    std::vector<double> abs_corrs(keep_idx.size());
    for (int i = 0; i < (int)keep_idx.size(); ++i)
      abs_corrs[i] = std::abs(corr_(keep_idx[i]));
    std::vector<double> sorted_ac = abs_corrs;
    std::sort(sorted_ac.begin(), sorted_ac.end());
    int qi = std::min((int)(opt_.trim_q * sorted_ac.size()),
                      (int)sorted_ac.size() - 1);
    double thresh = sorted_ac[qi];
    std::vector<int> trimmed;
    trimmed.reserve(keep_idx.size());
    for (int i = 0; i < (int)keep_idx.size(); ++i)
      if (abs_corrs[i] <= thresh) trimmed.push_back(keep_idx[i]);
    if ((int)trimmed.size() >= min_samples) keep_idx = std::move(trimmed);
  }

  const bool spherical_ag =
      (opt_.dummy_law == DummyLaw::SphericalAdaptiveGaussian);
  const int nt = (int)keep_idx.size();
  if (nt < min_samples) {
    // With too few reals, the sphere-constrained law falls back to the exact
    // isotropic spherical break.  The unconstrained laws keep the original
    // iid-Gaussian fallback.
    std::normal_distribution<double> N01(0.0, 1.0);
    double scale = 1.0 / std::sqrt(double(n_));
    double* row_k = vd_proj_.row(k).data();
    const int future_dims = std::max(n_ - k - 2, 0);
    for (int d : dummies) {
      if (vd_is_realized_[d]) continue;
      if (!spherical_ag) {
        row_k[d] = scale * N01(rng_);
        continue;
      }
      const double stick = std::max(vd_stick_(d), 0.0);
      if (stick <= opt_.eps) { row_k[d] = 0.0; continue; }
      double q;
      if (future_dims == 0) {
        q = (rng_() & 1ULL) ? 1.0 : -1.0;
      } else {
        const double u = vd_detail::beta_rv(
            0.5, 0.5 * double(future_dims), rng_);
        q = ((rng_() & 1ULL) ? 1.0 : -1.0)
            * std::sqrt(std::min(std::max(u, 0.0), 1.0));
      }
      row_k[d] = std::sqrt(stick) * q;
      vd_stick_(d) = std::max(stick * (1.0 - q * q), 0.0);
    }
    return;
  }

  std::normal_distribution<double> N01(0.0, 1.0);
  double* row_k = vd_proj_.row(k).data();

  // AdaptiveGaussian draws N(0,1); SubGaussian draws a unit-variance
  // generalized normal.  SphericalAdaptiveGaussian uses the same fitted
  // Gaussian conditional mean and variance as AdaptiveGaussian, but treats
  // the draw as a latent coordinate in an angular stick break.  Its completed
  // columns lie exactly on the unit sphere in the centered subspace.
  const bool sub_g = (opt_.dummy_law == DummyLaw::SubGaussian);
  auto choose_beta = [&](const std::vector<double>& innovations) {
    double beta = opt_.sg_beta;
    if (opt_.sg_adaptive && sub_g) {
      const bool has_previous =
          group_index >= 0
          && group_index < static_cast<int>(sg_beta_state_by_group_.size())
          && std::isfinite(sg_beta_state_by_group_[group_index]);
      const double previous = has_previous
          ? sg_beta_state_by_group_[group_index] : opt_.sg_beta;
      const double fitted = fit_subbotin_beta_truncated_(innovations, previous);
      if (has_previous) {
        const double weight = std::min(std::max(opt_.sg_beta_smooth, 0.0), 1.0);
        beta = (1.0 - weight) * previous + weight * fitted;
      } else {
        beta = fitted;
      }
      sg_beta_state_by_group_[group_index] = beta;
    } else if (sub_g && opt_.sg_tau > 0.0) {
      beta = 2.0 + (opt_.sg_beta - 2.0) * std::exp(-double(k) / opt_.sg_tau);
    }
    if (sub_g && group_index == 0) {
      if (static_cast<int>(sg_beta_path_.size()) <= k)
        sg_beta_path_.resize(k + 1, std::numeric_limits<double>::quiet_NaN());
      sg_beta_path_[k] = beta;
    }
    return beta;
  };

  // Map one fitted Gaussian coordinate to an angular coordinate.  The
  // chi-square draw represents the latent energy of the unrevealed centered
  // directions.  If latent ~ N(0, tau2), then q^2 is exactly
  // Beta(1/2, future_dims/2), recovering spherical stick-breaking.
  auto angular_sample = [&](double latent, double tau2,
                            int future_dims, int d) {
    const double stick = std::max(vd_stick_(d), 0.0);
    if (stick <= opt_.eps) {
      vd_stick_(d) = 0.0;
      return 0.0;
    }
    const double future_energy = (future_dims > 0)
        ? 2.0 * std::max(tau2, sreg) * vd_detail::gamma_mt(
              0.5 * double(future_dims), rng_)
        : 0.0;
    const double total_energy = latent * latent + future_energy;
    double q;
    if (total_energy > opt_.eps) {
      q = latent / std::sqrt(total_energy);
    } else {
      q = (rng_() & 1ULL) ? 1.0 : -1.0;
    }
    q = std::min(std::max(q, -1.0), 1.0);
    vd_stick_(d) = std::max(stick * (1.0 - q * q), 0.0);
    return std::sqrt(stick) * q;
  };

  if (k == 0) {
    // Marginal: N(mu, var) fitted from Z_proj_(keep, 0)
    double sum = 0.0, sum2 = 0.0;
    for (int j : keep_idx) {
      double z = Z_proj_(j, 0);
      sum += z; sum2 += z * z;
    }
    double mu = sum / nt;
    double var = (nt > 1) ? (sum2 / nt - mu * mu) * nt / (nt - 1.0) : sreg;
    var = std::max(var, sreg);
    double sd = std::sqrt(var);
    std::vector<double> raw_residuals;
    raw_residuals.reserve(nt);
    for (int j : keep_idx)
      raw_residuals.push_back(Z_proj_(j, 0) - mu);
    sd = fit_gaussian_scale_top_tail_(raw_residuals, sd);
    var = sd * sd;
    std::vector<double> innovations;
    innovations.reserve(nt);
    for (int j : keep_idx)
      innovations.push_back((Z_proj_(j, 0) - mu) / sd);
    const double sg_b = choose_beta(innovations);

    const int future_dims = std::max(n_ - k - 2, 0);
    double tau2 = sreg;
    if (spherical_ag && future_dims > 0) {
      double residual_energy = 0.0;
      for (int j : keep_idx) {
        const double z = Z_proj_(j, 0);
        residual_energy += std::max(1.0 - z * z, 0.0);
      }
      tau2 = std::max(
          residual_energy / (double(nt) * double(future_dims)), sreg);
    }

    for (int d : dummies) {
      if (vd_is_realized_[d]) continue;
      const double latent = mu + sd
          * (sub_g ? vd_detail::subgauss_rv(sg_b, rng_) : N01(rng_));
      row_k[d] = spherical_ag
          ? angular_sample(latent, tau2, future_dims, d)
          : latent;
    }
  } else {
    // Conditional Gaussian: alpha_k | alpha_{0:k-1}
    // Build (nt x (k+1)) matrix of trimmed Z values.
    MatC Zt(nt, k + 1);
    for (int i = 0; i < nt; ++i)
      for (int c = 0; c <= k; ++c)
        Zt(i, c) = Z_proj_(keep_idx[i], c);

    Vec mu_hat = Zt.colwise().mean();

    // Covariance (k+1 x k+1), regularized
    MatC centered = Zt.rowwise() - mu_hat.transpose();
    MatC Sigma(k + 1, k + 1);
    if (nt > 1)
      Sigma = (centered.transpose() * centered) / double(nt - 1);
    else
      Sigma = MatC::Identity(k + 1, k + 1);
    Sigma += sreg * MatC::Identity(k + 1, k + 1);

    // Partition: [0:k) = conditioning, [k] = target
    MatC S11 = Sigma.topLeftCorner(k, k);
    Vec  S12 = Sigma.topRightCorner(k, 1);
    double S22 = Sigma(k, k);

    // Solve S11 w = S12 via Cholesky
    Eigen::LLT<MatC> llt(S11);
    if (llt.info() != Eigen::Success)
      llt.compute(S11 + 1e-4 * MatC::Identity(k, k));
    Vec w = llt.solve(S12);

    double v_cond = std::max(S22 - S12.dot(w), sreg);
    double sd_cond = std::sqrt(v_cond);
    double mu_k = mu_hat(k);
    Vec mu_prev = mu_hat.head(k);
    std::vector<double> raw_residuals;
    raw_residuals.reserve(nt);
    for (int i = 0; i < nt; ++i) {
      double conditional_mean = mu_k;
      for (int c = 0; c < k; ++c)
        conditional_mean += w(c) * (Zt(i, c) - mu_prev(c));
      raw_residuals.push_back(Zt(i, k) - conditional_mean);
    }
    sd_cond = fit_gaussian_scale_top_tail_(raw_residuals, sd_cond);
    v_cond = sd_cond * sd_cond;
    std::vector<double> innovations;
    innovations.reserve(nt);
    for (double residual : raw_residuals)
      innovations.push_back(residual / sd_cond);
    const double sg_b = choose_beta(innovations);

    // The latent complement scale comes from the remaining unit-norm energy
    // of the same retained real columns.  This preserves the trace budget
    // without estimating an additional ambient covariance matrix.
    const int future_dims = std::max(n_ - k - 2, 0);
    double tau2 = sreg;
    if (spherical_ag && future_dims > 0) {
      double residual_energy = 0.0;
      for (int i = 0; i < nt; ++i) {
        double used = 0.0;
        for (int c = 0; c <= k; ++c) {
          const double z = Zt(i, c);
          used += z * z;
        }
        residual_energy += std::max(1.0 - used, 0.0);
      }
      tau2 = std::max(
          residual_energy / (double(nt) * double(future_dims)), sreg);
    }

    // For each unrealized dummy, compute conditional mean from its
    // previously sampled projections in rows 0..k-1.
    for (int d : dummies) {
      if (vd_is_realized_[d]) continue;
      double m_cond = mu_k;
      for (int c = 0; c < k; ++c)
        m_cond += w(c) * (vd_proj_(c, d) - mu_prev(c));
      const double latent = m_cond + sd_cond
          * (sub_g ? vd_detail::subgauss_rv(sg_b, rng_) : N01(rng_));
      row_k[d] = spherical_ag
          ? angular_sample(latent, tau2, future_dims, d)
          : latent;
    }
  }
  // AdaptiveGaussian and SubGaussian keep vd_stick_ = 1 for Gaussian
  // completion.  SphericalAdaptiveGaussian updates its stick above.
}

void VD_Base::realize_dummy_(int vd_idx) {
  const int j = T_realized_;
  const int nb = basis_size_();
  if (nb == 0) return;
  if (vd_proj_.rows() < nb) return;

  Eigen::VectorXd alphas = vd_proj_.topRows(nb).col(vd_idx);
  Vec explained(n_);
  explained.noalias() = basis_.leftCols(nb) * alphas;

  double stick = vd_stick_(vd_idx);
  if (stick < 0.0) stick = 0.0;
  const double r = std::sqrt(stick);

  std::normal_distribution<double> N01(0.0, 1.0);
  Eigen::VectorXd v(n_);
  for (int i = 0; i < n_; ++i) v(i) = N01(rng_);
  double mean = v.mean();
  v.array() -= mean;

  Vec x;
  if (opt_.dummy_law == DummyLaw::Spherical ||
      opt_.dummy_law == DummyLaw::AdaptiveBeta ||
      opt_.dummy_law == DummyLaw::SphericalAdaptiveGaussian) {
    auto u_opt = orthonormalize_(v);
    if (!u_opt) return;
    x = explained + r * (*u_opt);
  } else {
    const double scale = 1.0 / std::sqrt(double(n_));
    Vec g_perp = project_to_Vperp_(scale * v);
    if (g_perp.norm() < opt_.eps * std::sqrt(double(n_))) return;
    x = explained + g_perp;
  }

  if (X_realized_.cols() <= j)
    X_realized_.conservativeResize(Eigen::NoChange, j + 1);
  X_realized_.col(j) = x;

  Vec Xt_x;
  if (shared_products_()) {
    Xt_x.resize(p_);
    vd_detail::gemv_Xt(X_, x, Xt_x,
        opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
    Xt_realized_.push_back(Xt_x);
  }

  if (auto vo = orthonormalize_(x)) {
    const int col = basis_size_();
    basis_.col(col) = *vo;
    basis_indices_.push_back(VD_DUMMY_SENTINEL);
    const bool defer_virtual_update =
        opt_.adaptive_dummy_radius
        && (opt_.dummy_radius_scale_stopping ||
            calibration_dummies_ > 0)
        && opt_.dummy_law == DummyLaw::Spherical;
    if (!defer_virtual_update) {
      if (shared_products_())
        update_virtual_dummies_(&x, &Xt_realized_.back());
      else
        update_virtual_dummies_();
    }
  }

  if (corr_realized_.size() < j+1) corr_realized_.conservativeResize(j+1);
  corr_realized_(j) = x.dot(residuals_);

  if (vd_idx >= 0 && vd_idx < vd_corr_.size()) vd_corr_(vd_idx) = 0.0;
  if (vd_idx >= 0 && vd_idx < (int)vd_is_realized_.size())
    vd_is_realized_[vd_idx] = 1;
  vd_stick_(vd_idx) = 0.0;

  ++T_realized_;
  if (vd_idx < calibration_dummies_)
    ++T_calibration_realized_;
  else
    ++T_stopping_realized_;
}

// ---------- Cholesky ----------
bool VD_Base::chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v,
                          double s, double eps)
{
  const int t = R.rows();
  if (t == 0) {
    R.resize(1, 1);
    R(0, 0) = std::sqrt(std::max(s, eps));
    return true;
  }
  Eigen::VectorXd z = v;
  R.transpose().template triangularView<Eigen::Lower>().solveInPlace(z);
  double r2 = s - z.squaredNorm();
  const double floor = std::max(eps, 1e-14 * std::max(1.0, s));
  if (r2 <= floor) r2 = floor;
  R.conservativeResize(t+1, t+1);
  R.topRightCorner(t, 1) = z;
  R.row(t).setZero();
  R(t, t) = std::sqrt(r2);
  return true;
}

// ---------- Full correlation refresh ----------
void VD_Base::full_corr_refresh_() {
  // Real features: corr = X^T r
  if (shared_products_()) {
    corr_ = opt_.shared_real_cache->Xty();
    for (const auto& af : active_features_) {
      if (af.kind == ActiveFeature::Kind::Real) {
        const double coefficient = beta_(af.index);
        if (coefficient != 0.0)
          corr_.noalias() -= coefficient * (*real_cross_(af.index));
      } else if (af.kind == ActiveFeature::Kind::Persistent) {
        const double coefficient = beta_persistent_(af.index);
        if (coefficient != 0.0)
          corr_.noalias() -=
              coefficient * (*persistent_real_cross_(af.index));
      } else if (af.index >= 0 &&
                 af.index < (int)Xt_realized_.size()) {
        const double coefficient = beta_dummy_(af.index);
        if (coefficient != 0.0)
          corr_.noalias() -= coefficient * Xt_realized_[af.index];
      }
    }
  } else {
    vd_detail::gemv_Xt(X_, residuals_, corr_,
        opt_.mmap_fd, opt_.mmap_block_cols, scratch_ptr_());
  }

  // Virtual dummies: vd_corr = A_k^T (E_k^T r)
  const int nb = basis_size_();
  const int m_rows = std::min(nb, vd_rows_filled_);
  if (m_rows > 0 && L_ > 0) {
    Vec basis_proj = basis_.leftCols(nb).transpose() * residuals_;
    vd_corr_.noalias() =
        vd_proj_.topRows(m_rows).transpose() * basis_proj.head(m_rows);
    for (int d = 0; d < L_; ++d)
      if (vd_is_realized_[d]) vd_corr_(d) = 0.0;
  }

  // Realized dummies
  if (T_realized_ > 0) {
    corr_realized_.head(T_realized_).noalias() =
        X_realized_.leftCols(T_realized_).transpose() * residuals_;
  }

  refresh_persistent_correlations_();
}
