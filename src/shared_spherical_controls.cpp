#include "shared_spherical_controls.hpp"
#include <stdexcept>

SharedSphericalControls::SharedSphericalControls(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y,
    int controls,
    unsigned long long seed,
    double eps)
  : Xptr_(X.data()), yptr_(y.data()),
    n_((int)X.rows()), p_((int)X.cols()), controls_(controls),
    centered_dim_(std::max((int)X.rows() - 1, 0)),
    eps_(std::max(eps, 1e-14)), rng_(seed),
    master_basis_(n_, std::min(std::max(8, 1), std::max(centered_dim_, 1))),
    coordinates_(std::min(std::max(8, 1), std::max(centered_dim_, 1)),
                 std::max(controls, 0)),
    stick_(Vec::Ones(std::max(controls, 0))),
    materialized_(std::max(controls, 0)),
    real_cross_(std::max(controls, 0)),
    response_inner_(std::max(controls, 0), 0.0),
    response_ready_(std::max(controls, 0), 0)
{
  if (n_ <= 1) throw std::invalid_argument(
      "SharedSphericalControls requires at least two observations");
  if (y.size() != n_) throw std::invalid_argument(
      "SharedSphericalControls: X/y mismatch");
  if (controls_ <= 0) throw std::invalid_argument(
      "SharedSphericalControls: controls must be positive");
}

void SharedSphericalControls::ensure_capacity_(int need) {
  if (need <= master_basis_.cols()) return;
  const int capacity = std::min(
      centered_dim_,
      std::max(need, std::max(8, 2 * (int)master_basis_.cols())));
  master_basis_.conservativeResize(Eigen::NoChange, capacity);
  MatR enlarged(capacity, controls_);
  if (rank_ > 0) enlarged.topRows(rank_) = coordinates_.topRows(rank_);
  coordinates_.swap(enlarged);
}

Vec SharedSphericalControls::centered_unit_(
    const Eigen::Ref<const Vec>& direction,
    double& original_norm) const
{
  if (direction.size() != n_) throw std::invalid_argument(
      "SharedSphericalControls: direction has the wrong length");
  Vec centered = direction;
  centered.array() -= centered.mean();
  original_norm = centered.norm();
  if (!(original_norm > eps_ * std::sqrt(double(n_))))
    return Vec::Zero(n_);
  centered /= original_norm;
  return centered;
}

Vec SharedSphericalControls::projections(
    const Eigen::Ref<const Vec>& direction)
{
  std::lock_guard<std::mutex> lock(mu_);
  return projections_locked_(
      direction, nullptr, 0.0, -1.0, nullptr);
}

Vec SharedSphericalControls::projections_with_conditional_radius(
    const Eigen::Ref<const Vec>& direction,
    const std::vector<char>& excluded,
    double real_threshold,
    double target_probability,
    double& required_radius)
{
  if ((int)excluded.size() != controls_)
    throw std::invalid_argument(
        "SharedSphericalControls: excluded mask has the wrong length");
  std::lock_guard<std::mutex> lock(mu_);
  return projections_locked_(
      direction, &excluded, real_threshold, target_probability,
      &required_radius);
}

Vec SharedSphericalControls::projections_locked_(
    const Eigen::Ref<const Vec>& direction,
    const std::vector<char>* excluded,
    double real_threshold,
    double target_probability,
    double* required_radius)
{
  double direction_norm = 0.0;
  Vec unit = centered_unit_(direction, direction_norm);
  if (!(direction_norm > 0.0)) {
    if (required_radius) *required_radius = 1.0;
    return Vec::Zero(controls_);
  }

  Vec coefficients;
  Vec residual = unit;
  if (rank_ > 0) {
    coefficients = master_basis_.leftCols(rank_).transpose() * unit;
    residual.noalias() -= master_basis_.leftCols(rank_) * coefficients;
    // A second pass keeps the master basis stable when many path directions
    // are nearly contained in its current span.
    Vec correction = master_basis_.leftCols(rank_).transpose() * residual;
    residual.noalias() -= master_basis_.leftCols(rank_) * correction;
    coefficients += correction;
  }

  const double residual_norm = residual.norm();
  const double tolerance = 64.0 * eps_;
  Vec known = Vec::Zero(controls_);
  if (rank_ > 0) {
    known.noalias() =
        coordinates_.topRows(rank_).transpose() * coefficients;
  }

  // Calibrate from the pre-draw conditional law.  The known contribution is
  // the projection onto the revealed master basis.  The unknown contribution
  // is one coordinate of the remaining spherical stick in dimension
  // centered_dim_-rank_.  Materialized controls are fully deterministic.
  if (required_radius) {
    std::vector<double> means;
    std::vector<double> scales;
    means.reserve(controls_);
    scales.reserve(controls_);
    const int remaining_dimension = centered_dim_ - rank_;
    for (int j = 0; j < controls_; ++j) {
      if (excluded && (*excluded)[j]) continue;
      if (materialized_[j]) {
        means.push_back(direction.dot(*materialized_[j]));
        scales.push_back(0.0);
        continue;
      }
      means.push_back(direction_norm * known(j));
      scales.push_back(
          direction_norm * residual_norm
          * std::sqrt(std::max(stick_(j), 0.0)));
    }
    *required_radius = vd_detail::conditional_spherical_pool_radius(
        real_threshold, target_probability, means, scales,
        remaining_dimension);
  }

  int appended_rank = -1;
  if (residual_norm > tolerance && rank_ < centered_dim_) {
    ensure_capacity_(rank_ + 1);
    Vec q = residual / residual_norm;
    master_basis_.col(rank_) = q;
    appended_rank = rank_;

    const int future_dims = centered_dim_ - rank_ - 1;
    for (int j = 0; j < controls_; ++j) {
      double coordinate = 0.0;
      if (materialized_[j]) {
        coordinate = q.dot(*materialized_[j]);
      } else {
        const double remaining = std::max(stick_(j), 0.0);
        if (remaining > eps_) {
          double fraction;
          if (future_dims <= 0) {
            fraction = 1.0;
          } else {
            fraction = vd_detail::beta_rv(
                0.5, 0.5 * double(future_dims), rng_);
          }
          const double sign = (rng_() & 1ULL) ? 1.0 : -1.0;
          coordinate = sign * std::sqrt(
              remaining * std::min(std::max(fraction, 0.0), 1.0));
          stick_(j) = std::max(
              remaining - coordinate * coordinate, 0.0);
        }
      }
      coordinates_(rank_, j) = coordinate;
    }
    ++rank_;
  }

  Vec result = direction_norm * known;
  if (appended_rank >= 0) {
    result.noalias() +=
        direction_norm * residual_norm
        * coordinates_.row(appended_rank).transpose();
  }
  return result;
}

std::shared_ptr<const Vec>
SharedSphericalControls::materialize_locked_(int j) {
  if (j < 0 || j >= controls_) throw std::out_of_range(
      "SharedSphericalControls: control index out of range");
  if (materialized_[j]) return materialized_[j];

  Vec column = Vec::Zero(n_);
  if (rank_ > 0)
    column.noalias() =
        master_basis_.leftCols(rank_) * coordinates_.topRows(rank_).col(j);

  const double remaining = std::max(stick_(j), 0.0);
  if (remaining > eps_ && rank_ < centered_dim_) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Vec completion(n_);
    bool found = false;
    for (int attempt = 0; attempt < 32 && !found; ++attempt) {
      for (int i = 0; i < n_; ++i) completion(i) = normal(rng_);
      completion.array() -= completion.mean();
      if (rank_ > 0) {
        completion.noalias() -= master_basis_.leftCols(rank_)
            * (master_basis_.leftCols(rank_).transpose() * completion);
        completion.noalias() -= master_basis_.leftCols(rank_)
            * (master_basis_.leftCols(rank_).transpose() * completion);
      }
      const double norm = completion.norm();
      if (norm > eps_ * std::sqrt(double(n_))) {
        completion /= norm;
        found = true;
      }
    }
    if (!found) throw std::runtime_error(
        "SharedSphericalControls: could not complete a control");
    column.noalias() += std::sqrt(remaining) * completion;
  }

  // The master basis and its orthogonal completion both lie in the centered
  // subspace, and the stick construction already gives unit norm.  Do not
  // renormalize here: previously revealed coordinates must remain bitwise
  // consistent for every stream that has already queried this control.
  const double norm = column.norm();
  if (!(norm > eps_)) throw std::runtime_error(
      "SharedSphericalControls: degenerate materialized control");
  auto stored = std::make_shared<Vec>(std::move(column));
  materialized_[j] = stored;
  stick_(j) = 0.0;
  ++materialized_count_;
  return stored;
}

std::shared_ptr<const Vec> SharedSphericalControls::materialize(int j) {
  std::lock_guard<std::mutex> lock(mu_);
  return materialize_locked_(j);
}

std::shared_ptr<const Vec> SharedSphericalControls::real_cross(int j) {
  std::shared_ptr<const Vec> column;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (j < 0 || j >= controls_) throw std::out_of_range(
        "SharedSphericalControls: control index out of range");
    if (real_cross_[j]) return real_cross_[j];
    column = materialize_locked_(j);
  }

  auto product = std::make_shared<Vec>(p_);
  MapMatC X(Xptr_, n_, p_);
  vd_detail::gemv_Xt(X, *column, *product);

  std::lock_guard<std::mutex> lock(mu_);
  if (!real_cross_[j]) real_cross_[j] = product;
  return real_cross_[j];
}

double SharedSphericalControls::response_inner(int j) {
  std::shared_ptr<const Vec> column;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (j < 0 || j >= controls_) throw std::out_of_range(
        "SharedSphericalControls: control index out of range");
    if (response_ready_[j]) return response_inner_[j];
    column = materialize_locked_(j);
  }
  MapVec y(yptr_, n_);
  const double value = column->dot(y);
  std::lock_guard<std::mutex> lock(mu_);
  if (!response_ready_[j]) {
    response_inner_[j] = value;
    response_ready_[j] = 1;
  }
  return response_inner_[j];
}
