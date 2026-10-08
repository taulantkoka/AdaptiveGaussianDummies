// Exact, shared virtual representation of persistent spherical controls.
//
// The K randomized OMP streams generally have different local bases.  This
// object maintains one master orthonormal basis spanning every direction ever
// requested by any stream.  Each persistent control stores its coordinates in
// that master basis and a remaining spherical stick.  A newly requested
// direction either lies in the existing span or appends one master direction,
// in which case one exact conditional spherical coordinate is drawn for every
// control.  Thus all streams see projections of the same fixed columns.
#pragma once

#include "vd_common.hpp"
#include <atomic>
#include <memory>
#include <mutex>

class SharedSphericalControls {
public:
  SharedSphericalControls(
      const Eigen::Ref<const MatC>& X,
      const Eigen::Ref<const Vec>& y,
      int controls,
      unsigned long long seed,
      double eps);

  int size() const noexcept { return controls_; }
  int master_rank() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return rank_;
  }
  int materialized_count() const noexcept {
    return materialized_count_.load();
  }

  // Inner products between every persistent control and ``direction``.
  Vec projections(const Eigen::Ref<const Vec>& direction);

  // As above, but before revealing a genuinely new direction, compute the
  // common score multiplier whose conditional pool-maximum exceedance
  // probability is ``target_probability``.  ``excluded`` marks controls
  // already active in the calling stream.  The calculation uses each
  // control's revealed coordinates and remaining stick and is performed under
  // the same lock as the subsequent coordinate draw.
  Vec projections_with_conditional_radius(
      const Eigen::Ref<const Vec>& direction,
      const std::vector<char>& excluded,
      double real_threshold,
      double target_probability,
      double& required_radius);

  // Complete one persistent control in the centered observation-space
  // complement.  Repeated calls return the same column.
  std::shared_ptr<const Vec> materialize(int j);

  // Shared products used after a persistent control enters an OMP path.
  std::shared_ptr<const Vec> real_cross(int j);
  double response_inner(int j);

private:
  const double* Xptr_ = nullptr;
  const double* yptr_ = nullptr;
  int n_ = 0, p_ = 0, controls_ = 0, centered_dim_ = 0;
  double eps_ = 1e-12;

  mutable std::mutex mu_;
  std::mt19937_64 rng_;
  MatC master_basis_;
  MatR coordinates_;
  Vec stick_;
  int rank_ = 0;

  std::vector<std::shared_ptr<const Vec>> materialized_;
  std::vector<std::shared_ptr<const Vec>> real_cross_;
  std::vector<double> response_inner_;
  std::vector<unsigned char> response_ready_;
  std::atomic<int> materialized_count_{0};

  void ensure_capacity_(int need);
  Vec centered_unit_(const Eigen::Ref<const Vec>& direction,
                     double& original_norm) const;
  Vec projections_locked_(
      const Eigen::Ref<const Vec>& direction,
      const std::vector<char>* excluded,
      double real_threshold,
      double target_probability,
      double* required_radius);
  std::shared_ptr<const Vec> materialize_locked_(int j);
};
