// vd_base.hpp
// Base class for VD-* forward selectors.
// Owns all shared state: data views, VD pool, basis, Cholesky,
// model state, scratch buffers, and public accessors.
// Solvers (LARS, OMP, AFS) inherit and implement run().
#pragma once
#include "vd_common.hpp"
#include "shared_spherical_controls.hpp"

class VD_Base {
public:
  // ---------- Constructors ----------
  VD_Base(const double* Xptr, int n, int p,
          const double* yptr, int ny,
          int num_dummies, const VDOptions& o);

  VD_Base(const Eigen::Ref<const MatC>& X_in,
          const Eigen::Ref<const Vec>&  y_in,
          int num_dummies, const VDOptions& o);

  virtual ~VD_Base() = default;

  // Pure virtual: each solver implements its own run loop.
  virtual MatC run(int T = 1) = 0;

  // ---------- Public accessors ----------
  int num_dummies()          const noexcept { return stopping_dummies_; }
  int num_realized_dummies() const noexcept {
    return T_stopping_realized_;
  }
  int num_realized_calibration_dummies() const noexcept {
    return T_calibration_realized_;
  }
  int n_features()           const noexcept { return p_; }
  int n_persistent_controls() const noexcept {
    return persistent_controls_;
  }
  int n_samples()            const noexcept { return n_; }
  int basis_size()           const noexcept { return basis_size_(); }
  Eigen::VectorXd sg_beta_path_copy() const {
    Eigen::VectorXd out(static_cast<int>(sg_beta_path_.size()));
    for (int k = 0; k < out.size(); ++k) out(k) = sg_beta_path_[k];
    return out;
  }
  Eigen::VectorXd dummy_radius_path_copy() const {
    Eigen::VectorXd out(static_cast<int>(dummy_radius_path_.size()));
    for (int k = 0; k < out.size(); ++k)
      out(k) = dummy_radius_path_[k];
    return out;
  }

  Eigen::VectorXd beta_view_copy()   const { return beta_; }
  Eigen::VectorXd corr_view_copy()   const { return corr_; }
  Eigen::VectorXi active_indices()   const;

  Eigen::VectorXd corr_realized_view_copy() const {
    return corr_realized_.head(T_realized_);
  }
  Eigen::VectorXi is_dummy_realized_view() const {
    Eigen::VectorXi out(L_);
    for (int d = 0; d < L_; ++d) out(d) = vd_is_realized_[d] ? 1 : 0;
    return out;
  }

  std::vector<ActiveFeature> active_features_copy() const { return active_features_; }

  Eigen::VectorXd beta_real() const {
    Eigen::VectorXd out(p_);
    if (beta_.size() >= p_) out = beta_.head(p_);
    else                    out.setZero();
    return out;
  }
  
  Eigen::MatrixXd X_realized_copy() const {
    return X_realized_.leftCols(T_realized_);
  }

  Eigen::VectorXd realized_dummy_column_copy(int j) const {
    if (j < 0 || j >= T_realized_) {
      throw std::out_of_range("realized_dummy_column_copy: index out of range");
    }
    return X_realized_.col(j);
}
  const Vec&  normx_view()    const noexcept { return normx_; }
  const MatR& vd_proj_view()  const noexcept { return vd_proj_; }
  const Vec&  vd_corr_view()  const noexcept { return vd_corr_; }
  const Vec&  vd_stick_view() const noexcept { return vd_stick_; }

protected:
  // ---------- Problem sizes & options ----------
  int n_ = 0, p_ = 0, L_ = 0, persistent_controls_ = 0;
  int calibration_dummies_ = 0, stopping_dummies_ = 0;
  VDOptions opt_;

  // ---------- Data (mapped views) ----------
  MapMatC X_;
  MapVec  y_;

  // ---------- RNG ----------
  std::mt19937_64 rng_{0};

  // ---------- Global scalars ----------
  bool   standardized_ = false;
  double y_norm_ = 0.0;

  // ---------- Model state ----------
  Vec normx_;
  Vec beta_, beta_dummy_, beta_persistent_, mu_, residuals_, corr_;

  // ---------- Active set ----------
  std::vector<int> actives_;                    // real-feature indices
  std::vector<ActiveFeature> active_features_;  // all active (real + dummy)
  std::vector<char> is_active_;                 // O(1) lookup for reals
  std::vector<char> persistent_is_active_;

  // ---------- Basis ----------
  MatC basis_;
  std::vector<int> basis_indices_;
  inline int basis_size_() const noexcept {
    return static_cast<int>(basis_indices_.size());
  }

  // ---------- Reusable buffers ----------
  mutable Vec proj_coeffs_;
  mutable Vec ortho_buffer_;

  // ---------- Active columns & Cholesky ----------
  MatC X_active_;
  MatC chol_factor_;

  // ---------- Realized dummies ----------
  MatC X_realized_;
  Vec  corr_realized_;
  int  T_realized_ = 0;
  int  T_stopping_realized_ = 0;
  int  T_calibration_realized_ = 0;
  // X^T times each realized dummy, needed by the cached real-product engine.
  std::vector<Vec> Xt_realized_;

  // ---------- Virtual-dummy pool ----------
  MatR vd_proj_;
  Vec  vd_stick_, vd_corr_;
  std::vector<char> vd_is_realized_;
  std::vector<int>  vd_unrealized_idx_;
  int vd_rows_filled_ = 0, vd_rows_cap_ = 0;

  // Shared persistent spherical controls.  Correlations are requested against
  // the current residual through the common master basis, so no M-by-depth
  // coordinate table is duplicated in each stream.
  Vec persistent_corr_;
  double current_dummy_radius_ = 1.0;
  double conditional_dummy_radius_ = 1.0;
  bool conditional_dummy_radius_available_ = false;
  std::vector<double> dummy_radius_path_;

  // ---------- Real-variable projections (for AdaptiveGaussian) ----------
  MatC Z_proj_;   // (p, max_vd_proj): Z_proj_(j, k) = <X_j, e_k>

  // Per-stream strong references keep selected real Gram columns alive even
  // when the shared cache's retention budget has been reached.
  std::unordered_map<int, std::shared_ptr<const Vec>> real_cross_local_;

  // ---------- pread scratch ----------
  std::vector<double> gemv_scratch_;

  // ---------- Bookkeeping ----------
  int step_ = 0;

  // ---------- Shared setup ----------
  void init_common_state_();    // alloc + init everything above
  void initialize_basis_();
  void ensure_vd_rows_capacity_(int need);

  // ---------- Shared VD machinery ----------
  std::optional<Vec> orthonormalize_(const Vec& v) const;
  Vec  project_to_Vperp_(const Vec& z) const;
  void initialize_virtual_dummies_();
  void update_virtual_dummies_(const Vec* source = nullptr,
                               const Vec* Xt_source = nullptr);
  void prepare_stopping_conditional_radius_();
  void initialize_persistent_controls_();
  void update_persistent_controls_();
  void refresh_persistent_correlations_();
  void apply_persistent_score_ar1_();
  std::shared_ptr<const Vec> persistent_column_(int j);
  std::shared_ptr<const Vec> persistent_real_cross_(int j);
  double choose_dummy_radius_();
  void adaptive_gaussian_update_(int k);
  void adaptive_beta_update_(int k);
  // Group-aware AdaptiveGaussian helpers.
  void init_groups_();
  // Fit the law on ``keep_idx`` and draw the dummies listed in ``dummies``.
  void ag_fit_and_sample_(int k, const std::vector<int>& keep_idx,
                          const std::vector<int>& dummies, int group_index);
  double fit_subbotin_beta_truncated_(
      const std::vector<double>& residuals, double fallback) const;
  double fit_gaussian_scale_top_tail_(
      const std::vector<double>& residuals, double fallback,
      int tail_m = -1) const;
  std::vector<std::vector<int>> group_members_;   // real predictors per group
  std::vector<std::vector<int>> group_dummies_;   // dummy ids per group
  std::vector<double> sg_beta_state_by_group_;     // smoothed conditional shape
  std::vector<double> sg_beta_path_;               // group-0 shape by basis row
  // AdaptiveBeta: a dummy's "proxy" identity is a property of the COLUMN, not of
  // the step.  A real null that is aligned with the active subspace is heavy at
  // MANY steps (its projections are correlated across k, being projections of one
  // fixed vector).  Resampling the mixture component every step would give each
  // dummy T independent chances to be heavy, inflating its selection probability
  // above a null's and breaking the urn model's exchangeability.  So the label is
  // drawn ONCE and persists.
  std::vector<char> vd_heavy_;                    // per-dummy proxy label
  bool vd_heavy_assigned_ = false;
  void realize_dummy_(int vd_idx);
  bool shared_products_() const noexcept {
    return static_cast<bool>(opt_.shared_real_cache);
  }
  std::shared_ptr<const Vec> real_cross_(int j);

  // ---------- Shared Cholesky ----------
  static bool chol_append(MatC& R,
                          const Eigen::Ref<const Vec>& v,
                          double s, double eps);

  // ---------- GEMV convenience ----------
  double* scratch_ptr_() {
    return gemv_scratch_.empty() ? nullptr : gemv_scratch_.data();
  }
  void full_corr_refresh_();    // corr_ = X^T r, vd_corr, corr_realized

private:
  VD_Base(const VD_Base&) = delete;
  VD_Base& operator=(const VD_Base&) = delete;
};
