// trex.hpp — T-Rex selector with solver dispatch and calibration modes.
//
// Solver dispatch: SolverType selects LARS / OMP / AFS via VD_Base.
//
// Calibration modes:
//   FixedTL:       user-specified T and L, calibrate v only.
//   CalibrateT:    fix L = L_factor * p, search (T, v) grid with early stopping.
//   CalibrateL:    fix T, scan L = p, 2p, ... until FDP_hat(T=1, v=0.75) < α.
//   CalibrateBoth: calibrate L first, then search (T, v).
//
// Parallelism: OpenMP (falls back to sequential when not available).
#pragma once
#include "vd_base.hpp"
#include "vd_lars.hpp"
#include "vd_omp.hpp"
#include "vd_afs.hpp"
#include "shared_real_cache.hpp"
#include "shared_spherical_controls.hpp"
#include <memory>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---------- Enums ----------
enum class SolverType : uint8_t { LARS = 0, OMP = 1, AFS = 2, NNLS = 3 };
enum class CalibMode  : uint8_t { FixedTL = 0, CalibrateT = 1, CalibrateL = 2, CalibrateBoth = 3 };

// ---------- Options ----------
struct TRexOptions {
  double tFDR         = 0.2;
  int    K            = 20;        // number of random experiments
  int    L_factor     = 10;        // L = L_factor * p  (when L is fixed)
  int    L             = -1;       // >0 overrides L_factor * p for fixed-L modes
  int    T_stop       = -1;        // -1 = auto (min(L, n/2)); >0 = explicit ceiling/fixed T
  int    consecutive_dummy_stop = 0; // NNOMP dummy-run stop; 0 = cumulative T
  int    max_L_factor = 50;        // ceiling for L calibration

  int    stride_width  = 1;        // steps between early-stop checks
  bool   posthoc_mode  = false;    // posthoc (no early stop) vs strided early-stop
  int    max_stale_strides = 3;    // stop if no new reals enter for this many strides
  int    max_vd_proj   = 100;
  double eps           = 1e-12;
  bool   verbose       = true;
  unsigned long long seed = 0ULL;

  SolverType solver    = SolverType::LARS;
  CalibMode  calib     = CalibMode::CalibrateBoth;
  VDDummyLaw dummy_law = VDDummyLaw::Spherical;
  double     rho       = 1.0;      // AFS shrinkage
  double     trim_q    = 1.0;      // quantile trim for AdaptiveGaussian
  int        ag_tail_m = 0;        // 0 = pooled AG; >0 = truncated top-m scale fit
  double     deflation_v = 0.5;    // Phi_prime deflation threshold.  The theory only
                                   // needs the null count above v to be dominated by
                                   // an NHG; that holds a fortiori for v > 0.5, so
                                   // this may be raised (more conservative, and it
                                   // lets the dummy-law calibration use the SAME v).
  double     sg_beta   = 4.0;      // SubGaussian shape (2 = Gaussian, >2 lighter tails)
  bool       sg_adaptive = false;  // fit beta from each conditional residual row
  double     sg_fit_q_lo = 0.00;   // retained |residual| interval for shape fitting
  double     sg_fit_q_hi = 0.80;
  double     sg_beta_min = 1.0;
  double     sg_beta_max = 6.0;
  double     sg_beta_smooth = 0.5;
  double     sg_tau    = 0.0;      // >0: beta_k -> 2 as signals leave the fit pool
  double     vd_seed_corr = -1.0;  // >=0: random seed direction, y goes second
  double     sigma_reg = 1e-8;
  std::vector<int> groups;         // per-predictor group labels for AdaptiveGaussian

  int mmap_fd         = -1;
  int mmap_block_cols = 0;
  int n_threads       = 0;         // 0 = auto
  bool use_shared_gram_cache = true;  // false retains the original implementation
  double gram_cache_mb = 512.0;       // strong-retention budget; 0 = unlimited
  // Exact shared virtual persistent controls.  Supported by OMP.  The K
  // experiments reuse one master-basis representation while their stopping
  // dummy pools remain independent.
  int persistent_spherical_controls = 0;
  // Additional path-local calibration dummies.  They do not increment T.
  // Their geometry is independent across random experiments.  With a
  // positive sigma, each label receives one shared log-normal priority mark;
  // sigma zero gives exchangeable unit marks for event-mass calibration.
  int fresh_calibration_dummies = 0;
  double fresh_calibration_priority_sigma = 0.0;
  double dummy_priority_radius = 1.0;
  bool adaptive_dummy_radius = false;
  int dummy_radius_tail_m = 0;
  double dummy_radius_tail_fraction = 0.95;
  double dummy_radius_target_win_prob = -1.0;
  bool dummy_radius_urn_target = false;
  bool dummy_radius_account_for_stopping = false;
  int dummy_radius_active_upper_bound = 0;
  double dummy_radius_real_score_quantile = 1.0;
  int dummy_radius_real_score_rank = 0;
  double dummy_radius_envelope_theta = 1.0;
  double dummy_radius_noise_score_bound = 0.0;
  DummyRadiusCovarianceModel dummy_radius_covariance_model =
      DummyRadiusCovarianceModel::None;
  double dummy_radius_covariance_rho = 0.0;
  bool persistent_score_ar1 = false;
  bool dummy_radius_scale_stopping = false;
  double dummy_radius_smooth = 0.80;
  double dummy_radius_min = 1.0;
  double dummy_radius_max = 5.0;
};

// ---------- Result ----------
struct TRexResult {
  Eigen::VectorXi selected_var;
  double          v_thresh      = 0.0;
  int             T_stop        = 0;
  int             num_dummies   = 0;
  int             L_calibrated  = 0;
  Eigen::VectorXd V;
  Eigen::MatrixXd FDP_hat_mat;    // (T_stop × |V|)
  Eigen::MatrixXd Phi_mat;        // (T_stop × p)
  Eigen::VectorXd Phi_prime;
  // Mean final real-feature coefficient across the K experiments.  Features
  // absent from an experiment contribute zero, so this combines occurrence
  // frequency and fitted strength without thresholding.
  Eigen::VectorXd coefficient_mean;
  int             K = 0;
  long long       gram_cache_hits = 0;
  long long       gram_cache_misses = 0;
  long long       gram_cache_waits = 0;
  int             gram_cache_columns = 0;
  int             persistent_master_rank = 0;
  int             persistent_materialized = 0;
  double          dummy_radius_mean = 1.0;
  double          dummy_radius_min = 1.0;
  double          dummy_radius_max = 1.0;
};

// ---------- Selector ----------
class TRexSelector {
public:
  explicit TRexSelector(const TRexOptions& opts);

  TRexResult run(const Eigen::Ref<const MatC>& X,
                 const Eigen::Ref<const Vec>&  y);

private:
  TRexOptions opt_;
  std::shared_ptr<SharedRealCache> shared_real_cache_;
  std::shared_ptr<SharedSphericalControls> shared_spherical_controls_;
  std::shared_ptr<const Vec> fresh_calibration_priorities_;

  // ---- Solver factory ----
  std::unique_ptr<VD_Base> make_solver_(
      const Eigen::Ref<const MatC>& X,
      const Eigen::Ref<const Vec>& y,
      int num_dummies, unsigned long long seed, int T_stop) const;

  VDOptions make_vd_opts_(unsigned long long seed, int T_stop, int n) const;

  // ---- Execution paths ----
  TRexResult run_fixed_T_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int T_fixed, int num_dummies, int n_threads, const Vec& V);

  TRexResult run_posthoc_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int Tstop, int num_dummies, int n_threads, const Vec& V);

  TRexResult run_early_stop_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int Tstop, int num_dummies, int n_threads, const Vec& V);

  // ---- L calibration ----
  int calibrate_L_(
      const Eigen::Ref<const MatC>& X, const Eigen::Ref<const Vec>& y,
      int p, int n_threads);

  // ---- Statistical helpers ----
  Vec make_V_(int K, double eps) const;

  Vec Phi_prime_fun_(int p, int T_stop, int num_dummies,
                     const MatC& phi_T_mat, const Vec& Phi) const;

  Vec fdp_hat_(const Vec& V, const Vec& Phi, const Vec& Phi_prime) const;

  struct SelectResult {
    Eigen::VectorXi selected_var;
    double v_thresh;
  };

  SelectResult select_var_(int p, double tFDR, int T_stop,
                           const Eigen::MatrixXd& FDP_hat_mat,
                           const Eigen::MatrixXd& Phi_mat,
                           const Vec& V) const;

  // ---- Seeds ----
  std::vector<unsigned long long> make_seeds_(int K) const;
};
