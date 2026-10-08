// vd_omp.hpp — Virtual Dummy OMP (greedy argmax, full OLS refit)
#pragma once
#include "vd_base.hpp"

class VD_OMP : public VD_Base {
public:
  using VD_Base::VD_Base;  // inherit constructors

  MatC run(int T = 1) override;

private:
  Vec Xty_active_;

  struct Candidate {
    enum class Pool : uint8_t { Real, VD, Realized, Persistent };
    Pool pool; int index; double abs_corr;
  };
  std::optional<Candidate> find_best_candidate_();
  void append_to_factor_(const Vec& x_col);
  void ols_refit_();
  void nnls_refit_();  // nonnegative least squares on the active set

  void init_omp_();  // called once at start of run()
  bool omp_inited_ = false;
  int consecutive_stopping_entries_ = 0;
};
