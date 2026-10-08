// Shared, lazy real-design products for the K T-Rex experiments.
//
// X^T y is computed once.  A full Gram column X^T X_j is computed on first
// request and may then be reused by every random experiment.  Weak references
// let simultaneous users share a column even after the strong-retention budget
// is exhausted.
#pragma once

#include "vd_common.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

class SharedRealCache {
public:
  SharedRealCache(const Eigen::Ref<const MatC>& X,
                  const Eigen::Ref<const Vec>& y,
                  int mmap_fd, int mmap_block_cols,
                  std::size_t max_bytes);

  const Vec& Xty() const noexcept { return Xty_; }
  std::shared_ptr<const Vec> gram_column(int j);

  long long hits() const noexcept { return hits_.load(); }
  long long misses() const noexcept { return misses_.load(); }
  long long waits() const noexcept { return waits_.load(); }
  int retained_columns() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return retained_count_;
  }

private:
  const double* Xptr_ = nullptr;
  int n_ = 0, p_ = 0;
  int mmap_fd_ = -1, mmap_block_cols_ = 0;
  Vec Xty_;

  // 0 = idle, 1 = one thread is computing this column.
  std::vector<unsigned char> computing_;
  std::vector<std::weak_ptr<const Vec>> columns_;
  std::vector<std::shared_ptr<const Vec>> retained_;
  int max_retained_columns_ = 0;
  int retained_count_ = 0;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<long long> hits_{0}, misses_{0}, waits_{0};
};
