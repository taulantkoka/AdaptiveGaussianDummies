#include "shared_real_cache.hpp"
#include <stdexcept>

SharedRealCache::SharedRealCache(
    const Eigen::Ref<const MatC>& X,
    const Eigen::Ref<const Vec>& y,
    int mmap_fd, int mmap_block_cols,
    std::size_t max_bytes)
  : Xptr_(X.data()), n_((int)X.rows()), p_((int)X.cols()),
    mmap_fd_(mmap_fd), mmap_block_cols_(mmap_block_cols),
    Xty_(p_), computing_(p_, 0), columns_(p_), retained_(p_)
{
  if (y.size() != n_) throw std::invalid_argument("SharedRealCache: X/y mismatch");

  MapMatC Xm(Xptr_, n_, p_);
  Vec yy = y;
  std::vector<double> scratch;
  if (mmap_fd_ >= 0 && mmap_block_cols_ > 0)
    scratch.resize(std::size_t(mmap_block_cols_) * n_);
  vd_detail::gemv_Xt(Xm, yy, Xty_, mmap_fd_, mmap_block_cols_,
                     scratch.empty() ? nullptr : scratch.data());

  const std::size_t bytes_per_column = std::size_t(std::max(p_, 1)) * sizeof(double);
  if (max_bytes == 0) {
    max_retained_columns_ = p_;
  } else {
    max_retained_columns_ = (int)std::min<std::size_t>(
        std::size_t(p_), std::max<std::size_t>(1, max_bytes / bytes_per_column));
  }
}

std::shared_ptr<const Vec> SharedRealCache::gram_column(int j) {
  if (j < 0 || j >= p_) throw std::out_of_range("SharedRealCache: column out of range");

  {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      if (auto column = columns_[j].lock()) {
        ++hits_;
        return column;
      }
      if (!computing_[j]) {
        computing_[j] = 1;
        break;
      }
      ++waits_;
      cv_.wait(lock, [&]{ return !computing_[j]; });
    }
  }

  auto column = std::make_shared<Vec>(p_);
  MapMatC Xm(Xptr_, n_, p_);
  Vec xj = Xm.col(j);
  std::vector<double> scratch;
  if (mmap_fd_ >= 0 && mmap_block_cols_ > 0)
    scratch.resize(std::size_t(mmap_block_cols_) * n_);
  vd_detail::gemv_Xt(Xm, xj, *column, mmap_fd_, mmap_block_cols_,
                     scratch.empty() ? nullptr : scratch.data());

  {
    std::lock_guard<std::mutex> lock(mu_);
    columns_[j] = column;
    if (!retained_[j] && retained_count_ < max_retained_columns_) {
      retained_[j] = column;
      ++retained_count_;
    }
    computing_[j] = 0;
    ++misses_;
  }
  cv_.notify_all();
  return column;
}
