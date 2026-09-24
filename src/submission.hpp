#pragma once

#include <cassert>
#include <cstddef>
#include <new>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

// Every row starts on a 64-byte boundary
inline constexpr std::size_t CacheLineBytes = 64;
inline constexpr std::size_t DoublesPerCacheLine = CacheLineBytes / sizeof(double);  // 8

// std::vector only guarantees alignof(double) == 8 bytes. 
// This allocator makes sure that every row of the grid starts on a cache line boundary
template <typename T>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>&) {}

  T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{CacheLineBytes}));
  }
  void deallocate(T* p, std::size_t) {
    ::operator delete(p, std::align_val_t{CacheLineBytes});
  }

  template <typename U>
  bool operator==(const AlignedAllocator<U>&) const { return true; }
  template <typename U>
  bool operator!=(const AlignedAllocator<U>&) const { return false; }
};

// One aligned buffer, element (i, j) at i * stride + j. stride is cols rounded
// up to a multiple of 8 doubles, so every row starts aligned.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double, AlignedAllocator<double>> data_;

  static std::size_t round_up(std::size_t n) {
    return (n + DoublesPerCacheLine - 1) / DoublesPerCacheLine * DoublesPerCacheLine;
  }

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), stride_(round_up(cols)), data_(rows * stride_, 0.0) {}

  double& operator()(std::size_t i, std::size_t j) { return data_[i * stride_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data_[i * stride_ + j]; }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::size_t stride() const { return stride_; }

  // Pointer to the start of row i
  double*       row(std::size_t i)       { return data_.data() + i * stride_; }
  const double* row(std::size_t i) const { return data_.data() + i * stride_; }
};

// Computes one interior row. __restrict tells the compiler that out does not
// overlap up, mid or down, so it can vectorize without runtime overlap checks.
inline void stencil_row(const double* __restrict up, const double* __restrict mid,
                        const double* __restrict down, double* __restrict out, std::size_t cols) {
  const std::size_t end = cols == 0 ? 0 : cols - 1;
  #ifdef _OPENMP
    #pragma omp simd
  #endif
  for (std::size_t j = 1; j < end; ++j) {
    out[j] = 0.5 * mid[j] + 0.125 * (up[j] + down[j] + mid[j - 1] + mid[j + 1]);
  }
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  // Assertions to check that the grids are compatible
  assert(new_grid.rows() == rows && new_grid.cols() == cols && "old and new grids must be the same size");
  assert(new_grid.stride() == old_grid.stride() && "old and new grids must share a row layout");
  assert(&old_grid != &new_grid && "old and new must be separate grids");

  if (rows == 0 || cols == 0) return;

  // Boundary ring is copied unchanged
  for (std::size_t j = 0; j < cols; ++j) {
    new_grid(0, j) = old_grid(0, j);
    new_grid(rows - 1, j) = old_grid(rows - 1, j);
  }
  for (std::size_t i = 0; i < rows; ++i) {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);
  }

  // Interior cells
  for (std::size_t i = 1; i + 1 < rows; ++i) {
    stencil_row(old_grid.row(i - 1), old_grid.row(i), old_grid.row(i + 1), new_grid.row(i), cols);
  }
}
