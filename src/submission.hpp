#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.


// One vector, element (i, j) at i * cols + j
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

  double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  double  operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();
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
    for (std::size_t j = 1; j + 1 < cols; ++j) {
      new_grid(i, j) = 0.5 * old_grid(i, j) + 0.125 * (old_grid(i - 1, j) + old_grid(i + 1, j) + old_grid(i, j - 1) + old_grid(i, j + 1));
    }
  }
}
