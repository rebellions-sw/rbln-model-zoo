#ifndef RBLN_TENSOR_H
#define RBLN_TENSOR_H

#include <memory>
#include <string>
#include <vector>

template <typename T> class Tensor {
public:
  Tensor() : depth_(1), rows_(0), cols_(0) { array_.reserve(0); }
  Tensor(T val) : depth_(1), rows_(0), cols_(1) { array_.resize(1, T{val}); }
  Tensor(size_t row, size_t col) : depth_(1), rows_(row), cols_(col) {
    array_.resize(GetCapacity(), T{});
  }

  Tensor(const void *data, size_t row, size_t col)
      : depth_(1), rows_(row), cols_(col) {
    const T *ptr = static_cast<const T *>(data);
    array_.assign(ptr, ptr + GetCapacity());
  }

  Tensor(size_t depth, size_t row, size_t col)
      : depth_(depth), rows_(row), cols_(col) {
    array_.resize(GetCapacity(), T{});
  }

  ~Tensor() = default;

  Tensor(const Tensor &other) {
    array_ = other.array_;
    depth_ = other.depth_;
    rows_ = other.rows_;
    cols_ = other.cols_;
  }

  Tensor(Tensor &&other) {
    array_ = std::move(other.array_);
    depth_ = other.depth_;
    rows_ = other.rows_;
    cols_ = other.cols_;
  }

  T &operator[](size_t i) { return array_[i]; }
  T operator[](size_t i) const { return array_[i]; }

  Tensor operator=(const Tensor &other) {
    if (this != &other) {
      array_ = other.array_;
      depth_ = other.depth_;
      rows_ = other.rows_;
      cols_ = other.cols_;
    }
    return *this;
  }

  T &operator()(size_t r_idx, size_t c_idx) {
    if (r_idx >= rows_ || c_idx >= cols_) {
      throw std::out_of_range("Index out of bounds");
    }
    return array_[cols_ * r_idx + c_idx];
  }

  T &operator()(size_t col) {
    if (col >= cols_) {
      throw std::out_of_range("Index out of bounds");
    }
    return array_[col];
  }

  T operator()(size_t r_idx, size_t c_idx) const {
    if (r_idx >= rows_ || c_idx >= cols_) {
      throw std::out_of_range("Index out of bounds");
    }
    return array_[cols_ * r_idx + c_idx];
  }

  Tensor operator+(T val) {
    Tensor ret(rows_, cols_);
    for (auto r = 0; r < rows_; r++) {
      for (auto c = 0; c < cols_; c++) {
        ret(r, c) = array_[r * cols_ + c] + val;
      }
    }
    return ret;
  }

  void *GetData() { return array_.data(); }
  size_t GetRows() const { return rows_; }
  size_t GetCols() const { return cols_; }
  size_t GetDepth() const { return depth_; }
  size_t GetSize() const { return array_.size(); }
  void Ones() { std::fill(array_.begin(), array_.end(), T{1}); }
  void Zeros() { std::fill(array_.begin(), array_.end(), T{0}); }

  size_t GetCapacity(size_t r, size_t c) const {
    return std::max(1UL, r) * std::max(1UL, c);
  }

  size_t GetCapacity() const {
    return GetCapacity(rows_, cols_) * std::max(1UL, depth_);
  }

  void Resize(size_t row, size_t col) {
    rows_ = row;
    cols_ = col;
    array_.resize(GetCapacity(row, col));
  }

private:
  std::vector<T> array_;
  size_t depth_;
  size_t rows_;
  size_t cols_;
};
#endif
