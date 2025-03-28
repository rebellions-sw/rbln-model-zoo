#ifndef RBLN_LLAMA_OPS_H
#define RBLN_LLAMA_OPS_H

// Tensor operations implementation
namespace tensor_ops {

template <typename T>
Tensor<T> Reshape(const Tensor<T> &tensor, int row, int col) {
  if (tensor.GetCapacity(row, col) != tensor.GetCapacity()) {
    throw std::runtime_error("Cannot reshape: total size must remain the same");
  }

  Tensor<T> ret(tensor);
  std::vector<T> temp(tensor.GetSize());
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      size_t new_idx = i * col + j;
      size_t old_idx = (new_idx / col) * tensor.GetCols() + (new_idx % col);
      temp[new_idx] = tensor[old_idx];
    }
  }

  for (size_t i = 0; i < temp.size(); ++i) {
    ret[i] = temp[i];
  }
  ret.Resize(row, col);
  return ret;
}

template <typename T> Tensor<T> Reshape(const Tensor<T> &tensor, int col) {
  if (col != tensor.GetCapacity()) {
    throw std::runtime_error("Cannot reshape: total size must remain the same");
  }

  Tensor<T> ret(tensor);
  ret.Resize(0, col);
  return ret;
}

template <typename T> void Arange(Tensor<T> &tensor, int start, int stop) {
  tensor.Resize(0, stop - start);
  for (size_t i = 0; i < stop - start; ++i) {
    tensor[i] = static_cast<T>(start + i);
  }
}

template <typename T> Tensor<T> UnSqueeze(const Tensor<T> &tensor) {
  Tensor<T> ret(1, tensor.GetSize());
  for (size_t i = 0; i < tensor.GetSize(); ++i) {
    ret(0, i) = tensor[i];
  }
  return ret;
}

template <typename T> Tensor<T> SelectLastColumn(const Tensor<T> &tensor) {
  Tensor<T> result(tensor.GetRows(), 1);
  size_t last_col = tensor.GetCols() - 1;

  for (size_t i = 0; i < tensor.GetRows(); ++i) {
    result(i, 0) = tensor(i, last_col);
  }
  return result;
}

template <typename T>
Tensor<T> Pad(const Tensor<T> &tensor, size_t start_pos, size_t end_pos) {
  Tensor<T> padded(tensor.GetRows(), tensor.GetCols() + start_pos + end_pos);
  for (size_t i = 0; i < tensor.GetRows(); ++i) {
    for (size_t j = 0; j < tensor.GetCols(); ++j) {
      padded(i, start_pos + j) = tensor(i, j);
    }
  }
  return padded;
}

template <typename T>
Tensor<T> VerticalSlicing(Tensor<T> &tensor, size_t start_pos, size_t end_pos) {
  Tensor<T> ret(tensor);

  std::vector<T> temp(ret.GetCapacity(ret.GetRows(), (end_pos - start_pos)));
  for (size_t i = 0; i < ret.GetRows(); ++i) {
    for (size_t j = start_pos; j < end_pos; ++j) {
      temp[i * (end_pos - start_pos) + (j - start_pos)] = ret(i, j);
    }
  }
  for (size_t i = 0; i < temp.size(); ++i) {
    ret[i] = temp[i];
  }
  return ret;
}

template <typename T>
void SetCausalMask(Tensor<T> &tensor, const Tensor<T> &mask_tensor,
                   size_t start_pos, size_t end_pos) {
  if (end_pos > tensor.GetCols()) {
    throw std::out_of_range("Index range out of bounds");
  }
  for (size_t d = 0; d < tensor.GetDepth(); ++d) {
    for (size_t r = 0; r < tensor.GetRows(); ++r) {
      size_t base_idx = (d * tensor.GetRows() + r) * tensor.GetCols();
      for (size_t idx = start_pos; idx < end_pos; ++idx) {
        tensor[base_idx + idx] = mask_tensor(r, idx - start_pos);
      }
    }
  }
}

template <typename T>
Tensor<T> ConcatenateWithRange(const Tensor<T> &tensor, size_t start_pos,
                               size_t end_pos) {
  Tensor<T> range;
  tensor_ops::Arange(range, start_pos, end_pos);

  size_t total_cols = tensor.GetCols() + range.GetSize();
  Tensor<T> result(1, total_cols);
  for (size_t i = 0; i < tensor.GetCols(); ++i) {
    result(0, i) = tensor[i];
  }
  for (size_t i = 0; i < range.GetSize(); ++i) {
    result(0, tensor.GetCols() + i) = range[i];
  }
  return result;
}

template <typename T, typename T1>
Tensor<T1> GetArgmax(const Tensor<T> &tensor) {
  Tensor<T1> next_tokens(tensor.GetRows(), 1);
  for (size_t i = 0; i < tensor.GetRows(); ++i) {
    size_t max_idx = 0;
    T max_val = tensor(i, 0);

    for (size_t j = 1; j < tensor.GetCols(); ++j) {
      if (tensor(i, j) > max_val) {
        max_val = tensor(i, j);
        max_idx = j;
      }
    }
    next_tokens(i, 0) = static_cast<T1>(max_idx);
  }
  return next_tokens;
}

template <typename T>
Tensor<T> Concatenate(const Tensor<T> &tensor, const Tensor<T> &other) {
  Tensor<T> result(tensor.GetRows(), tensor.GetCols() + 1);
  for (size_t i = 0; i < tensor.GetRows(); ++i) {
    for (size_t j = 0; j < tensor.GetCols(); ++j) {
      result(i, j) = tensor(i, j);
    }
    result(i, tensor.GetCols()) = other(i, 0);
  }
  return result;
}

template <typename T>
void SetMaskAtPos(Tensor<T> &tensor, size_t pos, T value) {
  if (pos >= tensor.GetCols()) {
    throw std::out_of_range("Index out of bounds");
  }

  for (size_t i = 0; i < tensor.GetRows(); ++i) {
    tensor(i, pos) = value;
  }
}

template <typename T>
void SetMaskUpToPos(Tensor<T> &tensor, size_t batch_idx, size_t pos, T value) {
  if (pos > tensor.GetCols()) {
    throw std::out_of_range("Index out of bounds");
  }

  for (size_t r = 0; r < tensor.GetRows(); ++r) {
    for (size_t i = 0; i < pos; ++i) {
      tensor(r, i) = value;
    }
  }
}

template <typename T> Tensor<T> TriuMask(size_t row, size_t col) {
  Tensor<T> mask(row, col);
  mask.Ones();

  for (size_t i = 0; i < row; ++i) {
    for (size_t j = i + 1; j < col; ++j) {
      mask(i, j) = 0;
    }
  }
  return mask;
}

template <typename T>
Tensor<T> FilterByMask(const Tensor<T> &tensor, const Tensor<T> &mask,
                       size_t i) {
  size_t count = 0;
  for (size_t j = 0; j < mask.GetCols(); ++j) {
    if (mask(i, j) == 1) {
      count++;
    }
  }

  Tensor<T> result(1, count);
  size_t idx = 0;
  for (size_t j = 0; j < mask.GetCols(); ++j) {
    if (mask(i, j) == 1) {
      result[idx++] = tensor[j];
    }
  }
  return result;
}

} // namespace tensor_ops

#endif