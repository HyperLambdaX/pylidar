#pragma once

#include <cstddef>
#include <stdexcept>

namespace pylidar::core {

// Row-major 2D matrix view. Owns no memory by default — wraps a caller-supplied
// buffer (numpy array data pointer). 0-based indexing throughout.
template <typename T>
class Matrix2D {
public:
    Matrix2D(T* data, std::size_t rows, std::size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

private:
    T* data_;
    std::size_t rows_;
    std::size_t cols_;
};

}  // namespace pylidar::core
