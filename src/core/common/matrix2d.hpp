// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — common::Matrix2D<T>
//
// Owning 2D buffer in COLUMN-MAJOR layout (matches R/lidR's NumericMatrix and
// makes line-by-line translation of lidR C++ algorithms direct). Bindings layer
// transposes to/from row-major numpy arrays.
//
// Storage formula: element (r, c) lives at index `c * rows_ + r`.

#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pylidar::common {

template <class T>
class Matrix2D {
public:
    Matrix2D() noexcept = default;

    Matrix2D(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols),
          buf_(rows * cols == 0 ? nullptr : new T[rows * cols]()) {}

    Matrix2D(const Matrix2D&)            = delete;
    Matrix2D& operator=(const Matrix2D&) = delete;

    Matrix2D(Matrix2D&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), buf_(std::move(other.buf_)) {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix2D& operator=(Matrix2D&& other) noexcept {
        if (this != &other) {
            rows_ = other.rows_;
            cols_ = other.cols_;
            buf_  = std::move(other.buf_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_ * cols_; }
    bool        empty() const noexcept { return size() == 0; }

    T& at(std::size_t r, std::size_t c) noexcept {
        assert(r < rows_ && c < cols_);
        return buf_[c * rows_ + r];
    }
    const T& at(std::size_t r, std::size_t c) const noexcept {
        assert(r < rows_ && c < cols_);
        return buf_[c * rows_ + r];
    }

    // Bounds-checking access — throws std::out_of_range on OOB. Algorithm
    // code should use at() (assert-only) on the hot path.
    T& checked(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix2D index out of range");
        }
        return buf_[c * rows_ + r];
    }
    const T& checked(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix2D index out of range");
        }
        return buf_[c * rows_ + r];
    }

    T*       data()       noexcept { return buf_.get(); }
    const T* data() const noexcept { return buf_.get(); }

private:
    std::size_t          rows_ = 0;
    std::size_t          cols_ = 0;
    std::unique_ptr<T[]> buf_;
};

}  // namespace pylidar::common
