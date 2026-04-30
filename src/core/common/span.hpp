// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — common::Span<T>
//
// Minimal C++17-compatible stand-in for std::span (which is C++20). Just enough
// surface for the algorithm code in core/its/ to take "pointer + length" args
// without committing to C++20.

#pragma once

#include <cstddef>

namespace pylidar::common {

template <class T>
struct Span {
    T*          data = nullptr;
    std::size_t size = 0;

    constexpr T*       begin()       noexcept { return data; }
    constexpr T*       end()         noexcept { return data + size; }
    constexpr const T* begin() const noexcept { return data; }
    constexpr const T* end()   const noexcept { return data + size; }

    constexpr T&       operator[](std::size_t i)       noexcept { return data[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return data[i]; }

    constexpr bool empty() const noexcept { return size == 0; }
};

}  // namespace pylidar::common
