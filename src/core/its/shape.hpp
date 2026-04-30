// SPDX-License-Identifier: GPL-3.0-or-later
// pylidar — its::Shape
//
// Neighbourhood shape shared across ITS algorithms (smooth_height, lmf, ...).
// Integer values intentionally match lidR's int convention (1 = Square /
// rectangle, 2 = Circular / disc) so direct ports of LAS.cpp can keep the
// numeric branches without a translation layer.

#pragma once

namespace pylidar::its {

enum class Shape : int {
    Square   = 1,
    Circular = 2,
};

}  // namespace pylidar::its
