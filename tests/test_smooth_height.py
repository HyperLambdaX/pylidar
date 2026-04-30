"""Phase 1 acceptance tests — pylidar.smooth_height.

Four cases per task_plan.md:
  1. Random point cloud → output Z variance shrinks (smoothing actually
     averages neighbours).
  2. Flat ground (constant z) → output equals input (no spurious wiggle).
  3. sigma=0 with method="mean" → no error (sigma is unused for mean); output
     matches a sigma-irrelevant baseline.
  4. Empty (N=0) xyz → ValueError from the Python validator.
"""

from __future__ import annotations

import numpy as np
import pytest

import pylidar


def _grid_points_with_z(rng: np.random.Generator, n_side: int = 20,
                        spacing: float = 0.5) -> np.ndarray:
    """Regular XY grid with random Z noise — ideal smoothing candidate."""
    xs = np.arange(n_side, dtype=np.float64) * spacing
    ys = np.arange(n_side, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(xs, ys)
    zz = rng.standard_normal(xx.shape) * 2.0 + 10.0  # mean 10, std ~2
    xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return np.ascontiguousarray(xyz, dtype=np.float64)


def test_smooth_height_reduces_variance_circular_mean():
    """Random Z over a regular grid → mean-circular smoothing must lower
    the Z variance (each point pulled toward its neighbour mean)."""
    rng = np.random.default_rng(seed=42)
    xyz = _grid_points_with_z(rng, n_side=20, spacing=0.5)

    z_in  = xyz[:, 2].copy()
    z_out = pylidar.smooth_height(xyz, size=2.0, method="mean", shape="circular")

    assert z_out.shape == (xyz.shape[0],)
    assert z_out.dtype == np.float64
    assert np.all(np.isfinite(z_out))
    # Smoothing strictly reduces variance for non-degenerate noise.
    assert z_out.var() < z_in.var() * 0.5


def test_smooth_height_flat_ground_unchanged():
    """All Z = const → smoothing must yield the same const (mean / gaussian /
    square / circular all preserve constants)."""
    n_side = 10
    xs = np.arange(n_side, dtype=np.float64)
    ys = np.arange(n_side, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    z = np.full(xx.shape, 7.5, dtype=np.float64)
    xyz = np.ascontiguousarray(
        np.stack([xx.ravel(), yy.ravel(), z.ravel()], axis=1)
    )

    for method, shape in [
        ("mean",     "circular"),
        ("mean",     "square"),
        ("gaussian", "circular"),
        ("gaussian", "square"),
    ]:
        out = pylidar.smooth_height(xyz, size=2.0, method=method, shape=shape)
        assert np.allclose(out, 7.5, atol=1e-9), \
            f"flat ground perturbed by method={method} shape={shape}: " \
            f"max dev = {np.max(np.abs(out - 7.5))}"


def test_smooth_height_sigma_zero_is_harmless_for_mean():
    """sigma is irrelevant for method='mean'. sigma=0 must not raise and must
    produce the same output as the default sigma."""
    rng = np.random.default_rng(seed=7)
    xyz = _grid_points_with_z(rng, n_side=8, spacing=0.5)

    out_default = pylidar.smooth_height(xyz, size=1.5, method="mean",
                                        shape="circular")
    out_sigma_0 = pylidar.smooth_height(xyz, size=1.5, method="mean",
                                        shape="circular", sigma=0.0)

    assert np.allclose(out_default, out_sigma_0, atol=1e-12)

    # Conversely, sigma <= 0 with gaussian must raise ValueError.
    with pytest.raises(ValueError, match="sigma"):
        pylidar.smooth_height(xyz, size=1.5, method="gaussian",
                              shape="circular", sigma=0.0)


def test_smooth_height_empty_array_raises_value_error():
    empty = np.empty((0, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="N=0"):
        pylidar.smooth_height(empty, size=1.0, method="mean", shape="circular")


def test_smooth_height_rejects_wrong_dtype():
    """Sanity: float32 input is rejected with TypeError, not silently cast."""
    xyz_f32 = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        pylidar.smooth_height(xyz_f32, size=1.0, method="mean",
                              shape="circular")


def test_smooth_height_rejects_bad_shape_string():
    xyz = np.zeros((5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        pylidar.smooth_height(xyz, size=1.0, method="mean", shape="hexagon")
