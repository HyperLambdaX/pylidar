"""Minimal PCD (binary/ascii, xyz float) → LAS converter for testing.

Parses a PCD v0.7 header, reads the point payload, and writes a LAS 1.2
point-format-0 file with mm scaling. Only the x/y/z fields are carried.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import laspy


def read_pcd_xyz(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    # Header is ASCII, terminated by the line after "DATA ...".
    # Find the end of the header by locating the DATA line.
    header_lines = []
    idx = 0
    data_mode = None
    while True:
        nl = raw.index(b"\n", idx)
        line = raw[idx:nl].decode("ascii", errors="replace").strip()
        header_lines.append(line)
        idx = nl + 1
        if line.upper().startswith("DATA"):
            data_mode = line.split()[1].lower()
            break

    fields = sizes = types = counts = None
    npoints = None
    for line in header_lines:
        toks = line.split()
        if not toks:
            continue
        key = toks[0].upper()
        if key == "FIELDS":
            fields = toks[1:]
        elif key == "SIZE":
            sizes = [int(t) for t in toks[1:]]
        elif key == "TYPE":
            types = toks[1:]
        elif key == "COUNT":
            counts = [int(t) for t in toks[1:]]
        elif key == "POINTS":
            npoints = int(toks[1])

    if counts is None:
        counts = [1] * len(fields)

    print(f"  fields={fields} size={sizes} type={types} count={counts} "
          f"npoints={npoints} data={data_mode}")

    if data_mode == "ascii":
        arr = np.loadtxt(path, skiprows=len(header_lines))
        col = {f: i for i, f in enumerate(fields)}
        return np.ascontiguousarray(
            np.stack([arr[:, col["x"]], arr[:, col["y"]], arr[:, col["z"]]], axis=1),
            dtype=np.float64,
        )

    if data_mode != "binary":
        raise SystemExit(f"unsupported DATA mode: {data_mode}")

    # Build a numpy structured dtype from FIELDS/SIZE/TYPE/COUNT.
    np_type = {("F", 4): "f4", ("F", 8): "f8",
               ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4",
               ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
    dt = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        base = np_type[(t.upper(), s)]
        if c == 1:
            dt.append((f, base))
        else:
            dt.append((f, base, (c,)))
    struct = np.dtype(dt)
    payload = raw[idx:]
    expected = struct.itemsize * npoints
    payload = payload[:expected]
    pts = np.frombuffer(payload, dtype=struct, count=npoints)
    return np.ascontiguousarray(
        np.stack([pts["x"].astype(np.float64),
                  pts["y"].astype(np.float64),
                  pts["z"].astype(np.float64)], axis=1),
        dtype=np.float64,
    )


def write_las(xyz: np.ndarray, out_path: Path) -> None:
    # Drop non-finite points (PCD can carry NaN/inf for invalid returns).
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} non-finite points")
        xyz = xyz[finite]

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [float(xyz[:, 0].min()),
                      float(xyz[:, 1].min()),
                      float(xyz[:, 2].min())]
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(str(out_path))


if __name__ == "__main__":
    in_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser()
    print(f"reading {in_path}")
    xyz = read_pcd_xyz(in_path)
    print(f"  {xyz.shape[0]} points; "
          f"x[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}] "
          f"y[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}] "
          f"z[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_las(xyz, out_path)
    print(f"wrote {out_path}")
