#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _parse_meta_scalar(meta_raw) -> Dict:
    if meta_raw is None:
        return {}
    if isinstance(meta_raw, np.ndarray):
        if meta_raw.shape == ():
            meta_raw = meta_raw.item()
        else:
            return {}
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8", errors="ignore")
    if not isinstance(meta_raw, str):
        return {}
    text = meta_raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _cell_to_world(world_min: np.ndarray, resolution: float, gx: int, gy: int, z: float) -> Dict[str, float]:
    return {
        "x": float(world_min[0] + (float(gx) + 0.5) * resolution),
        "y": float(world_min[1] + (float(gy) + 0.5) * resolution),
        "z": float(z),
    }


def _pick_flow_points_from_walkable(free_out: np.ndarray) -> List[List[int]]:
    ys, xs = np.where(free_out > 0)
    if len(xs) < 20:
        raise RuntimeError(f"Walkable cells too few for stable flow points: {len(xs)}")

    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = (centered.T @ centered) / max(1, len(coords) - 1)
    vals, vecs = np.linalg.eigh(cov)
    main_axis = vecs[:, int(np.argmax(vals))]
    orth_axis = np.array([-main_axis[1], main_axis[0]], dtype=np.float64)

    proj_main = centered @ main_axis
    proj_orth = centered @ orth_axis

    q_low, q_high = np.quantile(proj_main, [0.12, 0.88])
    low_idx = np.where(proj_main <= q_low)[0]
    high_idx = np.where(proj_main >= q_high)[0]
    if len(low_idx) < 2 or len(high_idx) < 2:
        order = np.argsort(proj_main)
        k = max(2, len(order) // 8)
        low_idx = order[:k]
        high_idx = order[-k:]

    li1 = low_idx[int(np.argmin(proj_orth[low_idx]))]
    li2 = low_idx[int(np.argmax(proj_orth[low_idx]))]
    hi1 = high_idx[int(np.argmin(proj_orth[high_idx]))]
    hi2 = high_idx[int(np.argmax(proj_orth[high_idx]))]
    return [
        [int(xs[li1]), int(ys[li1])],
        [int(xs[li2]), int(ys[li2])],
        [int(xs[hi1]), int(ys[hi1])],
        [int(xs[hi2]), int(ys[hi2])],
    ]


def _disk_offsets(radius_cells: int) -> List[List[int]]:
    if radius_cells <= 0:
        return [[0, 0]]
    out: List[List[int]] = []
    rr2 = radius_cells * radius_cells
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= rr2:
                out.append([dx, dy])
    return out


def _draw_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = mask.shape
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        mask[y, x] = True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _build_walked_path_mask(
    roi_mask: np.ndarray,
    points_grid: List[List[int]],
    path_radius_cells: int,
    close_loop: bool,
) -> np.ndarray:
    h, w = roi_mask.shape
    line_mask = np.zeros((h, w), dtype=bool)
    if len(points_grid) < 2:
        return line_mask.astype(np.uint8)
    for i in range(len(points_grid) - 1):
        x0, y0 = int(points_grid[i][0]), int(points_grid[i][1])
        x1, y1 = int(points_grid[i + 1][0]), int(points_grid[i + 1][1])
        _draw_line(line_mask, x0, y0, x1, y1)
    if close_loop and len(points_grid) >= 3:
        x0, y0 = int(points_grid[-1][0]), int(points_grid[-1][1])
        x1, y1 = int(points_grid[0][0]), int(points_grid[0][1])
        _draw_line(line_mask, x0, y0, x1, y1)

    if path_radius_cells > 0:
        dil = np.zeros_like(line_mask, dtype=bool)
        ys, xs = np.where(line_mask)
        offsets = _disk_offsets(path_radius_cells)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dx, dy in offsets:
                xx = x + int(dx)
                yy = y + int(dy)
                if 0 <= xx < w and 0 <= yy < h:
                    dil[yy, xx] = True
        line_mask = dil
    return (line_mask & (roi_mask > 0)).astype(np.uint8)


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert carla walk ROI output to minimal runnable episode assets."
    )
    parser.add_argument("--annotated-npz", required=True, help="carla_walk_roi_annotator output npz (must contain roi_mask)")
    parser.add_argument("--base-npz", required=True, help="trusted full-map base npz (must contain free_grid/world_min/world_max)")
    parser.add_argument("--output-dir", required=True, help="scenario assets directory, e.g. assets/scenes/park")
    parser.add_argument("--polygon-json", default="", help="optional input polygon json from annotator")
    parser.add_argument("--preserve-walked-path", action="store_true", help="force keep walked polyline cells walkable")
    parser.add_argument("--walked-path-radius-cells", type=int, default=1, help="radius (cells) when preserving walked path")
    parser.add_argument("--walked-path-close-loop", action="store_true", help="also connect last point back to first point")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting output files")
    return parser.parse_args()


def main() -> None:
    args = build_parser()
    annotated_npz = Path(args.annotated_npz).expanduser().resolve()
    base_npz = Path(args.base_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    polygon_json = Path(args.polygon_json).expanduser().resolve() if args.polygon_json else None

    if not annotated_npz.exists():
        raise SystemExit(f"annotated npz not found: {annotated_npz}")
    if not base_npz.exists():
        raise SystemExit(f"base npz not found: {base_npz}")
    if polygon_json is not None and not polygon_json.exists():
        raise SystemExit(f"polygon json not found: {polygon_json}")

    out_npz = output_dir / "gridmap_roi.npz"
    out_flow = output_dir / "gridmap_roi_flow_points.json"
    out_poly = output_dir / "gridmap_roi_polygon.json"
    out_png = output_dir / "gridmap_roi.png"
    if (not args.overwrite) and any(p.exists() for p in [out_npz, out_flow, out_poly, out_png]):
        raise SystemExit("Target files already exist. Use --overwrite to replace existing outputs.")

    base = np.load(base_npz, allow_pickle=True)
    ann = np.load(annotated_npz, allow_pickle=True)
    base_arrays: Dict[str, np.ndarray] = {k: base[k] for k in base.files}
    ann_arrays: Dict[str, np.ndarray] = {k: ann[k] for k in ann.files}

    if "free_grid" not in base_arrays:
        raise SystemExit("base npz must contain free_grid")
    if "roi_mask" not in ann_arrays:
        raise SystemExit("annotated npz must contain roi_mask")
    if "world_min" not in base_arrays or "world_max" not in base_arrays:
        raise SystemExit("base npz must contain world_min/world_max")

    free_base = np.array(base_arrays["free_grid"], dtype=np.uint8)
    roi_mask = (np.array(ann_arrays["roi_mask"], dtype=np.uint8) > 0).astype(np.uint8)
    if free_base.shape != roi_mask.shape:
        raise SystemExit(f"shape mismatch: base free_grid {free_base.shape} vs roi_mask {roi_mask.shape}")

    free_out = ((free_base > 0) & (roi_mask > 0)).astype(np.uint8)
    walked_mask = np.zeros_like(free_out, dtype=np.uint8)
    if args.preserve_walked_path:
        if polygon_json is None:
            raise SystemExit("--preserve-walked-path requires --polygon-json")
        with open(polygon_json, "r", encoding="utf-8") as f:
            poly_payload = json.load(f)
        pts_grid = poly_payload.get("points_grid", None)
        if not isinstance(pts_grid, list) or len(pts_grid) < 2:
            raise SystemExit("polygon json must contain points_grid with at least 2 points for preserve-walked-path")
        walked_mask = _build_walked_path_mask(
            roi_mask=roi_mask,
            points_grid=pts_grid,
            path_radius_cells=max(0, int(args.walked_path_radius_cells)),
            close_loop=bool(args.walked_path_close_loop),
        )
        free_out = np.where(walked_mask > 0, 1, free_out).astype(np.uint8)

    occ_out = (free_out == 0).astype(np.uint8)
    walkable = int(free_out.sum())
    if walkable == 0:
        raise SystemExit("walkable cells are zero after merge. Check map/ROI coordinate alignment.")

    base_meta = _parse_meta_scalar(base_arrays["__meta__"]) if "__meta__" in base_arrays else {}
    resolution = float(base_meta.get("resolution", base_meta.get("parameters", {}).get("resolution", 0.5)))
    world_min = np.array(base_arrays["world_min"], dtype=np.float64)
    ground_z = float(base_meta.get("parameters", {}).get("ground_z", 0.0))

    points_grid = _pick_flow_points_from_walkable(free_out)
    points_world = [_cell_to_world(world_min, resolution, gx, gy, z=ground_z) for gx, gy in points_grid]

    output_dir.mkdir(parents=True, exist_ok=True)

    arrays_out = dict(base_arrays)
    arrays_out["occupied_grid"] = occ_out.astype(np.uint8)
    arrays_out["free_grid"] = free_out.astype(np.uint8)
    arrays_out["roi_mask"] = roi_mask.astype(np.uint8)
    base_meta["resolution"] = float(resolution)
    base_meta["manual_region"] = {
        "tool": "convert_roi_to_minimal_assets",
        "source_annotated_npz": str(annotated_npz),
        "source_base_npz": str(base_npz),
        "roi_cells": int(roi_mask.sum()),
        "walkable_cells_after_merge": int(walkable),
        "preserve_walked_path": bool(args.preserve_walked_path),
        "walked_path_cells_forced": int(walked_mask.sum()),
        "walked_path_radius_cells": int(max(0, int(args.walked_path_radius_cells))),
        "walked_path_close_loop": bool(args.walked_path_close_loop),
    }
    arrays_out["__meta__"] = np.array(json.dumps(base_meta, ensure_ascii=False), dtype=np.str_)
    np.savez_compressed(out_npz, **arrays_out)

    flow_payload = {
        "source_npz": out_npz.name,
        "resolution": float(resolution),
        "world_min": [float(world_min[0]), float(world_min[1])],
        "points_grid": points_grid,
        "points_world": points_world,
        "usage_note": "For bidirectional mode: points[0,1]=side A, points[2,3]=side B.",
    }
    with open(out_flow, "w", encoding="utf-8") as f:
        json.dump(flow_payload, f, indent=2, ensure_ascii=False)

    if polygon_json is not None:
        with open(polygon_json, "r", encoding="utf-8") as f:
            poly = json.load(f)
        poly["source_npz"] = out_npz.name
        with open(out_poly, "w", encoding="utf-8") as f:
            json.dump(poly, f, indent=2, ensure_ascii=False)

    if plt is not None:
        plt.imsave(
            out_png,
            np.where(occ_out > 0, 0, 255).astype(np.uint8),
            cmap="gray",
            vmin=0,
            vmax=255,
        )

    print(f"Saved: {out_npz}")
    print(f"Saved: {out_flow}")
    if polygon_json is not None:
        print(f"Saved: {out_poly}")
    if plt is not None:
        print(f"Saved: {out_png}")
    print(f"Stats: roi_cells={int(roi_mask.sum())}, walkable_cells={int(walkable)}, occupied_cells={int(occ_out.sum())}")
    print(f"Flow points(grid): {points_grid}")


if __name__ == "__main__":
    main()

