#!/usr/bin/env python3
"""Per-tile evaluation of inferred pedestrian networks vs OSM ground truth.

This is the per-tile pickle workflow: run prepare_tile_pkls.py first to produce
per-tile adjacency pickles (pixel (row,col) coords in a TILE px x TILE px tile)
for each source, then this script compares each source against the OSM GT
tile-by-tile and aggregates.

For every tile it computes:
  * count-based precision / recall / F1: an edge matches when it intersects a
    4 m buffer around the centroid of an edge in the other graph (geometry in
    EPSG:3857 from the tile bbox);
  * APLS: via the Go tool in evaluation/cityscale_metrics/apls/ (convert.py
    turns each pickle into JSON, then the ``apls`` binary scores it).

Aggregation: precision/recall/F1 are micro-averaged (sum matched / sum total)
and macro-averaged (mean of per-tile values); APLS is the mean over tiles with
a valid (non-NaN) score.

Examples
--------
  python evaluation/evaluate_cityscale_metrics.py --source sidewalkformer \
      --bbox_file tiles.json --gt_dir gt_pkls \
      --sidewalkformer_dir prediction_pkls --out_dir evaluation_results
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Transformer

BASE = os.path.dirname(os.path.abspath(__file__))
TILE = 2048  # tile pixel size, must match prepare_tile_pkls.py TILE_SIZE
APLS_DIR_DEFAULT = os.path.join(BASE, "cityscale_metrics", "apls")


# --------------------------------------------------------------------------- #
# Coordinate transforms + adjacency helpers
# --------------------------------------------------------------------------- #
_T_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def latlon_bbox_to_mercator_bbox(south, west, north, east):
    min_x, min_y = _T_4326_3857.transform(west, south)
    max_x, max_y = _T_4326_3857.transform(east, north)
    return float(min_x), float(min_y), float(max_x), float(max_y)


def pixel_to_mercator(row, col, h, w, merc_bbox):
    min_x, min_y, max_x, max_y = merc_bbox
    x = min_x + (col / (w - 1)) * (max_x - min_x)
    y = max_y - (row / (h - 1)) * (max_y - min_y)  # rows increase downward
    return float(x), float(y)


def _as_rc(t: Any) -> Tuple[int, int]:
    if isinstance(t, (tuple, list)) and len(t) == 2:
        return int(t[0]), int(t[1])
    if isinstance(t, str):
        a, b = t.strip().lstrip("(").rstrip(")").split(",")
        return int(a), int(b)
    raise ValueError(f"Unrecognized node key: {t!r}")


def load_adjdict_pickle(path: str) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {_as_rc(k): [_as_rc(v) for v in nbrs] for k, nbrs in raw.items()}


def adjdict_to_edges_gdf(adj, latlon_bbox_swne, h=TILE, w=TILE) -> gpd.GeoDataFrame:
    """Pixel adjacency dict -> undirected LineString edges in EPSG:3857."""
    south, west, north, east = latlon_bbox_swne
    merc = latlon_bbox_to_mercator_bbox(south, west, north, east)
    edges, seen = [], set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u == v:
                continue
            key = tuple(sorted([u, v]))
            if key in seen:
                continue
            seen.add(key)
            x1, y1 = pixel_to_mercator(u[0], u[1], h, w, merc)
            x2, y2 = pixel_to_mercator(v[0], v[1], h, w, merc)
            edges.append(LineString([(x1, y1), (x2, y2)]))
    return gpd.GeoDataFrame({"geometry": edges}, crs="EPSG:3857")


# --------------------------------------------------------------------------- #
# Count-based metric (centroid buffer)
# --------------------------------------------------------------------------- #
def centroid_buffer_counts(ref_edges: gpd.GeoDataFrame, pred_edges: gpd.GeoDataFrame,
                           buffer_m: float, min_len_m: float) -> dict:
    out = {"n_ref": 0, "n_pred": 0, "n_ref_matched": 0, "n_pred_matched": 0}
    if ref_edges.empty or pred_edges.empty:
        out["n_ref"] = len(ref_edges)
        out["n_pred"] = len(pred_edges)
        return out
    ref_m = ref_edges.to_crs(ref_edges.estimate_utm_crs() or "EPSG:3857")
    pred_m = pred_edges.to_crs(ref_m.crs)
    if min_len_m > 0:
        ref_m = ref_m[ref_m.geometry.length >= min_len_m]
        pred_m = pred_m[pred_m.geometry.length >= min_len_m]
    n_ref, n_pred = len(ref_m), len(pred_m)
    out["n_ref"], out["n_pred"] = n_ref, n_pred
    if n_ref == 0 or n_pred == 0:
        return out
    cbuf = gpd.GeoDataFrame(geometry=ref_m.geometry.centroid.buffer(buffer_m), crs=ref_m.crs)
    cbuf.index = np.arange(len(cbuf))
    pred_m = pred_m.reset_index(drop=True)
    hits_ref = gpd.sjoin(cbuf, pred_m, how="inner", predicate="intersects")
    out["n_ref_matched"] = len(set(hits_ref.index.unique()))
    hits_pred = gpd.sjoin(pred_m, cbuf, how="inner", predicate="intersects")
    out["n_pred_matched"] = len(set(hits_pred.index.unique()))
    return out


def prf(n_ref, n_pred, n_ref_matched, n_pred_matched) -> Tuple[float, float, float]:
    recall = n_ref_matched / n_ref if n_ref else 0.0
    precision = n_pred_matched / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# --------------------------------------------------------------------------- #
# APLS via the Go tool (convert.py + apls binary)
# --------------------------------------------------------------------------- #
def run_apls_on_pickles(apls_dir: str, gt_p: str, pred_p: str, work: str
                        ) -> Optional[Tuple[float, float, float]]:
    convert = os.path.join(apls_dir, "convert.py")
    apls_bin = os.path.join(apls_dir, "apls")
    gt_json = os.path.join(work, "gt.json")
    pred_json = os.path.join(work, "pred.json")
    res_txt = os.path.join(work, "res.txt")
    try:
        subprocess.run([sys.executable, convert, gt_p, gt_json],
                       check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, convert, pred_p, pred_json],
                       check=True, capture_output=True, text=True)
        r = subprocess.run([apls_bin, gt_json, pred_json, res_txt],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
        with open(res_txt) as f:
            vals = f.read().split()
        return float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Per-source evaluation
# --------------------------------------------------------------------------- #
def evaluate_source(name: str, pred_dir: str, gt_dir: str, bboxes: List[List[float]],
                    apls_dir: str, buffer_m: float, min_len_m: float,
                    do_apls: bool) -> dict:
    rows = []
    tot = {"n_ref": 0, "n_pred": 0, "n_ref_matched": 0, "n_pred_matched": 0}
    work = tempfile.mkdtemp(prefix=f"apls_{name}_")

    for idx, bbox in enumerate(bboxes):
        pred_p = os.path.join(pred_dir, f"tile_{idx}.p")
        gt_p = os.path.join(gt_dir, f"tile_{idx}.p")
        if not (os.path.exists(pred_p) and os.path.exists(gt_p)):
            continue
        gt_adj = load_adjdict_pickle(gt_p)
        pred_adj = load_adjdict_pickle(pred_p)
        if not gt_adj:  # no ground truth in this tile -> nothing to score
            continue

        south, west, north, east = bbox
        gt_edges = adjdict_to_edges_gdf(gt_adj, (south, west, north, east))
        pred_edges = adjdict_to_edges_gdf(pred_adj, (south, west, north, east))
        c = centroid_buffer_counts(gt_edges, pred_edges, buffer_m, min_len_m)
        for k in tot:
            tot[k] += c[k]
        p, r, f = prf(c["n_ref"], c["n_pred"], c["n_ref_matched"], c["n_pred_matched"])

        apls_mean = float("nan")
        if do_apls and pred_adj:
            res = run_apls_on_pickles(apls_dir, gt_p, pred_p, work)
            if res is not None:
                apls_mean = res[2]

        rows.append({"tile": idx, "precision": p, "recall": r, "f1": f,
                     "apls": apls_mean, **c})

    df = pd.DataFrame(rows)
    micro_p, micro_r, micro_f = prf(tot["n_ref"], tot["n_pred"],
                                    tot["n_ref_matched"], tot["n_pred_matched"])
    apls_valid = df["apls"].dropna() if not df.empty else pd.Series(dtype=float)
    summary = {
        "source": name,
        "n_tiles_scored": int(len(df)),
        "precision_micro": micro_p, "recall_micro": micro_r, "f1_micro": micro_f,
        "precision_macro": float(df["precision"].mean()) if not df.empty else 0.0,
        "recall_macro": float(df["recall"].mean()) if not df.empty else 0.0,
        "f1_macro": float(df["f1"].mean()) if not df.empty else 0.0,
        "apls_mean": float(apls_valid.mean()) if len(apls_valid) else float("nan"),
        "apls_n_tiles": int(len(apls_valid)),
        **{f"total_{k}": v for k, v in tot.items()},
    }
    return {"summary": summary, "per_tile": df}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source",
        choices=["sidewalkformer", "tile2net", "both"],
        default="sidewalkformer",
    )
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--sidewalkformer_dir", default=None)
    ap.add_argument("--tile2net_dir", default=None)
    ap.add_argument(
        "--bbox_file",
        required=True,
        help="JSON list of [south, west, north, east] tile bounds.",
    )
    ap.add_argument(
        "--apls_dir",
        default=APLS_DIR_DEFAULT,
        help="Directory containing convert.py and the locally built APLS binary.",
    )
    ap.add_argument("--buffer_m", type=float, default=4.0)
    ap.add_argument("--min_len_m", type=float, default=2.0)
    ap.add_argument("--no_apls", action="store_true")
    ap.add_argument("--out_dir", default="evaluation_results")
    args = ap.parse_args()

    if args.source in ("sidewalkformer", "both") and not args.sidewalkformer_dir:
        ap.error("--sidewalkformer_dir is required for the selected source")
    if args.source in ("tile2net", "both") and not args.tile2net_dir:
        ap.error("--tile2net_dir is required for the selected source")
    if not args.no_apls and not os.path.isfile(os.path.join(args.apls_dir, "apls")):
        ap.error(
            "APLS binary not found. Build it with: "
            "cd evaluation/cityscale_metrics/apls && go build -o apls .; "
            "or pass --no_apls."
        )

    with open(args.bbox_file) as f:
        bboxes = json.load(f)  # [[south, west, north, east], ...]
    print(f"[grid] {len(bboxes)} tiles from {os.path.basename(args.bbox_file)}")

    sources = []
    if args.source in ("sidewalkformer", "both"):
        sources.append(("sidewalkformer", args.sidewalkformer_dir))
    if args.source in ("tile2net", "both"):
        sources.append(("tile2net", args.tile2net_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = []
    for name, pred_dir in sources:
        if not os.path.isdir(pred_dir):
            print(f"[skip] {name}: missing pred dir {pred_dir}")
            continue
        print(f"\n[eval] {name}  (pred={pred_dir})")
        res = evaluate_source(name, pred_dir, args.gt_dir, bboxes, args.apls_dir,
                              args.buffer_m, args.min_len_m, not args.no_apls)
        s = res["summary"]
        res["per_tile"].to_csv(os.path.join(args.out_dir, f"per_tile_{name}.csv"), index=False)
        all_summaries.append(s)
        print(f"  tiles scored: {s['n_tiles_scored']}")
        print(f"  micro  P={s['precision_micro']:.4f}  R={s['recall_micro']:.4f}  F1={s['f1_micro']:.4f}")
        print(f"  macro  P={s['precision_macro']:.4f}  R={s['recall_macro']:.4f}  F1={s['f1_macro']:.4f}")
        print(f"  APLS   mean={s['apls_mean']:.4f}  (over {s['apls_n_tiles']} tiles)")

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n[done] wrote {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
