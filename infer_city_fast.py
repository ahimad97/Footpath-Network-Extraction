"""Throughput-oriented city-scale inference for very large areas.

Reuses the model forward, topology inference, merge and export of
``infer_city.py`` and produces the same global outputs. Differences:

* Coverage-based patch grid: instead of a fixed ``INFER_PATCHES_PER_EDGE``,
  each tile uses the fewest patches that cover it at ``--patch_overlap``
  (a 1024 px tile at PATCH_SIZE=1024 needs one patch). ``--patches_per_edge``
  forces a fixed count instead.
* Optional ``--sample_from_centerline`` node proposals (smaller graphs).
* A prefetching ``DataLoader`` decodes upcoming tiles while the GPU runs, and
  per-tile outputs are written by a bounded background thread pool.
* ``--bulk`` (default) writes only ``graph_core.npz`` and ``meta.json`` per tile;
  ``--no-bulk`` also writes the visualisation, masks and pickle.
* Resume state is an append-only log plus a single merge cache, so resuming a
  run with tens of thousands of tiles does not reopen every tile file.
"""

import argparse
import json
import math
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import infer_city as base
from infer_city import (
    CityScaleInferencer,
    choose_device,
    load_resume_state,
    merge_global_graph,
    read_manifest,
    save_global_graph,
    save_resume_state,
    snap_high_degree_clusters,
)
from sidewalkformer.utils import load_config


# ---------------------------------------------------------------------------
# Lightweight tile record: only the arrays the merge reads.
# ---------------------------------------------------------------------------


@dataclass
class SlimTile:
    """Just enough of a TileResult for merge_global_graph (duck-typed)."""

    tile_id: str
    nodes_global_xy: np.ndarray
    edges: np.ndarray
    edge_scores: np.ndarray
    edge_types: np.ndarray
    edge_is_synthetic: np.ndarray


_MERGE_CACHE_VERSION = 1
_MERGE_CACHE_NAME = "resume_merge_cache.pkl"


def _slim(tr) -> SlimTile:
    return SlimTile(
        tile_id=str(tr.tile_id),
        nodes_global_xy=tr.nodes_global_xy,
        edges=tr.edges,
        edge_scores=tr.edge_scores,
        edge_types=tr.edge_types,
        edge_is_synthetic=tr.edge_is_synthetic,
    )


def _merge_cache_path(out_dir: str) -> str:
    return os.path.join(out_dir, _MERGE_CACHE_NAME)


def _cache_payload(tr: SlimTile) -> dict:
    return {
        "nodes_global_xy": np.asarray(tr.nodes_global_xy, dtype=np.float64),
        "edges": np.asarray(tr.edges, dtype=np.int32),
        "edge_scores": np.asarray(tr.edge_scores, dtype=np.float32),
        "edge_types": np.asarray(tr.edge_types, dtype=np.uint8),
        "edge_is_synthetic": np.asarray(tr.edge_is_synthetic, dtype=bool),
    }


def _slim_from_cache(tile_id: str, payload: dict) -> SlimTile:
    tr = SlimTile(
        tile_id=str(tile_id),
        nodes_global_xy=np.asarray(payload["nodes_global_xy"], dtype=np.float64),
        edges=np.asarray(payload["edges"], dtype=np.int32),
        edge_scores=np.asarray(payload["edge_scores"], dtype=np.float32),
        edge_types=np.asarray(payload["edge_types"], dtype=np.uint8),
        edge_is_synthetic=np.asarray(payload["edge_is_synthetic"], dtype=bool),
    )
    base._validate_edge_metadata(
        tr.edges, tr.edge_scores, tr.edge_types, tr.edge_is_synthetic
    )
    return tr


def _load_merge_cache(out_dir: str) -> dict:
    """Load the optional one-file resume cache; corrupt/stale caches are ignored."""
    path = _merge_cache_path(out_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("version") != _MERGE_CACHE_VERSION:
            return {}
        tiles = cache.get("tiles")
        if not isinstance(tiles, dict):
            return {}
        # Validate entries now so an old/incomplete cache can never poison a merge.
        return {
            str(tile_id): _cache_payload(_slim_from_cache(str(tile_id), payload))
            for tile_id, payload in tiles.items()
        }
    except (EOFError, OSError, KeyError, TypeError, ValueError, pickle.UnpicklingError):
        print("[resume] ignoring an unreadable merge cache; rebuilding it from tile graphs.")
        return {}


def _write_merge_cache(out_dir: str, slim_tiles: List[SlimTile]) -> None:
    """Atomically persist merge arrays so future resumes avoid per-tile NPZ reads."""
    path = _merge_cache_path(out_dir)
    tmp_path = path + ".tmp"
    tiles = {str(tr.tile_id): _cache_payload(tr) for tr in slim_tiles}
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {"version": _MERGE_CACHE_VERSION, "tiles": tiles},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Coverage-based patch grid.
# ---------------------------------------------------------------------------


def coverage_patches_per_edge(h: int, w: int, patch_size: int, overlap: float) -> int:
    """Minimum patches-per-edge to cover the tile at the requested overlap fraction.

    Uses the longer side so the shorter side never under-samples. Returns 1 when the
    tile is not larger than a single patch (no oversampling).
    """
    L = max(int(h), int(w))
    if L <= patch_size:
        return 1
    overlap = min(max(float(overlap), 0.0), 0.95)
    step = patch_size * (1.0 - overlap)
    return max(1, int(math.ceil((L - patch_size) / step)) + 1)


# ---------------------------------------------------------------------------
# Prefetching dataset. Workers only decode images.
# ---------------------------------------------------------------------------


class TileImageDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        img_bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {rec['image_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Return a contiguous uint8 tensor so the worker->main handoff uses shared
        # memory instead of a pickle copy.
        return {
            "tile_id": str(rec["tile_id"]),
            "image": torch.from_numpy(np.ascontiguousarray(img_rgb)),
            "bbox": tuple(map(float, rec["bbox"])),
        }


def _worker_init(_worker_id: int) -> None:
    # Avoid OpenCV thread oversubscription across DataLoader workers.
    cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Minimal per-tile output + resume (bulk mode).
# ---------------------------------------------------------------------------


def save_tile_result_minimal(out_dir: str, tr) -> None:
    """Write only the small arrays needed for the merge and resume (uncompressed)."""
    base._validate_edge_metadata(
        tr.edges, tr.edge_scores, tr.edge_types, tr.edge_is_synthetic
    )
    tile_dir = os.path.join(out_dir, "tiles", str(tr.tile_id))
    os.makedirs(tile_dir, exist_ok=True)
    np.savez(
        os.path.join(tile_dir, "graph_core.npz"),
        nodes_rc=tr.nodes_rc_core.astype(np.int32),
        edges=tr.edges.astype(np.int32),
        edge_scores=tr.edge_scores.astype(np.float32),
        edge_probs=tr.edge_scores.astype(np.float32),
        edge_types=tr.edge_types.astype(np.uint8),
        edge_is_synthetic=tr.edge_is_synthetic.astype(bool),
        nodes_global_xy=tr.nodes_global_xy.astype(np.float64),
    )
    with open(os.path.join(tile_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tile_id": str(tr.tile_id),
                "bbox": list(map(float, tr.bbox)),
                "core_shape_hw": [int(tr.core_shape_hw[0]), int(tr.core_shape_hw[1])],
                "edge_type_labels": {"0": "sidewalk", "1": "crossing"},
            },
            f,
        )


def load_resumed_slim(tile_dir: str, rec: dict) -> SlimTile:
    """Load a previously-computed tile as a SlimTile (no masks required)."""
    with np.load(os.path.join(tile_dir, "graph_core.npz")) as g:
        edges = g["edges"].astype(np.int32)
        if "edge_scores" in g.files:
            scores = g["edge_scores"].astype(np.float32)
        elif "edge_probs" in g.files:
            scores = g["edge_probs"].astype(np.float32)
        elif len(edges) == 0:
            scores = np.zeros((0,), dtype=np.float32)
        else:
            raise ValueError("legacy graph_core.npz has no edge_scores or edge_probs")
        if "edge_types" in g.files:
            edge_types = g["edge_types"].astype(np.uint8)
        elif len(edges) == 0:
            edge_types = np.zeros((0,), dtype=np.uint8)
        else:
            raise ValueError("legacy graph_core.npz has no edge_types")
        edge_is_synthetic = (
            g["edge_is_synthetic"].astype(bool)
            if "edge_is_synthetic" in g.files
            else np.zeros((len(edges),), dtype=bool)
        )
        nodes_global_xy = g["nodes_global_xy"].astype(np.float64)
    base._validate_edge_metadata(edges, scores, edge_types, edge_is_synthetic)
    return SlimTile(
        tile_id=str(rec["tile_id"]),
        nodes_global_xy=nodes_global_xy,
        edges=edges,
        edge_scores=scores,
        edge_types=edge_types,
        edge_is_synthetic=edge_is_synthetic,
    )


def load_resumed_tiles_background(
    out_dir: str,
    records: List[dict],
    cached_tiles: dict,
) -> Tuple[List[SlimTile], List[dict]]:
    """Load old merge inputs without blocking GPU inference startup.

    Only records absent from the compact cache require a per-tile ``.npz`` open.
    Non-empty legacy tiles lacking scores/types are returned for one-time
    re-inference rather than silently producing an incomplete global graph.
    """
    resumed: List[SlimTile] = []
    legacy: List[dict] = []
    total = len(records)
    for index, rec in enumerate(records, start=1):
        tile_id = str(rec["tile_id"])
        payload = cached_tiles.get(tile_id)
        if payload is not None:
            resumed.append(_slim_from_cache(tile_id, payload))
        else:
            tile_dir = os.path.join(out_dir, "tiles", tile_id)
            graph_path = os.path.join(tile_dir, "graph_core.npz")
            if not os.path.exists(graph_path):
                legacy.append(rec)
            else:
                try:
                    tr = load_resumed_slim(tile_dir, rec)
                    resumed.append(tr)
                except ValueError:
                    legacy.append(rec)
        if index % 250 == 0 or index == total:
            print(
                f"[resume] loaded {index}/{total} completed tile graphs "
                f"({len(resumed)} ready, {len(legacy)} need reprocessing)"
            )

    return resumed, legacy


def load_resume_inputs_background(
    out_dir: str,
    records: List[dict],
) -> Tuple[List[SlimTile], List[dict]]:
    """Load the compact cache and any missing tile graphs off the GPU path."""
    cached_tiles = _load_merge_cache(out_dir)
    cached_count = sum(str(rec["tile_id"]) in cached_tiles for rec in records)
    print(
        f"[resume] {cached_count}/{len(records)} completed tile graphs found in "
        "the compact merge cache."
    )
    return load_resumed_tiles_background(out_dir, records, cached_tiles)


def _validate_manifest_record(rec: dict) -> None:
    """Validate manifest structure without touching the (possibly slow) filesystem."""
    for key in ("tile_id", "image_path", "bbox"):
        if key not in rec:
            raise ValueError(f"Manifest row missing key: {key}")
    if len(rec["bbox"]) != 4:
        raise ValueError(f"bbox must have 4 values for tile {rec['tile_id']}")


# ---------------------------------------------------------------------------
# Append-only done log: O(1) per tile instead of rewriting the JSON each time.
# ---------------------------------------------------------------------------


def _load_done(out_dir: str) -> set:
    done = set(load_resume_state(os.path.join(out_dir, "resume_state.json")).get("done", []))
    log_path = os.path.join(out_dir, "done_tiles.txt")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    done.add(t)
    return done


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fast city-scale SidewalkFormer inference")
    p.add_argument("--config", required=True, help="Path to model config yaml (PATCH_SIZE=1024)")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path (.ckpt)")
    p.add_argument("--manifest", required=True, help="Tile manifest (.json or .csv)")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--halo_px", type=int, default=0,
                   help="Pixels cropped from each tile side (0 for non-overlapping tiles)")

    # Merge arguments (same semantics as infer_city.py).
    p.add_argument("--snap_tol_m", type=float, default=2.0,
                   help="Node snap tolerance in metres (default: 2.0).")
    p.add_argument("--bridge_tol_m", type=float, default=-1.0,
                   help="Cross-tile seam-bridge tolerance in metres; -1 uses 3x snap_tol_m, 0 disables.")
    p.add_argument("--topo_merge_threshold", type=float, default=None)
    p.add_argument("--prob_min", type=float, default=None,
                   help="Per-tile edge-probability retention floor (overrides config TOPO_THRESHOLD).")
    p.add_argument("--junction_snap_tol_m", type=float, default=None,
                   help="High-degree junction snap tolerance in metres; 0 disables.")
    p.add_argument("--skip_merged_graph", action="store_true")

    # Device / precision.
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--no-bf16", dest="no_bf16", action="store_true")
    p.add_argument("--no-compile", dest="no_compile", action="store_true")
    p.add_argument("--gnn-chunk", type=int, default=16)

    # Patch grid and node proposals.
    p.add_argument("--patch_overlap", type=float, default=0.15,
                   help="Fraction of overlap between adjacent patches for the "
                        "coverage-based grid (default 0.15). Lower = fewer patches = faster.")
    p.add_argument("--patches_per_edge", type=int, default=None,
                   help="Force a fixed patches-per-edge (ignores --patch_overlap), e.g. the "
                        "config INFER_PATCHES_PER_EDGE to match infer_city.py.")
    p.add_argument("--sample_from_centerline", action="store_true",
                   help="Sample candidate nodes from the mask skeleton/centerline.")
    p.add_argument("--centerline_min_blob_px", type=int, default=None)
    p.add_argument("--centerline_thickness_px", type=int, default=None)

    # Prefetching and background saving.
    p.add_argument("--num_workers", type=int, default=6,
                   help="DataLoader workers decoding tiles ahead of the GPU (default 6).")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches prefetched per worker (only used when num_workers>0).")
    p.add_argument("--save_workers", type=int, default=4,
                   help="Background threads writing per-tile outputs (default 4).")
    bulk = p.add_mutually_exclusive_group()
    bulk.add_argument("--bulk", dest="bulk", action="store_true", default=True,
                      help="Minimal typed outputs (default): graph_core.npz + meta.json.")
    bulk.add_argument("--no-bulk", dest="bulk", action="store_false",
                      help="Write all per-tile outputs (viz PNG, masks, pickle) like infer_city.py.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    state_path = os.path.join(args.output_dir, "resume_state.json")
    done_log_path = os.path.join(args.output_dir, "done_tiles.txt")
    done = _load_done(args.output_dir)

    cfg = load_config(args.config)
    if args.topo_merge_threshold is not None:
        cfg.TOPO_MERGE_THRESHOLD = float(args.topo_merge_threshold)
    if args.prob_min is not None:
        cfg.TOPO_THRESHOLD = float(args.prob_min)
    if args.junction_snap_tol_m is not None:
        cfg.JUNCTION_SNAP_TOL_M = float(args.junction_snap_tol_m)
    if args.sample_from_centerline:
        cfg.SAMPLE_FROM_CENTERLINE = True
    if args.centerline_min_blob_px is not None:
        cfg.CENTERLINE_MIN_BLOB_PX = int(args.centerline_min_blob_px)
    if args.centerline_thickness_px is not None:
        cfg.CENTERLINE_THICKNESS_PX = int(args.centerline_thickness_px)

    patch_size = int(cfg.PATCH_SIZE)
    print(f"[cfg] PATCH_SIZE={patch_size}  patch_overlap={args.patch_overlap}  "
          f"centerline={bool(getattr(cfg, 'SAMPLE_FROM_CENTERLINE', False))}  "
          f"bulk={args.bulk}  num_workers={args.num_workers}  save_workers={args.save_workers}")

    device = choose_device(args.device)
    inferencer = CityScaleInferencer(
        cfg, args.checkpoint, device=device,
        use_bf16=not args.no_bf16,
        use_compile=not args.no_compile,
        gnn_chunk=int(args.gnn_chunk),
    )

    manifest = read_manifest(args.manifest)
    for rec in manifest:
        _validate_manifest_record(rec)
    resumed_records = [rec for rec in manifest if str(rec["tile_id"]) in done]
    pending = [rec for rec in manifest if str(rec["tile_id"]) not in done]
    if args.skip_merged_graph:
        print(
            f"[resume] {len(resumed_records)} completed candidates; global merge is skipped, "
            "so their tile graphs will not be loaded."
        )
    else:
        print(
            f"[resume] {len(resumed_records)} completed candidates; their merge inputs "
            "will load in the background."
        )
    print(f"[info] {len(pending)} new tiles to process; GPU inference starts immediately.")

    # Reading thousands of small resumed tile graphs (especially from network
    # storage) can take longer than inference, so it runs in the background and
    # is only awaited before the global merge. It starts after the DataLoader
    # workers exist: forking workers while a thread is active can hang on Linux.
    resume_executor = ThreadPoolExecutor(max_workers=1)
    resume_future = None

    def _start_resume_load() -> None:
        nonlocal resume_future
        if args.skip_merged_graph:
            return
        if resume_future is None:
            resume_future = resume_executor.submit(
                load_resume_inputs_background,
                args.output_dir,
                resumed_records,
            )

    def _run_records(
        records: List[dict],
        phase: str,
        on_workers_started=None,
    ) -> Tuple[List[SlimTile], int]:
        if not records:
            return [], 0

        loader = DataLoader(
            TileImageDataset(records),
            batch_size=1,
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=lambda b: b[0],
            prefetch_factor=(int(args.prefetch_factor) if args.num_workers > 0 else None),
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=_worker_init,
        )
        executor = ThreadPoolExecutor(max_workers=max(1, int(args.save_workers)))
        # Bound in-flight saves so at most a few tiles' full arrays are alive at once.
        inflight = threading.Semaphore(max(2, int(args.save_workers) * 4))
        lock = threading.Lock()
        done_log = open(done_log_path, "a", encoding="utf-8")
        slim_results: List[SlimTile] = []

        def _make_callback(tile_id: str):
            def _cb(fut) -> None:
                try:
                    exc = fut.exception()
                    if exc is not None:
                        print(f"[warn] save failed for {tile_id}: {exc}")
                        return
                    with lock:
                        done.add(tile_id)
                        done_log.write(tile_id + "\n")
                        done_log.flush()
                finally:
                    inflight.release()
            return _cb

        sample_iter = iter(loader)
        if on_workers_started is not None:
            on_workers_started()

        processed = 0
        t0 = time.time()
        for sample in sample_iter:
            tile_id = sample["tile_id"]
            img_rgb = sample["image"].numpy()
            bbox = sample["bbox"]
            h, w = img_rgb.shape[:2]
            if min(h, w) < patch_size:
                print(f"[skip] tile {tile_id}: {w}x{h} smaller than PATCH_SIZE={patch_size}")
                continue

            if args.patches_per_edge is not None:
                ppe = int(args.patches_per_edge)
            else:
                ppe = coverage_patches_per_edge(h, w, patch_size, args.patch_overlap)
            inferencer.cfg.INFER_PATCHES_PER_EDGE = ppe

            try:
                tr = inferencer.infer_tile(
                    tile_id=tile_id,
                    image_rgb=img_rgb,
                    bbox=bbox,
                    halo_px=args.halo_px,
                )
            except Exception as exc:  # keep the run going on a single bad tile
                print(f"[warn] inference failed for {tile_id}: {exc}")
                continue

            processed += 1
            if processed % 20 == 0:
                rate = processed / max(1e-6, time.time() - t0)
                print(
                    f"[run:{phase}] {processed}/{len(records)}  tile {tile_id}  {w}x{h}  "
                    f"ppe={ppe}  nodes={tr.nodes_rc_core.shape[0]}  "
                    f"edges={tr.edges.shape[0]}  ({rate:.2f} tiles/s)"
                )

            # Keep only the small arrays for the merge; hand the full result to the saver.
            slim_results.append(_slim(tr))
            inflight.acquire()
            if args.bulk:
                fut = executor.submit(save_tile_result_minimal, args.output_dir, tr)
            else:
                fut = executor.submit(base.save_tile_result, args.output_dir, tr, img_rgb, args.halo_px)
            fut.add_done_callback(_make_callback(tile_id))

        executor.shutdown(wait=True)
        done_log.close()
        return slim_results, processed

    fresh_slims, processed = _run_records(
        pending, "new", on_workers_started=_start_resume_load
    )
    if args.skip_merged_graph:
        resume_executor.shutdown(wait=True)
        save_resume_state(state_path, {"done": sorted(done)})
        print(f"[done] per-tile inference completed ({processed} tiles). Global merge skipped.")
        return

    _start_resume_load()
    assert resume_future is not None
    resumed_slims, legacy_records = resume_future.result()
    resume_executor.shutdown(wait=True)
    if legacy_records:
        print(
            f"[resume] {len(legacy_records)} completed tiles lack required graph metadata; "
            "reprocessing them once."
        )
    legacy_slims, legacy_processed = _run_records(legacy_records, "legacy")
    processed += legacy_processed
    slim_results = resumed_slims + fresh_slims + legacy_slims
    _write_merge_cache(args.output_dir, slim_results)
    # Write the JSON resume state once at the end (read by infer_city.py too).
    save_resume_state(state_path, {"done": sorted(done)})

    bridge_tol_m = args.bridge_tol_m if args.bridge_tol_m >= 0.0 else 3.0 * args.snap_tol_m
    topo_thresh = float(getattr(cfg, "TOPO_MERGE_THRESHOLD", 0.6))
    print(
        f"[merge] building global stitched graph  snap_tol_m={args.snap_tol_m:.2f}  "
        f"bridge_tol_m={bridge_tol_m:.2f}  topo_merge_threshold={topo_thresh:.2f}  "
        f"tiles={len(slim_results)} ..."
    )
    nodes_xy, edges, edge_scores, edge_types, edge_is_synthetic = merge_global_graph(
        slim_results,
        snap_tol_m=args.snap_tol_m,
        bridge_tol_m=bridge_tol_m,
        topo_merge_threshold=topo_thresh,
    )

    junction_tol_m = float(cfg.get("JUNCTION_SNAP_TOL_M", 3.0))
    if junction_tol_m > 0 and edges.shape[0] > 0:
        before_n, before_e = nodes_xy.shape[0], edges.shape[0]
        nodes_xy, edges, edge_scores, edge_types, edge_is_synthetic = snap_high_degree_clusters(
            nodes_xy,
            edges,
            radius_m=junction_tol_m,
            edge_scores=edge_scores,
            edge_types=edge_types,
            edge_is_synthetic=edge_is_synthetic,
        )
        print(
            f"[snap] junction snap radius_m={junction_tol_m:.2f}: "
            f"nodes {before_n} -> {nodes_xy.shape[0]}, edges {before_e} -> {edges.shape[0]}"
        )

    save_global_graph(
        args.output_dir,
        nodes_xy,
        edges,
        edge_scores,
        edge_types,
        edge_is_synthetic,
    )
    sidewalk_links = int(np.count_nonzero(edge_types == 0))
    crossing_links = int(np.count_nonzero(edge_types == 1))
    real_scores = edge_scores[~edge_is_synthetic]
    score_summary = ""
    if real_scores.size:
        score_summary = (
            f" prob[min/median/max]={real_scores.min():.3f}/"
            f"{np.median(real_scores):.3f}/{real_scores.max():.3f}"
        )
    print(
        f"[done] merged graph nodes={nodes_xy.shape[0]} edges={edges.shape[0]} "
        f"sidewalk_links={sidewalk_links} crossing_links={crossing_links}"
        f"{score_summary}"
    )


if __name__ == "__main__":
    main()
