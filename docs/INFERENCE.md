# Inference Guide

## Entry points

- `infer_city.py`: canonical tile inference, per-tile graph export, cross-tile
  stitching, and GeoJSON export.
- `infer_city_fast.py`: large-area runner reusing the canonical inference and
  merge implementation with image prefetching, background saving,
  coverage-based patch counts, and compact tile outputs.
- `infer_single.py`: single-image/debug workflow driven by a manifest. It also
  supports the `tile2net` post-processing baseline.

All three require a trained model checkpoint and imagery described by a
manifest: a JSON list of records, each with `tile_id`, `image_path`, and
`bbox` (`[minx, miny, maxx, maxy]` in WGS84).

## Build a manifest

`tools/build_manifest.py` turns a directory of downloaded tiles plus a JSON
list of `[south, west, north, east]` WGS84 boxes (one per tile, in download
order) into the manifest format:

```bash
python tools/build_manifest.py \
  --bbox_list /path/to/bounds.json \
  --image_dir /path/to/images \
  --image_pattern "image{i}.png" \
  --output /path/to/manifest.json \
  --strict
```

## Canonical command

```bash
python infer_city.py \
  --config config/sidewalkformer.yaml \
  --checkpoint weights/sidewalkformer.ckpt \
  --manifest /path/to/manifest.json \
  --output_dir /path/to/results \
  --device auto
```

Important merge controls include `--snap_tol_m`, `--bridge_tol_m`,
`--topo_merge_threshold`, `--junction_snap_tol_m`, and `--halo_px`. Use
`--skip_merged_graph` when only per-tile results are required.

## Fast command

```bash
python infer_city_fast.py \
  --config config/sidewalkformer.yaml \
  --checkpoint weights/sidewalkformer.ckpt \
  --manifest /path/to/manifest.json \
  --output_dir /path/to/results_fast \
  --sample_from_centerline \
  --patch_overlap 0.15 \
  --num_workers 6 \
  --save_workers 4 \
  --bulk
```

Use `--no-bulk` for visual diagnostic outputs. Before a full city run, compare
the canonical and fast runners on a small manifest with a fixed
`--patches_per_edge` value.

## Single-image command

Create a one-record manifest, then select that tile explicitly:

```bash
python infer_single.py \
  --config config/sidewalkformer.yaml \
  --checkpoint weights/sidewalkformer.ckpt \
  --manifest /path/to/one-tile-manifest.json \
  --image_ids tile_00001 \
  --output_dir /path/to/single_image_result \
  --device auto
```

## Resume behavior

The canonical runner records completed tile IDs in `resume_state.json`; the
fast runner additionally uses an append-only completion log and merge cache.
Re-running the same command with the same output directory skips valid
completed tiles and rebuilds the global merge from resumed and new results.

## Confidence and edge metadata

Per-patch observations are averaged per tile. Cross-tile duplicate edges retain
the highest confidence before the final `TOPO_MERGE_THRESHOLD` is applied.
Synthetic seam bridges are added after model thresholding, have `NaN`/`null`
confidence, and set `edge_is_synthetic=true`.

`edge_types` uses `0` for sidewalk and `1` for crossing. The global NPZ also
stores the corresponding `edge_type_names` array.

## From a mask to a network (no model)

`tools/mask_to_network.py` converts a single binary sidewalk mask (PNG, JPEG,
TIFF, or `.npy`) into a pedestrian graph using the Tile2Net
polygon → centreline pipeline. Use it to turn manual or third-party
segmentation masks into networks:

```bash
python tools/mask_to_network.py \
  --mask /path/to/mask.png \
  --bbox 151.205,-33.875,151.215,-33.865 \
  --output-dir /path/to/network_output
```

Use `--meters-per-pixel` instead of `--bbox` when the mask has no geographic
extent; the output is then in local metric coordinates.
