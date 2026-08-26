#!/usr/bin/env python3
"""Extract a pedestrian network from one binary segmentation mask.

Command-line wrapper around ``sidewalkformer.tile2net_postprocess``, which
runs Tile2Net's mask -> polygon -> centreline -> graph pipeline.

Examples
--------
Mask and geographic extent are known::

    python tools/mask_to_network.py \
        --mask mask.png \
        --bbox 151.205,-33.875,151.215,-33.865 \
        --output-dir network_output

For a mask without geographic coordinates, provide the imagery resolution in
metres per pixel.  The resulting GeoJSON is in a *local* metric coordinate
system, rather than a real-world map location::

    python tools/mask_to_network.py \
        --mask mask.png --meters-per-pixel 0.15 \
        --output-dir network_output

By default, every non-zero pixel is a pedestrian-network pixel (the usual
black-background / white-or-coloured-network convention).  For black lines on
a white background, add ``--foreground black``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse ``min_lon,min_lat,max_lon,max_lat`` and validate its order."""
    try:
        bbox = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "bbox must be min_lon,min_lat,max_lon,max_lat"
        ) from exc
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise argparse.ArgumentTypeError(
            "bbox must be min_lon,min_lat,max_lon,max_lat with min < max"
        )
    return bbox  # type: ignore[return-value]


def _read_mask(path: Path) -> np.ndarray:
    """Read a PNG/JPEG/TIFF image or a NumPy array without changing its values."""
    if not path.is_file():
        raise FileNotFoundError(f"Mask does not exist: {path}")
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for image masks. Install it in the sidewalkformer environment, "
                "or provide a .npy mask."
            ) from exc
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2-D mask or a colour image, received shape {array.shape}."
        )
    return array


def _default_threshold(values: np.ndarray) -> float:
    """Return the midpoint for binary images encoded as 0/1 or 0/255."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Mask contains no finite pixel values.")
    return 0.5 if float(finite.max()) <= 1.0 else 127.5


def binary_foreground_mask(mask: np.ndarray, foreground: str) -> np.ndarray:
    """Convert a binary grayscale/RGB(A) mask to uint8 labels {0, 1}.

    ``nonzero`` is intentionally the default: it is unambiguous for standard
    masks whose background is black and whose network is white or coloured.
    """
    if mask.ndim == 3:
        if mask.shape[2] not in (3, 4):
            raise ValueError(
                "Colour masks must have 3 (RGB) or 4 (RGBA) channels; "
                f"received {mask.shape[2]}."
            )
        colour = mask[..., :3]
        values = np.max(colour, axis=2)
        alpha_valid = mask[..., 3] > 0 if mask.shape[2] == 4 else None
    else:
        values = mask
        alpha_valid = None

    if foreground == "nonzero":
        network = values != 0
    else:
        threshold = _default_threshold(values)
        network = values > threshold if foreground == "white" else values < threshold
    if alpha_valid is not None:
        network &= alpha_valid

    network = np.asarray(network, dtype=np.uint8)
    if not network.any():
        raise ValueError(
            "No pedestrian-network pixels were found. Check that the mask uses "
            "non-zero network pixels, or use --foreground black for black lines "
            "on a white background."
        )
    if network.all():
        raise ValueError(
            "Every pixel was treated as foreground. The mask likely has a white "
            "background; rerun with --foreground black."
        )
    return network


def _local_bbox(height: int, width: int, metres_per_pixel: float) -> tuple[float, float, float, float]:
    """Create an EPSG:4326 bbox that represents a local EPSG:3857 metre grid."""
    if metres_per_pixel <= 0:
        raise ValueError("--meters-per-pixel must be greater than zero.")
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    max_x = width * metres_per_pixel
    max_y = height * metres_per_pixel
    min_lon, min_lat = to_wgs84.transform(0.0, 0.0)
    max_lon, max_lat = to_wgs84.transform(max_x, max_y)
    return min_lon, min_lat, max_lon, max_lat


def _write_geojson(lines_gdf, output_path: Path) -> None:
    """Write GeoJSON without relying on optional Fiona/pyogrio drivers."""
    output_path.write_text(lines_gdf.to_json(), encoding="utf-8")


def _write_outputs(
    output_dir: Path,
    nodes_pixel_rc: np.ndarray,
    edges: np.ndarray,
    lines_gdf,
    metadata: dict,
    has_geographic_bbox: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pedestrian_network_pixel_graph.npz",
        nodes_pixel_rc=nodes_pixel_rc,
        edges=edges,
    )
    np.savetxt(
        output_dir / "pedestrian_network_pixel_nodes.csv",
        nodes_pixel_rc,
        fmt="%d",
        delimiter=",",
        header="row,col",
        comments="",
    )
    np.savetxt(
        output_dir / "pedestrian_network_edges.csv",
        edges,
        fmt="%d",
        delimiter=",",
        header="source_node,target_node",
        comments="",
    )

    if has_geographic_bbox:
        _write_geojson(lines_gdf, output_dir / "pedestrian_network.geojson")
        metadata["centerlines"] = "pedestrian_network.geojson (EPSG:4326)"
    else:
        metric_lines = lines_gdf.to_crs("EPSG:3857")
        _write_geojson(metric_lines, output_dir / "pedestrian_network_local_metres.geojson")
        metadata["centerlines"] = (
            "pedestrian_network_local_metres.geojson (local metres; not georeferenced)"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _load_postprocess_module():
    """Import ``sidewalkformer.tile2net_postprocess`` (kept lazy so ``--help`` is fast)."""
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from sidewalkformer import tile2net_postprocess

    return tile2net_postprocess


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a pedestrian graph from a single binary mask using Tile2Net "
            "post-processing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mask", required=True, type=Path, help="Binary mask: PNG, JPEG, TIFF, or .npy.")
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument(
        "--bbox",
        type=_parse_bbox,
        help="Mask extent as min_lon,min_lat,max_lon,max_lat in EPSG:4326.",
    )
    location.add_argument(
        "--meters-per-pixel",
        type=float,
        help="Use when no real-world extent is available; outputs local metre coordinates.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the extracted graph.")
    parser.add_argument(
        "--foreground",
        choices=("nonzero", "white", "black"),
        default="nonzero",
        help="Which binary pixels represent pedestrian network.",
    )
    parser.add_argument(
        "--snap-tol",
        type=float,
        default=0.001,
        help="Endpoint matching tolerance in metres; preserves the underlying API default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.snap_tol < 0:
        raise ValueError("--snap-tol cannot be negative.")

    raw_mask = _read_mask(args.mask)
    binary_mask = binary_foreground_mask(raw_mask, args.foreground)
    height, width = binary_mask.shape
    geographic_bbox = args.bbox is not None
    bbox = args.bbox if geographic_bbox else _local_bbox(height, width, args.meters_per_pixel)

    postprocess = _load_postprocess_module()
    nodes_metric, edges, lines_gdf = postprocess.seg_mask_to_graph(
        binary_mask,
        bbox,
        class_map={1: "sidewalk"},
        snap_tol=args.snap_tol,
    )
    nodes_pixel_rc = postprocess.metric_coords_to_pixel(nodes_metric, bbox, (height, width))
    metadata = {
        "input_mask": str(args.mask.resolve()),
        "mask_shape": [int(height), int(width)],
        "foreground": args.foreground,
        "bbox_epsg4326": [float(value) for value in bbox],
        "spatial_mode": "geographic" if geographic_bbox else "local_metric",
        "meters_per_pixel": args.meters_per_pixel if not geographic_bbox else None,
        "node_count": int(len(nodes_pixel_rc)),
        "edge_count": int(len(edges)),
        "line_count": int(len(lines_gdf)),
        "pixel_graph": "pedestrian_network_pixel_graph.npz",
        "pixel_nodes": "pedestrian_network_pixel_nodes.csv",
        "edges": "pedestrian_network_edges.csv",
    }
    _write_outputs(
        args.output_dir, nodes_pixel_rc, edges, lines_gdf, metadata, geographic_bbox
    )
    print(f"Extracted {metadata['node_count']} nodes and {metadata['edge_count']} edges.")
    print(f"Outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
