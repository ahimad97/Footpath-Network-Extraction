"""Build a city-scale inference manifest JSON from a bbox list and image directory.

The bbox list is a JSON list of ``[south, west, north, east]`` 4-tuples in
WGS84, one entry per tile, in the same order the images were downloaded as
``image{i}.png``.

Inference scripts expect a manifest with ``tile_id``, ``image_path``, and a 4-value
``bbox`` in the order ``[minx, miny, maxx, maxy]`` (i.e. ``[west, south, east, north]``).
This script handles the reordering and validates that all images exist.

Usage:
    python tools/build_manifest.py \
        --bbox_list bounds.json \
        --image_dir images \
        --image_pattern "image{i}.png" \
        --output manifest.json
"""
import argparse
import json
import os
from typing import List


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bbox_list", required=True,
                   help="JSON list of [south, west, north, east] boxes, one per tile.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--image_pattern", default="image{i}.png",
                   help="Filename pattern, {i} is the bbox index (0-based by default).")
    p.add_argument("--start_index", type=int, default=0,
                   help="Index of the first image (0 if image0.png exists, 1 if image1.png).")
    p.add_argument("--output", required=True, help="Output manifest JSON path.")
    p.add_argument("--strict", action="store_true",
                   help="Error if any image file is missing instead of skipping it.")
    args = p.parse_args()

    with open(args.bbox_list, "r") as f:
        bbox_list = json.load(f)

    records: List[dict] = []
    missing: List[str] = []
    for i, bbox_sw_ne in enumerate(bbox_list):
        # bbox_list entries are [south, west, north, east]
        south, west, north, east = map(float, bbox_sw_ne)
        # Inference manifest expects [minx, miny, maxx, maxy] = [west, south, east, north]
        bbox = [west, south, east, north]
        tile_id = f"tile_{i:05d}"
        img_name = args.image_pattern.format(i=i + args.start_index)
        img_path = os.path.join(args.image_dir, img_name)
        if not os.path.exists(img_path):
            missing.append(img_path)
            if args.strict:
                raise FileNotFoundError(img_path)
            continue
        records.append({
            "tile_id": tile_id,
            "image_path": os.path.abspath(img_path),
            "bbox": bbox,
        })

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"wrote {len(records)} tiles -> {args.output}")
    if missing:
        print(f"  skipped {len(missing)} missing images (first: {missing[0]})")


if __name__ == "__main__":
    main()
