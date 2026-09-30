# Third-Party Notices

This repository contains curated source from the following projects under
`third_party/`. Their license files remain beside the corresponding source and
govern those components. The root MIT license applies only to
SidewalkFormer-owned code.

## SAM-Road

- Component: `third_party/sam_road/` (only `model.py` and the bundled
  Segment Anything package are vendored)
- Project: *Segment Anything Model for Road Network Graph Extraction*
- Upstream: <https://github.com/htcr/sam_road>
- License: MIT
- License file: `third_party/sam_road/LICENSE`
- Paper and upstream information: `third_party/sam_road/README.md`

## Segment Anything implementation used by SAM-Road

- Component: `third_party/sam_road/sam/segment_anything/`
- License: Apache License 2.0
- License file: `third_party/sam_road/sam/LICENSE`

## Tile2Net

- Component: `third_party/tile2net/`
- Upstream: <https://github.com/VIDA-NYU/tile2net>
- License: BSD 3-Clause
- License file: `third_party/tile2net/LICENSE`

The vendored Maine imagery source reads its key from
`TILE2NET_MAINE_API_KEY`; no API credential is embedded in this repository.

The Tile2Net segmentation implementation also includes NVIDIA-derived source:

- Component: `third_party/tile2net/src/tile2net/tileseg/`
- License: NVIDIA BSD-style redistribution license
- License file: `third_party/tile2net/src/tile2net/tileseg/LICENSE`

## APLS evaluation utility

- Component: `evaluation/cityscale_metrics/apls/` (`main.go`, `convert.py`)
- Origin: the APLS tool distributed with SAM-Road (from Sat2Graph), adapted
  for sidewalk networks.
- Compiled APLS binaries are not distributed. The Go sources and module files
  are included so users can build a local executable.

## Model weights and datasets

No third-party datasets or model weights are distributed. Their licenses and
terms must be reviewed at download time and are not replaced by this source
repository's licenses.
