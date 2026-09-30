# Evaluation

Predicted networks are compared with a reference network (for example,
OpenStreetMap sidewalks) on a grid of evaluation tiles. All input and output
paths are external.

1. `prepare_tile_pkls.py` converts Tile2Net outputs, the SidewalkFormer merged
   graph, and OSM ground truth (fetched with OSMnx) into aligned per-tile
   adjacency pickles.
2. `evaluate_cityscale_metrics.py` computes geometry-based precision, recall
   and F1 per tile, and APLS when the Go binary is available.

```bash
python evaluation/prepare_tile_pkls.py --help
python evaluation/evaluate_cityscale_metrics.py --help
```

Build the APLS tool locally (Go >= 1.18); the executable is git-ignored:

```bash
cd evaluation/cityscale_metrics/apls
go build -o apls .
```

`cityscale_metrics/apls/` also contains **K-APLS + geometry**
(`k_apls_geometry.py`, `k_apls_geometry_tiles.py`), which scores multiple
diverse routes and their geometric fidelity; see its
[README](cityscale_metrics/apls/README.md). Its synthetic tests run with:

```bash
python -m unittest discover -s evaluation/cityscale_metrics/apls -p "test_*.py"
```
