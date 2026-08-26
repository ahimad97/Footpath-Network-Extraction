# APLS

APLS is implemented in Go (adapted from the SAM-Road / Sat2Graph tooling).
Graph pickles are first converted to JSON, then scored:

```bash
python convert.py gt.p gt.json
python convert.py prop.p prop.json
go build -o apls . && ./apls gt.json prop.json result.txt
```

`result.txt` holds three numbers: `apls_gt apls_prop apls_mean`. The region,
margin and snapping parameters are set for sidewalk tiles in `main.go`
(see `main()` and `GPSInBound`). `evaluate_cityscale_metrics.py` runs these
steps for every evaluation tile.

Dependency: `github.com/dhconnelly/rtreego` (fetched by `go build`).

## K-APLS + Geometry (Python)

`k_apls_geometry.py` extends the length-only APLS comparison with up to `k`
diverse routes and a projected Hausdorff or Fréchet geometry score.

It accepts the `[nodes, edges]` APLS JSON files and GeoJSON
LineString/MultiLineString networks, including shapefiles such as Tile2Net
outputs. WGS84 inputs are automatically projected to the local UTM CRS before
distances are calculated.

```bash
python evaluation/cityscale_metrics/apls/k_apls_geometry.py \
  ground_truth.geojson prediction.geojson \
  --k 3 \
  --theta 0.6 \
  --tau 20 \
  --output result.json \
  --details-output result_details.json
```

The output reports:

- `k_apls_geometry_strict`: all sampled OD pairs, including unsnapped and
  disconnected pairs as zero;
- `k_apls_geometry_matched`: pairs where both endpoints snapped, still
  penalising disconnection;
- `k_apls_geometry_routable`: local path quality where a predicted route
  exists;
- `length_component_*`: the multi-route APLS-style length contribution;
- `geometry_component_*`: the normalized Hausdorff contribution;
- `vanilla_apls_*`: shortest-path length-only reference scores;
- `mean_aligned_hausdorff_m`: raw mean distance over existing aligned paths;
- `mean_aligned_length_score` and `mean_aligned_geometry_score`: path-pair
  fidelity before missing-route alternatives are inserted as zero;
- matching coverage, connectivity, and route availability;
- control-point matching and OD-pair counts; and
- optional per-OD route diagnostics.

### Path formulas

For each aligned route pair:

```text
L = clamp(1 - |length_gt - length_pred| / length_gt, 0, 1)
G = clamp(1 - Hausdorff(path_gt, path_pred) / tau, 0, 1)
S = w_length * L + w_geometry * G
```

Hausdorff distance is symmetric worst-case route displacement:

```text
h(A,B) = max over a in A [min over b in B distance(a,b)]
H(A,B) = max(h(A,B), h(B,A))
```

The implementation uses Shapely/GEOS discrete Hausdorff distance on route
polylines densified to approximately 5 m graph segments. This checks a dense
sequence of route vertices against the nearest point on the other polyline.

For an OD pair, length, geometry, and combined scores are each divided by the
number of accepted GT routes. Missing prediction alternatives therefore
contribute zero:

```text
L_OD = sum(aligned L) / number_of_GT_routes
G_OD = sum(aligned G) / number_of_GT_routes
S_OD = sum(aligned S) / number_of_GT_routes
S_OD = w_length * L_OD + w_geometry * G_OD
```

`vanilla_apls_*` is different from `length_component_*`: vanilla APLS compares
only the shortest path, while the length component evaluates up to `k` diverse
paths and penalizes missing alternatives.

The top-level `k_apls_geometry` and `vanilla_apls` fields are aliases for the
strict scores.

For a large region, do not report the strict global score alone. A long route
can be invalidated by one local break, so region-wide OD aggregation mainly
measures completeness. Report:

1. tile-averaged strict score for regional performance,
2. routable-path score for local path quality, and
3. tile coverage and route availability.

The decomposition is exact:

`strict score = route availability * routable-path score`

`matched score = connectivity rate * routable-path score`

These identities apply independently to the combined, length, and geometry
components.

By default, scoring is GT-to-prediction. Use `--symmetric` to also score
prediction-to-GT and average both directions.

Runtime controls:

- `--max-control-points 160` limits deterministic spatial control sampling;
- `--max-od-pairs 500` limits deterministic OD sampling;
- `--max-candidates 30` bounds the simple-path search per OD pair;
- passing `0` to either sampling cap keeps all available controls or OD pairs.

Run the synthetic tests with:

```bash
python -m unittest discover \
  -s evaluation/cityscale_metrics/apls \
  -p "test_k_apls_geometry.py"
```

### Regional tile comparison

Use `k_apls_geometry_tiles.py` to compare two city-scale networks over a set
of comparison bboxes. It clips both global networks geographically to the same
bboxes, so source tile indices do not need to match. The prediction may be a
SidewalkFormer `graph_merged.geojson` or any GeoJSON/shapefile line network
(for example, Tile2Net output).

```bash
python evaluation/cityscale_metrics/apls/k_apls_geometry_tiles.py \
  /path/to/ground_truth.geojson \
  /path/to/results/global/graph_merged.geojson \
  /path/to/comparison_bboxes.json \
  --output /path/to/k_apls_tiles.json \
  --csv-output /path/to/k_apls_tiles.csv
```
