"""SidewalkFormer: pedestrian-network extraction from aerial imagery."""

import os
import sys

# The SAM-Road and Tile2Net baselines are vendored under ``third_party/`` and
# imported as top-level ``sam_road`` and ``tile2net`` packages. SAM-Road
# imports its bundled Segment Anything code as the top-level ``sam`` package.
THIRD_PARTY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party"
)
for _path in (THIRD_PARTY_DIR, os.path.join(THIRD_PARTY_DIR, "sam_road")):
    if _path not in sys.path:
        sys.path.append(_path)

# Prepend so the vendored Tile2Net wins over any pip-installed copy.
_TILE2NET_SRC = os.path.join(THIRD_PARTY_DIR, "tile2net", "src")
if _TILE2NET_SRC not in sys.path:
    sys.path.insert(0, _TILE2NET_SRC)
