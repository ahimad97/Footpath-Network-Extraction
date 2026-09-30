# Heavy GIS/raster imports — wrapped so that tileseg (HRNet+OCR) can load
# independently even when raster dependencies (imageio, certifi, …) are absent.
try:
    from tile2net.raster.raster import Raster
except Exception:
    Raster = None