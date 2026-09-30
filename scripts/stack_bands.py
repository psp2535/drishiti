#!/usr/bin/env python3
"""Stack five single-band rasters into the GeoTIFF layout DRISHTI ingests.

Works offline on files you already have (Sentinel-2 L2A JP2 / GeoTIFF bands, Landsat C2 L2 SR bands, or any
single-band GeoTIFFs). Bands are resampled onto the grid of the FIRST band given (use the 10 m blue band).

  # Sentinel-2 L2A (baseline >= 04.00 has a -0.1 reflectance offset: pass --offset -0.1)
  python scripts/stack_bands.py --blue B02_10m.jp2 --green B03_10m.jp2 --red B04_10m.jp2 \
      --nir B08_10m.jp2 --swir16 B11_20m.jp2 --scl SCL_20m.jp2 \
      --date 2025-03-14T05:26:21Z --platform Sentinel-2A --sensor S2-MSI --scale 1e-4 --offset -0.1 \
      --out data/real/S2_20250314.tif

  # Landsat 8/9 Collection 2 Level-2 surface reflectance
  python scripts/stack_bands.py --blue SR_B2.TIF --green SR_B3.TIF --red SR_B4.TIF --nir SR_B5.TIF \
      --swir16 SR_B6.TIF --date 2025-03-20T05:10:00Z --platform Landsat-9 --sensor L9-OLI2 \
      --scale 2.75e-5 --offset -0.2 --out data/real/L9_20250320.tif

Output: tiled, deflate-compressed GeoTIFF (COG-friendly), uint16, band descriptions blue/green/red/nir/swir16,
tags ACQUISITION_DATETIME / PLATFORM / SENSOR / SCENE_ID / SCALE / OFFSET (+ an optional 'scl' 6th band).
"""
import argparse, os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

BANDS = ["blue", "green", "red", "nir", "swir16"]


def main():
    ap = argparse.ArgumentParser()
    for b in BANDS:
        ap.add_argument(f"--{b}", required=True)
    ap.add_argument("--scl", help="optional Sentinel-2 scene classification layer (nearest resampling)")
    ap.add_argument("--date", required=True, help="acquisition time, ISO 8601")
    ap.add_argument("--platform", default="unknown"); ap.add_argument("--sensor", default="unknown")
    ap.add_argument("--scale", type=float, default=1e-4); ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--scene-id"); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with rasterio.open(getattr(a, BANDS[0])) as ref:
        prof, H, W, tf, crs = ref.profile, ref.height, ref.width, ref.transform, ref.crs
    layers, names = [], []
    for b in BANDS + (["scl"] if a.scl else []):
        path = getattr(a, b)
        dst = np.zeros((H, W), np.uint16)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tf, dst_crs=crs, src_nodata=src.nodata if src.nodata is not None else 0,
                      dst_nodata=0, resampling=Resampling.nearest if b == "scl" else Resampling.bilinear)
        layers.append(dst); names.append(b)

    prof.update(driver="GTiff", count=len(layers), dtype="uint16", nodata=0, tiled=True,
                blockxsize=256, blockysize=256, compress="deflate", predictor=2)
    for k in ("photometric",):
        prof.pop(k, None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        for i, (arr, name) in enumerate(zip(layers, names), start=1):
            dst.write(arr, i); dst.set_band_description(i, name)
        dst.update_tags(ACQUISITION_DATETIME=a.date, PLATFORM=a.platform, SENSOR=a.sensor,
                        SCENE_ID=a.scene_id or os.path.splitext(os.path.basename(a.out))[0],
                        SCALE=str(a.scale), OFFSET=str(a.offset), SOURCE="scripts/stack_bands.py")
    print(f"wrote {a.out}: {len(layers)} bands, {W}x{H}, {crs}")


if __name__ == "__main__":
    main()
