"""Raster I/O: sensor-agnostic band mapping, AOI-grid reprojection, provenance hashing.

Accepts GeoTIFF / Cloud-Optimised GeoTIFF. Bands are matched by STAC common name or by
native band id (Sentinel-2 / Landsat C2), so S2, Landsat and Bhuvan products share one path.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform

BAND_ORDER = ["blue", "green", "red", "nir", "swir16"]
ALIASES = {
    "blue":   ["blue", "b02", "b2", "sr_b2", "band2"],
    "green":  ["green", "b03", "b3", "sr_b3", "band3"],
    "red":    ["red", "b04", "b4", "sr_b4", "band4"],
    "nir":    ["nir", "nir08", "b08", "b8", "b8a", "sr_b5", "band5"],
    "swir16": ["swir16", "swir1", "b11", "sr_b6", "band6"],
}


@dataclass
class Scene:
    arr: np.ndarray            # (5,H,W) float32 surface reflectance on the AOI grid
    valid: np.ndarray          # (H,W) bool, inside footprint & not nodata
    transform: object
    crs: str
    meta: dict = field(default_factory=dict)
    scl: np.ndarray | None = None  # Sentinel-2 scene classification, if the file carries an 'scl' band


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _band_index(descriptions):
    low = [(d or "").strip().lower() for d in descriptions]
    idx = {}
    for cn, al in ALIASES.items():
        for i, d in enumerate(low):
            if d in al:
                idx[cn] = i + 1
                break
    if len(idx) < 5:  # fall back to positional order
        idx = {cn: i + 1 for i, cn in enumerate(BAND_ORDER)}
    return idx


def read_scene(path: str, grid: dict | None = None) -> Scene:
    """Read a scene; if `grid` is given, reproject onto it (bilinear) so all dates align."""
    with rasterio.open(path) as src:
        tags = src.tags()
        scale = float(tags.get("SCALE", 1e-4))    # Sentinel-2 L2A: 1e-4 (and OFFSET -0.1 for baseline >= 04.00)
        offset = float(tags.get("OFFSET", 0.0))   # Landsat C2 L2 SR: SCALE 2.75e-5, OFFSET -0.2
        bidx = _band_index(src.descriptions)
        native_res = abs(src.transform.a)
        nod = src.nodata if src.nodata is not None else 0
        if grid is None:
            raw = np.stack([src.read(bidx[b]) for b in BAND_ORDER]).astype(np.float32)
            valid = np.all(raw != nod, axis=0)
            arr = np.where(valid, raw * scale + offset, 0).astype(np.float32)
            transform, crs = src.transform, src.crs.to_string()
        else:
            H, W = grid["shape"]
            dst_tf = rasterio.Affine(*grid["transform"][:6])
            arr = np.zeros((5, H, W), np.float32)
            for k, b in enumerate(BAND_ORDER):
                reproject(source=rasterio.band(src, bidx[b]), destination=arr[k],
                          src_nodata=nod, dst_nodata=0, dst_transform=dst_tf,
                          dst_crs=grid["crs"], resampling=Resampling.bilinear)
            valid = np.all(arr > 0, axis=0)
            arr = np.where(valid, arr * scale + offset, 0).astype(np.float32)
            transform, crs = dst_tf, grid["crs"]
        low = [(d or "").strip().lower() for d in src.descriptions]
        scl = None
        if "scl" in low:
            k = low.index("scl") + 1
            if grid is None:
                scl = src.read(k).astype(np.uint8)
            else:
                scl = np.zeros(grid["shape"], np.uint8)
                reproject(source=rasterio.band(src, k), destination=scl, dst_transform=dst_tf, dst_crs=grid["crs"],
                          src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    meta = dict(
        path=path, sha256=sha256_file(path), scene_id=tags.get("SCENE_ID") or path.rsplit("/", 1)[-1],
        acq_time=tags.get("ACQUISITION_DATETIME", ""), platform=tags.get("PLATFORM", "unknown"),
        sensor=tags.get("SENSOR", "unknown"), product_level=tags.get("PRODUCT_LEVEL", ""),
        native_res_m=native_res, source=tags.get("SOURCE", ""),
    )
    return Scene(arr=arr, valid=valid, transform=transform, crs=crs, meta=meta, scl=scl)


def pixel_to_map(transform, rows, cols):
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def map_to_lonlat(crs, xs, ys):
    lon, lat = warp_transform(crs, "EPSG:4326", list(np.atleast_1d(xs)), list(np.atleast_1d(ys)))
    return np.asarray(lon), np.asarray(lat)


def rgb_uint8(arr: np.ndarray, gain: float = 0.28, gamma: float = 0.8) -> np.ndarray:
    """Fixed (not per-image) stretch so before/after chips are radiometrically comparable."""
    rgb = np.stack([arr[2], arr[1], arr[0]], -1).astype(np.float32)
    return (np.clip(np.nan_to_num(rgb) / gain, 0, 1) ** gamma * 255).astype(np.uint8)
