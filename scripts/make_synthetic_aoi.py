#!/usr/bin/env python3
"""Synthetic multi-temporal, multi-sensor AOI with KNOWN changes and KNOWN confounders.

Why: lets us measure change precision / false alarms reproducibly before the organiser AOI
arrives. Confounders injected: crop seasonality, dry-season vegetation, monsoon cloud +
shadow, haze, radiometric gain shift, sub-pixel mis-registration, and a 30 m Landsat-like
acquisition mixed with 10 m Sentinel-2-like acquisitions.
Output: uint16 GeoTIFFs (reflectance x 1e4, STAC common band names) + ground truth.
"""
import argparse, json, os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi

H = W = 640
RES = 10.0
X0, Y0 = 300000.0, 1925000.0          # UTM 44N (synthetic location)
CRS = "EPSG:32644"
BANDS = ["blue", "green", "red", "nir", "swir16"]
SPEC = {  # surface reflectance, order = BANDS
    "water":  [0.060, 0.070, 0.040, 0.020, 0.010],
    "forest": [0.025, 0.050, 0.025, 0.320, 0.140],
    "grass":  [0.035, 0.070, 0.045, 0.300, 0.180],
    "crop_g": [0.030, 0.070, 0.035, 0.400, 0.170],
    "crop_m": [0.060, 0.100, 0.100, 0.300, 0.240],
    "crop_b": [0.100, 0.130, 0.170, 0.230, 0.310],
    "built":  [0.130, 0.140, 0.160, 0.210, 0.260],
    "road":   [0.080, 0.090, 0.100, 0.130, 0.150],
    "bare":   [0.130, 0.170, 0.210, 0.270, 0.340],
}
LC = {"water": 1, "veg": 2, "bare": 3, "built": 4}
DATES = [  # (date, platform, sensor)
    ("2024-01-15", "Sentinel-2A", "S2-MSI"),
    ("2024-04-10", "Landsat-9", "L9-OLI2"),
    ("2024-07-20", "Sentinel-2B", "S2-MSI"),
    ("2024-10-25", "Sentinel-2A", "S2-MSI"),
    ("2025-01-12", "Sentinel-2B", "S2-MSI"),
    ("2025-04-15", "Sentinel-2A", "S2-MSI"),
    ("2025-07-18", "Sentinel-2A", "S2-MSI"),
    ("2025-10-22", "Sentinel-2B", "S2-MSI"),
]
CROP_BY_MONTH = {1: "crop_g", 4: "crop_b", 7: "crop_m", 10: "crop_g"}


def snoise(rng, sigma):
    n = ndi.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return (n - n.mean()) / n.std()


def line_mask(p0, p1, half_w):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    (y0, x0), (y1, x1) = p0, p1
    dy, dx = y1 - y0, x1 - x0
    t = np.clip(((yy - y0) * dy + (xx - x0) * dx) / (dy * dy + dx * dx), 0, 1)
    return np.hypot(yy - (y0 + t * dy), xx - (x0 + t * dx)) <= half_w


def blocks(rng, box, size=(5, 8), gap=3, fill=0.8):
    """Random building footprints inside box=(r0,r1,c0,c1)."""
    m = np.zeros((H, W), bool)
    r0, r1, c0, c1 = box
    r = r0
    while r < r1:
        c = c0
        h = rng.integers(*size)
        while c < c1:
            w = rng.integers(*size)
            if rng.random() < fill:
                m[r:min(r + h, r1), c:min(c + w, c1)] = True
            c += w + gap
        r += h + gap
    return m


def build_world(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    center = 200 + 55 * np.sin(yy / 85.0) + 0.12 * yy
    d = np.abs(xx - center)
    L = {}
    L["water"] = d < 8
    L["grass"] = (d >= 8) & (d < 20)
    L["forest"] = np.hypot(yy - 470, xx - 470) < 85 + 14 * snoise(rng, 18)
    L["bare"] = np.hypot(yy - 520, xx - 260) < 48 + 8 * snoise(rng, 12)
    town = np.hypot(yy - 140, xx - 470) < 72 + 10 * snoise(rng, 15)
    town_roads = town & (((yy.astype(int) % 22) < 2) | ((xx.astype(int) % 22) < 2))
    L["built"] = town & ~town_roads & (snoise(rng, 3) > -1.0)
    L["road"] = town_roads | line_mask((0, 380), (639, 520), 1.2)
    L["pond"] = np.hypot(yy - 330, xx - 400) < 9
    order = ["water", "pond", "road", "built", "forest", "bare", "grass"]
    taken = np.zeros((H, W), bool)
    for k in order:
        L[k] = L[k] & ~taken
        taken |= L[k]
    L["town_grass"] = town & ~taken
    taken |= L["town_grass"]
    L["crop"] = ~taken
    fid = (yy // 28).astype(int) * 100 + (xx // 34).astype(int)
    ff = np.random.default_rng(7).uniform(0.9, 1.1, fid.max() + 1)
    L["field_gain"] = ff[fid].astype(np.float32)
    tex = rng.uniform(0.7, 1.35, (H // 2 + 1, W // 2 + 1))
    L["tex"] = np.kron(tex, np.ones((2, 2)))[:H, :W].astype(np.float32)
    L["center"] = center
    return L


def build_changes(rng, L):
    c = L["center"]
    C = {}
    cc = int(c[270, 0])
    C["C1"] = dict(type="construction", subtype="appearance", appear=3,
                   mask=blocks(rng, (245, 295, cc + 26, cc + 64)) & ~L["water"] & ~L["grass"],
                   note="new structures ~300 m east of river")
    fbox = np.zeros((H, W), bool); fbox[430:486, 400:456] = True
    C["C2"] = dict(type="clearance", subtype="disappearance", appear=4,
                   mask=fbox & L["forest"], note="forest patch cleared")
    yy, xx = np.mgrid[0:H, 0:W]
    C["C3"] = dict(type="water_increase", subtype="expansion", appear=5,
                   mask=(np.hypot(yy - 330, xx - 400) < 21) & ~L["pond"], note="reservoir/pond expanded")
    C["C4"] = dict(type="road", subtype="appearance", appear=5,
                   mask=line_mask((300, 560), (420, 639), 1.6), note="new road corridor")
    C["C5"] = dict(type="construction", subtype="expansion", appear=6,
                   mask=blocks(rng, (214, 246, 430, 510)) & ~L["built"] & ~L["road"],
                   note="settlement edge expansion (under monsoon cloud at onset)")
    return C


def render(i, date, L, C, rng):
    month = int(date[5:7])
    img = np.zeros((5, H, W), np.float32)
    lab = np.zeros((H, W), np.uint8)

    def put(mask, key, lc, gain=None):
        s = np.array(SPEC[key], np.float32)[:, None]
        g = 1.0 if gain is None else gain[mask][None, :]
        img[:, mask] = s * g
        lab[mask] = lc

    ck = CROP_BY_MONTH[month]
    put(L["crop"], ck, LC["bare"] if ck == "crop_b" else LC["veg"], L["field_gain"])
    dry = month == 4
    for k, key in (("grass", "grass"), ("town_grass", "grass"), ("forest", "forest")):
        s = np.array(SPEC[key]); b = np.array(SPEC["bare"])
        mix = (0.35 if key == "grass" else 0.10) if dry else 0.0
        img[:, L[k]] = ((1 - mix) * s + mix * b).astype(np.float32)[:, None]
        lab[L[k]] = LC["veg"]
    put(L["bare"], "bare", LC["bare"])
    put(L["water"], "water", LC["water"]); put(L["pond"], "water", LC["water"])
    put(L["road"], "road", LC["built"])
    put(L["built"], "built", LC["built"], L["tex"])
    for cid, ch in C.items():
        if i < ch["appear"]:
            continue
        m = ch["mask"]
        if ch["type"] == "construction": put(m, "built", LC["built"], L["tex"])
        elif ch["type"] == "clearance": put(m, "bare", LC["bare"])
        elif ch["type"] == "water_increase": put(m, "water", LC["water"])
        elif ch["type"] == "road": put(m, "road", LC["built"])
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    conf = {}
    if i in (2, 6):  # monsoon cloud + shadow
        n = snoise(rng, 20)
        if i == 6:
            yy, xx = np.mgrid[0:H, 0:W]
            n = np.maximum(n, 2.2 - np.hypot((yy - 230) / 45.0, (xx - 470) / 70.0) * 1.2)
        a = np.clip((n - 0.9) / 0.35, 0, 1).astype(np.float32)
        sh = ndi.shift(a, (14, 11), order=0)
        cloud = np.array([0.46, 0.46, 0.47, 0.49, 0.36], np.float32)[:, None, None]
        img = img * (1 - 0.65 * sh * (a < 0.05))
        img = img * (1 - a) + cloud * a * rng.uniform(0.9, 1.0, (1, H, W)).astype(np.float32)
        conf["cloud"] = a > 0.02
        conf["shadow"] = (sh > 0.02) & (a <= 0.02)
    if i == 3:  # haze, SW quadrant
        yy, xx = np.mgrid[0:H, 0:W]
        hz = np.clip(1.4 - np.hypot((yy - 520) / 160.0, (xx - 120) / 160.0), 0, 1).astype(np.float32)
        img += np.array([0.05, 0.035, 0.02, 0.01, 0.0], np.float32)[:, None, None] * hz
        conf["haze"] = hz > 0.2
    if i == 4:  # radiometric / illumination difference
        img = img * 0.88 + 0.008
        conf["radiometric_gain"] = True
    if i == 5:  # sub-pixel mis-registration
        img = np.stack([ndi.shift(b, (1.6, -1.2), order=1, mode="nearest") for b in img])
        conf["misregistration_px"] = [1.6, -1.2]
    return np.clip(img, 0.0001, 1), lab, conf


def write_tif(path, img, transform, tags, descr):
    prof = dict(driver="GTiff", height=img.shape[1], width=img.shape[2], count=img.shape[0],
                dtype="uint16", crs=CRS, transform=transform, tiled=True, blockxsize=128,
                blockysize=128, compress="deflate", nodata=0)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.round(img * 10000).astype(np.uint16))
        for b, d in enumerate(descr, 1):
            dst.set_band_description(b, d)
        dst.update_tags(**tags)
        dst.build_overviews([2, 4], rasterio.enums.Resampling.average)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--seed", type=int, default=2026, help="different seed = different world + confounders")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.gt, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    L = build_world(rng)
    C = build_changes(rng, L)
    gt = {"crs": CRS, "res": RES, "shape": [H, W], "origin": [X0, Y0],
          "dates": [d for d, _, _ in DATES], "changes": [], "confounders": {"injected": []}}
    arrays = {}
    for i, (date, plat, sensor) in enumerate(DATES):
        img, lab, conf = render(i, date, L, C, rng)
        sid = f"{'S2' if plat.startswith('Sentinel') else 'L9'}_{date.replace('-', '')}_SYN_AOI"
        tags = dict(ACQUISITION_DATETIME=f"{date}T05:15:00Z", PLATFORM=plat, SENSOR=sensor,
                    PRODUCT_LEVEL="L2A-like surface reflectance", SCALE="0.0001",
                    SOURCE=f"DRISHTI synthetic AOI generator v1 (seed {a.seed})", SCENE_ID=sid)
        tf = from_origin(X0, Y0, RES, RES)
        if plat.startswith("Landsat"):  # native 30 m + slightly different NIR response
            img[3] *= 0.96
            hh, ww = H // 3 + 1, W // 3 + 1
            pad = np.pad(img, ((0, 0), (0, hh * 3 - H), (0, ww * 3 - W)), mode="edge")
            img = pad.reshape(5, hh, 3, ww, 3).mean(axis=(2, 4))
            tf = from_origin(X0, Y0, 30.0, 30.0)
        write_tif(os.path.join(a.out, sid + ".tif"), img, tf, tags, BANDS)
        arrays[f"lc_{i}"] = lab
        for k, v in conf.items():
            gt["confounders"]["injected"].append(f"{k}@{date}")
            if isinstance(v, np.ndarray):
                arrays[f"{k}_{i}"] = v
        print("wrote", sid, img.shape)
    for cid, ch in C.items():
        arrays[f"change_{cid}"] = ch["mask"]
        gt["changes"].append(dict(id=cid, type=ch["type"], subtype=ch["subtype"],
                                  appear_date=DATES[ch["appear"]][0], appear_index=ch["appear"],
                                  area_px=int(ch["mask"].sum()), note=ch["note"]))
    arrays["crop_zone"] = L["crop"]
    arrays["forest_zone"] = L["forest"]
    arrays["town_zone"] = L["built"] | L["road"]
    np.savez_compressed(os.path.join(a.gt, "gt_arrays.npz"), **arrays)
    json.dump(gt, open(os.path.join(a.gt, "ground_truth.json"), "w"), indent=2)
    print(json.dumps(gt["changes"], indent=1))


if __name__ == "__main__":
    main()
