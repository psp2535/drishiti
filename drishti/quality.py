"""False-alarm suppression building blocks.

1. quality_mask        cloud / thin-cloud edge / cloud-shadow (uses Sentinel-2 SCL if present,
                       spectral + geometric tests otherwise)
2. coregister          sub-pixel phase correlation against the AOI reference acquisition
3. normalize           IR-MAD-style pseudo-invariant-feature (PIF) relative radiometric normalisation
4. classify_state      ratio-based land-state map (water / vegetation / bare / built) used by
                       the temporal-consistency change engine. Ratios are illumination-robust.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi

UNKNOWN, WATER, VEG, BARE, BUILT = 0, 1, 2, 3, 4
STATE_NAMES = {0: "unknown", 1: "water", 2: "vegetation", 3: "bare", 4: "built"}
EPS = 1e-6


def indices(a: np.ndarray) -> dict:
    b, g, r, n, s = a
    return dict(
        ndvi=(n - r) / (n + r + EPS),
        mndwi=(g - s) / (g + s + EPS),
        ndbi=(s - n) / (s + n + EPS),
        redness=(r - b) / (r + b + EPS),
        bright=(b + g + r) / 3.0,
    )


def quality_mask(a: np.ndarray, valid: np.ndarray, scl: np.ndarray | None = None,
                 sun_shift=(14, 11)) -> dict:
    if scl is not None:  # Sentinel-2 L2A Scene Classification Layer
        cloud = np.isin(scl, [8, 9, 10])
        shadow = np.isin(scl, [3])
        snow = np.isin(scl, [11])
    else:
        ix = indices(a)
        b = a[0]
        cloud = (b > 0.22) & (ix["bright"] > 0.22) & (ix["ndvi"] < 0.35)
        cloud = ndi.binary_opening(cloud, iterations=1)
        cloud = ndi.binary_dilation(cloud, iterations=4)          # thin/soft cloud edges
        # shadow search zone: cloud projected along the sun azimuth over a range of heights
        zone = np.zeros_like(cloud)
        for f in np.linspace(0.4, 1.6, 7):
            zone |= ndi.shift(cloud.astype(np.uint8), (sun_shift[0] * f, sun_shift[1] * f), order=0).astype(bool)
        ref_b = np.median(ix["bright"][valid & ~cloud]) if (valid & ~cloud).any() else 0.1
        dark = (ix["bright"] < 0.55 * ref_b) & (ix["mndwi"] < 0.15)
        shadow = ndi.binary_dilation(zone & dark & ~cloud, iterations=2)
        snow = (ix["mndwi"] > 0.4) & (ix["bright"] > 0.3)       # NDSI-like
    usable = valid & ~cloud & ~shadow & ~snow
    n = max(int(valid.sum()), 1)
    return dict(cloud=cloud, shadow=shadow, snow=snow, usable=usable,
                cloud_frac=float(cloud.sum() / n), shadow_frac=float(shadow.sum() / n),
                usable_frac=float(usable.sum() / n))


def coregister(a: np.ndarray, ref: np.ndarray, usable: np.ndarray, ref_usable: np.ndarray):
    """Estimate global (dy,dx) shift of `a` w.r.t. `ref` by phase correlation on NIR, then warp."""
    both = usable & ref_usable
    if both.mean() < 0.2:
        return a, (0.0, 0.0), 0.0
    def prep(x):
        x = x.copy().astype(np.float32)
        x[~both] = np.median(x[both])
        x = x - cv2.GaussianBlur(x, (0, 0), 8)  # high-pass: edges/structures, not season
        return x
    win = cv2.createHanningWindow(a.shape[1:][::-1], cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(prep(ref[3]), prep(a[3]), win)
    if abs(sx) > 20 or abs(sy) > 20 or resp < 0.05:
        return a, (0.0, 0.0), float(resp)
    M = np.float32([[1, 0, -sx], [0, 1, -sy]])
    out = np.stack([cv2.warpAffine(b, M, b.shape[::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE) for b in a])
    return out, (float(sy), float(sx)), float(resp)


def normalize(a: np.ndarray, ref: np.ndarray, usable: np.ndarray, ref_usable: np.ndarray,
              iters: int = 4, keep: float = 0.3):
    """Relative radiometric normalisation on pseudo-invariant features selected iteratively
    (IR-MAD-like: keep pixels whose multi-band residual is smallest, refit, repeat)."""
    m = usable & ref_usable
    if m.sum() < 500:
        return a, None, 0
    # Candidate PIFs: same illumination-robust state in both dates AND an inherently stable
    # surface class (water/built/bare). Stops seasonal vegetation from dominating the fit.
    s_cur, s_ref = classify_state(a, m), classify_state(ref, m)
    cand = m & (s_cur == s_ref) & np.isin(s_ref, [WATER, BUILT, BARE])
    if cand.sum() >= 300:
        m = cand
    X = a[:, m].T.astype(np.float64); Y = ref[:, m].T.astype(np.float64)
    sel = np.ones(len(X), bool)
    coefs = np.zeros((5, 2))
    for _ in range(iters):
        for k in range(5):
            coefs[k] = np.polyfit(X[sel, k], Y[sel, k], 1)
        res = (Y - (X * coefs[:, 0] + coefs[:, 1])) / (np.std(Y[sel], 0) + EPS)
        chi = (res ** 2).sum(1)
        sel = chi <= np.quantile(chi, keep)
    out = a * coefs[:, 0, None, None].astype(np.float32) + coefs[:, 1, None, None].astype(np.float32)
    return np.clip(out, 0.0001, 1.0), coefs.round(4).tolist(), int(sel.sum())


def classify_state(a: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Illumination-robust land-state map. In production this head is a small classifier
    trained on ESA WorldCover / Dynamic World labels over the same embeddings (see docs)."""
    ix = indices(a)
    s = np.full(a.shape[1:], BUILT, np.uint8)
    s[ix["redness"] >= 0.17] = BARE
    s[ix["ndvi"] >= 0.40] = VEG
    s[(ix["mndwi"] >= 0.15) & (ix["ndvi"] < 0.2)] = WATER
    s[~usable] = UNKNOWN
    return s
