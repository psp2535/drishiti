"""Incremental ingestion. One call per new acquisition; nothing already indexed is recomputed.

read (any CRS/res) → reproject to AOI grid → quality mask → co-register → radiometric
normalise → land-state map → tile → embed → append to index + provenance record + audit.
Idempotent: a file whose SHA-256 is already registered is skipped.
"""
from __future__ import annotations
import os, json, time
import numpy as np
from .raster import read_scene, pixel_to_map, map_to_lonlat
from .quality import quality_mask, coregister, normalize, classify_state, WATER, VEG, BARE, BUILT
from .store import Store, now_iso

PIPELINE_VERSION = "drishti-0.3.0"
TILE = 64


def cache_path(store: Store, scene_id: str) -> str:
    d = os.path.join(store.root, "cache"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{scene_id}.npz")


def load_cache(store: Store, scene_id: str):
    z = np.load(cache_path(store, scene_id))
    return z["arr"].astype(np.float32), z["usable"], z["state"]


def ingest_scene(path: str, store: Store, emb, tile: int = TILE, actor: str = "system") -> dict:
    t0 = time.time()
    from .raster import sha256_file
    sha = sha256_file(path)
    if store.db.execute("SELECT 1 FROM scenes WHERE sha256=?", (sha,)).fetchone():
        return dict(skipped=True, reason="already ingested", sha256=sha)
    m = store.meta()
    grid = json.loads(m["grid"]) if "grid" in m else None
    sc = read_scene(path, grid)
    steps = [dict(step="read", native_res_m=sc.meta["native_res_m"],
                  reprojected=grid is not None and abs(sc.meta["native_res_m"] - (grid or {}).get("res", 0)) > 1e-6)]
    if grid is None:
        grid = dict(shape=list(sc.arr.shape[1:]), transform=list(sc.transform)[:6], crs=sc.crs,
                    res=abs(sc.transform.a))
        store.set_meta(grid=json.dumps(grid))
    q = quality_mask(sc.arr, sc.valid, scl=sc.scl)
    steps.append(dict(step="quality_mask", method=("Sentinel-2 SCL (8/9/10 cloud, 3 shadow, 11 snow)" if sc.scl is not None
                                                   else "spectral cloud/shadow/snow tests + sun-geometry shadow projection"),
                      cloud_frac=round(q["cloud_frac"], 4), shadow_frac=round(q["shadow_frac"], 4)))
    ref_id = m.get("reference_scene")
    a, shift, resp, coefs, npif = sc.arr, (0.0, 0.0), 1.0, None, 0
    is_ref = 0
    if ref_id is None and q["usable_frac"] > 0.9:
        is_ref = 1
        store.set_meta(reference_scene=sc.meta["scene_id"])
        steps.append(dict(step="reference", note="selected as radiometric/geometric reference"))
    elif ref_id is not None:
        ra, ru, _ = load_cache(store, ref_id)
        a, shift, resp = coregister(a, ra, q["usable"], ru)
        steps.append(dict(step="coregister", method="phase correlation (NIR, high-pass)", ref=ref_id,
                          dy=round(shift[0], 3), dx=round(shift[1], 3), response=round(resp, 3)))
        a, coefs, npif = normalize(a, ra, q["usable"], ru)
        steps.append(dict(step="radiometric_normalise", method="class-consistent PIF + IR-MAD-style reweighting",
                          ref=ref_id, n_pif=npif, gain_offset=coefs))
    state = classify_state(a, q["usable"])
    steps.append(dict(step="land_state", method="ratio indices (NDVI/MNDWI/redness)"))
    sid = sc.meta["scene_id"]
    np.savez_compressed(cache_path(store, sid), arr=a.astype(np.float16), usable=q["usable"], state=state)

    # ---- tiles ----
    H, W = state.shape
    rows_meta, payload = [], []
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            u = q["usable"][r:r + tile, c:c + tile]
            s = state[r:r + tile, c:c + tile]
            n = max(int(u.sum()), 1)
            xs, ys = pixel_to_map(sc.transform, [r, r + tile - 1], [c, c + tile - 1])
            lon, lat = map_to_lonlat(sc.crs, [xs.mean()], [ys.mean()])
            rows_meta.append(dict(
                scene_id=sid, row=r // tile, col=c // tile, r0=r, c0=c, r1=r + tile, c1=c + tile,
                x_min=float(xs.min()), y_min=float(ys.min()), x_max=float(xs.max()), y_max=float(ys.max()),
                lon=float(lon[0]), lat=float(lat[0]), acq_time=sc.meta["acq_time"], sensor=sc.meta["sensor"],
                platform=sc.meta["platform"], usable_frac=float(u.mean()),
                water=float(((s == WATER) & u).sum() / n), veg=float(((s == VEG) & u).sum() / n),
                bare=float(((s == BARE) & u).sum() / n), built=float(((s == BUILT) & u).sum() / n)))
            payload.append((a[:, r:r + tile, c:c + tile], s, u))
    vecs = emb.embed_tiles(payload)
    steps.append(dict(step="embed", model=emb.name, model_fingerprint=emb.fingerprint(), tile_px=tile))
    ids = store.add_tiles(rows_meta, vecs)
    steps.append(dict(step="index", index="FAISS HNSW (M=32) + IDMap, incremental add", n_tiles=len(ids)))
    with store.lock:
        store.db.execute(
            "INSERT INTO scenes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, path, sha, sc.meta["platform"], sc.meta["sensor"], sc.meta["acq_time"],
             sc.meta["product_level"], sc.meta["native_res_m"], sc.crs, q["cloud_frac"], q["usable_frac"],
             shift[0], shift[1], resp, json.dumps(coefs), npif, is_ref, cache_path(store, sid), now_iso(),
             json.dumps(dict(pipeline=PIPELINE_VERSION, steps=steps))))
        store.db.commit()
    dt = time.time() - t0
    h = store.audit(actor, "INGEST", dict(scene_id=sid, sha256=sha, tiles=len(ids), seconds=round(dt, 2)))
    return dict(skipped=False, scene_id=sid, tiles=len(ids), seconds=dt, index_total=store.index.ntotal,
                usable_frac=q["usable_frac"], shift=shift, audit_hash=h)
