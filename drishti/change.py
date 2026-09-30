"""Multi-temporal change analysis with precision-first false-alarm suppression.

Idea: a real change is a PERSISTENT transition between land states; confounders are not.
  * cloud / shadow / snow        → masked to UNKNOWN; never evidence for or against change
  * haze, residual mis-registration, one-off radiometric glitches → transient: fail persistence
  * season (crop cycles, dry-season grass) → cyclic veg<->bare pixels are excluded from the
    clearance test; only stable-vegetation → bare breaks count
  * illumination / sensor differences → ratio-based states + PIF normalisation upstream
Every pixel's full usable history is scanned for the single best break (max pre-consistency ×
post-consistency). Objects = spatially grouped pixels sharing a transition.
"Earliest observation" = first acquisition where the new state is visible in usable pixels,
bracketed by the last usable acquisition that still shows the old state.
"""
from __future__ import annotations
import json, hashlib, time
import numpy as np
from scipy import ndimage as ndi
from .quality import WATER, VEG, BARE, BUILT, indices
from .ingest import load_cache
from .raster import pixel_to_map, map_to_lonlat
from .store import Store, now_iso

L1_NAMES = {1: "water", 2: "built", 3: "land"}
L2_NAMES = {1: "vegetation", 2: "bare"}


def detect_breaks(X: np.ndarray, allowed: np.ndarray, ncls: int, min_pre: int = 2, pure_pre: bool = False):
    T = X.shape[0]
    valid = X > 0
    oh = np.stack([(X == k) for k in range(1, ncls + 1)]).astype(np.int16)  # K,T,H,W
    cum = oh.cumsum(1); tot = cum[:, -1]
    vcum = valid.cumsum(0).astype(np.int16); vtot = vcum[-1]
    shape = X.shape[1:]
    best = np.zeros(shape, np.float32); bt = np.full(shape, -1, np.int16)
    bpre = np.zeros(shape, np.uint8); bpost = np.zeros(shape, np.uint8)
    for t in range(1, T):
        if not allowed[t]:
            continue
        pre = cum[:, t - 1]; post = tot - pre
        npre = vcum[t - 1]; npost = vtot - npre
        pm, qm = pre.argmax(0), post.argmax(0)
        pc = pre.max(0) / np.maximum(npre, 1); qc = post.max(0) / np.maximum(npost, 1)
        ok = (npre >= min_pre) & (npost >= 1) & (pm != qm)
        if pure_pre:
            ok &= pc >= 0.999
        sc = (pc * qc * ok).astype(np.float32)
        upd = sc > best + 1e-6
        best[upd] = sc[upd]; bt[upd] = t; bpre[upd] = pm[upd] + 1; bpost[upd] = qm[upd] + 1
    return dict(score=best, t=bt, pre=bpre, post=bpost)


def _elongation(rr, cc):
    if len(rr) < 5:
        return 1.0, 1.0
    cov = np.cov(np.stack([rr, cc]).astype(np.float64))
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 4 * np.sqrt(max(ev[0], 1e-6))
    return float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6))), float(len(rr) / major)


def analyze(store: Store, date_from: str | None = None, date_to: str | None = None,
            score_thr: float = 0.6, min_area_px: int = 15, actor: str = "system") -> dict:
    t_start = time.time()
    scenes = store.scenes()
    T = len(scenes)
    if T < 3:
        return dict(events=[], note="need >= 3 acquisitions")
    dates = [s["acq_time"][:10] for s in scenes]
    df, dt = date_from or dates[0], date_to or dates[-1]
    allowed = np.array([df <= d <= dt for d in dates])
    S = np.stack([load_cache(store, s["scene_id"])[2] for s in scenes])  # T,H,W states
    H, W = S.shape[1:]
    grid = json.loads(store.meta()["grid"])
    import rasterio
    tf = rasterio.Affine(*grid["transform"][:6]); res = grid["res"]

    L1 = np.zeros_like(S); L1[S == WATER] = 1; L1[S == BUILT] = 2; L1[(S == VEG) | (S == BARE)] = 3
    L2 = np.zeros_like(S); L2[S == VEG] = 1; L2[S == BARE] = 2
    b1 = detect_breaks(L1, allowed, 3)
    b2 = detect_breaks(L2, allowed, 2, pure_pre=True)

    # bookkeeping for the false-alarm report
    usable = S > 0
    naive = np.zeros((H, W), bool)
    for t in range(1, T):
        if allowed[t]:
            naive |= usable[t] & usable[t - 1] & (S[t] != S[t - 1])
    transitions = [
        (b1, (3, 2), "construction"), (b1, (3, 1), "water_increase"), (b1, (1, 3), "water_decrease"),
        (b1, (2, 3), "demolition"), (b1, (2, 1), "inundation"), (b1, (1, 2), "reclamation"),
        (b2, (1, 2), "clearance"),
    ]
    run_id = hashlib.sha1(f"{df}|{dt}|{T}|{score_thr}".encode()).hexdigest()[:10]
    cache_arr = {}

    def arr(i):
        if i not in cache_arr:
            cache_arr[i] = load_cache(store, scenes[i]["scene_id"])[0]
        return cache_arr[i]

    events, reported = [], np.zeros((H, W), bool)
    held = []
    dist_cache = {}

    def cloud_dist(d):  # distance (px) to nearest cloud/shadow/snow pixel in acquisition d
        if d not in dist_cache:
            dist_cache[d] = ndi.distance_transform_edt(usable[d]) if (~usable[d]).any() else np.full((H, W), 1e9)
        return dist_cache[d]
    for b, (pa, pb), ctype in transitions:
        m = (b["score"] >= score_thr) & (b["pre"] == pa) & (b["post"] == pb) & (b["t"] >= 0)
        if ctype != "clearance":
            m &= ~reported
        m = ndi.binary_opening(m, structure=np.ones((2, 2)))
        if m.sum() < min_area_px:
            continue
        lab, n = ndi.label(ndi.binary_dilation(m, iterations=3), structure=np.ones((3, 3)))
        lab = lab * m
        for i in range(1, n + 1):
            pix = lab == i
            area = int(pix.sum())
            if area < min_area_px:
                continue
            X = L2 if b is b2 else L1
            rr, cc = np.nonzero(pix)
            tmode = int(np.bincount(b["t"][pix].clip(0)).argmax())
            # earliest usable acquisition showing the new state on >= 50% of the object
            earliest, n_support = None, 0
            for d in range(tmode, T):
                if (X[d][pix] == pb).mean() >= 0.5:
                    earliest = d if earliest is None else earliest
                    n_support += 1
            if earliest is None:
                continue
            last_pre = None
            for d in range(earliest - 1, -1, -1):
                if (X[d][pix] == pa).mean() >= 0.5:
                    last_pre = d; break
            if last_pre is None:
                continue
            skipped = [dates[d] for d in range(last_pre + 1, earliest) if (X[d][pix] > 0).mean() < 0.5]
            cdist = float(cloud_dist(earliest)[pix].min())
            if n_support < 2 and cdist < 12:
                # single-acquisition evidence next to cloud/shadow = classic soft-edge artefact:
                # hold until the next acquisition confirms it (precision over recall)
                held.append(dict(type=ctype, date=dates[earliest], area_px=area, cloud_dist_px=round(cdist, 1)))
                continue
            q = float(np.mean([(X[d][pix] > 0).mean() for d in range(T)]))
            base = float(b["score"][pix].mean())
            persist = 1.0 if n_support >= 2 else 0.65
            size = 1 - np.exp(-area / 40.0)
            conf = round(base * persist * (0.6 + 0.4 * q) * (0.7 + 0.3 * size), 3)
            elong, width = _elongation(rr, cc)
            ring = ndi.binary_dilation(pix, iterations=6) & ~pix
            ring_pre = S[last_pre][ring]
            etype, sub = ctype, None
            if ctype == "construction":
                if elong >= 4 and width <= 5:
                    etype, sub = "road", "appearance"
                else:
                    sub = "expansion" if (ring_pre == BUILT).sum() >= 40 else "appearance"
            elif ctype == "water_increase":
                sub = "expansion" if (ring_pre == WATER).sum() >= 20 else "appearance"
            elif ctype == "water_decrease":
                sub = "contraction" if (S[earliest][ring] == WATER).mean() > 0.08 else "disappearance"
            elif ctype == "clearance":
                sub = "contraction" if (S[earliest][ring] == VEG).mean() > 0.3 else "disappearance"
            else:
                sub = "disappearance" if ctype == "demolition" else "appearance"
            r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
            xs, ys = pixel_to_map(tf, [r0, r1 - 1], [c0, c1 - 1])
            lon, lat = map_to_lonlat(grid["crs"], [float(xs.mean())], [float(ys.mean())])
            ia, ib = indices(arr(last_pre)), indices(arr(earliest))
            ev = dict(ndvi_before=float(ia["ndvi"][pix].mean()), ndvi_after=float(ib["ndvi"][pix].mean()),
                      mndwi_before=float(ia["mndwi"][pix].mean()), mndwi_after=float(ib["mndwi"][pix].mean()))
            # semantic evidence: embedding distance of the covering tile, before vs after
            tr, tc = (r0 + r1) // 2 // 64, (c0 + c1) // 2 // 64
            tids = [store.db.execute("SELECT tile_id FROM tiles WHERE scene_id=? AND row=? AND col=?",
                                     (scenes[d]["scene_id"], tr, tc)).fetchone() for d in (last_pre, earliest)]
            sem = None
            if all(tids):
                V = store.vectors()
                sem = float(1 - V[tids[0][0]] @ V[tids[1][0]])
            detail = dict(pre_state=(L2_NAMES if b is b2 else L1_NAMES)[pa],
                          post_state=(L2_NAMES if b is b2 else L1_NAMES)[pb],
                          area_px=area, persistence_obs=n_support, quality_frac=round(q, 3),
                          elongation=round(elong, 2), mean_width_px=round(width, 2),
                          cloud_distance_m=None if cdist > 1e8 else round(cdist * res, 1),
                          cloud_skipped_dates=skipped, evidence=ev, semantic_change=sem,
                          window=[df, dt], run_id=run_id, score_thr=score_thr,
                          method="temporal-consistency break detection (L1 water/built/land, L2 stable-veg→bare)")
            cid = hashlib.sha1(f"{etype}|{r0}|{c0}|{r1}|{c1}|{dates[earliest]}".encode()).hexdigest()[:12]
            e = dict(change_id=cid, run_id=run_id, type=etype, subtype=sub, status="pending",
                     confidence=conf, score=conf, provisional=int(n_support < 2), area_m2=area * res * res,
                     r0=r0, c0=c0, r1=r1, c1=c1, lon=float(lon[0]), lat=float(lat[0]),
                     x_min=float(xs.min()), y_min=float(ys.min()), x_max=float(xs.max()), y_max=float(ys.max()),
                     before_scene=scenes[last_pre]["scene_id"], after_scene=scenes[earliest]["scene_id"],
                     earliest_obs=dates[earliest], last_pre_obs=dates[last_pre], detail=json.dumps(detail),
                     created_at=now_iso())
            events.append(e)
            reported |= pix
    with store.lock:
        for e in events:
            cols = list(e.keys())
            store.db.execute(f"INSERT OR IGNORE INTO changes ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                             [e[c] for c in cols])
        ids = [e["change_id"] for e in events]
        # candidates from earlier runs that the new evidence no longer supports leave the queue
        cur = store.db.execute(f"UPDATE changes SET status='superseded' WHERE status='pending' AND change_id NOT IN ({','.join('?' * len(ids))})", ids) if ids else store.db.execute("UPDATE changes SET status='superseded' WHERE status='pending'")
        n_sup = cur.rowcount
        store.db.commit()
    stats = dict(window=[df, dt], acquisitions=T, naive_flip_px=int(naive.sum()),
                 reported_px=int(reported.sum()), events=len(events), held_for_confirmation=len(held),
                 superseded=n_sup,
                 suppression_ratio=round(1 - reported.sum() / max(naive.sum(), 1), 4),
                 seconds=round(time.time() - t_start, 2))
    store.audit(actor, "CHANGE_RUN", dict(run_id=run_id, **stats))
    events.sort(key=lambda e: -e["confidence"])
    return dict(run_id=run_id, stats=stats, events=events, mask=reported)
