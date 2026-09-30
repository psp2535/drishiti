"""Analyst workflow: discovery, review queue, decisions, feedback re-ranking, provenance export."""
from __future__ import annotations
import json
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import LogisticRegression
from .store import Store, now_iso
from .ingest import PIPELINE_VERSION

TYPES = ["construction", "road", "water_increase", "water_decrease", "clearance", "demolition",
         "inundation", "reclamation"]


# ---------------- discovery ----------------
def cluster_scene(store: Store, scene_id: str, min_cluster_size: int = 4) -> dict:
    rows = [dict(r) for r in store.db.execute("SELECT * FROM tiles WHERE scene_id=? ORDER BY tile_id", (scene_id,))]
    ids = np.array([r["tile_id"] for r in rows])
    V = np.asarray(store.vectors()[ids])
    lab = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(V)
    with store.lock:
        store.db.executemany("UPDATE tiles SET cluster=? WHERE tile_id=?", [(int(l), int(i)) for l, i in zip(lab, ids)])
        store.db.commit()
    clusters = {}
    for r, l in zip(rows, lab):
        if l < 0:
            continue
        clusters.setdefault(int(l), []).append(r)
    out = []
    for l, members in clusters.items():
        mids = [m["tile_id"] for m in members]
        c = np.asarray(store.vectors()[mids]).mean(0)
        medoid = mids[int(np.argmax(np.asarray(store.vectors()[mids]) @ c))]
        comp = {k: round(float(np.mean([m[k] for m in members])), 3) for k in ("water", "veg", "bare", "built")}
        out.append(dict(cluster=l, size=len(members), medoid=medoid, composition=comp,
                        tiles=[dict(tile_id=m["tile_id"], row=m["row"], col=m["col"]) for m in members]))
    out.sort(key=lambda c: -c["size"])
    return dict(scene_id=scene_id, n_tiles=len(rows), n_clusters=len(out), noise=int((lab < 0).sum()), clusters=out)


# ---------------- review queue ----------------
def _features(c: dict) -> list[float]:
    d = json.loads(c["detail"])
    oh = [1.0 if c["type"] == t else 0.0 for t in TYPES]
    return [c["confidence"], np.log1p(c["area_m2"]), float(c["provisional"]), d.get("quality_frac", 1.0),
            float(d.get("persistence_obs", 1)), float(d.get("semantic_change") or 0.0), d.get("elongation", 1.0) / 10] + oh


def feedback_model(store: Store):
    rows = [dict(r) for r in store.db.execute("SELECT * FROM changes WHERE status IN ('confirmed','rejected')")]
    y = [1 if r["status"] == "confirmed" else 0 for r in rows]
    if sum(y) < 2 or len(y) - sum(y) < 2:
        return None
    return LogisticRegression(C=1.0, max_iter=500).fit([_features(r) for r in rows], y)


def queue(store: Store, status: str = "pending", limit: int = 100) -> list[dict]:
    rows = [dict(r) for r in store.db.execute("SELECT * FROM changes WHERE status=?", (status,))]
    m = feedback_model(store)
    for r in rows:
        r["detail"] = json.loads(r["detail"])
        p = m.predict_proba([_features({**r, "detail": json.dumps(r["detail"])})])[0, 1] if m else None
        r["feedback_prob"] = None if p is None else round(float(p), 3)
        r["rank_score"] = round(r["confidence"] if p is None else 0.5 * r["confidence"] + 0.5 * p, 4)
    rows.sort(key=lambda r: -r["rank_score"])
    return rows[:limit]


def decide(store: Store, change_id: str, decision: str, analyst: str, note: str = "") -> dict:
    assert decision in ("confirmed", "rejected")
    if not store.db.execute("SELECT 1 FROM changes WHERE change_id=?", (change_id,)).fetchone():
        raise KeyError(change_id)
    h = store.audit(analyst, "DECISION", dict(change_id=change_id, decision=decision, note=note))
    with store.lock:
        store.db.execute("UPDATE changes SET status=? WHERE change_id=?", (decision, change_id))
        store.db.execute("INSERT INTO decisions (change_id,decision,analyst,note,ts,audit_hash) VALUES (?,?,?,?,?,?)",
                         (change_id, decision, analyst, note, now_iso(), h))
        store.db.commit()
    return dict(change_id=change_id, decision=decision, audit_hash=h)


# ---------------- export with provenance ----------------
def export_geojson(store: Store, statuses=("confirmed",)) -> dict:
    from rasterio.warp import transform as wt
    grid = json.loads(store.meta()["grid"])
    feats = []
    q = f"SELECT * FROM changes WHERE status IN ({','.join('?' * len(statuses))})"
    for c in [dict(r) for r in store.db.execute(q, statuses)]:
        xs = [c["x_min"], c["x_max"], c["x_max"], c["x_min"], c["x_min"]]
        ys = [c["y_min"], c["y_min"], c["y_max"], c["y_max"], c["y_min"]]
        lon, lat = wt(grid["crs"], "EPSG:4326", xs, ys)
        sc = {s: store.scene(s) for s in (c["before_scene"], c["after_scene"])}
        dec = [dict(r) for r in store.db.execute("SELECT decision,analyst,note,ts,audit_hash FROM decisions WHERE change_id=? ORDER BY id", (c["change_id"],))]
        feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[[list(p) for p in zip(lon, lat)]]),
                          properties=dict(
                              change_id=c["change_id"], type=c["type"], subtype=c["subtype"], status=c["status"],
                              confidence=c["confidence"], provisional=bool(c["provisional"]), area_m2=c["area_m2"],
                              earliest_obs=c["earliest_obs"], last_pre_obs=c["last_pre_obs"],
                              detail=json.loads(c["detail"]), decisions=dec,
                              provenance=dict(
                                  pipeline=PIPELINE_VERSION, embedder=store.meta().get("embedder"),
                                  grid_crs=grid["crs"],
                                  source_scenes=[dict(role=role, scene_id=s["scene_id"], sha256=s["sha256"], platform=s["platform"],
                                                      sensor=s["sensor"], acq_time=s["acq_time"], processing=json.loads(s["processing"]))
                                                 for role, s in (("before", sc[c["before_scene"]]), ("after", sc[c["after_scene"]]))]))))
    fc = dict(type="FeatureCollection", features=feats, exported_at=now_iso(), audit=store.verify_audit())
    store.audit("system", "EXPORT", dict(n=len(feats), statuses=list(statuses)))
    return fc
