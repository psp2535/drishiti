"""DRISHTI REST API + offline analyst UI (no CDN, no external calls).

  DRISHTI_STORE=data/store uvicorn drishti.api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import io, os, json, time, platform
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import Response, FileResponse, JSONResponse
from .store import Store
from .embed import get_embedder
from .ingest import load_cache, ingest_scene, PIPELINE_VERSION
from .raster import rgb_uint8
from .query import search_text, search_image, plan
from .change import analyze
from .review import cluster_scene, queue, decide, export_geojson
from .quality import STATE_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.environ.get("DRISHTI_STORE", os.path.join(ROOT, "data", "store"))
# public demo deployments: refuse ingest (it takes a server-side file path)
READONLY = os.environ.get("DRISHTI_READONLY", "0") == "1"
emb =get_embedder(os.path.join(ROOT, "models"), os.environ.get("DRISHTI_EMBEDDER", "auto"))
st = Store(STORE, dim=emb.dim, embedder_name=emb.name)
app = FastAPI(title="DRISHTI", version=PIPELINE_VERSION)
TYPE_COLORS = {"construction": (255, 90, 60), "road": (255, 200, 40), "water_increase": (60, 170, 255),
               "water_decrease": (120, 220, 255), "clearance": (190, 255, 60), "demolition": (230, 120, 255)}


def png(a: np.ndarray) -> Response:
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True)
    return Response(b.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@lru_cache(maxsize=16)
def scene_rgb(scene_id: str) -> np.ndarray:
    a, u, s = load_cache(st, scene_id)
    return rgb_uint8(a)


def scene_idx(scene_id):
    ids = [s["scene_id"] for s in st.scenes()]
    return ids.index(scene_id)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


@app.get("/api/stats")
def stats():
    m = st.meta(); sc = st.scenes()
    grid = json.loads(m["grid"]) if "grid" in m else None
    area = (grid["shape"][0] * grid["shape"][1] * grid["res"] ** 2 / 1e6) if grid else 0
    sb = st.storage_bytes()
    return dict(pipeline=PIPELINE_VERSION, embedder=emb.name, embedder_licence=emb.licence, dim=st.dim,
                scenes=len(sc), tiles=st.index.ntotal, area_km2=round(area, 2),
                sensors=sorted({s["sensor"] for s in sc}), date_range=[sc[0]["acq_time"][:10], sc[-1]["acq_time"][:10]] if sc else None,
                storage_bytes=sum(sb.values()), storage=sb, index="FAISS IndexIDMap2(HNSWFlat M=32, inner product)",
                changes=dict(st.db.execute("SELECT COUNT(*), SUM(status='pending') FROM changes").fetchone()),
                audit=st.verify_audit(), offline=True, external_calls=0, readonly=READONLY,
                host=dict(cpu=platform.processor() or platform.machine(), python=platform.python_version()))


@app.get("/api/scenes")
def scenes():
    out = []
    for s in st.scenes():
        s = dict(s); s["processing"] = json.loads(s["processing"]); s["norm_coefs"] = json.loads(s["norm_coefs"] or "null")
        out.append(s)
    return out


@app.post("/api/search/text")
def api_search_text(body: dict = Body(...)):
    t = time.perf_counter()
    r = search_text(st, emb, body["query"], body.get("filters"), int(body.get("k", 12)),
                    body.get("positives", []), body.get("negatives", []))
    r["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return r


@app.post("/api/search/image")
def api_search_image(body: dict = Body(...)):
    t = time.perf_counter()
    r = search_image(st, int(body["tile_id"]), body.get("filters"), int(body.get("k", 12)))
    r["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
    return r


@app.get("/api/plan")
def api_plan(q: str):
    return plan(q)


@app.post("/api/change/run")
def api_change(body: dict = Body(default={})):
    r = analyze(st, body.get("date_from"), body.get("date_to"), float(body.get("score_thr", 0.6)))
    return dict(run_id=r["run_id"], stats=r["stats"], n=len(r["events"]))


@app.get("/api/changes")
def api_changes(status: str | None = None):
    q = "SELECT * FROM changes" + (" WHERE status=?" if status else "") + " ORDER BY confidence DESC"
    rows = [dict(r) for r in st.db.execute(q, (status,) if status else ())]
    for r in rows:
        r["detail"] = json.loads(r["detail"])
    return rows


@app.get("/api/queue")
def api_queue(status: str = "pending"):
    return queue(st, status)


@app.post("/api/decision")
def api_decision(body: dict = Body(...)):
    try:
        return decide(st, body["change_id"], body["decision"], body.get("analyst", "analyst"), body.get("note", ""))
    except KeyError:
        raise HTTPException(404, "unknown change_id")


@app.get("/api/audit")
def api_audit(limit: int = 50):
    rows = [dict(r) for r in st.db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))]
    return dict(verify=st.verify_audit(), entries=rows)


@app.get("/api/discover/clusters")
def api_clusters(scene_id: str, min_cluster_size: int = 4):
    return cluster_scene(st, scene_id, min_cluster_size)


@app.get("/api/tile/{tile_id}")
def api_tile(tile_id: int):
    t = st.tiles([tile_id])
    if not t:
        raise HTTPException(404)
    return t[0]


@app.get("/api/timeline")
def api_timeline(change_id: str):
    c = st.db.execute("SELECT * FROM changes WHERE change_id=?", (change_id,)).fetchone()
    if not c:
        raise HTTPException(404)
    post = json.loads(c["detail"]).get("post_state", "")
    code = {v: k for k, v in STATE_NAMES.items()}.get(post)
    out = []
    for s in st.scenes():
        _, u, stt = load_cache(st, s["scene_id"])
        w = np.s_[c["r0"]:c["r1"], c["c0"]:c["c1"]]
        uu = u[w]; ss = stt[w]
        vals, cnt = np.unique(ss[uu], return_counts=True) if uu.any() else ([], [])
        dom = STATE_NAMES[int(vals[np.argmax(cnt)])] if len(cnt) else "no usable pixels"
        # share of the event's post-change state inside the box: what the analyst actually checks
        share = round(float((ss[uu] == code).mean()), 2) if (code is not None and uu.any()) else None
        out.append(dict(scene_id=s["scene_id"], date=s["acq_time"][:10], sensor=s["sensor"],
                        usable=round(float(uu.mean()), 2), dominant=dom, post_state=post, post_share=share))
    return out


@app.post("/api/ingest")
def api_ingest(body: dict = Body(...)):
    if READONLY:
        raise HTTPException(403, "ingest is disabled on this read-only deployment")
    r = ingest_scene(body["path"], st, emb, actor=body.get("actor", "analyst"))
    scene_rgb.cache_clear()
    if not r.get("skipped") and body.get("analyze", True):
        r["change_analysis"] = analyze(st)["stats"]
    return r


@app.post("/api/export")
def api_export(body: dict = Body(default={})):
    fc = export_geojson(st, tuple(body.get("statuses", ["confirmed"])))
    return JSONResponse(fc, headers={"Content-Disposition": "attachment; filename=drishti_changes.geojson"})


# ---------------- imagery ----------------
@app.get("/img/tile/{tile_id}.png")
def img_tile(tile_id: int, size: int = 160):
    t = st.tiles([tile_id])[0]
    a = scene_rgb(t["scene_id"])[t["r0"]:t["r1"], t["c0"]:t["c1"]]
    return png(np.asarray(Image.fromarray(a).resize((size, size), Image.NEAREST)))


@app.get("/img/scene/{scene_id}.png")
def img_scene(scene_id: str, size: int = 640):
    a = scene_rgb(scene_id)
    return png(np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR)))


@app.get("/img/chip/{change_id}/{which}.png")
def img_chip(change_id: str, which: str, pad: int = 24, size: int = 256):
    c = st.db.execute("SELECT * FROM changes WHERE change_id=?", (change_id,)).fetchone()
    if not c:
        raise HTTPException(404)
    sid = c["before_scene"] if which == "before" else c["after_scene"]
    a = scene_rgb(sid)
    H, W = a.shape[:2]
    r0, r1 = max(0, c["r0"] - pad), min(H, c["r1"] + pad)
    c0, c1 = max(0, c["c0"] - pad), min(W, c["c1"] + pad)
    side = max(r1 - r0, c1 - c0)
    im = Image.fromarray(a[r0:r0 + side, c0:c0 + side]).resize((size, size), Image.NEAREST)
    s = size / side
    d = ImageDraw.Draw(im)
    col = TYPE_COLORS.get(c["type"], (255, 255, 255))
    d.rectangle([(c["c0"] - c0) * s, (c["r0"] - r0) * s, (c["c1"] - c0) * s, (c["r1"] - r0) * s], outline=col, width=2)
    return png(np.asarray(im))
