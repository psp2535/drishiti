#!/usr/bin/env python3
"""Fast self-test (about 20 s on 1 CPU core). Exits non-zero on the first failure.

  python scripts/selftest.py

Covers: native and reprojected read paths, SCALE/OFFSET, the optional SCL band (via stack_bands.py), idempotent and
incremental ingest, change analysis, text/image search, review decision, GeoJSON export and audit-chain verification.
"""
import os, sys, json, shutil, subprocess, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import rasterio

from drishti.raster import read_scene
from drishti.store import Store
from drishti.embed import get_embedder
from drishti.ingest import ingest_scene
from drishti.change import analyze
from drishti.query import search_text, search_image
from drishti.review import decide, export_geojson

RAW = os.path.join(ROOT, "data", "raw")
ok = lambda msg: print(f"  ok  {msg}")


def check(cond, msg):
    if not cond:
        print(f"  FAIL {msg}"); sys.exit(1)
    ok(msg)


def main():
    if not os.path.exists(os.path.join(RAW, "S2_20251022_SYN_AOI.tif")):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_synthetic_aoi.py")], check=True,
                       cwd=ROOT, capture_output=True)
    scenes = sorted(os.listdir(RAW))
    ref, others = os.path.join(RAW, "S2_20240115_SYN_AOI.tif"), [os.path.join(RAW, s) for s in scenes]
    tmp = tempfile.mkdtemp(prefix="drishti_selftest_")
    try:
        print("raster reader")
        a = read_scene(ref)
        check(a.arr.shape == (5, 640, 640) and a.valid.all(), "native read path")
        grid = dict(shape=[640, 640], transform=list(a.transform)[:6], crs=a.crs)
        b = read_scene(os.path.join(RAW, "L9_20240410_SYN_AOI.tif"), grid)
        check(b.arr.shape == (5, 640, 640) and b.crs == a.crs, "reprojected read path (30 m -> 10 m grid)")
        c = read_scene(ref, grid)
        check(np.allclose(c.arr[:, c.valid], a.arr[:, c.valid], atol=1e-3), "reprojection onto own grid is identity")

        # split into single bands, restack with SCL + OFFSET through the helper script
        with rasterio.open(ref) as s:
            p = s.profile.copy(); p.update(count=1)
            for i, bn in enumerate(["blue", "green", "red", "nir", "swir16"], 1):
                with rasterio.open(f"{tmp}/{bn}.tif", "w", **p) as d:
                    d.write(s.read(i), 1)
            p.update(dtype="uint8", nodata=0)
            scl = np.full((s.height, s.width), 4, np.uint8); scl[:50, :50] = 9
            with rasterio.open(f"{tmp}/scl.tif", "w", **p) as d:
                d.write(scl, 1)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "stack_bands.py"),
                        *sum([[f"--{bn}", f"{tmp}/{bn}.tif"] for bn in ["blue", "green", "red", "nir", "swir16"]], []),
                        "--scl", f"{tmp}/scl.tif", "--date", "2024-01-15T05:30:00Z", "--offset", "0.01",
                        "--out", f"{tmp}/stacked.tif"], check=True, capture_output=True)
        d = read_scene(f"{tmp}/stacked.tif")
        check(d.scl is not None and (d.scl == 9).sum() == 2500, "SCL band carried through stack_bands.py")
        check(np.allclose(d.arr[:, d.valid], a.arr[:, d.valid] + 0.01, atol=1e-5), "OFFSET tag applied")
        e = read_scene(f"{tmp}/stacked.tif", grid)
        check(e.scl is not None and e.transform is not None, "SCL on the reprojected path")

        print("store, ingest, change, search, review")
        emb = get_embedder(os.path.join(ROOT, "models"), "lite")
        st = Store(f"{tmp}/store", dim=emb.dim, embedder_name=emb.name)
        first = [x for x in others if "20251022" not in x]
        for pth in sorted(first, key=lambda x: rasterio.open(x).tags()["ACQUISITION_DATETIME"]):
            ingest_scene(pth, st, emb)
        check(st.index.ntotal == 700, "7 scenes -> 700 tiles")
        r = ingest_scene(os.path.join(RAW, "S2_20251022_SYN_AOI.tif"), st, emb)
        check(st.index.ntotal == 800 and not r.get("skipped"), "incremental add -> 800 tiles")
        r = ingest_scene(os.path.join(RAW, "S2_20251022_SYN_AOI.tif"), st, emb)
        check(r.get("skipped") is True and st.index.ntotal == 800, "re-ingest of same file skipped (SHA-256)")
        res = analyze(st)
        check(len(res["events"]) == 5, f"change analysis -> 5 events (got {len(res['events'])})")
        types = sorted(ev["type"] for ev in res["events"])
        check(types == ["clearance", "construction", "construction", "road", "water_increase"], f"event types {types}")
        q = search_text(st, emb, "newly built structures near a river", k=5)
        check(len(q["results"]) >= 1 and any("construction" in " ".join(x["why"]) for x in q["results"]),
              "flagship query joins construction change")
        im = search_image(st, q["results"][0]["tile_id"], k=5)
        check(len(im["results"]) >= 1, "image-to-image search")
        cid = next(ev["change_id"] for ev in res["events"] if ev["type"] == "road")
        decide(st, cid, "confirmed", "selftest", "road visible")
        fc = export_geojson(st)
        check(len(fc["features"]) == 1 and fc["audit"]["ok"], "GeoJSON export + audit chain intact")
        print("ALL SELF-TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
