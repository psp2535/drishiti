#!/usr/bin/env python3
"""Build (or extend) the DRISHTI index from a folder of GeoTIFF/COG scenes.

  python scripts/build_index.py --src data/raw --store data/store            # initial build
  python scripts/build_index.py --add data/raw/NEW_SCENE.tif --store data/store  # incremental
Scenes are ingested chronologically; already-registered files (same SHA-256) are skipped,
so re-running is safe and never rebuilds the index.
"""
import argparse, glob, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rasterio
from drishti.embed import get_embedder
from drishti.store import Store
from drishti.ingest import ingest_scene
from drishti.change import analyze


def acq(p):
    with rasterio.open(p) as s:
        return s.tags().get("ACQUISITION_DATETIME", os.path.basename(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder of .tif scenes")
    ap.add_argument("--add", nargs="*", default=[], help="specific new scene(s) to ingest")
    ap.add_argument("--store", default="data/store")
    ap.add_argument("--exclude", nargs="*", default=[], help="basenames to hold out (for incremental demo)")
    ap.add_argument("--embedder", default="auto", choices=["auto", "lite", "remoteclip"])
    ap.add_argument("--no-analyze", action="store_true", help="skip automatic change analysis after ingest")
    a = ap.parse_args()
    emb = get_embedder(prefer=a.embedder)
    st = Store(a.store, dim=emb.dim, embedder_name=emb.name)
    paths = list(a.add)
    if a.src:
        paths += [p for p in glob.glob(os.path.join(a.src, "*.tif")) if os.path.basename(p) not in a.exclude]
    paths = sorted(set(paths), key=acq)
    t0 = time.time()
    for p in paths:
        r = ingest_scene(p, st, emb)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    if paths and not a.no_analyze:
        r = analyze(st)
        print("change analysis:", json.dumps(r["stats"]))
    print(f"done: {len(paths)} file(s) in {time.time() - t0:.2f}s; index size = {st.index.ntotal} tiles; "
          f"embedder = {emb.name}")


if __name__ == "__main__":
    main()
