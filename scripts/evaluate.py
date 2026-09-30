#!/usr/bin/env python3
"""Reproducible evaluation report (the numbers SIH asks for).

  python scripts/evaluate.py --raw data/raw --gt data/gt --store data/eval_store --out docs

Measures: indexed area, #scenes, #tiles, full build time, incremental ingest time (proves no
rebuild), storage footprint, query latency (p50/p95), hardware; change detection object/pixel
precision & recall, type accuracy, earliest-observation accuracy, false alarms inside
confounder zones vs a naive bi-temporal baseline; retrieval P@k / nDCG@10 on held-out queries;
and an index-only scale benchmark at CLIP dimensionality.
"""
import argparse, glob, json, os, platform, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage as ndi
import faiss, rasterio
from drishti.embed import get_embedder
from drishti.store import Store
from drishti.ingest import ingest_scene, load_cache
from drishti.change import analyze
from drishti.query import search_text, search_image
from drishti.quality import classify_state, WATER, VEG, BARE, BUILT
from drishti.raster import read_scene

TYPE_OK = {"construction": {"construction"}, "road": {"road", "construction"}, "clearance": {"clearance"},
           "water_increase": {"water_increase"}}


def hw():
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    try:
        mem = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1048576
    except Exception:
        mem = None
    return dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=round(mem, 1) if mem else None,
                gpu="none (CPU only)", os=platform.platform(), python=platform.python_version(),
                faiss=faiss.__version__, rasterio=rasterio.__version__)


def acq(p):
    with rasterio.open(p) as s:
        return s.tags()["ACQUISITION_DATETIME"]


def ndcg(rels, k=10):
    rels = rels[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(sorted(rels, reverse=True)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--store", default="data/eval_store"); ap.add_argument("--out", default="docs")
    ap.add_argument("--scale-n", type=int, default=100000)
    a = ap.parse_args()
    shutil.rmtree(a.store, ignore_errors=True)
    G = json.load(open(os.path.join(a.gt, "ground_truth.json"))); Z = np.load(os.path.join(a.gt, "gt_arrays.npz"))
    paths = sorted(glob.glob(os.path.join(a.raw, "*.tif")), key=acq)
    emb = get_embedder()
    st = Store(a.store, dim=emb.dim, embedder_name=emb.name)
    R = dict(hardware=hw(), embedder=emb.name, dataset=dict(source="synthetic AOI (seeded, see make_synthetic_aoi.py)",
             crs=G["crs"], res_m=G["res"], confounders_injected=G["confounders"]["injected"]))

    # ---------- build + incremental ingestion ----------
    t = time.perf_counter()
    for p in paths[:-1]:
        ingest_scene(p, st, emb)
    build_s = time.perf_counter() - t
    n_before = st.index.ntotal
    t = time.perf_counter(); r_inc = ingest_scene(paths[-1], st, emb); inc_s = time.perf_counter() - t
    H, W = G["shape"]
    R["index"] = dict(scenes=len(paths), tiles=st.index.ntotal, tile_px=64, tile_m=64 * G["res"],
                      area_km2=H * W * G["res"] ** 2 / 1e6, build_seconds_first_n_minus_1=round(build_s, 2),
                      incremental_ingest_seconds=round(inc_s, 2), tiles_before_incremental=n_before,
                      tiles_after_incremental=st.index.ntotal, full_rebuild=False,
                      reingest_same_file_skipped=ingest_scene(paths[-1], st, emb).get("skipped", False))

    # ---------- change detection ----------
    t = time.perf_counter(); res = analyze(st); ch_s = time.perf_counter() - t
    events = res["events"]
    scenes = st.scenes(); dates = [s["acq_time"][:10] for s in scenes]
    gt_changes = G["changes"]
    expected_earliest = {}
    for c in gt_changes:
        m = Z[f"change_{c['id']}"]
        for i in range(c["appear_index"], len(dates)):
            cl = Z[f"cloud_{i}"] if f"cloud_{i}" in Z.files else np.zeros_like(m)
            if (cl[m]).mean() < 0.5:
                expected_earliest[c["id"]] = dates[i]; break
    matched, tp_events, rows = set(), 0, []
    for e in events:
        box = np.zeros((H, W), bool); box[e["r0"]:e["r1"], e["c0"]:e["c1"]] = True
        hit = None
        for c in gt_changes:
            m = Z[f"change_{c['id']}"]
            if (box & m).sum() / m.sum() >= 0.3:
                hit = c; break
        ok_type = hit is not None and e["type"] in TYPE_OK[hit["type"]]
        rows.append(dict(event=e["change_id"], type=e["type"], subtype=e["subtype"], conf=e["confidence"],
                         provisional=bool(e["provisional"]), earliest=e["earliest_obs"],
                         gt=hit["id"] if hit else None, gt_type=hit["type"] if hit else None, type_ok=ok_type,
                         earliest_ok=bool(hit) and e["earliest_obs"] == expected_earliest[hit["id"]]))
        if hit:
            tp_events += 1; matched.add(hit["id"])
    gt_union = np.zeros((H, W), bool)
    for c in gt_changes:
        gt_union |= Z[f"change_{c['id']}"]
    rep = res["mask"].astype(bool)  # the exact pixels the change engine reported
    S = np.stack([load_cache(st, s["scene_id"])[2] for s in scenes])
    tol = ndi.binary_dilation(gt_union, iterations=1)
    px_p = float((rep & tol).sum() / max(rep.sum(), 1)); px_r = float((rep & gt_union).sum() / gt_union.sum())

    # naive baseline: raw bi-temporal state differencing between consecutive dates, no masks /
    # co-registration / normalisation / persistence, same object grouping & min area
    grid = json.loads(st.meta()["grid"])
    raw_states = []
    for p in paths:
        sc = read_scene(p, grid)
        raw_states.append(classify_state(sc.arr, sc.valid))
    naive = np.zeros((H, W), bool)
    for i in range(1, len(raw_states)):
        naive |= raw_states[i] != raw_states[i - 1]
    naive = ndi.binary_opening(naive, structure=np.ones((2, 2)))
    lab, n = ndi.label(ndi.binary_dilation(naive, iterations=3)); lab *= naive
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    naive_objs = [i + 1 for i, s in enumerate(sizes) if s >= 15]
    naive_tp = sum(1 for i in naive_objs if ((lab == i) & gt_union).any())
    conf_zones = dict(crop_seasonal=Z["crop_zone"] & ~gt_union, haze=Z["haze_3"] & ~gt_union,
                      cloud_or_shadow=(Z["cloud_2"] | Z["cloud_6"] | Z["shadow_2"] | Z["shadow_6"]) & ~gt_union)
    R["change"] = dict(
        window=res["stats"]["window"], seconds=round(ch_s, 2), events=len(events),
        object_precision=round(tp_events / max(len(events), 1), 3),
        object_recall=round(len(matched) / len(gt_changes), 3),
        type_accuracy=round(np.mean([r["type_ok"] for r in rows if r["gt"]]), 3) if tp_events else 0,
        earliest_obs_accuracy=round(np.mean([r["earliest_ok"] for r in rows if r["gt"]]), 3) if tp_events else 0,
        expected_earliest=expected_earliest, pixel_precision=round(px_p, 3), pixel_recall=round(px_r, 3),
        false_alarm_px_in_confounder_zones={k: int((rep & v).sum()) for k, v in conf_zones.items()},
        naive_baseline=dict(objects=len(naive_objs), true_objects=naive_tp,
                            object_precision=round(naive_tp / max(len(naive_objs), 1), 3),
                            false_alarm_px_in_confounder_zones={k: int((naive & v).sum()) for k, v in conf_zones.items()}),
        suppression_ratio_vs_raw_flips=res["stats"]["suppression_ratio"],
        raw_state_flip_px=res["stats"]["naive_flip_px"], reported_px=res["stats"]["reported_px"], per_event=rows)

    # ---------- retrieval ----------
    tiles = [dict(r) for r in st.db.execute("SELECT * FROM tiles")]
    didx = {s["scene_id"]: i for i, s in enumerate(scenes)}

    def frac(mask, t):
        return mask[t["r0"]:t["r1"], t["c0"]:t["c1"]].mean()

    def lcf(t, k):
        return (Z[f"lc_{didx[t['scene_id']]}"][t["r0"]:t["r1"], t["c0"]:t["c1"]] == k).mean()
    C1, C4 = Z["change_C1"], Z["change_C4"]
    queries = {
        "river": lambda t: lcf(t, 1) >= 0.10,
        "water body": lambda t: lcf(t, 1) >= 0.10,
        "dense settlement": lambda t: frac(Z["town_zone"], t) >= 0.40,
        "forest": lambda t: frac(Z["forest_zone"], t) >= 0.50,
        "open ground": lambda t: lcf(t, 3) >= 0.50,
        "newly built structures near a river": lambda t: C1[t["r0"]:t["r1"], t["c0"]:t["c1"]].sum() >= 20 and t["acq_time"][:10] >= "2024-10-25",
        "new road": lambda t: C4[t["r0"]:t["r1"], t["c0"]:t["c1"]].sum() >= 20 and t["acq_time"][:10] >= "2025-04-15",
    }
    qres, lat = {}, []
    for q, rel in queries.items():
        n_rel_locs = len({(t["row"], t["col"]) for t in tiles if rel(t)})
        r = search_text(st, emb, q, k=10)
        rels = [1 if rel(x) else 0 for x in r["results"]]
        k5 = min(5, n_rel_locs); k10 = min(10, n_rel_locs)
        qres[q] = dict(relevant_locations=n_rel_locs, returned=len(rels),
                       p_at_5=round(sum(rels[:5]) / max(k5, 1), 3), p_at_10=round(sum(rels[:10]) / max(k10, 1), 3),
                       ndcg_at_10=round(ndcg(rels + [0] * (10 - len(rels))), 3))
    town_seed = next(t for t in tiles if frac(Z["town_zone"], t) >= 0.6)
    ri = search_image(st, town_seed["tile_id"], k=10)
    rels = [1 if frac(Z["town_zone"], x) >= 0.3 else 0 for x in ri["results"]]
    n_rel = len({(t["row"], t["col"]) for t in tiles if frac(Z["town_zone"], t) >= 0.3} - {(town_seed["row"], town_seed["col"])})
    qres["image→image (settlement seed)"] = dict(relevant_locations=n_rel, returned=len(rels),
                                                 p_at_5=round(sum(rels[:5]) / max(min(5, n_rel), 1), 3),
                                                 p_at_10=round(sum(rels[:10]) / max(min(10, n_rel), 1), 3),
                                                 ndcg_at_10=round(ndcg(rels + [0] * (10 - len(rels))), 3))
    for i in range(200):  # latency (end-to-end planner + ANN + joins)
        q = list(queries)[i % len(queries)]
        t = time.perf_counter(); search_text(st, emb, q, k=10); lat.append((time.perf_counter() - t) * 1000)
    ann = []
    V = np.asarray(st.vectors())
    for i in range(500):
        t = time.perf_counter(); st.search(V[i % len(V)], 10); ann.append((time.perf_counter() - t) * 1000)
    R["retrieval"] = dict(queries=qres, macro_p_at_10=round(np.mean([v["p_at_10"] for v in qres.values()]), 3),
                          note="relevance judged from ground-truth land cover / change masks never seen by the system")
    R["latency_ms"] = dict(text_query_end_to_end_p50=round(np.percentile(lat, 50), 2),
                           text_query_end_to_end_p95=round(np.percentile(lat, 95), 2),
                           ann_only_p50=round(np.percentile(ann, 50), 3), ann_only_p95=round(np.percentile(ann, 95), 3))
    sb = st.storage_bytes()
    R["storage"] = dict(total_mb=round(sum(sb.values()) / 1048576, 2), files={k: v for k, v in sb.items()},
                        bytes_per_tile_index_and_vectors=int((sb.get("index.hnsw.faiss", 0) + sb.get("vectors.f32", 0)) / st.index.ntotal))

    # ---------- index-only scale benchmark at CLIP dimensionality ----------
    d, N = 512, a.scale_n
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((64, d)).astype(np.float32)
    X = centers[rng.integers(0, 64, N)] + 0.6 * rng.standard_normal((N, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    base = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = 80
    idx = faiss.IndexIDMap2(base)
    t = time.perf_counter(); idx.add_with_ids(X[: N - N // 10], np.arange(N - N // 10)); b_s = time.perf_counter() - t
    t = time.perf_counter(); idx.add_with_ids(X[N - N // 10:], np.arange(N - N // 10, N)); inc = time.perf_counter() - t
    Q = X[rng.integers(0, N, 200)] + 0.05 * rng.standard_normal((200, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    gt_I = np.argsort(-(Q @ X.T), axis=1)[:, :10]
    lat2 = []; rec = []
    p = faiss.SearchParametersHNSW(efSearch=128)
    for i in range(len(Q)):
        t = time.perf_counter(); _, I = idx.search(Q[i:i + 1], 10, params=p); lat2.append((time.perf_counter() - t) * 1000)
        rec.append(len(set(I[0]) & set(gt_I[i])) / 10)
    faiss.write_index(idx, "/tmp/scale.faiss")
    R["scale_benchmark"] = dict(vectors=N, dim=d, index="HNSW M=32 efC=80 efS=128", build_seconds=round(b_s, 1),
                                incremental_add_10pct_seconds=round(inc, 1), recall_at_10=round(float(np.mean(rec)), 3),
                                query_p50_ms=round(np.percentile(lat2, 50), 2), query_p95_ms=round(np.percentile(lat2, 95), 2),
                                index_mb=round(os.path.getsize("/tmp/scale.faiss") / 1048576, 1),
                                note="index-only; tiles at 256 px/10 m: 100k tiles ≈ 655,000 km² of Sentinel-2 coverage per date")
    os.makedirs(a.out, exist_ok=True)
    json.dump(R, open(os.path.join(a.out, "evaluation_report.json"), "w"), indent=2, default=str)
    write_md(R, os.path.join(a.out, "EVALUATION_REPORT.md"))
    print(json.dumps({k: R[k] for k in ("index", "latency_ms", "storage")}, indent=1, default=str))
    print(json.dumps({k: v for k, v in R["change"].items() if k != "per_event"}, indent=1, default=str))
    print(json.dumps(R["retrieval"], indent=1)); print(json.dumps(R["scale_benchmark"], indent=1))


def write_md(R, path):
    c, i, l, s, r, sb, h = R["change"], R["index"], R["latency_ms"], R["storage"], R["retrieval"], R["scale_benchmark"], R["hardware"]
    L = ["# DRISHTI reproducible evaluation report", "",
         f"Generated by `scripts/evaluate.py`. Embedder: `{R['embedder']}`. Data: {R['dataset']['source']}.", "",
         "## Indexed archive", "", "| Item | Value |", "|---|---|",
         f"| Indexed area | {i['area_km2']:.2f} km² |", f"| Scenes | {i['scenes']} (Sentinel-2-like 10 m + Landsat-like 30 m) |",
         f"| Tiles | {i['tiles']} ({i['tile_px']} px = {i['tile_m']:.0f} m) |",
         f"| Build time (first {i['scenes'] - 1} scenes, end to end) | {i['build_seconds_first_n_minus_1']} s |",
         f"| Incremental ingest of 1 new scene | {i['incremental_ingest_seconds']} s ({i['tiles_before_incremental']} → {i['tiles_after_incremental']} tiles, no rebuild) |",
         f"| Re-ingesting the same file | skipped by SHA-256: {i['reingest_same_file_skipped']} |",
         f"| Storage footprint | {s['total_mb']} MB total; {s['bytes_per_tile_index_and_vectors']} B/tile for index + vectors |",
         f"| Text query latency (planner + ANN + joins) | p50 {l['text_query_end_to_end_p50']} ms, p95 {l['text_query_end_to_end_p95']} ms |",
         f"| ANN only | p50 {l['ann_only_p50']} ms, p95 {l['ann_only_p95']} ms |",
         f"| Hardware | {h['cpu']}, {h['logical_cores']} logical core(s), {h['ram_gb']} GB RAM, {h['gpu']} |", "",
         "## Change analysis (generator ground truth, never seen by the pipeline)", "", "| Metric | DRISHTI | Naive bi-temporal baseline |", "|---|---|---|",
         f"| Reported objects | {c['events']} | {c['naive_baseline']['objects']} |",
         f"| Object precision | {c['object_precision']} | {c['naive_baseline']['object_precision']} |",
         f"| Object recall | {c['object_recall']} | – |",
         f"| Change-type accuracy | {c['type_accuracy']} | – |",
         f"| Earliest-observation accuracy | {c['earliest_obs_accuracy']} | – |",
         f"| Pixel precision / recall | {c['pixel_precision']} / {c['pixel_recall']} | – |"]
    for k in c["false_alarm_px_in_confounder_zones"]:
        L.append(f"| False-alarm px in {k.replace('_', ' ')} zone | {c['false_alarm_px_in_confounder_zones'][k]} | {c['naive_baseline']['false_alarm_px_in_confounder_zones'][k]:,} |")
    L += ["", f"Raw date-to-date state flips: {c['raw_state_flip_px']:,} px; reported as change: {c['reported_px']:,} px "
          f"({c['suppression_ratio_vs_raw_flips'] * 100:.1f}% suppressed).", "",
          "| Event | Type / subtype | Confidence | Earliest obs. | Matches GT | Type OK | Earliest OK |", "|---|---|---|---|---|---|---|"]
    for e in c["per_event"]:
        L.append(f"| {e['event']} | {e['type']} / {e['subtype']}{' (provisional)' if e['provisional'] else ''} | {e['conf']} | {e['earliest']} | {e['gt']} | {e['type_ok']} | {e['earliest_ok']} |")
    L += ["", "## Retrieval (relevance judged from generator ground truth)", "", "| Query | Relevant locations | P@5 | P@10 | nDCG@10 |", "|---|---|---|---|---|"]
    for q, v in r["queries"].items():
        L.append(f"| {q} | {v.get('relevant_locations', '–')} | {v.get('p_at_5', '–')} | {v['p_at_10']} | {v.get('ndcg_at_10', '–')} |")
    L += ["", f"Macro P@10: {r['macro_p_at_10']}. P@k is capped by the number of relevant locations when fewer than k exist.", "",
          "## Index scale benchmark (synthetic clustered 512-d vectors, CLIP-sized; not real embeddings)", "", "| Item | Value |", "|---|---|",
          f"| Vectors × dim | {sb['vectors']:,} × {sb['dim']} |", f"| Index | {sb['index']} |",
          f"| Build (first 90%) | {sb['build_seconds']} s; incremental +10%: {sb['incremental_add_10pct_seconds']} s |",
          f"| Recall@10 vs exact | {sb['recall_at_10']} |", f"| Query p50 / p95 | {sb['query_p50_ms']} / {sb['query_p95_ms']} ms |",
          f"| Index size | {sb['index_mb']} MB |", "", f"Note: {sb['note']}.", "",
          "## Honest limitations", "",
          "- Demo data is a simulated Sentinel-2-style AOI (seeded generator). Real Sentinel-2/Landsat scenes are messier; the organiser AOI will be run through the same script.",
          "- Robustness has been tested as confounder variation only (`scripts/seed_sweep.py`: new noise, cloud/shadow fields, building footprints, field textures on the same layout). It has NOT been tested on unseen geographic regions.",
          "- Perfect scores here reflect a controlled test set with known confounders; they are an upper bound, not an expected field accuracy.",
          "- Vehicle concentrations are NOT supported today: the demo embedder only understands concepts in its lexicon, and 10 m pixels are too coarse for vehicles. This needs the RemoteCLIP plug-in (or a fine-tuned successor) and sub-metre imagery.",
          "- Land-state rules are hand-set; production swaps in a classifier head trained on ESA WorldCover / Dynamic World labels."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
