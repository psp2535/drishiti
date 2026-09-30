#!/usr/bin/env python3
"""Robustness sweep: regenerate the synthetic AOI with different seeds (new noise, cloud/shadow fields,
building footprints and field textures) and re-run the full evaluation on each.

  python scripts/seed_sweep.py --seeds 11 42 7 --work /tmp/drishti_sweep --out docs
"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["events", "object_precision", "object_recall", "type_accuracy", "earliest_obs_accuracy",
        "pixel_precision", "pixel_recall"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 7])
ap.add_argument("--work", default="/tmp/drishti_sweep"); ap.add_argument("--out", default="docs")
a = ap.parse_args()
rows = []
for sd in a.seeds:
    w = os.path.join(a.work, f"seed{sd}")
    subprocess.run([sys.executable, f"{HERE}/scripts/make_synthetic_aoi.py", "--seed", str(sd),
                    "--out", f"{w}/raw", "--gt", f"{w}/gt"], check=True, capture_output=True)
    subprocess.run([sys.executable, f"{HERE}/scripts/evaluate.py", "--raw", f"{w}/raw", "--gt", f"{w}/gt",
                    "--store", f"{w}/store", "--out", f"{w}/docs", "--scale-n", "2000"], check=True, capture_output=True)
    R = json.load(open(f"{w}/docs/evaluation_report.json")); c = R["change"]
    rows.append(dict(seed=sd, **{k: c[k] for k in KEYS}, fa=c["false_alarm_px_in_confounder_zones"],
                     naive_fa=c["naive_baseline"]["false_alarm_px_in_confounder_zones"],
                     macro_p10=R["retrieval"]["macro_p_at_10"]))
    print(rows[-1], flush=True)
L = ["# Seed-robustness sweep", "", "Same AOI layout and change sites; every seed redraws sensor noise, cloud and shadow fields,",
     "building footprints and field textures (gain, mis-registration and haze stay fixed). Command: `python scripts/seed_sweep.py`.", "",
     "| Seed | Events | Obj P | Obj R | Type acc | Earliest acc | Pixel P / R | FA px crop / haze / cloud | Naive FA px crop / haze / cloud | Macro P@10 |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    f, n = r["fa"], r["naive_fa"]
    L.append(f"| {r['seed']} | {r['events']} | {r['object_precision']} | {r['object_recall']} | {r['type_accuracy']} | "
             f"{r['earliest_obs_accuracy']} | {r['pixel_precision']} / {r['pixel_recall']} | "
             f"{f['crop_seasonal']} / {f['haze']} / {f['cloud_or_shadow']} | "
             f"{n['crop_seasonal']:,} / {n['haze']:,} / {n['cloud_or_shadow']:,} | {r['macro_p10']} |")
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "SEED_SWEEP.md"), "w").write("\n".join(L) + "\n")
json.dump(rows, open(os.path.join(a.out, "seed_sweep.json"), "w"), indent=1)
