# DRISHTI — offline semantic search and change intelligence for satellite archives

Prototype for **Smart India Hackathon 2026, PS SIH26227** — *Semantic Retrieval and Multi-Temporal Change Analysis
of Satellite Imagery* (Ministry of Defence, Indian Army / DGIS). Team **Error_404$** (Team ID 124612).

DRISHTI lets an analyst ask an image archive questions like *"newly built structures near a river"*, returns ranked
locations with the date each change was first visible, shows before/after evidence, and records every decision in a
tamper-evident audit trail. It runs fully on-prem with no network access.

> **Read this first — scope of the evidence.** All results in this repository come from a **simulated
> Sentinel-2-style area of interest** produced by `scripts/make_synthetic_aoi.py` (40.96 km², 8 dates, injected
> cloud, shadow, haze, seasonal crop cycle, radiometric gain, mis-registration and a 30 m Landsat-like scene).
> Robustness was tested as **confounder variation** (different seeds on the same layout), **not** on unseen
> geographic regions. The demo used DRISHTI's own **lite spectral embedder**; the RemoteCLIP path is implemented but
> was not executed in our build environment (no PyTorch, no access to the weights). **Vehicle concentrations are not
> supported yet** (they need sub-metre imagery and a RemoteCLIP-class model). See `docs/EVALUATION_REPORT.md`.

---

## 1. What is in the box

| Path | What it is |
|---|---|
| `drishti/raster.py` | Sensor-agnostic GeoTIFF/COG reader: band aliases (Sentinel-2 B02–B11, Landsat C2 SR_B2–B6, named bands), SCALE/OFFSET, optional SCL band, reprojection onto the AOI grid, SHA-256 |
| `drishti/quality.py` | Cloud / shadow / snow mask (Sentinel-2 SCL if present, else spectral tests + sun-geometry shadow), phase-correlation co-registration, class-consistent PIF radiometric normalisation with IR-MAD-style reweighting, land-state classification |
| `drishti/embed.py` | `LiteEmbedder` (used in the demo, no weights) and `RemoteCLIPEmbedder` (OpenCLIP, loads only SHA-256-pinned local weights, tensors only) |
| `drishti/store.py` | SQLite metadata + FAISS `IndexIDMap2(HNSWFlat)`; incremental `add_with_ids`; filtered search; hash-chained audit log |
| `drishti/ingest.py` | Idempotent per-scene ingest (skips a file whose SHA-256 is already indexed), processing provenance per step |
| `drishti/change.py` | Time-series change engine: per-pixel break detection, persistence and cloud-proximity hold, typing, earliest usable observation, supersede-on-new-data |
| `drishti/query.py` | Query planner (subject / "near X" / "new"), neighbour relation check, change-table join, image-to-image search, Rocchio feedback |
| `drishti/review.py` | HDBSCAN site clustering, logistic-regression feedback re-ranking, decisions, GeoJSON export with provenance |
| `drishti/api.py` + `web/index.html` | FastAPI service + offline analyst UI (no CDN) |
| `scripts/` | `make_synthetic_aoi.py`, `build_index.py`, `evaluate.py`, `seed_sweep.py`, `stack_bands.py`, `selftest.py`, `screenshots.py` |
| `docs/` | `ARCHITECTURE.md`, `PROVENANCE.md`, `EVALUATION_REPORT.md` (+ `.json`), `SEED_SWEEP.md`, `screens/` |

## 2. Install (online machine)

Python 3.10+ (tested on 3.12.3, Linux x86-64).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras: `requirements-optional.txt` (RemoteCLIP embedder: PyTorch + OpenCLIP; UI screenshots: Playwright).

## 3. Install on an air-gapped machine

On a connected machine with the same OS / Python version:

```bash
pip download -r requirements.txt -d wheels/          # add requirements-optional.txt if you want RemoteCLIP
```

Copy the repository and `wheels/` across, then on the air-gapped machine:

```bash
pip install --no-index --find-links wheels/ -r requirements.txt
```

Nothing in DRISHTI downloads anything at runtime. The UI is plain HTML/JS served by the API (no CDN); the screenshot
script records every non-loopback request the UI makes and our run recorded **0**.

### Stage model weights (optional, RemoteCLIP)

1. On a connected machine, obtain `RemoteCLIP-ViT-B-32.pt` from the authors' release (Hugging Face repo
   `chendelong/RemoteCLIP`, see `docs/PROVENANCE.md`) and check its licence.
2. Pin it: `sha256sum RemoteCLIP-ViT-B-32.pt > RemoteCLIP-ViT-B-32.pt.sha256`
3. Copy both files into `models/`.

`get_embedder()` loads RemoteCLIP only if the checkpoint **and** its `.sha256` pin are present and match, and it
loads tensors only (`torch.load(..., weights_only=True)`). Otherwise it falls back to the lite embedder.
An index is tied to one embedder (dimension and name are stored), so rebuild the store after switching.

## 4. Quickstart with the simulated AOI (about 1 minute on 1 CPU core)

One command: `./run_demo.sh` (generates the data, builds the index, adds the 8th scene incrementally, starts the
UI on port 8000). Or step by step, from the repository root:

```bash
python scripts/make_synthetic_aoi.py                 # -> data/raw/*.tif (8 dates) + data/gt/
# build the index from the first 7 dates, then add the 8th incrementally (no rebuild)
python scripts/build_index.py --src data/raw --store data/store --exclude S2_20251022_SYN_AOI.tif
python scripts/build_index.py --add data/raw/S2_20251022_SYN_AOI.tif --store data/store
DRISHTI_STORE=data/store uvicorn drishti.api:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Try: *newly built structures near a river*, *new road*, *forest*, *dense settlement*; open **Change analysis**,
select the provisional *Construction · expansion* event to see its evidence strip (cloud-skipped date), confirm or
reject events in **Review queue**, export GeoJSON, and check the audit chain under **Provenance and system**.

## 5. Index build and incremental update procedure

* **Build:** `python scripts/build_index.py --src <folder of .tif> --store <store dir>`. Scenes are ingested in
  acquisition-time order. The first clear scene becomes the reference for co-registration and normalisation and
  defines the AOI grid; later scenes are reprojected onto it.
* **Incremental add:** `python scripts/build_index.py --add <new.tif> [...] --store <store dir>` (or
  `POST /api/ingest {"path": ...}`). New tiles are appended to the HNSW graph with `add_with_ids`; the existing
  index is never rebuilt. Change analysis re-runs automatically; alerts that the new imagery no longer supports are
  marked *superseded* (kept for audit, hidden from the queue).
* **Idempotency:** a file whose SHA-256 is already in the store is skipped.
* **What is stored per scene:** file SHA-256, platform, sensor, acquisition time, native resolution, usable
  fraction, co-registration shift, per-band radiometric gain/offset, pipeline version, and every processing step.
* **Options:** `--embedder auto|lite|remoteclip`, `--no-analyze`, `--exclude <basename ...>`.

## 6. Using real imagery

DRISHTI ingests a GeoTIFF/COG with five bands **blue, green, red, nir, swir16** (identified by band description,
else by position), reflectance = `DN × SCALE + OFFSET`, and tag `ACQUISITION_DATETIME`. `scripts/stack_bands.py`
builds that file from single-band products you already have, e.g. Sentinel-2 L2A (`B02 B03 B04 B08 B11`, optional
`SCL`) or Landsat 8/9 C2 L2 (`SR_B2..SR_B6`, `SCALE 2.75e-5`, `OFFSET -0.2`); run it with `-h` for examples.
All scenes of one AOI must overlap the reference scene's footprint.

*Status:* the reader, the SCL path, OFFSET handling and `stack_bands.py` are covered by `scripts/selftest.py` on files
derived from the simulated AOI. They
have **not yet been run on real Sentinel-2 / Landsat downloads**; that is the next step (see `docs/EVALUATION_REPORT.md`).
Bhuvan / IRS band mappings and Sentinel-1 SAR are roadmap items.

## 7. Self-test and reproducible evaluation

```bash
python scripts/selftest.py                    # ~20 s: read paths, SCL, OFFSET, incremental ingest, change, search, export, audit
python scripts/evaluate.py                    # rebuilds data/eval_store, writes docs/EVALUATION_REPORT.md + .json
python scripts/seed_sweep.py --seeds 11 42 7  # confounder-variation sweep -> docs/SEED_SWEEP.md
```

`evaluate.py` reports area, number of scenes and tiles, build time, incremental ingest time, storage, query latency
(end-to-end and ANN-only), hardware, change metrics against ground truth (object and pixel precision/recall, type
accuracy, earliest-observation accuracy, false-alarm pixels in confounder zones, naive-differencing baseline),
retrieval P@5 / P@10 / nDCG@10, and a 100k × 512-d HNSW scale benchmark. Use `--scale-n` to shrink the benchmark on
small machines. Timings vary run to run on shared hardware; accuracy metrics are deterministic for a given seed.

## 8. API summary

`GET /api/stats` · `GET /api/scenes` · `POST /api/search/text {"query", "filters", "k", "positives", "negatives"}` · `POST /api/search/image
{"tile_id"}` · `GET /api/plan?q=` · `POST /api/change/run` · `GET /api/changes` · `GET /api/queue` ·
`POST /api/decision` · `GET /api/audit` · `GET /api/discover/clusters` · `GET /api/tile/{id}` ·
`GET /api/timeline?change_id=` · `POST /api/ingest {"path"}` · `POST /api/export` (GeoJSON) · images under `/img/`.

## 9. Known limitations

* Evidence is from simulated data; perfect scores there are an upper bound, not expected field accuracy.
* Vehicle concentrations / small objects: not supported by the lite embedder, and 10 m pixels are too coarse.
* The lite embedder understands only the concepts in its lexicon (water, vegetation, forest, bare, built, road,
  texture, shore); open-vocabulary queries need RemoteCLIP or a fine-tuned successor.
* Radiometric normalisation is linear per band; strong BRDF / terrain effects are not modelled yet.
* Single AOI grid per store; multi-AOI deployments use one store per AOI (or sharding by region).

## 10. Licence

Source code: MIT (see `LICENSE`). Third-party components and data licences: `docs/PROVENANCE.md`.
