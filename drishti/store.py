"""Persistent store: SQLite (metadata, provenance, candidates, decisions, hash-chained audit)
+ FAISS HNSW vector index (append-only, incremental: no rebuild on new acquisitions)
+ raw vector file (exact re-ranking and small filtered searches)."""
from __future__ import annotations
import os, json, time, sqlite3, hashlib, threading
import numpy as np
import faiss

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS scenes (
  scene_id TEXT PRIMARY KEY, path TEXT, sha256 TEXT UNIQUE, platform TEXT, sensor TEXT,
  acq_time TEXT, product_level TEXT, native_res_m REAL, crs TEXT, cloud_frac REAL,
  usable_frac REAL, coreg_dy REAL, coreg_dx REAL, coreg_resp REAL, norm_coefs TEXT,
  n_pif INTEGER, is_reference INTEGER, cache_path TEXT, ingested_at TEXT, processing TEXT);
CREATE TABLE IF NOT EXISTS tiles (
  tile_id INTEGER PRIMARY KEY, scene_id TEXT, row INTEGER, col INTEGER,
  r0 INTEGER, c0 INTEGER, r1 INTEGER, c1 INTEGER, x_min REAL, y_min REAL, x_max REAL, y_max REAL,
  lon REAL, lat REAL, acq_time TEXT, sensor TEXT, platform TEXT, usable_frac REAL,
  water REAL, veg REAL, bare REAL, built REAL, cluster INTEGER DEFAULT -1);
CREATE INDEX IF NOT EXISTS ix_tiles_time ON tiles(acq_time);
CREATE INDEX IF NOT EXISTS ix_tiles_loc ON tiles(row, col);
CREATE TABLE IF NOT EXISTS changes (
  change_id TEXT PRIMARY KEY, run_id TEXT, type TEXT, subtype TEXT, status TEXT,
  confidence REAL, score REAL, provisional INTEGER, area_m2 REAL, r0 INTEGER, c0 INTEGER,
  r1 INTEGER, c1 INTEGER, lon REAL, lat REAL, x_min REAL, y_min REAL, x_max REAL, y_max REAL,
  before_scene TEXT, after_scene TEXT, earliest_obs TEXT, last_pre_obs TEXT, detail TEXT,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, change_id TEXT, decision TEXT, analyst TEXT, note TEXT,
  ts TEXT, audit_hash TEXT);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, payload TEXT,
  prev_hash TEXT, hash TEXT);
"""


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, root: str, dim: int | None = None, embedder_name: str | None = None):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.lock = threading.RLock()
        self.db = sqlite3.connect(os.path.join(root, "drishti.sqlite"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.index_path = os.path.join(root, "index.hnsw.faiss")
        self.vec_path = os.path.join(root, "vectors.f32")
        m = self.meta()
        if "dim" in m:
            self.dim = int(m["dim"])
            if embedder_name and m.get("embedder") != embedder_name:
                raise RuntimeError(f"Index built with {m.get('embedder')}, not {embedder_name}. "
                                   "Use a separate store root per embedder.")
        else:
            if dim is None:
                raise RuntimeError("new store needs dim")
            self.dim = dim
            self.set_meta(dim=dim, embedder=embedder_name or "unknown", created_at=now_iso())
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            base = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efConstruction = 80
            self.index = faiss.IndexIDMap2(base)
        self._vecs = None

    # ---------- meta ----------
    def meta(self) -> dict:
        return {r["key"]: r["value"] for r in self.db.execute("SELECT key,value FROM meta")}

    def set_meta(self, **kv):
        with self.lock:
            for k, v in kv.items():
                self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                                (k, v if isinstance(v, str) else json.dumps(v)))
            self.db.commit()

    # ---------- vectors ----------
    def vectors(self) -> np.ndarray:
        n = os.path.getsize(self.vec_path) // (4 * self.dim) if os.path.exists(self.vec_path) else 0
        if self._vecs is None or len(self._vecs) != n:
            self._vecs = (np.memmap(self.vec_path, dtype=np.float32, mode="r", shape=(n, self.dim))
                          if n else np.zeros((0, self.dim), np.float32))
        return self._vecs

    def add_tiles(self, rows: list[dict], vecs: np.ndarray):
        with self.lock:
            start = self.db.execute("SELECT COALESCE(MAX(tile_id),-1)+1 FROM tiles").fetchone()[0]
            assert start == len(self.vectors()), "vector file / table out of sync"
            ids = np.arange(start, start + len(rows), dtype=np.int64)
            cols = list(rows[0].keys())
            self.db.executemany(
                f"INSERT INTO tiles (tile_id,{','.join(cols)}) VALUES (?,{','.join('?' * len(cols))})",
                [(int(i), *[r[c] for c in cols]) for i, r in zip(ids, rows)])
            with open(self.vec_path, "ab") as f:
                f.write(np.ascontiguousarray(vecs, np.float32).tobytes())
            self.index.add_with_ids(np.ascontiguousarray(vecs, np.float32), ids)  # incremental
            faiss.write_index(self.index, self.index_path)
            self.db.commit()
            self._vecs = None
            return ids

    # ---------- filtered search ----------
    def filter_ids(self, f: dict | None):
        if not f:
            return None
        where, args = [], []
        if f.get("date_from"): where.append("acq_time >= ?"); args.append(f["date_from"])
        if f.get("date_to"): where.append("acq_time <= ?"); args.append(f["date_to"] + "T23:59:59Z")
        if f.get("sensors"):
            where.append(f"sensor IN ({','.join('?' * len(f['sensors']))})"); args += f["sensors"]
        if f.get("bbox"):  # lon_min, lat_min, lon_max, lat_max
            a, b, c, d = f["bbox"]
            where.append("lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?"); args += [a, c, b, d]
        if f.get("scene_ids"):
            where.append(f"scene_id IN ({','.join('?' * len(f['scene_ids']))})"); args += f["scene_ids"]
        if f.get("min_usable") is not None: where.append("usable_frac >= ?"); args.append(f["min_usable"])
        if not where:
            return None
        return np.array([r[0] for r in self.db.execute(
            f"SELECT tile_id FROM tiles WHERE {' AND '.join(where)}", args)], np.int64)

    def search(self, q: np.ndarray, k: int = 20, filters: dict | None = None, exact_limit: int = 20000):
        """Adaptive: selective filters → exact scan of the candidate subset; otherwise HNSW
        with an ID selector (no post-filter recall loss)."""
        q = np.ascontiguousarray(q.reshape(1, -1), np.float32)
        ids = self.filter_ids(filters)
        if ids is not None and len(ids) == 0:
            return np.array([]), np.array([], np.int64)
        if ids is not None and len(ids) <= exact_limit:
            s = self.vectors()[ids] @ q[0]
            o = np.argsort(-s)[:k]
            return s[o], ids[o]
        k2 = min(k, self.index.ntotal)
        if ids is None:
            D, I = self.index.search(q, k2, params=faiss.SearchParametersHNSW(efSearch=max(64, 2 * k2)))
        else:
            params = faiss.SearchParametersHNSW(sel=faiss.IDSelectorBatch(ids), efSearch=max(128, 4 * k2))
            D, I = self.index.search(q, k2, params=params)
        m = I[0] >= 0
        return D[0][m], I[0][m]

    # ---------- rows ----------
    def tiles(self, ids) -> list[dict]:
        if len(ids) == 0:
            return []
        rows = self.db.execute(f"SELECT * FROM tiles WHERE tile_id IN ({','.join(str(int(i)) for i in ids)})")
        d = {r["tile_id"]: dict(r) for r in rows}
        return [d[int(i)] for i in ids if int(i) in d]

    def scenes(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM scenes ORDER BY acq_time")]

    def scene(self, scene_id) -> dict | None:
        r = self.db.execute("SELECT * FROM scenes WHERE scene_id=?", (scene_id,)).fetchone()
        return dict(r) if r else None

    # ---------- audit (tamper-evident hash chain) ----------
    def audit(self, actor: str, action: str, payload: dict) -> str:
        with self.lock:
            last = self.db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last[0] if last else "GENESIS"
            ts = now_iso()
            body = json.dumps(payload, sort_keys=True, default=str)
            h = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{body}".encode()).hexdigest()
            self.db.execute("INSERT INTO audit (ts,actor,action,payload,prev_hash,hash) VALUES (?,?,?,?,?,?)",
                            (ts, actor, action, body, prev, h))
            self.db.commit()
            return h

    def verify_audit(self) -> dict:
        prev = "GENESIS"; n = 0
        for r in self.db.execute("SELECT * FROM audit ORDER BY seq"):
            h = hashlib.sha256(f"{prev}|{r['ts']}|{r['actor']}|{r['action']}|{r['payload']}".encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return dict(ok=False, broken_at=r["seq"], checked=n)
            prev = h; n += 1
        return dict(ok=True, checked=n, head=prev)

    def storage_bytes(self) -> dict:
        out = {}
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isfile(p):
                out[name] = os.path.getsize(p)
            elif os.path.isdir(p):
                out[name + "/"] = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
        return out
