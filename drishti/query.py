"""Change-aware semantic retrieval.

A natural-language query is split by a light planner into
  subject   : what the tile must look like             → embedding similarity (ANN)
  relation  : "near / along / beside <X>"               → X must be present in the tile or its
                                                           8 neighbours at the SAME acquisition
  temporal  : "new / newly / recent / expanded …"       → tile must be covered by a detected
                                                           change event of a matching type that
                                                           happened on or before that acquisition
So "newly built structures near a river" ≠ "built structures": it is answered by fusing the
vector index, the tile grid topology and the change-event table. Works identically with the
RemoteCLIP embedder (subject/relation text go through the CLIP text tower).
"""
from __future__ import annotations
import re
import numpy as np
from .store import Store

TEMPORAL = {"new", "newly", "recent", "recently", "fresh", "emerging", "appeared", "expanded",
            "expanding", "latest", "since"}
RELATIONS = ["next to", "adjacent to", "close to", "near", "along", "beside", "by the", "around"]
CHANGE_FOR = {"built": ["construction"], "structure": ["construction"], "building": ["construction"],
              "settlement": ["construction"], "road": ["road"], "track": ["road"], "water": ["water_increase"],
              "reservoir": ["water_increase"], "pond": ["water_increase"], "flood": ["water_increase", "inundation"],
              "cleared": ["clearance"], "clearing": ["clearance"], "bare": ["clearance"], "forest": ["clearance"]}


def plan(text: str) -> dict:
    t = " " + text.lower().strip() + " "
    toks = set(re.findall(r"[a-z]+", t))
    temporal = bool(toks & TEMPORAL)
    subject, relation = t, None
    for r in RELATIONS:
        k = t.find(" " + r + " ")
        if k >= 0:
            subject, relation = t[:k], t[k + len(r) + 2:]
            break
    subj_clean = " ".join(w for w in re.findall(r"[a-z\-]+", subject) if w not in TEMPORAL)
    change_types = sorted({c for w in re.findall(r"[a-z]+", subject) for key, cs in CHANGE_FOR.items()
                           if w.startswith(key) for c in cs}) if temporal else []
    return dict(query=text, subject=subj_clean.strip() or text, relation=(relation or "").strip() or None,
                temporal=temporal, change_types=change_types)


def rocchio(q, V_pos, V_neg, a=1.0, b=0.6, c=0.3):
    v = a * q
    if len(V_pos): v = v + b * V_pos.mean(0)
    if len(V_neg): v = v - c * V_neg.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def search_text(store: Store, emb, text: str, filters: dict | None = None, k: int = 12,
                positives=(), negatives=(), dedupe: bool = True) -> dict:
    p = plan(text)
    V = store.vectors()
    q = emb.embed_text(p["subject"])
    q = rocchio(q, V[list(positives)] if positives else [], V[list(negatives)] if negatives else [])
    n_cand = min(store.index.ntotal, max(200, k * 25))
    D, I = store.search(q, n_cand, filters)
    rows = store.tiles(I)
    sim = {int(i): float(d) for d, i in zip(D, I)}
    rel_q = emb.embed_text(p["relation"]) if p["relation"] else None
    out = []
    for r in rows:
        score = sim[r["tile_id"]]
        why = [f"semantic {score:.2f}"]
        if rel_q is not None:
            nb = store.db.execute(
                "SELECT tile_id FROM tiles WHERE scene_id=? AND row BETWEEN ? AND ? AND col BETWEEN ? AND ?",
                (r["scene_id"], r["row"] - 1, r["row"] + 1, r["col"] - 1, r["col"] + 1)).fetchall()
            rs = float(max(V[[n[0] for n in nb]] @ rel_q))
            score += 0.35 * rs
            why.append(f"'{p['relation']}' nearby {rs:.2f}")
            if rs < 0.3:
                continue
        if p["temporal"]:
            ct = p["change_types"] or ["construction", "road", "water_increase", "clearance"]
            ev = store.db.execute(
                f"SELECT change_id,type,earliest_obs,confidence FROM changes WHERE type IN ({','.join('?' * len(ct))}) "
                "AND status != 'rejected' AND earliest_obs <= ? AND NOT (x_max < ? OR x_min > ? OR y_max < ? OR y_min > ?)",
                (*ct, r["acq_time"][:10], r["x_min"], r["x_max"], r["y_min"], r["y_max"])).fetchall()
            if not ev:
                continue
            best = max(ev, key=lambda e: e[3])
            score += 0.5 * best[3]
            why.append(f"change {best[1]} since {best[2]} (conf {best[3]:.2f})")
            r["change_id"] = best[0]
        u = float(r.get("usable_frac") or 0.0)
        if u < 0.25:  # mostly cloud/shadow: nothing an analyst could verify
            continue
        if u < 0.95:  # cloud / shadow / haze-masked tiles are demoted so the clearest view of a place wins
            score -= 0.6 * (1 - u)
            why.append(f"usable {u:.0%}")
        r["score"] = round(score, 4); r["why"] = why
        out.append(r)
    out.sort(key=lambda r: -r["score"])
    if dedupe:  # one hit per location (best acquisition)
        seen, ded = set(), []
        for r in out:
            if (r["row"], r["col"]) not in seen:
                seen.add((r["row"], r["col"])); ded.append(r)
        out = ded
    return dict(plan=p, results=out[:k], candidates=len(rows))


def search_image(store: Store, tile_id: int, filters: dict | None = None, k: int = 12,
                 exclude_same_location: bool = True) -> dict:
    V = store.vectors()
    q = V[tile_id]
    seed = store.tiles([tile_id])[0]
    D, I = store.search(q, min(store.index.ntotal, k * 12), filters)
    rows = store.tiles(I)
    sim = {int(i): float(d) for d, i in zip(D, I)}
    seen, out = set(), []
    for r in rows:
        sim[r["tile_id"]] -= 0.6 * (1 - float(r.get("usable_frac") or 0.0))
    for r in sorted(rows, key=lambda r: -sim[r["tile_id"]]):
        loc = (r["row"], r["col"])
        if exclude_same_location and loc == (seed["row"], seed["col"]):
            continue
        if loc in seen:
            continue
        seen.add(loc)
        r["score"] = round(sim[r["tile_id"]], 4); r["why"] = [f"visual+semantic {r['score']:.2f}"]
        out.append(r)
    return dict(seed=seed, results=out[:k])
