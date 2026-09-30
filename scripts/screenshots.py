#!/usr/bin/env python3
"""Drive the real UI with headless Chromium and capture screenshots for docs / the SIH deck.
Runs the API in-process (loopback only), so it also proves the UI needs no internet."""
import os, sys, threading, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DRISHTI_STORE", "data/store")
import uvicorn
from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/screens"
os.makedirs(OUT, exist_ok=True)
PORT = 8765


def serve():
    from drishti.api import app
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")).run()


threading.Thread(target=serve, daemon=True).start()
time.sleep(3)
base = f"http://127.0.0.1:{PORT}"
external = []

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 940}, device_scale_factor=1.5)
    pg.on("request", lambda r: external.append(r.url) if not r.url.startswith(base) and not r.url.startswith("blob:") and not r.url.startswith("data:") else None)
    pg.goto(base); pg.wait_for_load_state("networkidle"); time.sleep(0.8)
    # 1. change-aware semantic search
    pg.screenshot(path=f"{OUT}/1_search.png")
    # 2. change analysis: run, select the cloud-affected provisional event
    pg.click("nav button[data-v=change]"); time.sleep(0.6)
    pg.click("#rerun"); time.sleep(2.0)
    evs = json.loads(pg.evaluate("JSON.stringify(S.changes.filter(c=>c.status!=='superseded').map(c=>[c.change_id,c.type,c.provisional]))"))
    target = next((c[0] for c in evs if c[2] == 1), evs[0][0])
    pg.evaluate(f"selectChange('{target}')"); time.sleep(1.2)
    pg.screenshot(path=f"{OUT}/2_change.png")
    # 3. review: confirm the road event first so the audit trail + feedback show, then open construction
    pg.click("nav button[data-v=review]"); time.sleep(1.0)
    road = next(c[0] for c in evs if c[1] == "road")
    cons = next(c[0] for c in evs if c[1] == "construction" and c[2] == 0)
    pg.evaluate(f"loadQueue('{road}')"); time.sleep(0.8)
    pg.fill("#note", "Road corridor visible in both post-change acquisitions; matches alignment of new track.")
    pg.click("#ok"); time.sleep(1.0)
    pg.evaluate(f"loadQueue('{cons}')"); time.sleep(1.4)
    pg.evaluate("window.scrollTo(0,0);document.querySelectorAll('main,section,.view').forEach(e=>e.scrollTop=0)"); time.sleep(0.3)
    pg.screenshot(path=f"{OUT}/3_review.png")
    # 4. discovery from a seed site
    pg.click("nav button[data-v=discover]"); time.sleep(1.6)
    pg.screenshot(path=f"{OUT}/4_discover.png")
    # 5. provenance / system
    pg.click("nav button[data-v=system]"); time.sleep(1.0)
    pg.screenshot(path=f"{OUT}/5_system.png")
    b.close()
print("screens saved to", OUT, "| non-loopback requests made by UI:", len(external), external[:3])
