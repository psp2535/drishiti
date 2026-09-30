#!/usr/bin/env python3
"""Deploy DRISHTI as a Hugging Face Docker Space (public read-only demo, see Dockerfile).

  pip install huggingface_hub && hf auth login          # once, with a token that has write access
  python scripts/deploy_hf_space.py <hf-user>/<space>    # add --private to hide the Space
The Space builds the Dockerfile itself; the simulated data is regenerated there, never uploaded.
Re-running pushes the current code as a new commit and triggers a rebuild.
"""
import argparse, os, sys
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".venv", "data", "__pycache__", ".git", "wheels"}
SKIP_FILES = {"README.md"}  # replaced by the Space card below
CARD = """---
title: DRISHTI
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: Semantic search + change analysis for satellite archives
---

"""


def files():
    for d, dirs, names in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for n in names:
            rel = os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, "/")
            if rel in SKIP_FILES or n.endswith(".pyc") or (rel.startswith("models/") and n != "README.md"):
                continue
            yield rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="<hf-user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    # Space card = HF front matter + the project README, so the scope-of-evidence note stays visible
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        card = (CARD + f.read()).encode("utf-8")
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=os.path.join(ROOT, p)) for p in files()]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
    c = api.create_commit(a.repo_id, repo_type="space", operations=ops, commit_message="Deploy DRISHTI")
    print(f"pushed {len(ops)} files: {c.commit_url}")
    print(f"build logs + app: https://huggingface.co/spaces/{a.repo_id}")
    print(f"direct URL:       https://{a.repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    sys.exit(main())
