"""Tile / text embedders behind one interface.

RemoteCLIPEmbedder : production path. OpenCLIP ViT-B/32 with RemoteCLIP weights (Apache-2.0 as listed; see docs/PROVENANCE.md),
                     loaded from a LOCAL checkpoint (no network). Text and imagery share one space.
LiteEmbedder       : zero-dependency fallback (numpy only) used for CI, low-spec laptops and this
                     demo. Tiles map to [concept part | visual part]; text maps to the concept part
                     through a small remote-sensing lexicon. Same API, so everything downstream
                     (index, planner, discovery, feedback) is unchanged when weights are staged.
"""
from __future__ import annotations
import os, re, hashlib
import numpy as np
from scipy import ndimage as ndi
from .quality import indices, WATER, VEG, BARE, BUILT
from .raster import rgb_uint8

CONCEPTS = ["water", "vegetation", "forest", "bare", "built", "road", "texture", "shore"]
LEXICON = {
    "water": ["water", "river", "rivers", "lake", "pond", "reservoir", "canal", "stream", "flood",
              "flooded", "wetland", "dam"],
    "vegetation": ["vegetation", "green", "crop", "crops", "field", "fields", "farmland", "agriculture",
                   "agricultural", "plantation", "grass", "orchard", "cultivated", "cropland"],
    "forest": ["forest", "forests", "tree", "trees", "jungle", "woodland", "wooded", "canopy"],
    "bare": ["bare", "barren", "soil", "open", "ground", "sand", "dry", "cleared", "clearing",
             "quarry", "empty", "dug", "excavation"],
    "built": ["building", "buildings", "structure", "structures", "settlement", "settlements",
              "urban", "built", "built-up", "houses", "house", "construction", "compound", "camp",
              "town", "village", "shed", "sheds", "facility"],
    "road": ["road", "roads", "highway", "track", "tracks", "street", "streets", "runway",
             "corridor", "path"],
    "texture": ["vehicle", "vehicles", "concentration", "concentrations", "clutter", "equipment",
                "containers", "dense", "large"],
    "shore": ["riverbank", "bank", "banks", "shore", "riverside", "waterfront", "coast", "embankment"],
}


def _l2(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


class LiteEmbedder:
    name = "drishti-lite-spectral-v1"
    licence = "MIT (this repository; no pretrained weights)"
    WC, WV = 0.8, 0.6

    def __init__(self):
        self.dim = len(CONCEPTS) + 44

    def concept_vec(self, a, s, u):
        n = max(int(u.sum()), 1)
        fr = [((s == k) & u).sum() / n for k in (WATER, VEG, BARE, BUILT)]
        ix = indices(a)
        forest = ((s == VEG) & u & (a[1] < 0.058) & (ix["ndvi"] > 0.7)).sum() / n
        built = (s == BUILT) & u
        thin = built & ~ndi.binary_opening(built, structure=np.ones((4, 4)))
        nir = a[3]
        loc_sd = np.sqrt(np.clip(ndi.uniform_filter(nir ** 2, 3) - ndi.uniform_filter(nir, 3) ** 2, 0, None))
        texture = float(np.clip(loc_sd[u].mean() / 0.04, 0, 1)) if u.any() else 0.0
        water = (s == WATER)
        shore = (ndi.binary_dilation(water, iterations=3) & ~water & u).sum() / n
        # Hellinger (sqrt) encoding: a tile that is 25 % river should still "contain a river"
        c = [fr[0], fr[1], forest, fr[2], fr[3], thin.sum() / n, texture, min(1.0, shore * 6)]
        return np.sqrt(np.clip(np.array(c, np.float32), 0, None))

    def visual_vec(self, a, s, u):
        if not u.any():
            return np.zeros(44, np.float32)
        ix = indices(a)
        f = []
        for b in range(5):
            f += [(a[b][u].mean() - 0.12) / 0.15, a[b][u].std() / 0.08]
        for k in ("ndvi", "mndwi", "ndbi", "redness"):
            f += [ix[k][u].mean(), ix[k][u].std() * 2]
        h, _ = np.histogram(ix["ndvi"][u], bins=8, range=(-0.6, 1.0))
        f += list(h / max(h.sum(), 1))
        H, W = s.shape
        for qs in (np.s_[:H // 2, :W // 2], np.s_[:H // 2, W // 2:], np.s_[H // 2:, :W // 2], np.s_[H // 2:, W // 2:]):
            qu = u[qs]; n = max(int(qu.sum()), 1)
            f += [((s[qs] == k) & qu).sum() / n for k in (WATER, VEG, BARE, BUILT)]
        gy, gx = np.gradient(a[3])
        g = np.hypot(gx, gy)[u]
        f += [g.mean() / 0.02, (g > 0.03).mean()]
        return np.array(f, np.float32)

    def embed_tiles(self, tiles):
        out = np.zeros((len(tiles), self.dim), np.float32)
        for i, (a, s, u) in enumerate(tiles):
            c = self.concept_vec(a, s, u); v = self.visual_vec(a, s, u)
            out[i] = np.concatenate([self.WC * _l2(c), self.WV * _l2(v)])
        return _l2(out)

    def parse_concepts(self, text: str) -> np.ndarray:
        toks = re.findall(r"[a-z\-]+", text.lower())
        w = np.zeros(len(CONCEPTS), np.float32)
        for j, c in enumerate(CONCEPTS):
            w[j] = sum(t in LEXICON[c] for t in toks)
        if "open ground" in text.lower():
            w[CONCEPTS.index("bare")] += 1
        return w

    def embed_text(self, text: str) -> np.ndarray:
        w = self.parse_concepts(text)
        if w.sum() == 0:
            w[:] = 1.0 / len(w)
        # light structure: roads are built surfaces, forests are vegetation, water implies a shoreline
        w[CONCEPTS.index("built")] += 0.3 * w[CONCEPTS.index("road")]
        w[CONCEPTS.index("vegetation")] += 0.5 * w[CONCEPTS.index("forest")]
        w[CONCEPTS.index("shore")] += 0.4 * w[CONCEPTS.index("water")]
        v = np.concatenate([_l2(w), np.zeros(self.dim - len(CONCEPTS), np.float32)])
        return _l2(v[None])[0]

    def fingerprint(self) -> str:
        return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]


class RemoteCLIPEmbedder:
    """OpenCLIP + RemoteCLIP checkpoint, fully offline (weights staged under models/)."""
    licence = "RemoteCLIP weights: Apache-2.0 as listed by the release (verify, see docs/PROVENANCE.md); OpenCLIP code: MIT"
    TEMPLATES = ["a satellite image of {}.", "an aerial photograph of {}.", "a remote sensing image of {}."]

    def __init__(self, ckpt_path: str, arch: str = "ViT-B-32", device: str = "cpu"):
        import torch, open_clip  # imported lazily: not required for the lite path
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=None)
        digest = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
        pin = ckpt_path + ".sha256"  # staged next to the weights; refuse anything that does not match
        if not os.path.exists(pin):
            raise RuntimeError(f"refusing unpinned weights: create {os.path.basename(pin)} (see README, 'Stage model weights')")
        want = open(pin).read().split()[0].strip().lower()
        if want != digest:
            raise RuntimeError(f"checkpoint hash mismatch: expected {want[:16]}…, got {digest[:16]}…")
        try:  # tensors only: never execute pickled code from a checkpoint
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # very old torch without weights_only
            raise RuntimeError("torch>=1.13 required for safe (weights_only) checkpoint loading")
        self.model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
        self.model.eval().to(device)
        self.tok = open_clip.get_tokenizer(arch)
        self.device = device
        self.name = f"remoteclip-{arch}"
        self.dim = self.model.visual.output_dim
        self._fp = digest[:16]

    def embed_tiles(self, tiles, batch=64):
        from PIL import Image
        vecs = []
        with self.torch.no_grad():
            for i in range(0, len(tiles), batch):
                ims = [self.preprocess(Image.fromarray(rgb_uint8(a))) for a, _, _ in tiles[i:i + batch]]
                x = self.torch.stack(ims).to(self.device)
                vecs.append(self.model.encode_image(x).float().cpu().numpy())
        return _l2(np.concatenate(vecs))

    def embed_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            t = self.tok([tp.format(text) for tp in self.TEMPLATES]).to(self.device)
            e = self.model.encode_text(t).float().cpu().numpy()
        return _l2(_l2(e).mean(0)[None])[0]

    def fingerprint(self) -> str:
        return self._fp


def get_embedder(models_dir: str = "models", prefer: str = "auto"):
    ckpt = os.path.join(models_dir, "RemoteCLIP-ViT-B-32.pt")
    if prefer in ("auto", "remoteclip") and os.path.exists(ckpt):
        try:
            return RemoteCLIPEmbedder(ckpt)
        except Exception as e:  # torch/open_clip missing → degrade gracefully, never call out
            print(f"[drishti] RemoteCLIP unavailable ({e}); using LiteEmbedder")
    return LiteEmbedder()
