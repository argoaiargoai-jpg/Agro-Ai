"""Build data/manifest.csv: every image, its label, and a leak-free train/val/test split.

Run:  python prepare_data.py
Steps
  1. PlantVillage: group images of the same physical leaf (official leaf-map.json + perceptual-hash
     near-duplicates that are rotations/flips of each other), then split BY GROUP.
  2. PlantDoc: map folder names to PlantVillage classes, drop near-duplicates between its train and test sets.
  3. COCO val2017: keep photos that contain no plants, split randomly.
Nothing here trains a model; it only decides which image may be used for what.
"""
import bisect
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from common import *  # noqa: F401,F403

SEED = 42
BLOCK, BUFFER = 32, 6   # unmapped files: group 32 consecutive numbers; purge 6 numbers around split borders
HASH_DIST = 6          # <= 6 of 64 differing bits (in any of 8 rotations/flips) => "same photo"
random.seed(SEED)


# ---------------------------------------------------------------- perceptual hashing
def _bits(a: np.ndarray) -> int:
    b = (a[:8, 1:] > a[:8, :-1]).flatten()
    return int("".join("1" if x else "0" for x in b), 2)


def orientation_hashes(path: str) -> list[int]:
    try:
        im = Image.open(path)
        im.draft("L", (64, 64))
        a = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((9, 9), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return [0] * 8
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [_bits(r), _bits(r[:, ::-1])]
    return out


def hash_all(paths: list[str]) -> list[list[int]]:
    with Pool(8) as p:
        return p.map(orientation_hashes, paths, chunksize=256)


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def popcount(x: int) -> int: return bin(x).count("1")


def union_near_duplicates(dsu: DSU, hashes: list[list[int]], restrict=None, max_bucket=300) -> int:
    """LSH on 8 chunks of 8 bits: two hashes within 7 bits must share a chunk (pigeonhole)."""
    buckets = defaultdict(set)
    for i, hs in enumerate(hashes):
        for h in set(hs):
            for j in range(8):
                buckets[(j, (h >> (8 * j)) & 255)].add(i)
    merged = 0
    for i, hs in enumerate(hashes):
        cand = set()
        for j in range(8):
            b = buckets[(j, (hs[0] >> (8 * j)) & 255)]
            if len(b) <= max_bucket:
                cand |= b
        for c in cand:
            if c <= i or (restrict and restrict(i, c)):
                continue
            if min(popcount(hs[0] ^ h2) for h2 in hashes[c]) <= HASH_DIST and dsu.find(i) != dsu.find(c):
                dsu.union(i, c); merged += 1
    return merged


def group_split(groups: dict, fracs=(0.70, 0.15, 0.15)) -> dict:
    """groups: gid -> n_images. Largest groups first, each goes to the split furthest below its target share,
    so even classes made of a few big groups get something in every split. Returns gid -> split."""
    names = ["train", "val", "test"]
    total = sum(groups.values())
    target = {n: f * total for n, f in zip(names, fracs) if f > 0}
    have = Counter(); out = {}
    order = sorted(groups, key=lambda g: (-groups[g], random.random()))
    for k, g in enumerate(order):
        open_ = [n for n in target if have[n] == 0] if k < len(target) else []   # first groups seed each split
        pick = open_[0] if open_ else max(target, key=lambda n: target[n] - have[n])
        out[g] = pick; have[pick] += groups[g]
    return out


# ---------------------------------------------------------------- PlantVillage
def build_plantvillage() -> pd.DataFrame:
    root = PV_DIR / "color"
    lm = json.load(open(PV_DIR / "leaf-map.json"))
    rows = []
    for cls in sorted(p.name for p in root.iterdir()):
        for f in sorted((root / cls).iterdir()):
            stem = f.stem
            k = (stem.split("___", 1)[1] if "___" in stem else stem).lower()
            leaf = next((v for v in lm.get(k, []) if v.split(":::")[0] == cls), None)
            rows.append(dict(path=str(f.relative_to(DATA)), label=cls, leaf=leaf))
    df = pd.DataFrame(rows)
    print(f"[PV] {len(df)} images, leaf-map covers {df.leaf.notna().sum()}")
    dsu = DSU(len(df))
    by_leaf = defaultdict(list)
    for i, l in enumerate(df.leaf):
        if pd.notna(l): by_leaf[l].append(i)   # NaN is truthy: a bare `if l` merged ALL unmapped files
    for idxs in by_leaf.values():
        for i in idxs[1:]: dsu.union(idxs[0], i)
    # Files NOT covered by the leaf-map: nearby file numbers within a prefix are (almost always) the same
    # leaf photographed several times, so block them together (BLOCK consecutive numbers per group).
    nums = []
    for p in df.path:
        stem = p.split("/")[-1].rsplit(".", 1)[0]; nm = (stem.split("___", 1)[1] if "___" in stem else stem).lower()
        m = re.match(r"(.*?)\s*(\d+)$", nm); nums.append((m.group(1), int(m.group(2))) if m else (nm, -1))
    df["prefix"] = [a for a, _ in nums]; df["num"] = [b for _, b in nums]
    blocks = {}
    for i, (l, pf, nu, leaf) in enumerate(zip(df.label, df.prefix, df.num, df.leaf)):
        if pd.isna(leaf) and nu >= 0:
            key = (l, pf, nu // BLOCK)
            if key in blocks: dsu.union(blocks[key], i)
            else: blocks[key] = i
    labels = df.label.tolist()
    print("[PV] hashing …"); hashes = hash_all([str(DATA / p) for p in df.path])
    n = union_near_duplicates(dsu, hashes, restrict=lambda i, c: labels[i] != labels[c])
    print(f"[PV] near-duplicate merges (same class): {n}")
    df["group"] = [f"pv{dsu.find(i)}" for i in range(len(df))]
    df["split"] = ""
    df["role"] = ""
    for cls, sub in df.groupby("label"):
        crop = crop_of(cls)
        if crop == UNKNOWN_NOVEL_VAL_CROP:
            df.loc[sub.index, ["split", "role"]] = ["val", "unknown_novel"]; continue
        if crop == UNKNOWN_NOVEL_TEST_CROP:
            df.loc[sub.index, ["split", "role"]] = ["test", "unknown_novel"]; continue
        sizes = sub.group.value_counts().to_dict()
        sp = group_split(sizes)
        df.loc[sub.index, "split"] = sub.group.map(sp)
        df.loc[sub.index, "role"] = "unknown_seen" if crop in UNKNOWN_SEEN_CROPS else "known"
    big = df.groupby("group").size().sort_values(ascending=False)
    print(f"[PV] groups: {len(big)}, largest group = {big.iloc[0]} images")
    # Purge: an unmapped image within BUFFER file-numbers of an image that sits in a DIFFERENT split could be the
    # same leaf, so it is excluded from every split (split="excluded"). Exact leaf-map groups are never purged.
    idx = defaultdict(list)
    for i, (l, pf, nu) in enumerate(zip(df.label, df.prefix, df.num)):
        if nu >= 0: idx[(l, pf)].append((nu, i))
    for v in idx.values(): v.sort()
    splits = df.split.tolist(); leafs = df.leaf.tolist(); drop = []
    for v in idx.values():
        keys = [a for a, _ in v]
        for nu, i in v:
            if not pd.isna(leafs[i]): continue
            lo, hi = bisect.bisect_left(keys, nu - BUFFER), bisect.bisect_right(keys, nu + BUFFER)
            if any(splits[j] != splits[i] for _, j in v[lo:hi]): drop.append(i)
    df.loc[drop, "split"] = "excluded"
    print(f"[PV] purged {len(drop)} unmapped images lying next to another split (leakage buffer)")
    df["label_out"] = [UNKNOWN if r.startswith("unknown") else l for l, r in zip(df.label, df.role)]
    df["source"] = "plantvillage"; df["domain"] = "lab"
    need = df[df.role.isin(["known", "unknown_seen"])]
    ct = need.groupby(["label", "split"]).size().unstack(fill_value=0)
    for c in ("train", "val", "test"): assert (ct[c] > 0).all(), f"class missing from {c}: {ct[ct[c]==0].index.tolist()}"
    print("[PV] smallest per-split class counts:", ct.min().to_dict())
    return df


# ---------------------------------------------------------------- PlantDoc
def build_plantdoc() -> pd.DataFrame:
    out_root = PROCESSED / "plantdoc"
    rows = []
    for split in ("train", "test"):
        for d in sorted((PD_DIR / split).iterdir()):
            if not d.is_dir() or d.name not in PLANTDOC_MAP: continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
                dest = out_root / split / d.name / (f.stem[:60] + ".jpg")
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                    im.thumbnail((384, 384)); im.save(dest, quality=92)
                except Exception as e:
                    print("  skip", f.name, e); continue
                tgt = PLANTDOC_MAP[d.name]
                rows.append(dict(path=str(dest.relative_to(DATA)), pd_split=split, folder=d.name,
                                 label=tgt or d.name, label_out=tgt or UNKNOWN,
                                 role="known" if tgt else ("unknown_seen" if d.name.startswith("Squash") else "unknown_novel")))
    df = pd.DataFrame(rows)
    print(f"[PD] {len(df)} usable images ({(df.pd_split=='train').sum()} train / {(df.pd_split=='test').sum()} test)")
    hashes = hash_all([str(DATA / p) for p in df.path])
    dsu = DSU(len(df)); union_near_duplicates(dsu, hashes)
    df["group"] = [f"pd{dsu.find(i)}" for i in range(len(df))]
    # any group touching the TEST set is removed from train/val  (no test leakage)
    test_groups = set(df[df.pd_split == "test"].group)
    leaked = df[(df.pd_split == "train") & df.group.isin(test_groups)]
    print(f"[PD] dropping {len(leaked)} train images that are near-duplicates of test images")
    df = df.drop(leaked.index).reset_index(drop=True)
    df["split"] = ""
    df.loc[df.pd_split == "test", "split"] = "test"
    tr = df[df.pd_split == "train"]
    for _, sub in tr.groupby("label_out" if False else "label"):
        sp = group_split(sub.group.value_counts().to_dict(), fracs=(0.85, 0.15, 0.0))
        df.loc[sub.index, "split"] = sub.group.map(sp).replace({"test": "train"})
    df["source"] = "plantdoc"; df["domain"] = "field"
    return df.drop(columns=["pd_split", "folder"])


# ---------------------------------------------------------------- COCO
def build_coco() -> pd.DataFrame:
    ann = json.load(open(COCO_DIR / "annotations" / "instances_val2017.json"))
    plant_ids = {c["id"] for c in ann["categories"] if c["name"] in ("potted plant", "vase")}
    has_plant = {a["image_id"] for a in ann["annotations"] if a["category_id"] in plant_ids}
    imgs = [i for i in ann["images"] if i["id"] not in has_plant]
    print(f"[COCO] {len(ann['images'])} images, {len(has_plant)} contain potted plants/vases -> excluded, {len(imgs)} kept")
    z = zipfile.ZipFile(COCO_DIR / "val2017.zip")
    random.shuffle(imgs)
    n = len(imgs); rows = []
    out_root = PROCESSED / "coco"; out_root.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(imgs):
        split = "train" if k < 0.6 * n else ("val" if k < 0.8 * n else "test")
        dest = out_root / i["file_name"]
        with z.open(f"val2017/{i['file_name']}") as fh:
            im = Image.open(fh).convert("RGB"); im.thumbnail((384, 384)); im.save(dest, quality=92)
        rows.append(dict(path=str(dest.relative_to(DATA)), label=NON_PLANT, label_out=NON_PLANT, split=split,
                         group=f"coco{i['id']}", role="non_plant", source="coco", domain="other"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.concat([build_plantvillage(), build_plantdoc(), build_coco()], ignore_index=True)
    df.to_csv(MANIFEST, index=False)
    print("\nSaved", MANIFEST, len(df))
    print(df.groupby(["source", "split", "role"]).size().to_string())
