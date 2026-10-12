# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# CSP CST 627 100  Deep Learning Neural Networks
# Week 05 Assignment: Latency, Memory, Accuracy: A Measured Trade-off Board
# Objective: Load a small public multimodal set (photos + human captions) for the benchmark.
#
# Dataset: Hugging Face `intro/flickr8k` (Flickr8k photos with five human captions each; the
# dataset card declares license CC0-1.0). It ships no class labels, so labels are DERIVED:
#   * label      = the scene category whose keywords occur most often in captions 1-4
#                  (needs at least 2 hits and no tie), so the label is a weak, caption-based label;
#   * text input = caption 0, which the label rule never reads;
#   * image      = the photo, centre-cropped and resized to 64 x 64 grayscale.
# Both modalities carry real signal, and neither defines the label exactly, as in the synthetic set.
# Held-out data = the official dev and test splits; training data = the official train split.
#
# Usage:
#   python scripts/public_data.py          # download the subset, print its summary
# =============================================================================
# endregion
# =============================================================================
# Step 1: Download the Subset and Build Tensors
# Reads the three small `metadata.csv` files, derives labels, picks a fixed class-capped subset
# with a seeded shuffle, downloads only those images over plain HTTPS (cached under data/), and
# returns arrays plus a provenance record. The subset list is written to results/ so a reader
# can see exactly which photos were used.
# =============================================================================
"""
https://huggingface.co/datasets/intro/flickr8k
https://creativecommons.org/publicdomain/zero/1.0/
https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.fit
"""

import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

DATASET_ID = "intro/flickr8k"
HF_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
CATEGORIES = {  # scene category -> keyword pattern, matched against lower-cased captions
    "dog": r"\b(dog|dogs|puppy|puppies)\b",
    "water": r"\b(water|beach|ocean|sea|pool|lake|river|swim\w*|surf\w*|wave|waves|boat|splash\w*)\b",
    "snow_mountain": r"\b(snow|snowy|mountain|mountains|ski\w*|hill|hiking|climb\w*|rock|cliff)\b",
    "ball_sport": r"\b(ball|soccer|football|basketball|baseball|volleyball|tennis|game|team|player|players)\b",
    "street_crowd": r"\b(street|city|sidewalk|crowd|building|buildings|road|bicycle|bike|motorcycle|car)\b",
    "child": r"\b(child|children|kid|kids|girl|boy|baby|toddler)\b",
}
CLASSES = list(CATEGORIES)
MIN_HITS = 2      # a label needs at least this many keyword hits across captions 1-4 ...
VOCAB_SIZE = 800  # ... and the text vocabulary is the most frequent words of the training captions
MAXLEN = 24


def _download(url, dest, tries=5):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == tries - 1:
                raise RuntimeError(f"download failed after {tries} tries: {url} ({exc})") from exc
            time.sleep(1.5 * (attempt + 1))  # back off; Hugging Face may rate-limit bursts


def derive_label(row):
    """Scene category from captions 1-4 only (caption 0 stays the unseen text input), or None."""
    text = " ".join(str(row[f"caption_{i}"]).lower() for i in range(1, 5))
    hits = {name: len(re.findall(pattern, text)) for name, pattern in CATEGORIES.items()}
    best = max(hits.values())
    top = [name for name, n in hits.items() if n == best]
    return top[0] if best >= MIN_HITS and len(top) == 1 else None


def _words(caption):
    return re.findall(r"[a-z']+", str(caption).lower())


def _pick(df, per_class, rng):
    parts = []
    for name in CLASSES:
        pool = df[df["label"] == name].sort_values("file_name")
        take = rng.permutation(len(pool))[:per_class]
        parts.append(pool.iloc[np.sort(take)])
    return pd.concat(parts, ignore_index=True)


def _images(df, cache, workers=8):
    paths = [cache / row.origin / row.file_name for row in df.itertuples()]
    urls = [f"{HF_BASE}/{row.origin}/{row.file_name}" for row in df.itertuples()]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_download, urls, paths))
    out = np.empty((len(df), 1, 64, 64), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            out[i, 0] = np.asarray(ImageOps.fit(im.convert("L"), (64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
    return out


def load(root, train_per_class=500, test_per_class=150, seed=1234):
    """Return dict(imgs_tr, toks_tr, y_tr, imgs_te, toks_te, y_te, vocab, classes, info, subset)."""
    root = Path(root)
    cache = root / "data" / "flickr8k"
    meta = {}
    for split in ("train", "dev", "test"):
        csv = _download(f"{HF_BASE}/{split}/metadata.csv", cache / split / "metadata.csv")
        df = pd.read_csv(csv)
        df["origin"] = split
        df["label"] = df.apply(derive_label, axis=1)
        meta[split] = df.dropna(subset=["label"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    train = _pick(meta["train"], train_per_class, rng)
    held = _pick(pd.concat([meta["dev"], meta["test"]], ignore_index=True), test_per_class, rng)
    train["role"], held["role"] = "train", "test"

    counts = pd.Series([w for c in train["caption_0"] for w in _words(c)]).value_counts()
    vocab = {"<unk>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(counts.index[: VOCAB_SIZE - 1])})  # 0 = pad

    def tokens(df):
        out = np.zeros((len(df), MAXLEN), dtype=np.int64)
        for i, caption in enumerate(df["caption_0"]):
            ids = [vocab.get(w, 1) for w in _words(caption)][:MAXLEN]
            out[i, : len(ids)] = ids
        return out

    y = lambda df: np.array([CLASSES.index(v) for v in df["label"]], dtype=np.int64)

    # Context for the report: how far a plain keyword rule on the text input gets (no model at all).
    def keyword_guess(caption):
        hits = {name: len(re.findall(p, str(caption).lower())) for name, p in CATEGORIES.items()}
        return max(hits, key=hits.get) if max(hits.values()) > 0 else None
    keyword_acc = float(np.mean([keyword_guess(c) == l for c, l in zip(held["caption_0"], held["label"])]))

    subset = pd.concat([train, held], ignore_index=True)[["role", "origin", "file_name", "label", "caption_0"]]
    info = dict(
        dataset=DATASET_ID, url=f"https://huggingface.co/datasets/{DATASET_ID}",
        license="CC0-1.0 as declared on the dataset card (provenance of the underlying Flickr photos not independently verified)",
        label_rule=f"scene category with the most keyword hits in captions 1-4 (min {MIN_HITS}, no tie); caption 0 is the text input",
        categories=CATEGORIES, classes=CLASSES, image_input="centre crop, 64x64 grayscale", text_vocab=len(vocab) + 1,
        train_per_class=train_per_class, test_per_class=test_per_class, n_train=len(train), n_test=len(held),
        train_class_counts=train["label"].value_counts().to_dict(), test_class_counts=held["label"].value_counts().to_dict(),
        chance=1.0 / len(CLASSES), keyword_rule_on_text_acc=keyword_acc, seed=seed)
    return dict(
        imgs_tr=_images(train, cache), toks_tr=tokens(train), y_tr=y(train),
        imgs_te=_images(held, cache), toks_te=tokens(held), y_te=y(held),
        vocab=vocab, classes=CLASSES, info=info, subset=subset)


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    data = load(here)
    out = here / "results" / "public_subset.csv"
    out.parent.mkdir(exist_ok=True)
    data["subset"].to_csv(out, index=False)
    for key in ("n_train", "n_test", "train_class_counts", "test_class_counts", "text_vocab", "keyword_rule_on_text_acc"):
        print(f"{key}: {data['info'][key]}")
    print(f"wrote {out.relative_to(here)}")
