"""Real mahsulot rasmlari benchmarki — GroceryStoreDataset (Klasson va boshq., WACV 2019).

https://github.com/marcusklasson/GroceryStoreDataset  (data/external/ ga klonlanadi)

Protokol (few-shot, open-set — aynan Rasta ssenariysi):
  * Sinflar tasodifiy 75% "katalogda bor" / 25% "noma'lum" ga bo'linadi.
  * Katalog: har tanish SKU uchun rasmiy packshot (iconic) + k ta real rasm
    (train bo'limidan), k ∈ {0, 1, 3, 5}.  k=0 — faqat packshot bilan onboarding.
  * So'rovlar: test bo'limidagi BARCHA rasmlar (tanish + noma'lum sinflar).
  * Har bo'linish/k uchun: closed-set top-1, AUROC (tanish vs noma'lum),
    halol kalibrlangan open-set aniqligi (τ juft seed'larda tanlanib, toq
    seed'larda tekshiriladi va aksincha).

Embeddinglar bir marta hisoblanib keshlanadi (data/external/cache/).

  python scripts/benchmark_grocery.py --subset packages --seeds 20
  python scripts/benchmark_grocery.py --subset all --seeds 10
"""
import argparse
import csv
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from rasta.calibrate import Rec, report
from rasta.config import ROOT, resolve_device
from rasta.features import color_descriptor

DS = ROOT / "data" / "external" / "GroceryStoreDataset" / "dataset"
CACHE = ROOT / "data" / "external" / "cache"

# (nom, model, resize_mode) — har biri uchun cls va mean alohida saqlanadi
BACKBONES = [
    ("dinov2b_crop", "facebook/dinov2-base", "crop"),
    ("dinov2b_pad", "facebook/dinov2-base", "pad"),
    ("dinov2b_stretch", "facebook/dinov2-base", "stretch"),
    ("clipL_stretch", "openai/clip-vit-large-patch14", "stretch"),
    ("dinov2L_crop", "facebook/dinov2-large", "crop"),
    ("siglip2b_stretch", "google/siglip2-base-patch16-224", "stretch"),
]


def load_index(subset: str):
    classes = {}
    with open(DS / "classes.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["Class ID (int)"])
            classes[cid] = {
                "name": row["Class Name (str)"],
                "coarse": row["Coarse Class Name (str)"],
                "iconic": row["Iconic Image Path (str)"].lstrip("/"),
                "packages": "/Packages/" in row["Iconic Image Path (str)"],
            }
    keep = {c for c, v in classes.items() if subset == "all" or v["packages"]}

    def read(split):
        items = []
        for line in (DS / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path, cid, _ = [x.strip() for x in line.split(",")]
            if int(cid) in keep:
                items.append((path, int(cid)))
        return items

    return classes, keep, read("train"), read("test")


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        return np.array(im)


def compute_features(paths: list[str], key: str, backbones=None, root: Path = DS):
    """paths -> {<backbone>_cls, <backbone>_mean, color}. Har backbone alohida keshlanadi."""
    CACHE.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    imgs = None

    def cached(name, fn):
        nonlocal imgs
        f = CACHE / f"{key}__{name}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if list(z["paths"]) == paths:
                return {k: z[k] for k in z.files if k != "paths"}
        if imgs is None:
            imgs = [load_rgb(root / p) for p in paths]
        t0 = time.time()
        out = fn(imgs)
        np.savez_compressed(f, paths=np.array(paths, dtype=object), **out)
        print(f"  [{name}] {len(imgs)} rasm, {time.time() - t0:.1f}s")
        return out

    from rasta.embedder import Embedder

    F = cached("color", lambda ims: {
        "color": np.stack([color_descriptor(im) for im in ims]),
        "aspect": np.array([im.shape[0] / max(im.shape[1], 1) for im in ims], np.float32)})
    for name, model, mode in (backbones or BACKBONES):
        def run(ims, model=model, mode=mode, name=name):
            if "dinov2" in model:
                emb = Embedder(model, device=device, batch_size=64, resize_mode=mode,
                               pooling="cls+mean")
                v = emb.embed(ims)
                d = v.shape[1] // 2
                nrm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
                return {f"{name}_cls": nrm(v[:, :d]), f"{name}_mean": nrm(v[:, d:])}
            emb = Embedder(model, device=device, batch_size=64, resize_mode=mode)
            return {f"{name}_cls": emb.embed(ims)}
        try:
            F.update(cached(name, run))
        except OSError as e:  # model yuklanmagan bo'lsa
            print(f"  [{name}] o'tkazib yuborildi: {e}")
    return F


def get_vec(F, idx, emb: str):
    """emb: '<backbone>_cls' | '<backbone>_mean' | '<backbone>_cls+mean'."""
    if emb.endswith("_cls+mean"):
        b = emb[: -len("_cls+mean")]
        v = np.concatenate([F[f"{b}_cls"][idx], F[f"{b}_mean"][idx]], axis=1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return F[emb][idx]


def run_protocol(F, n_gallery, gal_labels_all, train_idx, train_lab, q_idx, q_lab,
                 classes_all, emb, w_color, shots, seeds, unknown_frac=0.25,
                 w_shape=0.0, shape_sigma=0.25, return_recs=False):
    recs, groups = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        cls = np.array(sorted(classes_all))
        rng.shuffle(cls)
        n_unk = max(1, int(round(len(cls) * unknown_frac)))
        unknown = set(cls[:n_unk].tolist())
        known = [c for c in cls if c not in unknown]

        g_idx, g_lab = [], []
        for c in known:
            if n_gallery is not None:
                g_idx.append(n_gallery[c]); g_lab.append(c)      # iconic packshot
            pool = [i for i, l in zip(train_idx, train_lab) if l == c]
            if shots and pool:
                pick = rng.choice(pool, size=min(shots, len(pool)), replace=False)
                g_idx.extend(pick.tolist()); g_lab.extend([c] * len(pick))
        g_idx = np.array(g_idx); g_lab = np.array(g_lab)

        G = get_vec(F, g_idx, emb); Q = get_vec(F, q_idx, emb)
        S = Q @ G.T
        if w_color > 0:
            C = np.clip(F["color"][q_idx] @ F["color"][g_idx].T, 0, 1)
            S = S - w_color * S + w_color * C
        if w_shape > 0:
            la = np.log(F["aspect"])
            d = la[q_idx][:, None] - la[g_idx][None, :]
            S = S - w_shape * S + w_shape * np.exp(-(d * d) / (2 * shape_sigma ** 2))
        # SKU bo'yicha maksimum
        uniq = np.array(known)
        per = np.full((len(q_idx), len(uniq)), -1.0, dtype=np.float32)
        for j, c in enumerate(uniq):
            per[:, j] = S[:, g_lab == c].max(axis=1)
        order = np.argsort(-per, axis=1)
        top1 = uniq[order[:, 0]]
        s1 = per[np.arange(len(q_idx)), order[:, 0]]
        s2 = per[np.arange(len(q_idx)), order[:, 1]] if len(uniq) > 1 else np.zeros(len(q_idx))
        for t, p, a, b in zip(q_lab, top1, s1, s2):
            truth = "unknown" if t in unknown else str(t)
            recs.append(Rec(truth, str(p), float(a), float(a - b)))
            groups.append(seed)
    return (report(recs, groups), recs) if return_recs else report(recs, groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["packages", "all"], default="packages")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--out", default=None)
    ap.add_argument("--full", action="store_true", help="DINOv2 mean/cls+mean variantlari ham")
    ap.add_argument("--wc", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    args = ap.parse_args()

    classes, keep, train, test = load_index(args.subset)
    iconic = [classes[c]["iconic"] for c in sorted(keep)]
    paths = iconic + [p for p, _ in train] + [p for p, _ in test]
    print(f"[+] subset={args.subset}: {len(keep)} sinf, {len(train)} train, "
          f"{len(test)} test, {len(iconic)} packshot")
    F = compute_features(paths, f"grocery_{args.subset}")

    n_gallery = {c: i for i, c in enumerate(sorted(keep))}
    off = len(iconic)
    train_idx = list(range(off, off + len(train))); train_lab = [c for _, c in train]
    off2 = off + len(train)
    q_idx = np.arange(off2, off2 + len(test)); q_lab = [c for _, c in test]

    embs = []
    for name, model, _ in BACKBONES:
        if f"{name}_cls" not in F:
            continue
        embs.append(f"{name}_cls")
        if "dinov2" in model and args.full:
            embs += [f"{name}_mean", f"{name}_cls+mean"]
    # Ansambllar: vizual (DINOv2) + matn/logotip (CLIP/SigLIP)
    def norm(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    for a, b in (("dinov2b_crop_cls", "clipL_stretch_cls"),
                 ("dinov2L_crop_cls", "clipL_stretch_cls"),
                 ("dinov2b_crop_cls", "siglip2b_stretch_cls"),
                 ("clipL_stretch_cls", "siglip2b_stretch_cls")):
        if a in F and b in F:
            name = f"ens[{a.split('_')[0]}+{b.split('_')[0]}]_cls"
            F[name] = norm(np.concatenate([np.sqrt(0.4) * F[a], np.sqrt(0.6) * F[b]], 1))
            embs.append(name)
    results = []
    for emb in embs:
        for wc in args.wc:
            for k in args.shots:
                r = run_protocol(F, n_gallery, None, train_idx, train_lab, q_idx, q_lab,
                                 keep, emb, wc, k, args.seeds)
                row = {"emb": emb, "w_color": wc, "shots": k, **r}
                results.append(row)
                print(f"{emb:<26} wc={wc:<4} k={k}  closed={r['closed_acc']:.3f}  "
                      f"auroc={r['auroc']:.3f}  open(holdout)={r['holdout']['open_acc']:.3f}  "
                      f"n={r['n']}")
    out = ROOT / (args.out or f"outputs/benchmark_grocery_{args.subset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[✓] {out}")


if __name__ == "__main__":
    main()
