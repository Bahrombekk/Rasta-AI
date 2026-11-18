"""Foto arxividan (aptash) SKU katalogi nomzodlarini tayyorlash — operator uchun.

Bosqichlar (har biri keshlanadi, qayta ishga tushirsa davom etadi):
  extract  — muzlatgich/javon rasmlaridan mahsulotlarni kesish + embedding/rang/shakl
  cluster  — GPU k-means + birlashtirish + saralash (ko'p do'konda uchraganlar birinchi)
  review   — har klaster uchun montaj + operator sahifasi (review.html)

  python scripts/build_catalog_candidates.py --max-images 3000
  python scripts/build_catalog_candidates.py --stages cluster review --top 400

Natija: data/catalog_build/
  shards/*.npz        — crop'lar: rasm, quti, embedding, rang, nisbat
  clusters.json       — klasterlar (hajm, do'konlar soni, zichlik, a'zolar)
  review/review.html  — operator sahifasi: nom yozib "Yorliqlarni saqlash" -> labels.json
Keyin: python scripts/import_catalog_labels.py --labels data/catalog_build/review/labels.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from rasta.config import ROOT, Config

APT = ROOT / "data" / "external" / "aptash"
OUT = ROOT / "data" / "catalog_build"
OUTLET_RE = re.compile(r"img-[a-z]\d_(\d+)-\d+\.jpg")


def select_images(labels=("fridge", "shelf"), min_p=0.5, max_images=None, seed=0):
    idx = json.loads((APT / "index.json").read_text())
    cl = json.loads((APT / "classes.json").read_text())
    sel = [(m, f) for (m, f, _), c in zip(idx, cl)
           if c and c["label"] in labels and c["p"] >= min_p and c["bright"] >= 35]
    rng = np.random.default_rng(seed)
    rng.shuffle(sel)
    return sel[:max_images] if max_images else sel


def filter_boxes(boxes, w, h, min_conf=0.3, min_side=20):
    out = []
    for b in boxes:
        bw, bh = b.width, b.height
        if b.score < min_conf or min(bw, bh) < min_side:
            continue
        if not (0.3 <= bh / max(bw, 1) <= 6.0):
            continue
        if bw * bh > 0.25 * w * h:                       # butun eshik/muzlatgich
            continue
        if b.cx < 0.14 * w and b.cy < 0.11 * h:          # ilova logotipi (yuqori chap)
            continue
        out.append(b)
    return out


def stage_extract(cfg, images, shard_size=500):
    from rasta.config import resolve_device
    from rasta.detector import build_detector, suppress_nested
    from rasta.embedder import build_embedder, crop_boxes
    from rasta.features import aspect_ratio, color_descriptors
    from rasta.imageio import load_image

    (OUT / "shards").mkdir(parents=True, exist_ok=True)
    (OUT / "images.json").write_text(json.dumps(images))
    device = resolve_device(cfg.get("device", "auto"))
    det = build_detector(cfg, device)
    emb = build_embedder(cfg, device)
    t0 = time.time()
    for s in range(0, len(images), shard_size):
        f = OUT / "shards" / f"{s // shard_size:05d}.npz"
        if f.exists():
            continue
        E, C, A, B, I, S, Q = [], [], [], [], [], [], []
        for k, (m, name) in enumerate(images[s:s + shard_size]):
            img = load_image(APT / m / name)
            h, w = img.shape[:2]
            boxes = filter_boxes(suppress_nested(det.detect(img)), w, h)
            if not boxes:
                continue
            crops = crop_boxes(img, boxes)
            E.append(emb.embed(crops).astype(np.float16))
            C.append(color_descriptors(crops).astype(np.float16))
            A += [aspect_ratio(b.height, b.width) for b in boxes]
            B += [[b.x1, b.y1, b.x2, b.y2] for b in boxes]
            S += [b.score for b in boxes]
            Q += [crop_quality(c) for c in crops]
            I += [s + k] * len(boxes)
        np.savez(f, emb=np.concatenate(E), color=np.concatenate(C),
                 aspect=np.array(A, np.float32), box=np.array(B, np.float32),
                 img=np.array(I, np.int32), conf=np.array(S, np.float32),
                 quality=np.array(Q, np.float32))
        done = min(s + shard_size, len(images))
        print(f"  extract {done}/{len(images)} rasm, {time.time() - t0:.0f}s", flush=True)


def load_shards():
    fs = sorted((OUT / "shards").glob("*.npz"))
    z = [np.load(f) for f in fs]
    return {k: np.concatenate([x[k] for x in z])
            for k in ("emb", "color", "aspect", "box", "img", "conf", "quality")}


def crop_quality(crop) -> tuple[float, float]:
    """(keskinlik = Laplacian dispersiyasi, kontrast = yorqinlik std)."""
    import cv2
    g = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.std())


def good_mask(D, min_width=26, min_sharp=700, min_conf=0.45, min_contrast=20):
    """Katalog uchun yaroqli crop'lar. Aptash'da xira/kichik/blikli crop'lar
    (eni ~24px, keskinlik ~960) "har do'konda bor" axlat klasterlarini hosil qiladi;
    aniq mahsulot klasterlarida eni ~32px, keskinlik ~2760."""
    w = D["box"][:, 2] - D["box"][:, 0]
    q = D["quality"]
    return (w >= min_width) & (q[:, 0] >= min_sharp) & (D["conf"] >= min_conf) & (q[:, 1] >= min_contrast)


def stage_cluster(cfg, per_cluster=40, merge_thr=0.94, min_size=8):
    from rasta.catalog_build import cluster_stats, merge_clusters, spherical_kmeans
    from rasta.cluster import build_cluster_features

    images = json.loads((OUT / "images.json").read_text())
    D = load_shards()
    keep = good_mask(D)
    print(f"  sifat filtri: {int(keep.sum())}/{len(keep)} crop katalog uchun yaroqli")
    sel = np.where(keep)[0]
    n = len(sel)
    outlets = np.array([int(OUTLET_RE.match(images[i][1]).group(1)) for i in D["img"][sel]])
    X = build_cluster_features(D["emb"][sel].astype(np.float32),
                               D["color"][sel].astype(np.float32), D["aspect"][sel],
                               color_weight=cfg.get("clustering.color_weight", 0.6),
                               shape_weight=cfg.get("clustering.shape_weight", 0.3))
    k = max(2, n // per_cluster)
    t0 = time.time()
    labels, cents = spherical_kmeans(X, k)
    merged = merge_clusters(cents, labels, thr=merge_thr)
    stats = cluster_stats(X, merged, outlets, min_size=min_size)
    print(f"  {n} crop -> k-means {k} -> birlashtirilgan {len(np.unique(merged))} -> "
          f">= {min_size} a'zoli: {len(stats)} klaster ({time.time() - t0:.0f}s)")
    full = -np.ones(len(keep), dtype=int)
    full[sel] = merged
    np.save(OUT / "labels.npy", full)
    js = [{"cid": c.cid, "size": c.size, "n_outlets": c.n_outlets,
           "tightness": round(c.tightness, 4), "score": round(c.score, 3),
           "members": sel[c.members[:200]].tolist()} for c in stats]
    (OUT / "clusters.json").write_text(json.dumps(js))
    covered = sum(c.size for c in stats)
    print(f"  klasterlarga tushgan crop'lar: {covered}/{n} ({100 * covered / n:.0f}%)")
    return stats


def stage_review(top=300, per_row=12):
    from PIL import Image, ImageDraw

    from rasta.catalog_build import ClusterInfo, pick_examples

    images = json.loads((OUT / "images.json").read_text())
    D = load_shards()
    outlets = np.array([int(OUTLET_RE.match(images[i][1]).group(1)) for i in D["img"]])
    clusters = json.loads((OUT / "clusters.json").read_text())[:top]
    rdir = OUT / "review"
    (rdir / "strips").mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    def crop(i):
        m, name = images[D["img"][i]]
        if (m, name) not in cache:
            if len(cache) > 300:
                cache.clear()
            cache[(m, name)] = Image.open(APT / m / name).convert("RGB")
        x1, y1, x2, y2 = D["box"][i]
        return cache[(m, name)].crop((int(x1), int(y1), int(x2), int(y2)))

    TW, TH = 72, 132
    rows = []
    for c in clusters:
        info = ClusterInfo(c["cid"], c["size"], c["n_outlets"], c["tightness"],
                           np.array(c["members"]))
        ex = pick_examples(info, outlets, per_row)
        strip = Image.new("RGB", (per_row * (TW + 3), TH), (255, 255, 255))
        for k, i in enumerate(ex):
            im = crop(i)
            im.thumbnail((TW, TH))
            strip.paste(im, (k * (TW + 3) + (TW - im.width) // 2, TH - im.height))
        fn = f"strips/c{c['cid']:05d}.jpg"
        strip.save(rdir / fn, quality=88)
        rows.append({**{k: c[k] for k in ("cid", "size", "n_outlets", "tightness")}, "strip": fn})

    # Umumiy ko'rinish (PNG sahifalar) — tez ko'zdan kechirish uchun
    per_page = 25
    for p in range(0, len(rows), per_page):
        page = Image.new("RGB", (110 + per_row * (TW + 3), per_page * (TH + 6)), (255, 255, 255))
        d = ImageDraw.Draw(page)
        for r, row in enumerate(rows[p:p + per_page]):
            y = r * (TH + 6)
            d.text((4, y + 40), f"#{p + r + 1}\nc{row['cid']}\n{row['size']} crop\n"
                                f"{row['n_outlets']} do'kon\nz={row['tightness']:.2f}", fill=(0, 0, 0))
            page.paste(Image.open(rdir / row["strip"]), (110, y))
        page.save(rdir / f"page_{p // per_page + 1:03d}.jpg", quality=85)

    # Bitta mustaqil HTML: ma'lumot va montaj rasmlari ichiga joylanadi (operatorga
    # bitta fayl yuborish kifoya; tashqi fayl/internet kerak emas).
    import base64
    inline = []
    for row in rows:
        b64 = base64.b64encode((rdir / row["strip"]).read_bytes()).decode("ascii")
        inline.append({**row, "strip": "data:image/jpeg;base64," + b64})
    tpl = (ROOT / "tools" / "catalog_review.html").read_text(encoding="utf-8")
    data = "<script>window.CLUSTERS = " + json.dumps(inline) + ";</script>"
    html = tpl.replace('<script src="clusters.js"></script>', data)
    (rdir / "review.html").write_text(html, encoding="utf-8")
    mb = (rdir / "review.html").stat().st_size / 1e6
    print(f"  {len(rows)} klaster montaji -> {rdir / 'review.html'} ({mb:.1f} MB, mustaqil fayl)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["extract", "cluster", "review"])
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--per-cluster", type=int, default=40, help="k = crop'lar / shu son")
    ap.add_argument("--merge-thr", type=float, default=0.94)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--top", type=int, default=300, help="Operatorga ko'rsatiladigan klasterlar")
    ap.add_argument("--out", default=None, help="Natija papkasi (standart: data/catalog_build)")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = ROOT / args.out
    cfg = Config.load().override(args.set)
    if "extract" in args.stages:
        imgs = select_images(max_images=args.max_images)
        prev = OUT / "images.json"
        if prev.exists() and json.loads(prev.read_text()) != imgs:
            raise SystemExit(f"{OUT} boshqa rasm to'plami bilan qurilgan — papkani o'chiring "
                             f"yoki --out o'zgartiring.")
        print(f"[+] {len(imgs)} ta muzlatgich/javon rasmi")
        stage_extract(cfg, imgs)
    if "cluster" in args.stages:
        stage_cluster(cfg, args.per_cluster, args.merge_thr, args.min_size)
    if "review" in args.stages:
        stage_review(args.top)


if __name__ == "__main__":
    main()
