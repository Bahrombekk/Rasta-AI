"""Frontal muzlatgichlarni kesib chiqarish va belgilash uchun to'plam tanlash.

fridge_frontal.py natijasidan (data/fridge_frontal_all.json):
  1. frontal deb topilgan har bir muzlatgich qutisi (+ --margin chet) kesilib data/fridge_frontal/ ga
     yoziladi (asl rasm o'zgarmaydi); crops.json da har kesmaning manbasi va siljishi (x0, y0) bor,
     shu bilan kesmadagi qutilar asl rasmga qaytariladi;
  2. kesmalardan --n tasi belgilash uchun tanlanadi: SigLIP embedding'larida k-means, har klasterdan
     eng tipik rasm, bitta agentdan ko'pi bilan --per-agent ta -> --list.

    python scripts/make_fridge_batch.py --n 100 --list data/label_batches/f1.txt
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APT = ROOT / "data" / "external" / "aptash"


def quality(crops_dir, meta):
    """Kesma sifati belgilari, crops_dir/quality.json da keshlanadi."""
    path = crops_dir / "quality.json"
    q = {f["name"]: f for f in json.loads(path.read_text())} if path.exists() else {}
    for n in meta:
        if n in q:
            continue
        g = cv2.imread(str(crops_dir / n), cv2.IMREAD_GRAYSCALE)
        s = 800 / g.shape[0]
        g8 = cv2.resize(g, (max(1, int(g.shape[1] * s)), 800), interpolation=cv2.INTER_AREA)
        w, h = meta[n]["size"]
        q[n] = {"name": n, "w": w, "h": h, "aspect": h / w, "sharp": float(cv2.Laplacian(g8, cv2.CV_64F).var()),
                "bright": float(g.mean()), "glare": float((g > 245).mean())}
    path.write_text(json.dumps(list(q.values())), encoding="utf-8")
    return q


def good_crop(f):
    """f1 to'plamida qo'lda tashlangan 20 kesmadan 17 tasini chiqaradi: yassi (bir nechta eshik yoki
    noto'g'ri kesilgan), xira, juda qorong'i yoki yorug', past o'lchamli kesmalar."""
    return (1.5 <= f["aspect"] <= 3.2 and f["sharp"] >= 500 and 90 <= f["bright"] <= 150
            and f["glare"] <= 0.03 and f["h"] >= 700)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontal", default="data/fridge_frontal_all.json")
    ap.add_argument("--crops", default="data/fridge_frontal")
    ap.add_argument("--margin", type=float, default=0.03, help="kesma cheti, quti o'lchamiga nisbatan")
    ap.add_argument("--min-side", type=int, default=250, help="bundan kichik kesmalar tashlanadi (px)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-agent", type=int, default=2)
    ap.add_argument("--list", default="data/label_batches/f1.txt")
    ap.add_argument("--exclude", nargs="*", default=[], help="oldingi to'plam ro'yxatlari")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", action="store_true",
                    help="faqat sifatli kesmalar (good_crop): tik muzlatgich, keskin, me'yoriy yorug'lik")
    args = ap.parse_args()

    res = json.loads((ROOT / args.frontal).read_text(encoding="utf-8"))
    crops_dir = ROOT / args.crops
    crops_dir.mkdir(parents=True, exist_ok=True)
    meta_path = crops_dir / "crops.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    for rel, r in res.items():
        if not r["frontal"]:
            continue
        name = Path(rel).stem + "_fridge.jpg"
        if name in meta:
            continue
        x0, y0, x1, y1 = r["fridge"]
        w, h = r["size"]
        mx, my = args.margin * (x1 - x0), args.margin * (y1 - y0)
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
        if min(x1 - x0, y1 - y0) < args.min_side:
            continue
        img = cv2.imread(str(ROOT / rel))
        cv2.imwrite(str(crops_dir / name), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta[name] = {"source": rel, "offset": [x0, y0], "size": [x1 - x0, y1 - y0],
                      "tilt_h": r["tilt_h"], "spread_h": r["spread_h"]}
    meta_path.write_text(json.dumps(meta, indent=0), encoding="utf-8")
    print(f"frontal kesmalar: {len(meta)} -> {crops_dir}")

    # belgilash to'plami
    index = json.loads((APT / "index.json").read_text())
    row = {f"data/external/aptash/{d}/{n}": i for i, (d, n, _) in enumerate(index)}
    emb = np.load(APT / "siglip_emb.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6
    skip = set()
    for f in args.exclude:
        skip |= {Path(l).name for l in (ROOT / f).read_text(encoding="utf-8").split()}
    names = [n for n in sorted(meta) if n not in skip]
    if args.quality:
        q = quality(crops_dir, meta)
        before = len(names)
        names = [n for n in names if good_crop(q[n])]
        print(f"sifat filtri: {before} -> {len(names)} kesma")
    x = emb[[row[meta[n]["source"]] for n in names]]
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(names))
    c = x[rng.choice(len(x), n, replace=False)]
    for _ in range(30):
        a = (x @ c.T).argmax(1)
        for k in range(n):
            m = x[a == k]
            if len(m):
                c[k] = m.mean(0) / (np.linalg.norm(m.mean(0)) + 1e-6)
    sim = x @ c.T
    a = sim.argmax(1)
    agent = lambda nm: (re.match(r"img-([^-]+)-", nm) or [None, nm])[1]
    used, sel = Counter(), []
    for k in np.argsort(-np.bincount(a, minlength=n)):
        for j in np.argsort(-sim[:, k]):
            if a[j] == k and used[agent(names[j])] < args.per_agent:
                sel.append(names[j])
                used[agent(names[j])] += 1
                break
    out = ROOT / args.list
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{args.crops}/{nm}" for nm in sel) + "\n", encoding="utf-8")
    print(f"belgilash to'plami: {len(sel)} kesma ({len(used)} agent) -> {out}")


if __name__ == "__main__":
    main()
