""""Boshqa mahsulotlar" bazasi (negative bank) — open-set tanishni kuchaytirish uchun.

Katalog kichik bo'lsa, har qanday shisha katalogdagi eng yaqin SKU'ga "yopishadi"
(masalan bitta "Черноголовка" katalogida Lipton, Sprite, Pepsi shishalari ham shu nom
bilan tanilgan). Bu baza — foto arxividan (scripts/build_catalog_candidates.py)
olingan minglab REAL mahsulot klasterlarining markazlari. Tanishda so'rov ham
katalogga, ham shu bazaga solishtiriladi: bazadagi biror mahsulotga ko'proq o'xshasa
— "noma'lum". Katalogdagi SKU'ning o'ziga o'xshash klasterlar avtomatik chiqariladi.

  python scripts/build_negative_bank.py --build data/catalog_build_full
Natija: models/negatives/aptash_v1.npz
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np

import build_catalog_candidates as B
from rasta.config import ROOT, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="data/catalog_build_full")
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--out", default="models/negatives/aptash_v1.npz")
    args = ap.parse_args()

    B.OUT = ROOT / args.build
    D = B.load_shards()
    labels = np.load(B.OUT / "labels.npy")
    emb = D["emb"].astype(np.float32)
    col = D["color"].astype(np.float32)
    cids = [c for c in np.unique(labels) if c >= 0 and (labels == c).sum() >= args.min_size]
    E, C, A, S = [], [], [], []
    for c in cids:
        m = labels == c
        e = emb[m].mean(0); E.append(e / np.linalg.norm(e))
        k = col[m].mean(0); C.append(k / np.linalg.norm(k))
        A.append(float(np.median(D["aspect"][m])))
        S.append(int(m.sum()))
    cfg = Config.load()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=np.stack(E).astype(np.float16), color=np.stack(C).astype(np.float16),
                        aspect=np.array(A, np.float32), size=np.array(S, np.int32),
                        embedder=str(cfg.get("embedder.model")))
    print(f"[✓] {len(E)} ta mahsulot klasteri -> {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"embedder: {cfg.get('embedder.model')}")


if __name__ == "__main__":
    main()
