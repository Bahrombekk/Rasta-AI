"""Klasterlash benchmarki: noma'lum mahsulotlarni NIMA bo'yicha guruhlash kerak?

Savol: rang bo'yichami, shakl bo'yichami, yoki embedding bo'yichami?
Real javon crop'larida (RP2K) har xil xususiyat to'plamlari bilan HDBSCAN
ishlatiladi va natija haqiqiy SKU yorliqlari bilan solishtiriladi:

  ARI / NMI  — klasterlar haqiqiy SKU'larga qanchalik mos (1.0 = mukammal)
  purity     — klaster ichidagi eng ko'p SKU ulushi (operator uchun: bitta nom
               berganda nechta crop to'g'ri yorliq oladi)
  coverage   — shovqin (-1) bo'lmagan crop'lar ulushi

Har tajriba: tasodifiy N ta SKU (noma'lum mahsulotlar oqimi), har biridan
hamma test crop'lari; --trials marta takrorlanadi.

  python scripts/benchmark_clustering.py --trials 20 --skus 30
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np

from benchmark_rp2k import RP
import benchmark_grocery as G
from rasta.cluster import build_cluster_features, cluster_features, purity_and_coverage
from rasta.config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--skus", type=int, default=30, help="Har tajribadagi noma'lum SKU soni")
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--emb", default="siglip2b_stretch_cls")
    ap.add_argument("--out", default="outputs/benchmark_clustering.json")
    args = ap.parse_args()

    train = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                   for p in (RP / "train").rglob("*.jpg"))
    test = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                  for p in (RP / "test").rglob("*.jpg"))
    paths = [p for p, _ in train] + [p for p, _ in test]
    F = G.compute_features(paths, "rp2k", root=RP)
    q = np.arange(len(train), len(paths))
    lab = np.array([c for _, c in test])
    classes = sorted(set(lab.tolist()))

    emb = F[args.emb]
    col = F["color"]
    asp = F["aspect"]
    la = np.log(asp)[:, None]

    def only_color(i):
        return col[i]

    def only_shape(i):
        # shakl: tomonlar nisbati + (agar mavjud bo'lsa) kenglik/balandlik nisbati — 1D
        return np.concatenate([la[i], np.zeros((len(i), 1))], 1)

    def color_shape(i):
        v = np.concatenate([col[i], 0.5 * la[i]], 1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    variants = {
        "faqat_rang": only_color,
        "faqat_shakl": only_shape,
        "rang+shakl": color_shape,
        "embedding": lambda i: build_cluster_features(emb[i], None, None),
        "embedding+rang": lambda i: build_cluster_features(emb[i], col[i], None, 0.6, 0),
        "embedding+rang+shakl": lambda i: build_cluster_features(emb[i], col[i], asp[i], 0.6, 0.3),
        "embedding+rang(0.3)+shakl": lambda i: build_cluster_features(emb[i], col[i], asp[i], 0.3, 0.3),
        "embedding+rang(1.0)+shakl": lambda i: build_cluster_features(emb[i], col[i], asp[i], 1.0, 0.3),
    }

    results = {k: [] for k in variants}
    for t in range(args.trials):
        rng = np.random.default_rng(t)
        pick = set(rng.choice(classes, size=min(args.skus, len(classes)), replace=False).tolist())
        idx = q[np.isin(lab, list(pick))]
        truth = [str(x) for x in lab[np.isin(lab, list(pick))]]
        for name, fn in variants.items():
            labels = cluster_features(fn(idx), min_cluster_size=args.min_cluster)
            results[name].append(purity_and_coverage(labels, truth))

    summary = {}
    print(f"{'xususiyat':<28}{'ARI':>7}{'NMI':>7}{'purity':>8}{'coverage':>9}{'klaster':>8}")
    for name, rs in results.items():
        s = {k: float(np.mean([r[k] for r in rs])) for k in ("ARI", "NMI", "purity", "coverage",
                                                             "n_clusters")}
        summary[name] = {k: round(v, 4) for k, v in s.items()}
        print(f"{name:<28}{s['ARI']:>7.3f}{s['NMI']:>7.3f}{s['purity']:>8.3f}"
              f"{s['coverage']:>9.3f}{s['n_clusters']:>8.1f}")
    out = ROOT / args.out
    out.write_text(json.dumps({"args": vars(args), "true_clusters": args.skus,
                               "results": summary}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[✓] {out}  (haqiqiy SKU soni har tajribada: {args.skus})")


if __name__ == "__main__":
    main()
