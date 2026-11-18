"""Yakuniy jadval: ESKI tizim vs YANGI tizim — barcha real datasetlarda.

ESKI : DINOv2-base, markaziy kesish, CLS, rangsiz, shaklsiz
YANGI: DINOv2-base(crop, cls+mean) x0.4 + SigLIP2-base(stretch) x0.6,
       rang 0.25, shakl 0.05   (config.yaml standarti)

Embeddinglar keshdan olinadi (avval benchmark_grocery.py / benchmark_rp2k.py
ishga tushirilgan bo'lishi kerak). Natija: outputs/benchmark_summary.json

  python scripts/benchmark_summary.py --seeds 20
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np

import benchmark_grocery as G
from benchmark_rp2k import RP
from rasta.config import ROOT


def norm(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def add_new(F):
    dm = norm(np.concatenate([F["dinov2b_crop_cls"], F["dinov2b_crop_mean"]], 1))
    F["NEW"] = norm(np.concatenate([np.sqrt(0.4) * dm, np.sqrt(0.6) * F["siglip2b_stretch_cls"]], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    datasets = {}
    tr = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                for p in (RP / "train").rglob("*.jpg"))
    te = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                for p in (RP / "test").rglob("*.jpg"))
    F = G.compute_features([p for p, _ in tr] + [p for p, _ in te], "rp2k", root=RP)
    add_new(F)
    datasets["RP2K (javon crop'lari, 120 SKU)"] = (
        F, None, list(range(len(tr))), [c for _, c in tr],
        np.arange(len(tr), len(tr) + len(te)), [c for _, c in te], {c for _, c in tr}, (1, 3, 5), 0.05)
    for sub, title in (("packages", "Grocery qadoqlar (31 SKU)"), ("all", "Grocery hammasi (81 sinf)")):
        classes, keep, train, test = G.load_index(sub)
        ic = [classes[c]["iconic"] for c in sorted(keep)]
        F2 = G.compute_features(ic + [p for p, _ in train] + [p for p, _ in test], f"grocery_{sub}")
        add_new(F2)
        off, off2 = len(ic), len(ic) + len(train)
        datasets[title] = (F2, {c: i for i, c in enumerate(sorted(keep))}, list(range(off, off2)),
                           [c for _, c in train], np.arange(off2, off2 + len(test)),
                           [c for _, c in test], keep, (0, 3, 5), 0.0)  # to'liq fotolar: shakl yo'q

    rows = []
    for name, (F, ng, ti, tl, qi, ql, cl, shots, ws) in datasets.items():
        for k in shots:
            old = G.run_protocol(F, ng, None, ti, tl, qi, ql, cl, "dinov2b_crop_cls", 0.0, k, args.seeds)
            new = G.run_protocol(F, ng, None, ti, tl, qi, ql, cl, "NEW", 0.25, k, args.seeds, w_shape=ws)
            row = {"dataset": name, "shots": k, "queries": new["n"],
                   "old": {"closed": old["closed_acc"], "auroc": old["auroc"],
                           "open": old["holdout"]["open_acc"]},
                   "new": {"closed": new["closed_acc"], "auroc": new["auroc"],
                           "open": new["holdout"]["open_acc"], "tau": new["holdout"]["tau"]}}
            rows.append(row)
            print(f"{name:<34} k={k}  closed {old['closed_acc']:.3f} -> {new['closed_acc']:.3f}   "
                  f"AUROC {old['auroc']:.3f} -> {new['auroc']:.3f}   "
                  f"open {old['holdout']['open_acc']:.3f} -> {new['holdout']['open_acc']:.3f}   "
                  f"(n={new['n']})", flush=True)
    (ROOT / "outputs" / "benchmark_summary.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] outputs/benchmark_summary.json — jami so'rovlar: {sum(r['queries'] for r in rows) * 2}")


if __name__ == "__main__":
    main()
