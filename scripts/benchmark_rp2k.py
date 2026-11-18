"""Real javon crop'lari benchmarki — RP2K (scripts/fetch_rp2k.py bilan yuklangan qism).

Protokol benchmark_grocery.py bilan bir xil (few-shot, open-set, 25% sinf
noma'lum), lekin katalog faqat real javon crop'laridan (k ∈ {1, 3, 5}),
packshot yo'q. Crop'lar haqiqiy bo'lgani uchun SHAKL (tomonlar nisbati) ham
sinaladi.

  python scripts/benchmark_rp2k.py --seeds 20
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np

import benchmark_grocery as G
from rasta.config import ROOT

RP = ROOT / "data" / "external" / "rp2k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--wc", type=float, nargs="+", default=[0.0, 0.15, 0.3])
    ap.add_argument("--ws", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--out", default="outputs/benchmark_rp2k.json")
    args = ap.parse_args()

    train = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                   for p in (RP / "train").rglob("*.jpg"))
    test = sorted((str(p.relative_to(RP)).replace("\\", "/"), int(p.parent.name))
                  for p in (RP / "test").rglob("*.jpg"))
    classes = sorted({c for _, c in train})
    paths = [p for p, _ in train] + [p for p, _ in test]
    print(f"[+] RP2K: {len(classes)} SKU, {len(train)} katalog-nomzod, {len(test)} test crop")
    F = G.compute_features(paths, "rp2k", root=RP)

    tr_idx = list(range(len(train))); tr_lab = [c for _, c in train]
    q_idx = np.arange(len(train), len(paths)); q_lab = [c for _, c in test]

    norm = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
    embs = [f"{n}_cls" for n, _, _ in G.BACKBONES if f"{n}_cls" in F]
    embs += [e for e in ("dinov2b_crop_cls+mean", "dinov2b_stretch_cls+mean") if e.split("_cls")[0] + "_mean" in F]
    for a, b in (("dinov2b_crop_cls", "clipL_stretch_cls"), ("dinov2L_crop_cls", "clipL_stretch_cls"),
                 ("dinov2b_crop_cls", "siglip2b_stretch_cls"), ("clipL_stretch_cls", "siglip2b_stretch_cls")):
        if a in F and b in F:
            name = f"ens[{a.split('_')[0]}+{b.split('_')[0]}]_cls"
            F[name] = norm(np.concatenate([np.sqrt(0.4) * F[a], np.sqrt(0.6) * F[b]], 1))
            embs.append(name)

    results = []
    for emb in embs:
        for wc in args.wc:
            for ws in args.ws:
                for k in args.shots:
                    r = G.run_protocol(F, None, None, tr_idx, tr_lab, q_idx, q_lab, set(classes),
                                       emb, wc, k, args.seeds, w_shape=ws)
                    results.append({"emb": emb, "w_color": wc, "w_shape": ws, "shots": k, **r})
                    print(f"{emb:<30} wc={wc:<4} ws={ws:<4} k={k} closed={r['closed_acc']:.3f} "
                          f"auroc={r['auroc']:.3f} open(holdout)={r['holdout']['open_acc']:.3f} n={r['n']}",
                          flush=True)
    out = ROOT / args.out
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[✓] {out}")


if __name__ == "__main__":
    main()
