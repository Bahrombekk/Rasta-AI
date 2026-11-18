"""Qo'lda belgilash uchun har xil rasmlar to'plamini tanlash (aptash agent rasmlari).

Muzlatgich va javon rasmlaridan (classes.json), sifatli (ravshan, xira emas) va bir-biriga
o'xshamaydiganlarini tanlaydi: SigLIP embedding'larida k-means, har klasterdan eng tipik rasm,
bitta agentdan ko'pi bilan --per-agent ta. data/gt_det dagi rasmlar va yaqin dublikatlar chiqariladi.

    python scripts/select_label_batch.py --n 100 --out data/label_batches/b1.txt
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APT = ROOT / "data" / "external" / "aptash"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shelf-frac", type=float, default=0.3, help="javon ulushi, qolgani muzlatgich")
    ap.add_argument("--per-agent", type=int, default=3)
    ap.add_argument("--min-p", type=float, default=0.8, help="sahna klassi ishonchi")
    ap.add_argument("--exclude", nargs="*", default=[], help="oldingi to'plam ro'yxatlari")
    ap.add_argument("--out", default="data/label_batches/b1.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = json.loads((APT / "index.json").read_text())
    cls = json.loads((APT / "classes.json").read_text())
    emb = np.load(APT / "siglip_emb.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6

    skip = set()
    for p in (ROOT / "data" / "gt_det").glob("*.det.json"):
        skip.add(Path(json.loads(p.read_text(encoding="utf-8"))["image"]).name)
    for f in args.exclude:
        skip |= {Path(l).name for l in Path(f).read_text(encoding="utf-8").split()}

    ok = [i for i, c in enumerate(cls) if c and c["label"] in ("fridge", "shelf") and c["p"] >= args.min_p
          and index[i][1] not in skip and 60 <= c["bright"] <= 200]
    sharp = np.array([cls[i]["sharp"] for i in ok])
    ok = [i for i, s in zip(ok, sharp) if s >= np.percentile(sharp, 25)]  # eng xira chorakni tashlash

    agent = lambda i: (re.match(r"img-([^-]+)-", index[i][1]) or [None, index[i][1]])[1]
    rng = np.random.default_rng(args.seed)
    chosen = []
    for label, n in (("shelf", round(args.n * args.shelf_frac)), ("fridge", args.n - round(args.n * args.shelf_frac))):
        pool = np.array([i for i in ok if cls[i]["label"] == label])
        x = emb[pool]
        # k-means (sferik): har klasterdan uning markaziga eng yaqin rasm, ya'ni tipik rasm
        # (eng uzoq nuqta tanlovi g'ayrioddiy rasmlarni olardi: bo'sh muzlatgich, plakat)
        c = x[rng.choice(len(x), n, replace=False)]
        for _ in range(30):
            a = (x @ c.T).argmax(1)
            for k in range(n):
                m = x[a == k]
                if len(m):
                    c[k] = m.mean(0) / (np.linalg.norm(m.mean(0)) + 1e-6)
        sim, a = x @ c.T, (x @ c.T).argmax(1)
        used, sel = Counter(), []
        for k in np.argsort(-np.bincount(a, minlength=n)):  # avval katta klasterlar
            for j in np.argsort(-sim[:, k]):
                i = int(pool[j])
                if a[j] == k and used[agent(i)] < args.per_agent:
                    sel.append(i)
                    used[agent(i)] += 1
                    break
        chosen += sel
        print(f"{label}: {len(sel)} (hovuz {len(pool)}, agentlar {len(used)})")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"data/external/aptash/{index[i][0]}/{index[i][1]}" for i in chosen) + "\n",
                   encoding="utf-8")
    print(f"{len(chosen)} rasm -> {out}")


if __name__ == "__main__":
    main()
