"""Real javon GT bo'yicha konfiguratsiyalar sweep'i (ablation).

Embedder preprocessing / pooling, rang va shakl og'irliklari, rang veto'si,
chegaralar va kontekstni bir jarayonda solishtiradi (model bir marta yuklanadi).

  python scripts/sweep_shelf.py --out outputs/sweep_shelf.json
"""
import argparse
import itertools
import json

import _bootstrap  # noqa: F401

from rasta.catalog import SkuCatalog
from rasta.config import ROOT, Config
from eval_shelf import build_pipe, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", nargs="+", default=["data/gt/cola_cans_shelf.gt.json"])
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--out", default="outputs/sweep_shelf.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    base = Config.load()
    base.set("catalog.path", ":memory:")
    base.set("detector.backend", "simple")
    pipe = build_pipe(base)

    grid = {
        "embedder.resize_mode": ["crop", "pad", "stretch"],
        "embedder.pooling": ["cls", "cls+mean"],
        "recognition.w_color": [0.0, 0.2, 0.35],
        "recognition.color_min": [0.0, 0.5, 0.6],
        "recognition.sim_threshold": [0.5, 0.55, 0.6, 0.65],
        "context.enabled": [False, True],
    }
    if args.quick:
        grid["recognition.sim_threshold"] = [0.55]
        grid["recognition.color_min"] = [0.0, 0.55]
    keys = list(grid)
    rows = []
    last_embed = None
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = base.copy()
        for k, v in zip(keys, combo):
            cfg.set(k, v)
        emb_key = (cfg.get("embedder.resize_mode"), cfg.get("embedder.pooling"))
        if emb_key != last_embed:
            pipe.embedder.resize_mode, pipe.embedder.pooling = emb_key
            pipe.embedder._dim = None
            last_embed = emb_key
        for gt in args.gt:
            cfg.set("catalog.collection", f"sweep_{emb_key[0]}_{emb_key[1].replace('+', '_')}")
            pipe.catalog = SkuCatalog(pipe.embedder.dim, client=pipe.catalog.client,
                                      collection=cfg.get("catalog.collection"))
            ev, _, _ = run(cfg, ROOT / gt, "gt", args.shots, pipe=pipe, verbose=False)
            ev = {k: v for k, v in ev.items() if k != "confusions"}
            row = dict(zip(keys, combo), gt=gt, **ev)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    rows.sort(key=lambda r: (-r["accuracy"], r["false_accept"]))
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== TOP 10 ===")
    for r in rows[:10]:
        print(r)


if __name__ == "__main__":
    main()
