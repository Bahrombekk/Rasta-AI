"""OOS benchmarki: butunlay bo'sh javon qatorlarini topish (javon chetlari moduli).

N ta sintetik 4 qavatli javon; har birida bitta qator BUTUNLAY bo'sh + tasodifiy
bo'sh kataklar, real kamera buzilishlari bilan. Taqqoslash:
  oos.empty_shelves = false  (faqat qator-asosli OOS — bo'sh qatorni ko'rmaydi)
  oos.empty_shelves = true   (javon chetlari bo'yicha bo'sh zonalar)

  python scripts/benchmark_oos.py --n 40
"""
import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np

from benchmark_synthetic import FixedDetector, slot_center
from rasta import synthetic as S
from rasta.config import ROOT, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    from rasta.pipeline import RastaPipeline

    results = {}
    for flag in (False, True):
        cfg = Config.load().override(["catalog.path=:memory:", "detector.backend=simple"])
        cfg.set("oos.empty_shelves", flag)
        pipe = RastaPipeline(cfg)
        det = FixedDetector()
        pipe.detector = det
        rows_found = 0
        by_row = {r: [0, 0] for r in range(4)}
        slots = covered = false_boxes = 0
        for i in range(args.n):
            rng = np.random.default_rng(100 + i)
            er = int(rng.integers(0, 4))
            img, boxes, _, empties = S.generate_shelf(
                rows=4, cols=8, empty_slots=int(rng.integers(1, 4)), seed=500 + i,
                products=S.ALL_PRODUCTS, block_size=2, return_empties=True, empty_rows=(er,))
            img = S.distort(img, seed=i, strength=1.0)
            det.boxes = boxes
            res = pipe.analyze_image(img)
            pts = [slot_center(r, c) for r, c in empties]

            def inside(g, x, y):
                return g.x1 <= x <= g.x2 and g.y1 - 5 <= y <= g.y2 + 5

            hit_row = sum(any(inside(g, *slot_center(er, c)) for g in res.gaps)
                          for c in range(8)) >= 6
            rows_found += hit_row
            by_row[er][0] += hit_row
            by_row[er][1] += 1
            slots += len(pts)
            covered += sum(any(inside(g, x, y) for g in res.gaps) for x, y in pts)
            false_boxes += sum(1 for g in res.gaps if not any(inside(g, x, y) for x, y in pts))
        results[f"empty_shelves={flag}"] = {
            "empty_rows_found": f"{rows_found}/{args.n}",
            "by_row(top=0)": {str(k): f"{v[0]}/{v[1]}" for k, v in by_row.items()},
            "empty_slot_recall": round(covered / slots, 4),
            "false_oos_boxes": false_boxes,
        }
        print(json.dumps({f"empty_shelves={flag}": results[f"empty_shelves={flag}"]},
                         ensure_ascii=False), flush=True)
    (ROOT / "outputs" / "benchmark_oos.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
