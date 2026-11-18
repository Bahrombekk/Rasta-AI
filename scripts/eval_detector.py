"""Detektorni qo'lda tekshirilgan GT qutilari bilan baholash va variantlarni solishtirish.

GT: data/gt_det/*.det.json (tools/box_editor.html da tuzatilgan, status "done").
Tayyorlash: python scripts/prelabel_detector.py --n 40

Foydalanish:
  python scripts/eval_detector.py
  python scripts/eval_detector.py --variant "tta: detector.tta.enabled=true" \
      --variant "tiles: detector.tta.enabled=true detector.tta.tiles=2"
  python scripts/eval_detector.py --device cpu --all      # draft fayllar ham
Har variant: "nom: kalit=qiymat kalit=qiymat" (config ustidan).
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from rasta.config import ROOT, Config, resolve_device, resolve_path
from rasta.det_eval import evaluate
from rasta.detector import build_detector, suppress_nested
from rasta.imageio import load_image


def load_gt(paths: list[Path], include_draft: bool):
    out = []
    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        if g.get("status") != "done" and not include_draft:
            continue
        out.append((p, g))
    return out


def parse_variant(s: str) -> tuple[str, list[str]]:
    name, _, rest = s.partition(":")
    return name.strip(), rest.split()


def run_variant(cfg: Config, gts, device: str):
    det = build_detector(cfg, device)
    items, per_image, t = [], [], 0.0
    for p, g in gts:
        img = load_image(resolve_path(g["image"]))
        t0 = time.perf_counter()
        boxes = det.detect(img)
        if cfg.get("detector.suppress_nested", True):
            boxes = suppress_nested(boxes, cfg.get("detector.contain_thr", 0.8))
        t += time.perf_counter() - t0
        pred = np.array([[b.x1, b.y1, b.x2, b.y2, b.score] for b in boxes]).reshape(-1, 5)
        gt = np.array(g.get("boxes", []), dtype=float).reshape(-1, 4)
        ign = np.array(g.get("ignore", []), dtype=float).reshape(-1, 4)
        items.append((pred, gt, ign))
        r = evaluate([items[-1]])
        per_image.append({"file": p.name, "gt": r.gt, "pred": r.pred, "recall": round(r.recall, 3),
                          "precision": round(r.precision, 3)})
    rep = evaluate(items).as_dict()
    rep["ms_per_image"] = round(1000 * t / max(1, len(gts)), 1)
    return rep, per_image


def main():
    ap = argparse.ArgumentParser(description="Detektor baholash (AP50, recall, sanash xatosi)")
    ap.add_argument("--gt", default="data/gt_det", help="papka yoki bitta .det.json")
    ap.add_argument("--variant", action="append", default=[],
                    help='"nom: kalit=qiymat ..." (bir necha marta berish mumkin)')
    ap.add_argument("--set", action="append", default=[], help="hamma variantlar uchun")
    ap.add_argument("--device", default=None)
    ap.add_argument("--all", action="store_true", help="status 'draft' fayllarni ham olish")
    ap.add_argument("--out", default="outputs/benchmark_detector.json")
    ap.add_argument("--per-image", action="store_true")
    args = ap.parse_args()

    src = resolve_path(args.gt)
    files = sorted(src.glob("*.det.json")) if src.is_dir() else [src]
    gts = load_gt(files, args.all)
    if not gts:
        raise SystemExit(f"GT topilmadi ({src}). Avval prelabel_detector.py + tools/box_editor.html; "
                         "draft fayllar uchun --all.")
    base = Config.load().override(args.set)
    device = args.device or resolve_device(base.get("device", "auto"))
    variants = [("base", [])] + [parse_variant(v) for v in args.variant]

    results = {}
    print(f"{len(gts)} ta rasm, device={device}")
    print(f"{'variant':<14}{'AP50':>7}{'recall':>8}{'prec':>7}{'F1':>7}{'MAE':>7}{'MAPE%':>7}{'ms':>8}{'bestF1':>8}{'@thr':>6}")
    for name, sets in variants:
        rep, per = run_variant(base.copy().override(sets), gts, device)
        results[name] = {"set": sets, **rep, **({"per_image": per} if args.per_image else {})}
        print(f"{name:<14}{rep['ap50']:>7.3f}{rep['recall']:>8.3f}{rep['precision']:>7.3f}"
              f"{rep['f1']:>7.3f}{rep['count_mae']:>7.1f}{rep['count_mape']:>7.1f}{rep['ms_per_image']:>8.0f}"
              f"{rep['best_f1']:>8.3f}{rep['best_thr']:>6.2f}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": [str(p.name) for p, _ in gts], "variants": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[✓] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
