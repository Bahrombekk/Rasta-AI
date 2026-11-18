"""Real javon rasmi bo'yicha baholash (qo'lda belgilangan ground-truth bilan).

GT fayli (data/gt/*.gt.json): rasm, katalog uchun namuna qutilar
("catalog_boxes") va har bir mahsulot qutisi + yorlig'i ("objects").
Yorliq "unknown" — katalogda yo'q mahsulot (tizim uni rad etishi kerak),
"ignore" — baholanmaydi.

Ikki rejim:
  --mode gt    : GT qutilari detektor o'rniga beriladi -> faqat TANISH sifati
  --mode full  : haqiqiy detektor -> butun pipeline (detektsiya + tanish)

Foydalanish:
  python scripts/eval_shelf.py
  python scripts/eval_shelf.py --set recognition.w_color=0 --set recognition.color_min=0
  python scripts/eval_shelf.py --calibrate      # τ va margin'ni GT bo'yicha tanlash
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image, ImageEnhance

from rasta.config import ROOT, Config
from rasta.imageio import load_image
from rasta.types import Box


class FixedDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image):
        return list(self.boxes)


def augment(crop: Image.Image, n: int) -> list[np.ndarray]:
    """Bitta crop'dan n ta variant (yorqinlik/kontrast)."""
    out = [np.array(crop)]
    factors = [0.85, 1.15, 0.95, 1.1, 1.05, 0.9, 1.2]
    for i, f in enumerate(factors[: n - 1]):
        e = ImageEnhance.Brightness(crop).enhance(f)
        e = ImageEnhance.Contrast(e).enhance(1.0 + (i - 2) * 0.06)
        out.append(np.array(e))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collect_records(result, gt_objects):
    """Kalibrlash uchun: har GT obyekt -> (haqiqat, top-1, ball, margin)."""
    from rasta.calibrate import Rec
    recs = []
    for obj in gt_objects:
        if obj["label"] == "ignore":
            continue
        best, bd = 0.0, None
        for d in result.detections:
            v = iou(obj["box"], [d.box.x1, d.box.y1, d.box.x2, d.box.y2])
            if v > best:
                best, bd = v, d
        if bd is None or best < 0.5 or not (bd.match and bd.match.candidates):
            continue
        recs.append(Rec(obj["label"], bd.match.candidates[0]["sku_id"],
                        bd.match.score, bd.match.margin))
    return recs


def evaluate(result, gt_objects) -> dict:
    dets = result.detections
    used = set()
    n = correct = known_n = known_ok = unk_n = unk_ok = false_accept = missed = 0
    confusions = []
    for obj in gt_objects:
        lab = obj["label"]
        if lab == "ignore":
            continue
        best, bi = 0.0, -1
        for i, d in enumerate(dets):
            if i in used:
                continue
            v = iou(obj["box"], [d.box.x1, d.box.y1, d.box.x2, d.box.y2])
            if v > best:
                best, bi = v, i
        n += 1
        if lab == "unknown":
            unk_n += 1
        else:
            known_n += 1
        if best < 0.5:
            missed += 1
            confusions.append((lab, "<topilmadi>"))
            continue
        used.add(bi)
        m = dets[bi].match
        pred = m.sku_id if (m and m.is_known) else "unknown"
        if pred == lab:
            correct += 1
            if lab == "unknown":
                unk_ok += 1
            else:
                known_ok += 1
        else:
            confusions.append((lab, pred))
            if lab == "unknown" or (pred != "unknown"):
                false_accept += 1
    return {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct, "n": n,
        "known_acc": round(known_ok / known_n, 4) if known_n else 0.0,
        "unknown_reject": round(unk_ok / unk_n, 4) if unk_n else 0.0,
        "false_accept": false_accept,
        "missed": missed,
        "confusions": confusions,
    }


def build_pipe(cfg):
    from rasta.pipeline import RastaPipeline
    return RastaPipeline(cfg)


def run(cfg: Config, gt_path: Path, mode: str, shots: int, pipe=None, verbose=True):
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    img_path = ROOT / gt["image"]
    image = load_image(img_path)

    cfg = cfg.copy()
    cfg.set("catalog.path", ":memory:")
    cfg.set("catalog.collection", "eval_" + gt_path.stem.replace(".", "_"))
    cfg.set("detector.backend", "simple")  # YOLO faqat full rejimda kerak
    if pipe is None:
        pipe = build_pipe(cfg)
    else:
        pipe.cfg = cfg
        from rasta.catalog import RecognitionParams
        pipe.params = RecognitionParams.from_cfg(cfg)
    pipe.catalog.reset()

    pil = Image.fromarray(image)
    for sku, box in gt["catalog_boxes"].items():
        pipe.add_product(sku, sku, augment(pil.crop(tuple(box)), shots), brand=sku)

    if mode == "gt":
        pipe.detector = FixedDetector([Box(*o["box"], 1.0) for o in gt["objects"]])
    else:
        from rasta.detector import build_detector
        full = cfg.copy()
        full.set("detector.backend", "yolo")
        pipe.detector = build_detector(full, pipe.device)

    result = pipe.analyze_image(image, image_path=str(gt["image"]))
    ev = evaluate(result, gt["objects"])
    if verbose:
        print(json.dumps({k: v for k, v in ev.items() if k != "confusions"}))
        if ev["confusions"]:
            from collections import Counter
            print("  xatolar:", dict(Counter(f"{a}->{b}" for a, b in ev["confusions"])))
    return ev, result, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/gt/cola_cans_shelf.gt.json")
    ap.add_argument("--mode", choices=["gt", "full"], default="gt")
    ap.add_argument("--shots", type=int, default=6, help="Katalog uchun augment variantlar")
    ap.add_argument("--set", action="append", default=[], help="config o'zgartirish a.b=v")
    ap.add_argument("--config", default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="GT bo'yicha τ/margin tanlash (kontekst o'chiriladi)")
    args = ap.parse_args()
    cfg = Config.load(args.config).override(args.set)
    if args.calibrate:
        from rasta.calibrate import report
        cfg.set("context.enabled", False)
        cfg.set("recognition.sim_threshold", 0.0)   # barcha nomzodlar yozib olinadi
        cfg.set("recognition.min_margin", 0.0)
        _, result, _ = run(cfg, ROOT / args.gt, args.mode, args.shots, verbose=False)
        gt = json.loads((ROOT / args.gt).read_text(encoding="utf-8"))
        rep = report(collect_records(result, gt["objects"]))
        print(json.dumps(rep, indent=1))
        o = rep["oracle"]
        print(f"\nTavsiya (config.yaml -> recognition): sim_threshold: {o['tau']}  "
              f"min_margin: {o['margin']}")
        print("Eslatma: bitta rasmda kalibrlash optimistik — bir nechta GT rasm ishlating.")
        return
    run(cfg, ROOT / args.gt, args.mode, args.shots)


if __name__ == "__main__":
    main()
