"""Perspektivani to'g'rilash (homografiya) foydasini o'lchash.

Qo'lda belgilangan real javon (data/gt/cola_cans_shelf.gt.json, 73 mahsulot)
rasmi sun'iy ravishda qiyshaytiriladi — agent javonni yon tomondan / pastdan
suratga olganday (keystone + aylanish). Zona = javonning 4 burchagi. So'ng:
  A) oddiy zona tahlili (kesish + kattalashtirish)
  B) homografiya bilan "old tomondan" ko'rinishga keltirib tahlil
Ikkalasida: detektsiya recall (GT bilan IoU>=0.5), SKU aniqligi, qatorlar soni.

  python scripts/benchmark_rectify.py
"""
import json

import _bootstrap  # noqa: F401

import cv2
import numpy as np

from eval_shelf import augment, evaluate, iou
from rasta.config import ROOT, Config
from rasta.imageio import load_image
from rasta.roi import make_region
from rasta.types import Box


def warp_setup(w, h, yaw, pitch, rot):
    """yaw: o'ng chet qisqaradi (yon tomondan ko'rinish), pitch: tepasi qisqaradi."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dy = yaw * h / 2
    dx = pitch * w / 2
    dst = np.float32([[dx, 0], [w - dx, dy], [w - dx, h - dy], [0, h]])
    c = dst.mean(0)
    R = cv2.getRotationMatrix2D(tuple(map(float, c)), rot, 1.0)
    dst = (np.hstack([dst, np.ones((4, 1))]) @ R.T).astype(np.float32)
    dst -= dst.min(0) - 40                     # chegara
    W, H = (dst.max(0) + 40).astype(int)
    return cv2.getPerspectiveTransform(src, dst), (int(W), int(H)), dst


def warp_box(M, b):
    c = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]], float).T
    q = M @ c
    q = q[:2] / q[2]
    return [float(q[0].min()), float(q[1].min()), float(q[0].max()), float(q[1].max())]


def recall(result, gt_objs):
    hit = 0
    n = 0
    for o in gt_objs:
        if o["label"] == "ignore":
            continue
        n += 1
        if any(iou(o["box"], [d.box.x1, d.box.y1, d.box.x2, d.box.y2]) >= 0.5
               for d in result.detections):
            hit += 1
    return hit / n


def main():
    from PIL import Image

    from rasta.pipeline import RastaPipeline

    gt = json.loads((ROOT / "data/gt/cola_cans_shelf.gt.json").read_text(encoding="utf-8"))
    img = load_image(ROOT / gt["image"])
    h, w = img.shape[:2]
    cfg = Config.load().override(["catalog.path=:memory:", "context.enabled=false"])
    pipe = RastaPipeline(cfg)
    pil = Image.fromarray(img)
    for sku, box in gt["catalog_boxes"].items():          # katalog — asl (frontal) rasmdan
        pipe.add_product(sku, sku, augment(pil.crop(tuple(box)), 6), brand=sku)

    # javon maydoni (rasm chetlaridagi devorsiz): asl rasmdagi 4 burchak
    shelf = np.float32([[20, 30], [w - 15, 30], [w - 15, h - 25], [20, h - 25]])
    cases = [(0, 0, 0), (0.15, 0, 0), (0.3, 0, 0), (0.45, 0, 0), (0.3, 0.15, 0),
             (0.3, 0, 8), (0.45, 0.2, -6)]
    rows = []
    for yaw, pitch, rot in cases:
        M, (W, H), _ = warp_setup(w, h, yaw, pitch, rot)
        wimg = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderValue=(40, 40, 40))
        # kichraytirish — agent rasmlari kabi (uzun tomoni 800 px)
        f = 800 / max(W, H)
        wimg = cv2.resize(wimg, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        S = np.diag([f, f, 1.0]) @ M
        objs = [{"box": warp_box(S, o["box"]), "label": o["label"]} for o in gt["objects"]]
        q = cv2.perspectiveTransform(shelf[None], S)[0]
        region = make_region(q, "javon")
        row = {"yaw": yaw, "pitch": pitch, "rot": rot}
        for mode in (False, True):
            pipe.cfg.set("roi.rectify", mode)
            res = pipe.analyze_region(wimg, region)
            ev = evaluate(res, objs)
            key = "rectify" if mode else "plain"
            row[key] = {"recall": round(recall(res, objs), 3), "acc": ev["accuracy"],
                        "n_det": len(res.detections), "rows": res.rows}
        rows.append(row)
        print(f"yaw={yaw:<4} pitch={pitch:<4} rot={rot:<3} | oddiy: recall {row['plain']['recall']:.3f} "
              f"aniqlik {row['plain']['acc']:.3f} qator {row['plain']['rows']} | "
              f"to'g'rilangan: recall {row['rectify']['recall']:.3f} aniqlik {row['rectify']['acc']:.3f} "
              f"qator {row['rectify']['rows']}", flush=True)
    (ROOT / "outputs" / "benchmark_rectify.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
