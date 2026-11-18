"""To'liq offline demo — internetsiz butun pipeline'ni namoyish qiladi.

Qadamlar:
  1. Sintetik katalog yaratish + few-shot onboarding (vektor bazaga qo'shish).
  2. Sintetik javon rasmi (ba'zi kataklar bo'sh).
  3. To'liq tahlil: tanish + OOS + KPI.
  4. Ground-truth bilan solishtirish (tanish aniqligi) + belgilangan rasm.

Foydalanish:
  python scripts/demo.py
  python scripts/demo.py --detector simple   # OpenCV detektorni ham sinash
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from rasta.config import Config, ROOT
from rasta.pipeline import RastaPipeline
from rasta.visualize import draw_result
from rasta.imageio import save_image
from rasta.types import Box, Detection
from rasta import synthetic


class GtDetector:
    """Ground-truth qutilarni qaytaruvchi detektor (deterministik demo uchun)."""
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, image):
        return list(self._boxes)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Rasta offline demo")
    ap.add_argument("--detector", choices=["gt", "simple"], default="gt",
                    help="gt = ground-truth qutilar; simple = OpenCV detektor")
    ap.add_argument("--variants", type=int, default=5, help="Har SKU uchun packshot soni")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Konfiguratsiyani demo uchun sozlaymiz (alohida katalog, tanlangan detektor)
    cfg = Config.load()
    cfg.raw["catalog"]["path"] = "data/qdrant_demo"
    cfg.raw["catalog"]["recreate_on_mismatch"] = True  # katalog baribir qayta yaratiladi
    cfg.raw["catalog"]["collection"] = "demo_skus"
    # Har doim 'simple' bilan quramiz (YOLO yuklab olishning oldini oladi);
    # 'gt' rejimida detektor keyinroq GtDetector bilan almashtiriladi.
    cfg.raw["detector"]["backend"] = "simple"

    print("[+] Modellar yuklanmoqda (DINOv2 embedding)...")
    pipe = RastaPipeline(cfg)

    # ---- 1. Katalogni tozalab, few-shot onboarding ----
    pipe.catalog.reset()

    print(f"[+] Sintetik katalog yaratilmoqda ({args.variants} variant/SKU)...")
    catalog = synthetic.generate_catalog(variants=args.variants)
    for sku_id, images in catalog.items():
        name = next(p.name for p in synthetic.PRODUCTS if p.sku_id == sku_id)
        brand = synthetic.BRAND_OF[sku_id]
        pipe.add_product(sku_id, name, images, brand=brand, source="packshot")
    print(f"[✓] Katalog: {pipe.catalog.count()} vektor, "
          f"{len(pipe.catalog.list_skus())} SKU.")

    # ---- 2. Sintetik javon ----
    print("[+] Sintetik javon rasmi yaratilmoqda...")
    shelf_img, gt_boxes, gt_labels = synthetic.generate_shelf(
        rows=3, cols=8, empty_slots=5, seed=7
    )
    save_image(shelf_img, out_dir / "demo_shelf.jpg")

    # ---- 3. Tahlil ----
    if args.detector == "gt":
        pipe.detector = GtDetector(gt_boxes)
    print(f"[+] Tahlil (detektor: {args.detector})...")
    result = pipe.analyze_image(
        shelf_img, image_path="demo_shelf.jpg", brand_of=synthetic.BRAND_OF
    )

    # ---- 4. Natijalar ----
    k = result.kpi
    print("\n===== DEMO NATIJASI =====")
    print(f"Javon qatorlari: {result.rows}")
    print(f"Mahsulotlar: {k['num_products']} "
          f"(tanildi: {k['num_known']}, noma'lum: {k['num_unknown']}, "
          f"daraja: {k['recognition_rate_pct']}%)")
    print("\n--- Share of Shelf ---")
    for sku, info in k["share_of_shelf"].items():
        print(f"  {info['name']:<16} facings={info['facings']:<3} {info['share_pct']}%")
    print("\n--- Brend ulushi ---")
    for brand, info in k["brand_share"].items():
        print(f"  {brand:<12} facings={info['facings']:<3} {info['share_pct']}%")
    print(f"\n--- OOS ---")
    print(f"  Haqiqiy bo'sh kataklar: 5 (yaratilganda)")
    print(f"  Aniqlangan bo'sh joylar: {k['oos']['num_gaps']}, "
          f"taxminiy yo'q facing: {k['oos']['est_missing_facings']}")
    print(f"  Barcha qatorlarda takrorlangan (bo'linma yoki butun ustun bo'sh): "
          f"{k['oos'].get('num_divider', 0)}  -> rasmda ko'k punktir")

    # Tanish aniqligini GT bilan solishtirish (faqat gt detektorda mos keladi)
    if args.detector == "gt":
        correct = 0
        for det, gt_label in zip(result.detections, gt_labels):
            if det.match and det.match.is_known and det.match.sku_id == gt_label:
                correct += 1
        acc = 100.0 * correct / len(gt_labels) if gt_labels else 0.0
        brand_ok = sum(1 for det, gl in zip(result.detections, gt_labels)
                       if det.match and det.match.level == "brand"
                       and det.match.brand == synthetic.BRAND_OF[gl])
        wrong = sum(1 for det, gl in zip(result.detections, gt_labels)
                    if det.match and det.match.is_known and det.match.sku_id != gl)
        print(f"\n--- Tanish aniqligi (GT bilan) ---")
        print(f"  SKU to'g'ri: {correct}/{len(gt_labels)} = {acc:.1f}%")
        print(f"  Brend darajasida to'g'ri (SKU egizaklar 0.5L/1.0L): {brand_ok}")
        print(f"  Noto'g'ri SKU (xato): {wrong}")
    print("=========================\n")

    # Belgilangan rasm + JSON
    vis = draw_result(shelf_img, result)  # noqa
    save_image(vis, out_dir / "demo_annotated.jpg")
    with open(out_dir / "demo_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"[✓] Belgilangan rasm: {out_dir / 'demo_annotated.jpg'}")
    print(f"[✓] JSON: {out_dir / 'demo_result.json'}")


if __name__ == "__main__":
    main()
