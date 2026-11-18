"""Real rasmdan few-shot onboarding + to'liq tahlil (namoyish uchun).

Rastadan bir nechta namuna mahsulotni crop qilib katalogga o'rgatadi
(model qayta o'qitilmaydi), so'ng butun rastani tanib, KPI va OOS chiqaradi.
"""
import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from rasta.config import Config, ROOT
from rasta.pipeline import RastaPipeline
from rasta.visualize import cluster_montage, draw_result
from rasta.imageio import load_image, save_image

IMAGE = "data/samples/cola_cans_shelf.jpg"

# (sku_id, nom, brend, x1,y1,x2,y2) — 1280x852 rasmdagi namuna qutilar
PRODUCTS = [
    ("wine_beverino", "Beverino Vino Rosso", "Beverino", 62, 96, 128, 248),
    ("pepsi",         "Pepsi",               "PepsiCo",  158, 300, 232, 448),
    ("cocacola",      "Coca-Cola",           "Coca-Cola", 486, 295, 560, 448),
    ("fanta",         "Fanta",               "Coca-Cola", 424, 462, 484, 604),
    ("lemonsoda",     "Lemon Soda",          "LemonSoda", 915, 298, 988, 448),
    ("sanpellegrino", "San Pellegrino",      "SanPellegrino", 492, 462, 548, 604),
]

BRAND_OF = {p[0]: p[2] for p in PRODUCTS}


def augment(crop: Image.Image, n: int = 6) -> list[np.ndarray]:
    """Bitta crop'dan bir nechta variant (yorqinlik/kontrast/kichik masshtab)."""
    variants = [np.array(crop)]
    factors = [0.85, 1.15, 0.95, 1.1, 1.05]
    for i, f in enumerate(factors[: n - 1]):
        e = ImageEnhance.Brightness(crop).enhance(f)
        e = ImageEnhance.Contrast(e).enhance(1.0 + (i - 2) * 0.06)
        variants.append(np.array(e))
    return variants


def main():
    cfg = Config.load()
    # Alohida katalog (asosiy bazani ifloslantirmaslik uchun)
    cfg.raw["catalog"]["path"] = "data/qdrant_real"
    cfg.raw["catalog"]["recreate_on_mismatch"] = True  # katalog baribir qayta yaratiladi
    cfg.raw["catalog"]["collection"] = "cola_shelf"

    print("[+] Modellar yuklanmoqda...")
    pipe = RastaPipeline(cfg)
    pipe.catalog.reset()

    # ---- Few-shot onboarding ----
    im = Image.open(ROOT / IMAGE).convert("RGB")
    print(f"[+] {len(PRODUCTS)} ta namuna mahsulot katalogga o'rgatilmoqda (few-shot)...")
    for sku, name, brand, x1, y1, x2, y2 in PRODUCTS:
        crop = im.crop((x1, y1, x2, y2))
        variants = augment(crop, n=6)
        pipe.add_product(sku, name, variants, brand=brand)
        print(f"    [✓] {name:<20} ({len(variants)} variant)")
    print(f"[✓] Katalog: {pipe.catalog.count()} vektor, "
          f"{len(pipe.catalog.list_skus())} SKU.")

    # ---- To'liq tahlil ----
    print("[+] Butun rasta tahlil qilinmoqda...")
    shelf = load_image(ROOT / IMAGE)
    result = pipe.analyze_image(shelf, image_path=IMAGE, brand_of=BRAND_OF)

    k = result.kpi
    print("\n===== REAL RASTA TAHLILI =====")
    print(f"Javon qatorlari: {result.rows}")
    print(f"Mahsulotlar: {k['num_products']} "
          f"(tanildi: {k['num_known']}, noma'lum: {k['num_unknown']}, "
          f"daraja: {k['recognition_rate_pct']}%)")
    print("\n--- Share of Shelf ---")
    for sku, info in k["share_of_shelf"].items():
        print(f"  {info['name']:<20} facings={info['facings']:<3} {info['share_pct']}%")
    print("\n--- Brend ulushi ---")
    for brand, info in k["brand_share"].items():
        print(f"  {brand:<16} facings={info['facings']:<3} {info['share_pct']}%")
    print(f"\n--- OOS ---")
    print(f"  Bo'sh joylar: {k['oos']['num_gaps']}, "
          f"taxminiy yo'q facing: {k['oos']['est_missing_facings']}")
    print("==============================\n")

    mont = cluster_montage(shelf, result)
    if mont is not None:
        save_image(mont, ROOT / "outputs" / "cola_real_unknown_clusters.jpg")
        print("[✓] Noma'lum klasterlar: outputs/cola_real_unknown_clusters.jpg")
    vis = draw_result(shelf, result)
    save_image(vis, ROOT / "outputs" / "cola_real_recognized.jpg")
    with open(ROOT / "outputs" / "cola_real_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print("[✓] Belgilangan rasm: outputs/cola_real_recognized.jpg")
    print("[✓] JSON: outputs/cola_real_result.json")


if __name__ == "__main__":
    main()
