"""Yangi mahsulotni katalogga qo'shish (FEW-SHOT — model qayta o'qitilmaydi).

Bu tizimning asosiy xususiyati: yangi SKU = katalogga bir nechta rasm.
Har rasm uchun 3 xil belgi saqlanadi: DINOv2 vektori, rang deskriptori,
tomonlar nisbati (shakl) — gibrid tanish uchun.

Foydalanish:
  python scripts/add_product.py --sku cola_05 --name "Cola 0.5L" --brand ColaCo --images path/to/dir
  python scripts/add_product.py --sku cola_05 --name "Cola 0.5L" --images a.jpg b.jpg --replace
  python scripts/add_product.py --list
  python scripts/add_product.py --delete cola_05
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

import _bootstrap  # noqa: F401

from rasta.config import Config
from rasta.imageio import load_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(f for f in path.rglob("*") if f.suffix.lower() in IMG_EXT))
        elif path.is_file() and path.suffix.lower() in IMG_EXT:
            files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser(description="SKU'ni katalogga few-shot qo'shish")
    ap.add_argument("--sku", help="SKU identifikatori")
    ap.add_argument("--name", help="Mahsulot nomi")
    ap.add_argument("--brand", default=None, help="Brend (ixtiyoriy)")
    ap.add_argument("--images", nargs="+", help="Rasm fayllari yoki papka(lar)")
    ap.add_argument("--replace", action="store_true", help="SKU vektorlarini yangilari bilan almashtirish")
    ap.add_argument("--augment", choices=["auto", "yes", "no"], default="auto",
                    help="auto: faqat 1 ta rasm bo'lsa augmentatsiya (standart)")
    ap.add_argument("--source", choices=["auto", "packshot", "crop"], default="auto",
                    help="packshot = toza katalog rasmi (pastroq τ), crop = real javon kesmasi")
    ap.add_argument("--list", action="store_true", help="Katalogdagi SKU'larni ko'rsatish")
    ap.add_argument("--delete", default=None, help="SKU'ni katalogdan o'chirish")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = Config.load(args.config).override(args.set)
    cfg.set("detector.backend", "simple")  # onboarding'da detektor kerak emas
    from rasta.pipeline import RastaPipeline
    pipe = RastaPipeline(cfg)

    if args.list:
        skus = pipe.catalog.list_skus()
        brands = pipe.catalog.brand_map()
        print(f"Katalog: {pipe.catalog.count()} vektor, {len(skus)} SKU")
        for s, n in sorted(skus.items()):
            print(f"  {s:<24} {n:>4} vektor  {brands.get(s, '')}")
        return
    if args.delete:
        pipe.catalog.delete_sku(args.delete)
        print(f"[✓] '{args.delete}' o'chirildi. Qoldi: {pipe.catalog.count()} vektor.")
        return
    if not (args.sku and args.name and args.images):
        ap.error("--sku, --name va --images kerak (yoki --list / --delete)")

    files = collect_images(args.images)
    if not files:
        raise SystemExit("Rasm topilmadi. --images to'g'ri yo'l ekanini tekshiring.")
    print(f"[+] {len(files)} ta rasm topildi.")

    images = [load_image(f) for f in files]
    aug = {"auto": None, "yes": True, "no": False}[args.augment]
    src = None if args.source == "auto" else args.source
    with pipe.catalog.replace_sku(args.sku) if args.replace else nullcontext():
        n = pipe.add_product(args.sku, args.name, images, brand=args.brand, augment=aug, source=src)

    print(f"[✓] '{args.name}' ({args.sku}) katalogga qo'shildi: {n} ta vektor.")
    print(f"    Katalogda jami: {pipe.catalog.count()} vektor, "
          f"{len(pipe.catalog.list_skus())} SKU.")


if __name__ == "__main__":
    main()
