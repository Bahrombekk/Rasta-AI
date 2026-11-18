"""Katalogni saqlangan mahsulot rasmlaridan qayta indekslash.

Onboarding mantiqi yaxshilanganda (masalan packshot fonini kesish) eski vektorlar
eskirib qoladi. Bu skript har SKU uchun data/catalog_images/<sku>/*.jpg rasmlaridan
vektorlarni qayta hisoblaydi (nom va brend saqlanadi). Real javon crop'laridan
qo'shilgan (rasmi saqlanmagan) SKU'larga tegmaydi.

  python scripts/reindex_catalog.py            # hammasi
  python scripts/reindex_catalog.py --dry-run
Server ishlab turganda katalog fayli band bo'ladi — u holda API'dan foydalaning:
  curl -X POST http://localhost:8090/api/reindex
"""
import argparse

import _bootstrap  # noqa: F401

from rasta.config import Config, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = Config.load().override(args.set)
    cfg.set("detector.backend", "simple")
    from rasta.pipeline import RastaPipeline
    pipe = RastaPipeline(cfg)
    images_dir = resolve_path(cfg.get("web.images_dir", "data/catalog_images"))
    info = pipe.catalog.sku_info()
    print(f"[+] Katalog: {len(info)} SKU")
    # Veb-ilova bilan bir xil qoida: "c*" — javon crop'lari (augmentatsiyasiz), qolgani packshot
    from rasta.web import app as web
    web.State.images_dir = images_dir
    skipped = 0
    for sku, v in sorted(info.items()):
        files = web.image_files(sku)
        if not files:
            print(f"  {sku}: rasmlari saqlanmagan — o'tkazib yuborildi")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {sku}: {len(files)} rasm (eski vektorlar: {v['count']})")
            continue
        n = web.rebuild_sku(pipe, sku, v["name"], v.get("brand"), web.catalog_metadata(v))
        print(f"  {sku}: {len(files)} rasm -> {n} vektor (oldin {v['count']})")
    if not args.dry_run and not skipped:
        pipe.catalog.mark_current()
        print("[✓] Katalog joriy embedder bilan belgilandi")

if __name__ == "__main__":
    main()
