"""Operator oqimi: noma'lum mahsulotlar klasterlarini ko'rish va katalogga qo'shish.

1-qadam — ko'rish (montaj yaratiladi, har qator = bitta klaster):
  python scripts/review_unknowns.py --image shelf.jpg
  -> outputs/<rasm>_unknown_clusters.jpg

2-qadam — klasterga nom berib katalogga qo'shish (barcha crop'lari bilan):
  python scripts/review_unknowns.py --image shelf.jpg \
      --assign "0=chin8:Chin8 Neri:SanPellegrino" --assign "1=oransoda:Oransoda:Campari"

Format: <klaster>=<sku_id>:<nom>[:<brend>]
Klasterlash deterministik (bir xil rasm + katalog -> bir xil klaster raqamlari).
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from rasta.config import Config
from rasta.embedder import crop_boxes
from rasta.imageio import load_image, save_image
from rasta.visualize import cluster_montage, draw_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--assign", action="append", default=[])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = Config.load(args.config).override(args.set)
    from rasta.pipeline import RastaPipeline
    pipe = RastaPipeline(cfg)

    image = load_image(args.image)
    result = pipe.analyze_image(image, image_path=args.image)
    clusters = {k: v for k, v in result.clusters.items() if k != "noise"}
    print(f"[+] Noma'lum: {result.kpi['num_unknown']}, klasterlar: {len(clusters)}, "
          f"shovqin: {result.clusters.get('noise', {}).get('size', 0)}")
    for k, v in clusters.items():
        print(f"    klaster {k}: {v['size']} ta crop")

    stem = Path(args.image).stem
    mont = cluster_montage(image, result)
    if mont is not None:
        p = Path(args.out) / f"{stem}_unknown_clusters.jpg"
        save_image(mont, p)
        print(f"[✓] Montaj: {p}")

    if not args.assign:
        return
    for spec in args.assign:
        cid, _, rest = spec.partition("=")
        parts = rest.split(":")
        sku, name = parts[0], parts[1] if len(parts) > 1 else parts[0]
        brand = parts[2] if len(parts) > 2 else None
        if cid not in clusters:
            print(f"[!] klaster {cid} topilmadi — o'tkazib yuborildi")
            continue
        dets = [result.detections[i] for i in clusters[cid]["detections"]]
        crops = crop_boxes(image, [d.box for d in dets])
        n = pipe.add_product(sku, name, crops, brand=brand, augment=False)  # real crop'lar
        print(f"[✓] klaster {cid} -> '{name}' ({sku}): {n} vektor qo'shildi")

    result2 = pipe.analyze_image(image, image_path=args.image)
    k1, k2 = result.kpi, result2.kpi
    print(f"[✓] Qayta tahlil: tanish {k1['recognition_rate_pct']}% -> {k2['recognition_rate_pct']}%")
    save_image(draw_result(image, result2), Path(args.out) / f"{stem}_after_review.jpg")


if __name__ == "__main__":
    main()
