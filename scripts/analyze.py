"""Javon rasmini (yoki papkadagi barcha rasmlarni) tahlil qilish:
SKU tanish, OOS, KPI, noma'lumlar klasterlari + belgilangan rasm.

Foydalanish:
  python scripts/analyze.py --image shelf.jpg
  python scripts/analyze.py --image shelf.jpg --json outputs/shelf.json
  python scripts/analyze.py --image data/samples/ --out outputs/batch   # papka
  python scripts/analyze.py --image shelf.jpg --set recognition.sim_threshold=0.6

Zona (poligon) bo'yicha — faqat kerakli muzlatgich/javon (tools/roi_editor.html):
  python scripts/analyze.py --image rasm.jpg --roi rasm.roi.json
  python scripts/analyze.py --image papka/ --roi zonalar.roi.json      # batch JSON
  python scripts/analyze.py --image rasm.jpg --roi-points "95,105;390,150;362,985;110,1040"
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from rasta.config import Config
from rasta.imageio import load_image, save_image
from rasta.visualize import cluster_montage, draw_result, draw_results

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def print_summary(result):
    k = result.kpi
    print("\n===== TAHLIL NATIJASI =====")
    print(f"Rasm: {result.image_path}  ({result.image_size[0]}x{result.image_size[1]})")
    print(f"Javon qatorlari: {result.rows}   (orqa fon qutilari: {len(result.background)})")
    print(f"Mahsulotlar: {k['num_products']} "
          f"(tanildi: {k['num_known']}, noma'lum: {k['num_unknown']}, "
          f"tanish darajasi: {k['recognition_rate_pct']}%)")
    print(f"Turli SKU: {k['distinct_skus']}   Noma'lum klasterlar: {k.get('num_unknown_clusters', 0)}")

    if k["share_of_shelf"]:
        print("\n--- Share of Shelf ---")
        for sku, info in k["share_of_shelf"].items():
            print(f"  {info['name']:<22} facings={info['facings']:<3} "
                  f"ulush={info['share_pct']}%")

    if k["brand_share"]:
        print("\n--- Brend ulushi ---")
        for brand, info in k["brand_share"].items():
            print(f"  {brand:<16} facings={info['facings']:<3} "
                  f"ulush={info['share_pct']}%")

    oos = k["oos"]
    print("\n--- OOS (bo'sh joy) ---")
    if k.get("scene_frontal") is False:
        print(f"  [!] Sahna frontal javon emas (javon chetida turgan mahsulotlar: "
              f"{100 * k['scene_shelf_support']:.0f}%) — OOS 'shubhali' deb belgilandi. "
              f"Javonni to'g'ridan-to'g'ri suratga oling.")
    print(f"  Bo'sh joylar: {oos['num_gaps']}, taxminiy yo'q facing: "
          f"{oos['est_missing_facings']}, shubhali: {oos.get('num_suspect', 0)}, "
          f"bo'linma?: {oos.get('num_divider', 0)}")
    for row, cnt in oos["by_row"].items():
        print(f"    Qator {row}: {cnt} ta yo'q")
    if oos.get("empty_shelf_zones"):
        print(f"    Butunlay bo'sh javon zonalari: {oos['empty_shelf_zones']} "
              f"(~{oos['empty_shelf_facings']} facing)")
    t = result.timings_ms
    if t:
        print(f"\nVaqt: {t.get('total', 0):.0f} ms (detektsiya {t.get('detect', 0):.0f}, "
              f"tanish {t.get('recognize', 0):.0f}, OOS {t.get('oos', 0):.0f})")
    print("===========================\n")


def load_roi_map(path: str) -> dict:
    """Zona JSON -> {rasm_nomi: [regions-dict]} (bitta rasm yoki batch formati)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for it in data.get("items", [data]):
        out[Path(str(it.get("image", ""))).name] = it.get("regions", [])
    return out


def regions_for(f: Path, image, roi_map: dict, args):
    from rasta.roi import make_region, parse_points
    h, w = image.shape[:2]
    if args.roi_points:
        return [make_region(parse_points(args.roi_points), "zona", (w, h))]
    regs = roi_map.get(f.name)
    if regs is None and len(roi_map) == 1 and len(list(roi_map.values())[0]) and args.roi             and not Path(args.image).is_dir():
        regs = list(roi_map.values())[0]          # bitta rasm + bitta JSON: nom farq qilsa ham
    if not regs:
        return None
    return [make_region(r["points"], r.get("name", f"zona{i + 1}"), (w, h))
            for i, r in enumerate(regs)]


def main():
    ap = argparse.ArgumentParser(description="Javon rasmini tahlil qilish")
    ap.add_argument("--image", required=True, help="Javon rasmi yoki papka")
    ap.add_argument("--out", default="outputs", help="Natijalar papkasi")
    ap.add_argument("--json", default=None, help="Natija JSON fayli (bitta rasm uchun)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="config o'zgartirish a.b=v")
    ap.add_argument("--no-vis", action="store_true", help="Rasm chizmaslik")
    ap.add_argument("--roi", default=None,
                    help="Zona JSON (tools/roi_editor.html): bitta rasm yoki batch {items:[...]}")
    ap.add_argument("--roi-points", default=None, help='Bitta zona: "x1,y1;x2,y2;..."')
    ap.add_argument("--roi-only", action="store_true",
                    help="Zonasi belgilanmagan rasmlarni o'tkazib yuborish")
    args = ap.parse_args()

    src = Path(args.image)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    if not files:
        raise SystemExit("Rasm topilmadi.")

    cfg = Config.load(args.config).override(args.set)
    print("[+] Modellar yuklanmoqda...")
    from rasta.pipeline import RastaPipeline
    pipe = RastaPipeline(cfg)
    if pipe.catalog.count() == 0:
        print("[!] Katalog bo'sh — barcha mahsulotlar 'noma'lum' bo'ladi. "
              "Avval scripts/add_product.py bilan SKU qo'shing.")

    roi_map = load_roi_map(args.roi) if args.roi else {}
    out_dir = Path(args.out)
    batch = []
    for f in files:
        image = load_image(f)
        regions = regions_for(f, image, roi_map, args)
        if regions is None and args.roi_only:
            continue
        if regions:
            print(f"[+] Tahlil qilinmoqda: {f}  ({len(regions)} ta zona)")
            results = pipe.analyze_regions(image, regions, image_path=str(f))
            for r in results:
                print(f"\n##### ZONA: {r.roi.name}  — to'ldirilganlik: javon qatorlari {r.kpi.get('roi_shelf_fill_pct')}% "
                      f"{r.kpi.get('roi_row_fill_pct')}, maydon {r.kpi.get('roi_fill_pct')}%")
                print_summary(r)
            if not args.no_vis:
                save_image(draw_results(image, results), out_dir / f"{f.stem}_zones.jpg")
            json_path = out_dir / f"{f.stem}_zones.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps({"image": str(f), "zones": [r.to_dict() for r in results]},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[✓] {json_path}")
            for r in results:
                batch.append({"image": str(f), "zone": r.roi.name,
                              "fill_pct": r.kpi.get("roi_fill_pct"),
                              "shelf_fill_pct": r.kpi.get("roi_shelf_fill_pct"),
                              **{k: r.kpi[k] for k in ("num_products", "num_known",
                                                       "recognition_rate_pct", "distinct_skus")},
                              "oos_gaps": r.kpi["oos"]["num_gaps"]})
            continue
        print(f"[+] Tahlil qilinmoqda: {f}")
        result = pipe.analyze_image(image, image_path=str(f))
        print_summary(result)
        if not args.no_vis:
            save_image(draw_result(image, result), out_dir / f"{f.stem}_annotated.jpg")
            mont = cluster_montage(image, result)
            if mont is not None:
                save_image(mont, out_dir / f"{f.stem}_unknown_clusters.jpg")
        json_path = Path(args.json) if (args.json and len(files) == 1) else out_dir / f"{f.stem}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"[✓] {json_path}")
        batch.append({"image": str(f), **{k: result.kpi[k] for k in
                      ("num_products", "num_known", "recognition_rate_pct", "distinct_skus")},
                      "oos_gaps": result.kpi["oos"]["num_gaps"]})

    if len(files) > 1:
        (out_dir / "batch_summary.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[✓] Umumiy xulosa: {out_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
