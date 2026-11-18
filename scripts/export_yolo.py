"""*.det.json qoralamalarini labelImg uchun YOLO papkasiga chiqarish (sinflar: 0 product, 1 empty).

Qoralamada faqat product (0) bo'ladi; javondagi bo'sh joylar (empty, 1) qo'lda chiziladi.

    python scripts/export_yolo.py --src data/gt_det_f1 --out data/label_f1

Natija: <out>/images/*.jpg (nusxa), <out>/labels/*.txt ("0 cx cy w h", 0..1), <out>/labels/classes.txt.
labelImg: Open Dir -> <out>/images, Change Save Dir -> <out>/labels, chap paneldan "YOLO" formatini tanlang.
Mavjud .txt fayllar ustidan yozilmaydi (tuzatilgan belgilar saqlanadi); --force bilan yoziladi.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "classes.txt").write_text("product\nempty\n", encoding="utf-8")
    n_img = n_box = 0
    for p in sorted((ROOT / args.src).glob("*.det.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        src = ROOT / rec["image"]
        dst = out / "images" / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        lab = out / "labels" / (src.stem + ".txt")
        if lab.exists() and not args.force:
            continue
        w, h = rec["size"]
        rows = []
        for x0, y0, x1, y1 in rec["boxes"]:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 > 1 and y1 - y0 > 1:
                rows.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        lab.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        n_img += 1
        n_box += len(rows)
    print(f"{n_img} rasm, {n_box} quti -> {out}")


if __name__ == "__main__":
    main()
