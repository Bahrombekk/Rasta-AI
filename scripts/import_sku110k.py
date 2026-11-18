"""SKU-110K arxividan (SKU110K_fixed.tar.gz) baholash to'plami va o'qitish datasetini tayyorlash.

Litsenziya: SKU-110K faqat akademik/notijorat foydalanish uchun. Tijoriy mahsulotda
o'z rasmlaringizda (data/gt_det) qo'shimcha fine-tune qiling.

  # 1) Detektorlarni solishtirish uchun test qismidan N ta rasm (arxiv ochilmaydi):
  python scripts/import_sku110k.py --eval 300
  python scripts/eval_detector.py --gt data/gt_det_sku110k --variant "..."
  # 2) O'qitish uchun to'liq YOLO formatidagi dataset:
  python scripts/import_sku110k.py --yolo
  python scripts/train_detector.py --data data/external/sku110k/yolo/sku110k.yaml ...

Arxiv: http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz (~13.6 GB)
"""
import argparse
import csv
import io
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from rasta.config import ROOT, resolve_path


def read_annotations(data: bytes) -> dict[str, list]:
    """CSV: image_name,x1,y1,x2,y2,class,image_width,image_height (sarlavhasiz)."""
    out = defaultdict(list)
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        if len(row) < 8 or not row[1].strip().lstrip("-").isdigit():
            continue
        name, x1, y1, x2, y2, _, w, h = row[:8]
        out[name].append((float(x1), float(y1), float(x2), float(y2), int(w), int(h)))
    return out


def eval_subset(archive: Path, n: int, img_dir: Path, gt_dir: Path) -> None:
    step = max(1, 2936 // n)
    wanted = {f"test_{i}.jpg" for i in range(0, 2936, step)}
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    ann = None
    with tarfile.open(archive, "r|gz") as t:          # oqim: arxiv diskka ochilmaydi
        for m in t:
            base = m.name.rsplit("/", 1)[-1]
            if base in wanted and m.isfile():
                (img_dir / base).write_bytes(t.extractfile(m).read())
            elif base == "annotations_test.csv":
                ann = read_annotations(t.extractfile(m).read())
    if ann is None:
        raise SystemExit("annotations_test.csv arxivda topilmadi")
    k = 0
    for p in sorted(img_dir.glob("test_*.jpg")):
        boxes = ann.get(p.name)
        if not boxes:
            continue
        rec = {"image": p.relative_to(ROOT).as_posix(), "status": "done", "source": "SKU-110K test",
               "size": [boxes[0][4], boxes[0][5]],
               "boxes": [[round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)] for b in boxes],
               "ignore": []}
        (gt_dir / f"{p.stem}.det.json").write_text(json.dumps(rec), encoding="utf-8")
        k += 1
    print(f"[✓] {k} ta test rasm -> {gt_dir.relative_to(ROOT)}")


def yolo_dataset(archive: Path, out: Path) -> None:
    raw = out.parent / "SKU110K_fixed"
    if not raw.exists():
        print("Arxiv ochilmoqda (bir necha daqiqa)...")
        with tarfile.open(archive, "r:gz") as t:
            # Python 3.10.11 da extractall(filter=...) yo'q — yo'llarni o'zimiz tekshiramiz
            safe = [m for m in t.getmembers() if (m.isfile() or m.isdir())
                    and m.name.startswith("SKU110K_fixed/") and ".." not in Path(m.name).parts]
            t.extractall(out.parent, members=safe)
    for split in ("train", "val", "test"):
        ann = read_annotations((raw / "annotations" / f"annotations_{split}.csv").read_bytes())
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for name, boxes in ann.items():
            src = raw / "images" / name
            if not src.exists():
                continue
            dst = out / "images" / split / name
            if not dst.exists():
                shutil.move(src, dst)
            lines = []
            for x1, y1, x2, y2, w, h in boxes:
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
            n += 1
        print(f"  {split}: {n} rasm")
    (out / "sku110k.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: object\n", encoding="utf-8")
    print(f"[✓] {(out / 'sku110k.yaml').relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="SKU-110K import (baholash / YOLO dataset)")
    ap.add_argument("--archive", default="data/external/sku110k/SKU110K_fixed.tar.gz")
    ap.add_argument("--eval", type=int, default=0, help="test qismidan shuncha rasm (GT bilan)")
    ap.add_argument("--yolo", action="store_true", help="to'liq YOLO formatidagi dataset")
    args = ap.parse_args()
    archive = resolve_path(args.archive)
    if args.eval:
        eval_subset(archive, args.eval, resolve_path("data/external/sku110k/eval"),
                    resolve_path("data/gt_det_sku110k"))
    if args.yolo:
        yolo_dataset(archive, resolve_path("data/external/sku110k/yolo"))
    if not (args.eval or args.yolo):
        ap.print_help()


if __name__ == "__main__":
    main()
