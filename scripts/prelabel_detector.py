"""Detektor GT'si uchun qoralama: rasmlarni tanlab, kuchli (TTA) detektor bilan
oldindan belgilaydi -> data/gt_det/*.det.json (status "draft").

Keyin tools/box_editor.html da qutilarni tuzatasiz (noto'g'risini o'chirish,
o'tkazib yuborilganini chizish) va "done" holatida saqlaysiz. Noldan chizishdan
bir necha barobar tez.

Rasmlar har xil agent/do'kondan teng tanlanadi (fayl nomidagi agent prefiksi bo'yicha).

Foydalanish:
  python scripts/prelabel_detector.py --n 40
  python scripts/prelabel_detector.py --src data/samples --n 8 --device cpu
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from rasta.config import ROOT, Config, resolve_device, resolve_path
from rasta.detector import build_detector, suppress_nested
from rasta.imageio import load_image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pick(files: list[Path], n: int, seed: int) -> list[Path]:
    """Agent prefiksi (img-a0_10129-...) bo'yicha navbatma-navbat tanlash."""
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"(img-[^-]+)-", f.name)
        groups[m.group(1) if m else f.parent.name].append(f)
    rnd = random.Random(seed)
    keys = list(groups)
    rnd.shuffle(keys)
    for k in keys:
        rnd.shuffle(groups[k])
    out = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser(description="Detektor GT qoralamasi (prelabel)")
    ap.add_argument("--src", default="data/external/aptash")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/gt_det")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-tta", action="store_true", help="oddiy detektor bilan")
    args = ap.parse_args()

    cfg = Config.load()
    if not args.no_tta:
        cfg.set("detector.tta.enabled", True)
    cfg.override(args.set)
    device = args.device or resolve_device(cfg.get("device", "auto"))
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(p.read_text(encoding="utf-8"))["image"] for p in out.glob("*.det.json")}

    files = [p for p in resolve_path(args.src).rglob("*") if p.suffix.lower() in EXTS]
    files = [p for p in files if p.relative_to(ROOT).as_posix() not in done]
    chosen = pick(sorted(files), args.n, args.seed)
    det = build_detector(cfg, device)
    for i, p in enumerate(chosen, 1):
        img = load_image(p)
        boxes = det.detect(img)
        if cfg.get("detector.suppress_nested", True):
            boxes = suppress_nested(boxes, cfg.get("detector.contain_thr", 0.8))
        rel = p.relative_to(ROOT).as_posix()
        rec = {"image": rel, "status": "draft", "size": [int(img.shape[1]), int(img.shape[0])],
               "boxes": [[round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)] for b in boxes],
               "scores": [round(b.score, 3) for b in boxes], "ignore": []}
        (out / f"{p.stem}.det.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"[{i}/{len(chosen)}] {rel}: {len(boxes)} quti")
    print(f"[✓] {out} — endi tools/box_editor.html da tuzating")


if __name__ == "__main__":
    main()
