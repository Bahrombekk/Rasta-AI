"""Muzlatgich rasmlari: asosiy muzlatgichni topish, kesish va to'g'ridan (frontal) olinganini baholash.

1. Grounding DINO (Apache 2.0) "refrigerator" qutilari; bir nechta muzlatgichni o'rab olgan quti
   tashlanadi; asosiy muzlatgich = ishonch x sqrt(maydon) x markazga yaqinlik.
2. Frontallik kesilgan muzlatgichda, rasm chiziqlari bo'yicha (Canny + Hough):
   - javon chiziqlari (gorizontalga yaqin) qiyaligi: frontal rasmda ~0, qiyshiqda yuqori va pastki
     javonlar har xil tomonga og'adi (perspektiva);
   - yon chetlar (vertikalga yaqin) qiyaligi faqat ma'lumot uchun yoziladi.
   tilt_h = uzunlik bo'yicha o'rtacha |burchak| (gradus), spread_h = javon burchaklarining tarqoqligi.
3. Natija: --out JSON (har rasm: muzlatgich qutisi, tilt, spread, chiziqlar soni, frontal True/False).

    python scripts/fridge_frontal.py --list data/label_batches/b1.txt --out data/fridge_frontal_b1.json
    python scripts/fridge_frontal.py --all --out data/fridge_frontal_all.json   # barcha aptash muzlatgichlari
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APT = ROOT / "data" / "external" / "aptash"


def find_fridges(model, proc, img, thr=0.25):
    inp = proc(images=img, text="refrigerator. glass door fridge.", return_tensors="pt").to("cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        o = model(**inp)
    r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=thr, text_threshold=thr,
                                                    target_sizes=[img.size[::-1]])[0]
    return r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()


def main_fridge(b, s, w, h):
    if not len(b):
        return None, 0.0
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ix = (np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])).clip(0)
    iy = (np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])).clip(0)
    inside = ix * iy / a[None, :] > 0.8
    np.fill_diagonal(inside, False)
    group = inside.sum(1) >= 2  # bir nechta muzlatgichni o'rab olgan quti
    if (~group).any():
        b, s = b[~group], s[~group]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (w * h)
    cx = (b[:, 0] + b[:, 2]) / 2 / w
    k = int((s * np.sqrt(area) * (1 - np.abs(cx - 0.5))).argmax())
    return b[k], float(s[k])


def line_angles(crop):
    """Hough segmentlari: (burchak gradusda [-90, 90), uzunlik)."""
    h, w = crop.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 40, 120)
    L = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=60, minLineLength=int(0.25 * min(w, h)),
                        maxLineGap=int(0.02 * max(w, h)))
    if L is None:
        return np.zeros((0, 2))
    x1, y1, x2, y2 = L.reshape(-1, 4).astype(float).T
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    ang = (ang + 90) % 180 - 90
    return np.stack([ang, np.hypot(x2 - x1, y2 - y1)], 1)


def frontality(crop):
    a = line_angles(crop)
    hor = a[np.abs(a[:, 0]) < 30]  # javon chiziqlari
    ver = a[np.abs(a[:, 0]) > 60]  # yon chetlar
    ver = np.stack([90 - np.abs(ver[:, 0]), ver[:, 1]], 1) if len(ver) else ver
    wmean = lambda x: float(np.average(np.abs(x[:, 0]), weights=x[:, 1])) if len(x) else None
    spread = float(np.sqrt(np.cov(hor[:, 0], aweights=hor[:, 1]))) if len(hor) > 2 else None
    return {"tilt_h": wmean(hor), "tilt_v": wmean(ver), "spread_h": spread, "n_h": int(len(hor)), "n_v": int(len(ver))}


def is_frontal(f, max_tilt=3.5, max_spread=5.0):
    """Faqat javon chiziqlari bo'yicha. Yon chetlar o'lchovi kesilgan muzlatgich chetidan shovqinli:
    70 ta qo'lda ko'rilgan rasmda frontal rasmlarda ham 6-9 gradus chiqdi."""
    if f.get("n_h", 0) < 3 or f.get("tilt_h") is None:
        return False
    return f["tilt_h"] <= max_tilt and (f["spread_h"] or 0) <= max_spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="rasm ro'yxati (RASTA ichidagi nisbiy yo'llar)")
    ap.add_argument("--all", action="store_true", help="aptash'dagi barcha muzlatgich rasmlari")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()
    if args.all:
        index = json.loads((APT / "index.json").read_text())
        cls = json.loads((APT / "classes.json").read_text())
        paths = [f"data/external/aptash/{d}/{n}" for (d, n, _), c in zip(index, cls) if c and c["label"] == "fridge"]
    else:
        paths = [p for p in (ROOT / args.list).read_text(encoding="utf-8").split()]
    out = ROOT / args.out
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}  # davom ettirish
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).cuda().eval()
    for k, rel in enumerate(paths, 1):
        if rel in res:
            continue
        img = Image.open(ROOT / rel).convert("RGB")
        w, h = img.size
        box, score = main_fridge(*find_fridges(model, proc, img), w, h)
        rec = {"size": [w, h], "fridge": None, "score": score}
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            crop = cv2.cvtColor(np.array(img)[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            rec.update(fridge=[x0, y0, x1, y1], **frontality(crop))
        rec["frontal"] = is_frontal(rec)
        res[rel] = rec
        if k % 200 == 0 or k == len(paths):
            out.write_text(json.dumps(res), encoding="utf-8")
            print(f"{k}/{len(paths)} frontal {sum(r['frontal'] for r in res.values())}", flush=True)
    out.write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
