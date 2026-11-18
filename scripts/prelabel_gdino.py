"""Belgilash qoralamasi toza (Apache 2.0) model bilan: Grounding DINO, SKU-110K'siz.

SKU-110K faqat notijorat, undan o'qitilgan detektor chiqishi ham hosila bo'ladi. Shuning uchun
tijoriy o'qitish ma'lumotining qoralamasi Grounding DINO (IDEA-Research/grounding-dino-base) bilan
chiziladi, keyin tools/box_editor.html da qo'lda tuzatiladi.

Har rasm: butun rasm + bir-biriga kiradigan bo'laklar (kichik mahsulotlar uchun), NMS bilan
birlashtiriladi. Chiqariladi: rasmning chorakidan katta qutilar, ichida 3 tadan ko'p quti bor
"qator" qutilari, chap yuqori burchakdagi agent ilovasi logotipi.

    python scripts/prelabel_gdino.py --list data/label_batches/b1.txt --out data/gt_det_b1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "bottle. can. package. box. bag. jar. product."


def boxes_in(model, proc, img, thr):
    inp = proc(images=img, text=PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        o = model(**inp)
    r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=thr, text_threshold=thr,
                                                    target_sizes=[img.size[::-1]])[0]
    return r["boxes"].float().cpu(), r["scores"].float().cpu()


def detect(model, proc, img, thr, grid):
    w, h = img.size
    b, s = boxes_in(model, proc, img, thr)
    allb, alls = [b], [s]
    gx, gy = grid
    tw, th = w / gx * 1.3, h / gy * 1.3  # 30% ustma-ust
    for i in range(gx):
        for j in range(gy):
            x0 = min(max(0, i * w / gx - (tw - w / gx) / 2), w - tw)
            y0 = min(max(0, j * h / gy - (th - h / gy) / 2), h - th)
            crop = img.crop((int(x0), int(y0), int(x0 + tw), int(y0 + th)))
            b, s = boxes_in(model, proc, crop, thr)
            if len(b):
                # bo'lak chetiga tegib turgan qutilar kesilgan mahsulot: ularni butun rasm beradi
                cw, ch = crop.size
                inner = ((b[:, 0] > 2) | (x0 < 1)) & ((b[:, 1] > 2) | (y0 < 1)) & \
                        ((b[:, 2] < cw - 2) | (x0 + tw > w - 1)) & ((b[:, 3] < ch - 2) | (y0 + th > h - 1))
                b, s = b[inner] + torch.tensor([int(x0), int(y0), int(x0), int(y0)]), s[inner]
                allb.append(b)
                alls.append(s)
    b, s = torch.cat(allb), torch.cat(alls)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ok = (area < 0.25 * w * h) & (area > 0)
    logo = (b[:, 2] < 0.12 * w) & (b[:, 3] < 0.09 * h)  # agent ilovasi belgisi
    b, s = b[ok & ~logo], s[ok & ~logo]
    keep = torchvision.ops.nms(b, s, 0.5)
    b, s = b[keep], s[keep]
    # "qator" qutisi: ichida 3 tadan ko'p boshqa quti
    if len(b):
        x0 = torch.maximum(b[:, None, 0], b[None, :, 0]); y0 = torch.maximum(b[:, None, 1], b[None, :, 1])
        x1 = torch.minimum(b[:, None, 2], b[None, :, 2]); y1 = torch.minimum(b[:, None, 3], b[None, :, 3])
        inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        contained = (inter / area[None, :] > 0.8).float()  # [i, j]: j asosan i ichida
        contained.fill_diagonal_(0)
        row = contained.sum(1) > 3
        b, s = b[~row], s[~row]
    return b.numpy(), s.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--grid", default="2,3", help="bo'laklar: ustun,qator")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).cuda().eval()
    grid = tuple(int(v) for v in args.grid.split(","))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    paths = (ROOT / args.list).read_text(encoding="utf-8").split()
    for k, rel in enumerate(paths, 1):
        dst = out / (Path(rel).stem + ".det.json")
        if dst.exists():  # tuzatilgan qoralamani hech qachon ustidan yozmaymiz
            continue
        img = Image.open(ROOT / rel).convert("RGB")
        b, s = detect(model, proc, img, args.thr, grid)
        rec = {"image": rel, "status": "draft", "size": list(img.size),
               "boxes": [[round(float(v), 1) for v in x] for x in b],
               "scores": [round(float(v), 3) for v in s], "ignore": [],
               "source": f"grounding-dino-base thr {args.thr} grid {args.grid}"}
        dst.write_text(json.dumps(rec), encoding="utf-8")
        print(f"{k}/{len(paths)} {Path(rel).name}: {len(b)}", flush=True)


if __name__ == "__main__":
    main()
