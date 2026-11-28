"""Klasterlash benchmarki — «bitta javon»ga o'xshash to'plamlar (RP2K oynalari).

RP2K kesmalari har xil rasmlardan olingan, shuning uchun tasodifiy SKU to'plami haqiqiy tahlilga
o'xshamaydi: bitta javon rasmida hamma quti bir xil yorug'likda turadi va bir brendning ta'mlari
yonma-yon bo'ladi. RP2K fayl nomi «<kun>_<annotatsiya raqami>»: bir kunda ketma-ket raqamli kesmalar
bir do'kon/javondan. Har oyna — 60 ta ketma-ket kesma (o'rtacha 58 quti, 21 SKU, 9 tasi yakka).

Ko'rsatkichlar operator nuqtai nazaridan: har guruhga bitta nom beriladi (ko'pchilik SKU), guruhsiz
qutilar bittalab nomlanadi.
  xato   — noto'g'ri nom olgan qutilar ulushi
  amal   — nomlash amallari soni / haqiqiy SKU soni (1.0 — ideal)

Oynalarning birinchi yarmi — sozlash, ikkinchi yarmi — mustaqil tekshiruv.

  python scripts/benchmark_cluster_windows.py            # kerak bo'lsa ~140 MB yuklaydi
"""
import argparse
import collections
import json
import random
import re
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401

import numpy as np

import benchmark_grocery as G
from fetch_rp2k import URL
from rasta.cluster import cluster_unknown_features, purity_and_coverage, split_variants
from rasta.config import Config, ROOT

RP = ROOT / "data" / "external" / "rp2k"
OUT = ROOT / "data" / "external" / "rp2k_win"


def choose_windows(info, count, width, seed):
    by_day = collections.defaultdict(list)
    for name, size, *_ in info:
        parts = name.split("/")
        found = re.match(r"(\d{4})_(\d+)", parts[3]) if len(parts) == 4 and size > 0 else None
        if found:
            by_day[found.group(1)].append((int(found.group(2)), parts[2], name))
    days = sorted(day for day, rows in by_day.items() if len(rows) >= 3000)
    rows_of = {day: sorted(by_day[day]) for day in days}
    rng = random.Random(seed)
    windows = []
    for _ in range(20000):
        if len(windows) == count:
            break
        rows = rows_of[rng.choice(days)]
        start = rng.randrange(0, len(rows) - width)
        if rows[start + width - 1][0] - rows[start][0] > 4 * width:      # raqamlar uzilgan — boshqa javon
            continue
        window = [row for row in rows[start:start + width] if row[1] != "others"]
        if len(window) >= 25 and len({row[1] for row in window}) >= 5:
            windows.append(window)
    return windows


def fetch(windows, info, workers=12):
    import requests

    meta = {row[0]: row for row in info}
    jobs, index = [], []
    for number, window in enumerate(windows):
        for position, (_, sku, name) in enumerate(window):
            path = f"{number:03d}/{position:03d}.jpg"
            index.append({"win": number, "path": path, "cls": sku})
            if not (OUT / path).exists():
                jobs.append((name, OUT / path))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    local = threading.local()

    def one(job):
        name, target = job
        _, _, offset, packed, kind = meta[name]
        for attempt in range(4):
            try:
                if not hasattr(local, "session"):
                    local.session = requests.Session()
                end = offset + 30 + len(name.encode("utf-8")) + 512 + packed
                reply = local.session.get(URL, headers={"Range": f"bytes={offset}-{end}"}, timeout=60)
                reply.raise_for_status()
                body = reply.content
                name_len, extra_len = struct.unpack("<HH", body[26:30])
                data = body[30 + name_len + extra_len: 30 + name_len + extra_len + packed]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zlib.decompress(data, -15) if kind == 8 else data)
                return
            except Exception as error:  # noqa: BLE001
                local.session = requests.Session()
                if attempt == 3:
                    print(f"    [!] {target}: {error}")

    if jobs:
        print(f"[+] {len(jobs)} ta kesma yuklanadi...")
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, jobs))


def operator_cost(labels, truth):
    labels = np.asarray(labels)
    wrong = 0
    for group in set(labels[labels >= 0].tolist()):
        members = [truth[i] for i in np.flatnonzero(labels == group)]
        wrong += len(members) - collections.Counter(members).most_common(1)[0][1]
    actions = len(set(labels[labels >= 0].tolist())) + int((labels < 0).sum())
    quality = purity_and_coverage(labels, truth)
    return {"wrong": wrong / len(labels), "actions": actions / len(set(truth)),
            "bcubed_f1": quality["bcubed_f1"], "false_merge": quality["false_merge"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="outputs/benchmark_cluster_windows.json")
    args = ap.parse_args()

    if not (OUT / "index.json").exists():
        info = json.load(open(RP / "index_offsets.json", encoding="utf-8"))     # scripts/fetch_rp2k.py yaratadi
        fetch(choose_windows(info, args.windows, args.width, args.seed), info)
    index = [row for row in json.load(open(OUT / "index.json", encoding="utf-8")) if (OUT / row["path"]).exists()]
    backbones = [b for b in G.BACKBONES if b[0] in ("dinov2b_crop", "siglip2b_stretch")]
    F = G.compute_features([row["path"] for row in index], "rp2k_win", backbones=backbones, root=OUT)
    dino = G.get_vec(F, np.arange(len(index)), "dinov2b_crop_cls+mean")
    embedding = np.concatenate([np.sqrt(0.4) * dino, np.sqrt(0.6) * F["siglip2b_stretch_cls"]], axis=1)
    window_of = np.array([row["win"] for row in index])
    truth = np.array([row["cls"] for row in index])

    cfg = Config.load()
    min_size = cfg.get("clustering.min_cluster_size", 2)

    def previous(vectors, colors, aspects):
        return cluster_unknown_features(vectors, colors, aspects, cfg)

    def current(vectors, colors, aspects):
        enough = len(vectors) >= int(cfg.get("clustering.variant_min_boxes", 8))
        return split_variants(previous(vectors, colors, aspects), vectors, colors, min_size,
                              max_color=float(cfg.get("clustering.variant_color", 0.12)),
                              max_embedding=cfg.get("clustering.variant_embedding", 0.65),
                              center=vectors.mean(axis=0) if enough else None)

    numbers = sorted(set(window_of.tolist()))
    half = len(numbers) // 2
    summary = {}
    for part, chosen in (("calibration", numbers[:half]), ("holdout", numbers[half:])):
        for name, method in (("previous", previous), ("current", current)):
            rows = []
            for number in chosen:
                pick = window_of == number
                rows.append(operator_cost(method(embedding[pick], F["color"][pick], F["aspect"][pick]),
                                          truth[pick].tolist()))
            summary.setdefault(part, {})[name] = {key: round(float(np.mean([row[key] for row in rows])), 4)
                                                  for key in rows[0]}
    result = json.dumps(summary, indent=2)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
