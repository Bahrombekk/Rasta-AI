"""Operator yorliqlarini (review.html -> labels.json) SKU katalogiga import qilish.

Har "✓ SKU" klasteridan centroidga eng yaqin va iloji boricha TURLI do'konlardan
--per-cluster ta crop olinadi (turli yorug'lik/burchak = mustahkam katalog).
Bir xil sku_id bir nechta klasterda bo'lsa — hammasi bitta SKU'ga qo'shiladi.
Real javon crop'lari bo'lgani uchun augmentatsiya qilinmaydi (source="crop").

  python scripts/import_catalog_labels.py --labels data/catalog_build/review/labels.json
  python scripts/import_catalog_labels.py --labels labels.json --replace --dry-run
"""
import argparse
import json
from collections import defaultdict

import _bootstrap  # noqa: F401

import numpy as np
from PIL import Image

import build_catalog_candidates as B
from build_catalog_candidates import APT, OUTLET_RE
from rasta.catalog_build import ClusterInfo, pick_examples
from rasta.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="data/catalog_build",
                    help="build_catalog_candidates.py --out papkasi")
    ap.add_argument("--labels", default=None, help="standart: <build>/review/labels.json")
    ap.add_argument("--per-cluster", type=int, default=20)
    ap.add_argument("--replace", action="store_true", help="SKU'ning eski vektorlarini o'chirish")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    from rasta.config import ROOT
    B.OUT = ROOT / args.build
    OUT = B.OUT
    load_shards = B.load_shards
    args.labels = args.labels or str(OUT / "review" / "labels.json")

    labels = [v for v in json.loads(open(args.labels, encoding="utf-8").read())["labels"]
              if v.get("status") == "ok" and v.get("sku_id")]
    if not labels:
        raise SystemExit("labels.json da '✓ SKU' belgilangan klaster yo'q.")
    clusters = {c["cid"]: c for c in json.loads((OUT / "clusters.json").read_text())}
    images = json.loads((OUT / "images.json").read_text())
    D = load_shards()
    outlets = np.array([int(OUTLET_RE.match(images[i][1]).group(1)) for i in D["img"]])

    by_sku = defaultdict(list)
    for v in labels:
        by_sku[v["sku_id"]].append(v)
    print(f"[+] {len(labels)} klaster -> {len(by_sku)} SKU")

    pipe = None
    if not args.dry_run:
        cfg = Config.load().override(args.set)
        cfg.set("detector.backend", "simple")
        from rasta.pipeline import RastaPipeline
        pipe = RastaPipeline(cfg)

    total = 0
    for sku, vs in sorted(by_sku.items()):
        crops, n_out = [], set()
        for v in vs:
            c = clusters.get(v["cid"])
            if c is None:
                print(f"  [!] klaster {v['cid']} clusters.json da yo'q — o'tkazildi")
                continue
            info = ClusterInfo(c["cid"], c["size"], c["n_outlets"], c["tightness"],
                               np.array(c["members"]))
            for i in pick_examples(info, outlets, args.per_cluster):
                m, name = images[D["img"][i]]
                x1, y1, x2, y2 = (int(t) for t in D["box"][i])
                with Image.open(APT / m / name) as im:
                    crops.append(np.array(im.convert("RGB").crop((x1, y1, x2, y2))))
                n_out.add(outlets[i])
        if not crops:
            continue
        name, brand = vs[0]["name"], vs[0].get("brand") or None
        print(f"  {sku:<36} {name:<30} {brand or '':<14} {len(crops):>3} crop, "
              f"{len(n_out)} do'kon, {len(vs)} klaster")
        if pipe is not None:
            if args.replace:
                pipe.catalog.delete_sku(sku)
            total += pipe.add_product(sku, name, crops, brand=brand, augment=False, source="crop")
    if pipe is not None:
        print(f"[✓] {total} vektor qo'shildi. Katalog: {pipe.catalog.count()} vektor, "
              f"{len(pipe.catalog.list_skus())} SKU.")


if __name__ == "__main__":
    main()
