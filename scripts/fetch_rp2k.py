"""RP2K (Peng va boshq., 2020) datasetidan kichik qism yuklab olish.

RP2K — 2388 SKU, do'kon javonlaridan kesilgan 380K+ mahsulot rasmi (real javon
crop'lari; ko'plari juda o'xshash qadoqlar). To'liq arxiv 6.3 GB, shuning uchun
ZIP range so'rovlari bilan faqat tanlangan sinflarning bir qismi olinadi.

  python scripts/fetch_rp2k.py --classes 120 --train 5 --test 20
Natija: data/external/rp2k/{train,test}/<sinf_idx>/*.jpg + labels.json

Mustaqil (sozlashda ishlatilmagan) ikkinchi to'plam — birinchisidagi sinflarsiz:
  python scripts/fetch_rp2k.py --name rp2k_b --classes 300 --train 0 --test 20 --seed 1 --exclude rp2k
"""
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401

from rasta.config import ROOT

URL = "https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip"
OUT = ROOT / "data" / "external" / "rp2k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=120)
    ap.add_argument("--train", type=int, default=5)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--name", default="rp2k", help="data/external ichidagi papka nomi")
    ap.add_argument("--exclude", default=None, help="shu papkadagi (labels.json) sinflar olinmaydi")
    args = ap.parse_args()
    global OUT
    index_dir = OUT
    OUT = ROOT / "data" / "external" / args.name

    import requests

    # Indeks: (nom, hajm, lokal sarlavha offseti, siqilgan hajm, siqish turi)
    idx_file = index_dir / "index_offsets.json"
    if not idx_file.exists():
        from remotezip import RemoteZip
        OUT.mkdir(parents=True, exist_ok=True)
        with RemoteZip(URL) as z:
            json.dump([(i.filename, i.file_size, i.header_offset, i.compress_size,
                        i.compress_type) for i in z.infolist()],
                      open(idx_file, "w", encoding="utf-8"), ensure_ascii=False)
    info = json.load(open(idx_file, encoding="utf-8"))
    meta = {r[0]: r for r in info}

    files: dict[tuple[str, str], list[str]] = {}
    for name, size, *_ in info:
        parts = name.split("/")
        if len(parts) == 4 and parts[3] and size > 0:
            files.setdefault((parts[1], parts[2]), []).append(name)
    classes = sorted({c for (_, c) in files})
    skip = set()
    if args.exclude:
        skip = set(json.load(open(ROOT / "data" / "external" / args.exclude / "labels.json",
                                  encoding="utf-8")).values())
    ok = [c for c in classes if c not in skip and len(files.get(("train", c), [])) >= args.train
          and len(files.get(("test", c), [])) >= args.test]
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(ok, min(args.classes, len(ok))))

    jobs = []
    labels = {}
    for ci, c in enumerate(chosen):
        labels[str(ci)] = c
        for split, k in (("train", args.train), ("test", args.test)):
            pick = rng.sample(sorted(files[(split, c)]), k)
            for j, name in enumerate(pick):
                dst = OUT / split / f"{ci:04d}" / f"{j:03d}.jpg"
                if not dst.exists():
                    jobs.append((name, dst))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"[+] {len(chosen)} sinf, {len(jobs)} ta rasm yuklanadi...")

    local = threading.local()
    done = [0]
    lock = threading.Lock()

    def read_member(sess, name):
        """ZIP a'zosini bitta range so'rov bilan o'qish (lokal sarlavhani tahlil qilib)."""
        import struct
        import zlib
        _, _, off, csize, ctype = meta[name]
        nb = len(name.encode("utf-8"))
        end = off + 30 + nb + 512 + csize
        r = sess.get(URL, headers={"Range": f"bytes={off}-{end}"}, timeout=60)
        r.raise_for_status()
        buf = r.content
        assert buf[:4] == b"PK\x03\x04", "lokal sarlavha topilmadi"
        n, m = struct.unpack("<HH", buf[26:30])
        data = buf[30 + n + m: 30 + n + m + csize]
        if ctype == 8:
            data = zlib.decompress(data, -15)
        return data

    def fetch(job):
        name, dst = job
        if not hasattr(local, "s"):
            local.s = requests.Session()
        for attempt in range(4):
            try:
                data = read_member(local.s, name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"    [!] {dst}: {e}")
                local.s = requests.Session()
        with lock:
            done[0] += 1
            if done[0] % 250 == 0:
                print(f"    {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(fetch, jobs))
    print(f"[✓] {OUT}")


if __name__ == "__main__":
    main()
