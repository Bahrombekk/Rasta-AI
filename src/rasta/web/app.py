"""Rasta veb-ilovasi (FastAPI): admin — mahsulot katalogi, asosiy sahifa — javon tahlili.

Ishga tushirish:  python scripts/serve.py          ->  http://127.0.0.1:8000
Admin parol (ixtiyoriy):  RASTA_ADMIN_PASSWORD muhit o'zgaruvchisi.
"""
from __future__ import annotations

import hmac
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from ..config import ROOT, Config, resolve_path

STATIC = Path(__file__).parent / "static"
MAX_SIDE = 2000                    # katta telefon rasmlari shu o'lchamgacha kichraytiriladi
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000            # "dekompressiya bombasi": kichik fayl, ulkan rasm
# Bir-biridan aniq farqlanadigan ranglar: natijadagi mahsulotlarga ulush tartibida beriladi,
# shunda jadvaldagi rang belgisi rasmdagi ramka bilan bir qarashda mos keladi.
PALETTE = ["#2563eb", "#e11d48", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#ea580c",
           "#db2777", "#65a30d", "#0d9488", "#9333ea", "#ca8a04", "#4f46e5", "#dc2626"]
OTHER = "#94a3b8"


class State:
    pipe = None
    lock = threading.Lock()
    cfg: Optional[Config] = None
    images_dir: Path = ROOT / "data" / "catalog_images"
    public = False      # server tarmoqqa ochiq (0.0.0.0) — parolsiz admin yopiladi


def get_pipe():
    if State.pipe is None:
        with State.lock:
            if State.pipe is None:
                from ..pipeline import RastaPipeline
                State.cfg = State.cfg or Config.load()
                State.pipe = RastaPipeline(State.cfg)
                State.images_dir = resolve_path(State.cfg.get("web.images_dir",
                                                              "data/catalog_images"))
    return State.pipe


def check_admin(token: Optional[str]):
    """Parol o'rnatilgan bo'lsa — talab qilinadi. O'rnatilmagan bo'lsa admin faqat lokal
    serverda (127.0.0.1) ochiq; tarmoqqa ochilgan serverda (Docker, tunnel) yopiq.
    RASTA_ADMIN_OPEN=1 — ataylab parolsiz ochiq admin (sinov/demo uchun)."""
    pw = os.environ.get("RASTA_ADMIN_PASSWORD")
    if not pw:
        if State.public and os.environ.get("RASTA_ADMIN_OPEN") != "1":
            raise HTTPException(403, "Admin o'chirilgan: serverda RASTA_ADMIN_PASSWORD o'rnatilmagan")
        return
    if not hmac.compare_digest((token or "").encode(), pw.encode()):
        raise HTTPException(401, "Admin paroli noto'g'ri")


def read_upload(f: UploadFile) -> bytes:
    """Faylni o'qish — chegaradan oshganini butunlay xotiraga olmasdan."""
    data = f.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Rasm juda katta (20 MB dan ko'p)")
    return data


def read_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Rasm juda katta (20 MB dan ko'p)")
    try:
        im = Image.open(io.BytesIO(data))       # faqat sarlavha o'qiladi
    except Exception:
        raise HTTPException(400, "Rasmni o'qib bo'lmadi")
    if im.width * im.height > MAX_PIXELS:
        raise HTTPException(413, "Rasm o'lchami juda katta")
    try:
        im = ImageOps.exif_transpose(im).convert("RGB")    # telefon rasmi burilishi
    except Exception:
        raise HTTPException(400, "Rasmni o'qib bo'lmadi")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.array(im)


def slug(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", s.strip().lower()).strip("_")
    return s[:48] or f"sku_{int(time.time())}"


def thumb_url(sku: str) -> Optional[str]:
    d = State.images_dir / sku
    if d.is_dir():
        fs = sorted(d.glob("*.jpg"))
        if fs:
            return f"/media/{sku}/{fs[0].name}"
    return None


def image_files(sku: str) -> list[Path]:
    d = State.images_dir / sku
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def save_images(sku: str, imgs: list, prefix: str = "", max_side: Optional[int] = 800) -> list[Path]:
    """Mahsulot rasmlarini saqlash. prefix "c" — javondan o'rgatilgan crop (qayta
    indekslashda augmentatsiyasiz, "crop" sifatida qo'shiladi)."""
    d = State.images_dir / sku
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for f in d.glob("*.jpg") if (m := re.search(r"(\d+)$", f.stem))]
    start = max(nums, default=-1) + 1
    saved = []
    try:
        for i, a in enumerate(imgs):
            im = Image.fromarray(a)
            if max_side:
                im.thumbnail((max_side, max_side))
            p = d / f"{prefix}{start + i:03d}.jpg"
            saved.append(p)
            im.save(p, quality=92 if prefix else 88)
    except Exception:
        for p in saved:
            p.unlink(missing_ok=True)
        raise
    return saved


def add_with_images(pipe, sku: str, name: str, imgs: list, brand: Optional[str],
                    prefix: str = "", max_side: Optional[int] = 800, **kw) -> int:
    """Avval rasmlar diskka, keyin vektorlar katalogga. Vektor qo'shish xato bersa —
    saqlangan rasmlar o'chiriladi: rasm va katalog bir-biridan ajralib qolmaydi
    (qayta indekslash aynan shu rasmlardan qiladi)."""
    saved = save_images(sku, imgs, prefix=prefix, max_side=max_side)
    try:
        return pipe.add_product(sku, name, imgs, brand=brand, **kw)
    except Exception:
        for p in saved:
            p.unlink(missing_ok=True)
        raise


def rebuild_sku(pipe, sku: str, name: str, brand: Optional[str]) -> int:
    """SKU vektorlarini saqlangan rasmlardan qaytadan yaratish (rasm o'chirilganda,
    onboarding mantiqi o'zgarganda). Javon crop'lari va katalog rasmlari alohida."""
    files = image_files(sku)
    def load(paths):
        for path in paths:
            with Image.open(path) as image:
                yield np.array(image.convert("RGB"))

    crops = [f for f in files if f.stem.startswith("c")]
    shots = [f for f in files if not f.stem.startswith("c")]
    n = 0
    with pipe.catalog.replace_sku(sku):
        if shots:
            n += pipe.add_product(sku, name, list(load(shots)), brand=brand)
        if crops:
            n += pipe.add_product(sku, name, list(load(crops)), brand=brand,
                                  augment=False, source="crop")
    return n


def color_map(result) -> dict[str, str]:
    order = list(result.kpi["share_of_shelf"].keys())        # ulush bo'yicha kamayish tartibida
    return {sku: (PALETTE[i] if i < len(PALETTE) else OTHER) for i, sku in enumerate(order)}


def geometry(result) -> dict:
    """Brauzer mahsulotlar, bo'sh joylar va zonani chizishi uchun koordinatalar.

    Koordinatalar tahlil qilingan rasm pikselida (javobdagi "size").
    """
    r1 = lambda v: round(float(v), 1)  # noqa: E731
    dets = []
    for d in result.detections:
        m = d.match
        dets.append({
            "b": [r1(d.box.x1), r1(d.box.y1), r1(d.box.x2), r1(d.box.y2)],
            "s": m.sku_id if (m and m.is_known) else None,
            "brand": m.brand if (m and m.level == "brand") else None,
        })
    def gap_boxes(gaps):
        return [[r1(g.x1), r1(g.y1), r1(g.x2), r1(g.y2)] for g in gaps]

    roi = [[r1(x), r1(y)] for x, y in result.roi.points] if result.roi is not None else None
    return {"detections": dets, "gaps": gap_boxes(result.gaps),
            "suspect_gaps": gap_boxes(result.suspects), "divider_gaps": gap_boxes(result.dividers),
            "roi": roi}


def summarize(result, colors: dict[str, str]) -> dict:
    k = result.kpi
    names = {}
    for det in result.detections:
        if det.match and det.match.is_known:
            names[det.match.sku_id] = (det.match.name, det.match.brand)
    products = [{
        "sku_id": sku, "name": info["name"], "brand": names.get(sku, (None, None))[1],
        "facings": info["facings"], "share_pct": info["share_pct"],
        "color": colors.get(sku, OTHER), "thumb": thumb_url(sku)}
        for sku, info in k["share_of_shelf"].items()]
    brands = [{"brand": b, "facings": v["facings"], "share_pct": v["share_pct"]}
              for b, v in k.get("brand_share", {}).items()]
    known_share = sum(p["share_pct"] for p in products)
    oos = k["oos"]
    return {
        "num_products": k["num_products"],
        "num_known": k["num_known"],
        "num_unknown": k["num_unknown"] - k.get("num_brand_only", 0),
        "recognition_rate": k["recognition_rate_pct"],
        "fill_pct": k.get("roi_shelf_fill_pct"),
        "area_fill_pct": k.get("roi_fill_pct"),
        "row_fill": k.get("roi_row_fill_pct"),
        "rows": result.rows,
        "oos_gaps": oos["num_gaps"],
        "oos_facings": oos["est_missing_facings"],
        "oos_status": oos.get("status", "estimated"),
        "frontal": k.get("scene_frontal"),
        "products": products,
        "brands": brands,
        "unknown_share_pct": round(max(0.0, 100.0 - known_share), 2),
        "time_ms": round(result.timings_ms.get("total", 0)),
    }


def create_app(cfg: Optional[Config] = None, public: bool = False) -> FastAPI:
    State.cfg = cfg
    State.public = public
    app = FastAPI(title="Rasta AI", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/admin")
    def admin():
        return FileResponse(STATIC / "admin.html")

    @app.get("/media/{sku}/{name}")
    def media(sku: str, name: str):
        p = (State.images_dir / sku / name).resolve()
        if State.images_dir.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/products")
    def products():
        pipe = get_pipe()
        out = []
        with State.lock:
            for sku, v in pipe.catalog.sku_info().items():
                photos = image_files(sku)
                out.append({"sku_id": sku, "name": v["name"], "brand": v.get("brand"),
                            "images": len(photos), "vectors": v["count"],
                            "thumb": f"/media/{sku}/{photos[0].name}" if photos else None})
        out.sort(key=lambda r: (r.get("brand") or "", r["name"]))
        return {"products": out}

    @app.post("/api/products")
    def add_product(name: str = Form(...), brand: str = Form(""), sku_id: str = Form(""),
                    files: list[UploadFile] = File(...),
                    x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        name = name.strip()
        if not name:
            raise HTTPException(400, "Mahsulot nomini kiriting")
        sku = slug(sku_id or name)
        imgs = [read_image(read_upload(f)) for f in files if f.filename]
        if not imgs:
            raise HTTPException(400, "Kamida bitta rasm yuklang")
        pipe = get_pipe()
        with State.lock:
            if sku in pipe.catalog.sku_info():
                raise HTTPException(409, f"Bu nomdagi mahsulot allaqachon bor ({sku}) — rasmni "
                                         "o'sha mahsulot sahifasidan qo'shing yoki boshqa nom bering")
            n = add_with_images(pipe, sku, name, imgs, brand.strip() or None)
        return {"sku_id": sku, "vectors": n, "images": len(imgs)}

    def product_or_404(pipe, sku: str) -> dict:
        info = pipe.catalog.sku_info()
        if sku not in info:
            raise HTTPException(404, "Mahsulot topilmadi")
        return info[sku]

    @app.get("/api/products/{sku}")
    def product(sku: str):
        pipe = get_pipe()
        with State.lock:
            v = product_or_404(pipe, sku)
            return {"sku_id": sku, "name": v["name"], "brand": v.get("brand"), "vectors": v["count"],
                    "images": [{"name": f.name, "url": f"/media/{sku}/{f.name}",
                                "shelf": f.stem.startswith("c")} for f in image_files(sku)]}

    @app.patch("/api/products/{sku}")
    def edit_product(sku: str, name: str = Form(...), brand: str = Form(""),
                     x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        name = name.strip()
        if not name:
            raise HTTPException(400, "Mahsulot nomini kiriting")
        with State.lock:
            product_or_404(pipe, sku)
            pipe.catalog.update_sku(sku, name=name, brand=brand.strip() or None)
        return {"sku_id": sku, "name": name, "brand": brand.strip() or None}

    @app.post("/api/products/{sku}/images")
    def add_images(sku: str, files: list[UploadFile] = File(...),
                   x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        imgs = [read_image(read_upload(f)) for f in files if f.filename]
        if not imgs:
            raise HTTPException(400, "Kamida bitta rasm yuklang")
        with State.lock:
            v = product_or_404(pipe, sku)
            n = add_with_images(pipe, sku, v["name"], imgs, v.get("brand"))
        return {"sku_id": sku, "added": len(imgs), "vectors": n}

    @app.delete("/api/products/{sku}/images/{name}")
    def delete_image(sku: str, name: str, x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            v = product_or_404(pipe, sku)
            f = State.images_dir / sku / name
            files = image_files(sku)
            if f not in files:
                raise HTTPException(404, "Rasm topilmadi")
            if len(files) == 1:
                raise HTTPException(400, "Oxirgi rasmni o'chirib bo'lmaydi — mahsulotni o'chiring")
            bak = f.with_suffix(".jpg.del")
            f.rename(bak)
            try:
                n = rebuild_sku(pipe, sku, v["name"], v.get("brand"))
            except Exception:
                bak.rename(f)
                raise
            bak.unlink()
        return {"sku_id": sku, "deleted": name, "vectors": n}

    @app.delete("/api/products/{sku}")
    def delete_product(sku: str, x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            product_or_404(pipe, sku)
            pipe.catalog.delete_sku(sku)
            for f in image_files(sku):
                f.unlink()
        return {"deleted": sku}

    @app.post("/api/teach")
    def teach(image: UploadFile = File(...), boxes: str = Form(...), sku_id: str = Form(""),
              name: str = Form(""), brand: str = Form(""),
              x_admin_token: Optional[str] = Header(None)):
        """Javon rasmidagi tanlangan qutilarni katalogga qo'shish (real crop'lar bilan o'rgatish).

        Internet rasmiga qaraganda javondan olingan crop ancha ishonchli: u xuddi shu
        sharoitda (yorug'lik, burchak, oyna) olingan. boxes — tahlil javobidagi
        koordinatalar ("size" bo'yicha), sku_id — mavjud mahsulot yoki name — yangi.
        """
        check_admin(x_admin_token)
        img = read_image(read_upload(image))
        try:
            bx = np.asarray(json.loads(boxes), dtype=float)
            if bx.size == 0:
                raise HTTPException(400, "Kamida bitta mahsulotni tanlang")
            if bx.ndim != 2 or bx.shape[1] != 4 or len(bx) > 600 or not np.isfinite(bx).all():
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise HTTPException(400, "Qutilar noto'g'ri")
        name = name.strip()
        if not sku_id and not name:
            raise HTTPException(400, "Mahsulot nomini kiriting")
        h, w = img.shape[:2]
        crops = []
        for x1, y1, x2, y2 in bx:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops.append(np.ascontiguousarray(img[y1:y2, x1:x2]))
        if not crops:
            raise HTTPException(400, "Tanlangan qutilar juda kichik")
        pipe = get_pipe()
        sku = sku_id or slug(name)
        with State.lock:
            if sku_id:
                existing = product_or_404(pipe, sku)
                name, brand = existing["name"], existing.get("brand") or ""
            elif sku in pipe.catalog.sku_info():
                raise HTTPException(409, f"Bu nomdagi mahsulot allaqachon bor ({sku}) — "
                                         "ro'yxatdan mavjud mahsulotni tanlang")
            n = add_with_images(pipe, sku, name, crops, brand.strip() or None, prefix="c",
                                max_side=None, augment=False, source="crop")
        return {"sku_id": sku, "name": name, "added": len(crops), "vectors": n}

    @app.post("/api/reindex")
    def reindex(x_admin_token: Optional[str] = Header(None)):
        """Katalogni saqlangan mahsulot rasmlaridan qayta indekslash (onboarding
        mantiqi yaxshilanganda eski vektorlarni yangilash uchun)."""
        check_admin(x_admin_token)
        pipe = get_pipe()
        done, skipped = [], []
        with State.lock:
            for sku, v in pipe.catalog.sku_info().items():
                files = image_files(sku)
                if not files:
                    skipped.append(sku)
                    continue
                n = rebuild_sku(pipe, sku, v["name"], v.get("brand"))
                done.append({"sku_id": sku, "images": len(files), "vectors": n})
            if not skipped:
                pipe.catalog.mark_current()    # hamma SKU joriy embedder bilan qayta qurildi
        return {"reindexed": done, "skipped": skipped}

    @app.post("/api/analyze")
    def analyze(image: UploadFile = File(...), roi: str = Form(""),
                roi_normalized: bool = Form(False)):
        img = read_image(read_upload(image))
        region = None
        if roi.strip():
            from ..roi import make_region
            try:
                points = np.asarray(json.loads(roi), dtype=float)
                if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 100
                        or not np.isfinite(points).all()):
                    raise ValueError
                h, w = img.shape[:2]
                if roi_normalized:
                    points = points * [w, h]
                if (points < 0).any() or (points > [w, h]).any():
                    raise ValueError
                # A polygon with zero area cannot be analyzed or rectified.
                x, y = points.T
                area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
                if area < 1:
                    raise ValueError
                region = make_region(points, "Zona")
            except (ValueError, TypeError, OverflowError):
                raise HTTPException(400, "Zona noto'g'ri")
        pipe = get_pipe()
        with State.lock:
            res = (pipe.analyze_region(img, region) if region is not None
                   else pipe.analyze_image(img))
            summary = summarize(res, color_map(res))
        return JSONResponse({"summary": summary, **geometry(res),
                             "size": [int(img.shape[1]), int(img.shape[0])]})

    @app.get("/api/health")
    def health():
        stale = getattr(State.pipe.catalog, "stale", None) if State.pipe is not None else None
        return {"ok": True, "loaded": State.pipe is not None, "catalog_stale": bool(stale)}

    return app


app = create_app()
