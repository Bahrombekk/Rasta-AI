"""Rasta veb-ilovasi (FastAPI): admin — mahsulot katalogi, asosiy sahifa — javon tahlili.

Ishga tushirish:  python scripts/serve.py          ->  http://127.0.0.1:8000
Admin parol (ixtiyoriy):  RASTA_ADMIN_PASSWORD muhit o'zgaruvchisi.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import copy
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from rasta.config import ROOT, Config, resolve_path
from rasta.inference import InferenceUnavailable, RemoteInference
from .accounts import Accounts, SESSION_COOKIE
from .session_api import session_router
from .workspace_api import router as workspace_router
from .field_api import router as field_router

STATIC = ROOT / "frontend" / "dist"
MAX_SIDE = 2000                    # katta telefon rasmlari shu o'lchamgacha kichraytiriladi
OCR_MAX_SIDE = 6000                # yorliq yozuvi asl o'lchamga yaqin rasmdan o'qiladi
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000            # "dekompressiya bombasi": kichik fayl, ulkan rasm
# Katalog to'plami (.zip): bir korxonada tayyorlangan katalogni boshqasiga ko'chirish uchun
BUNDLE_MAX_BYTES = 300 * 1024 * 1024
BUNDLE_MAX_UNPACKED = 1024 * 1024 * 1024
BUNDLE_MAX_PRODUCTS = 3000
BUNDLE_MAX_IMAGES = 40             # bitta mahsulotga
BUNDLE_IMAGE = re.compile(r"c?[0-9]{3}\.jpg")
BUNDLE_SKU = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ_]{1,64}")
# Bir-biridan aniq farqlanadigan ranglar: natijadagi mahsulotlarga ulush tartibida beriladi,
# shunda jadvaldagi rang belgisi rasmdagi ramka bilan bir qarashda mos keladi.
PALETTE = ["#2563eb", "#e11d48", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#ea580c",
           "#db2777", "#65a30d", "#0d9488", "#9333ea", "#ca8a04", "#4f46e5", "#dc2626"]
OTHER = "#94a3b8"
organization_scope: ContextVar[int | None] = ContextVar("organization_scope", default=None)


class State:
    pipe = None
    lock = threading.Lock()
    analysis_lock = threading.Lock()
    cfg: Optional[Config] = None
    inference: Optional[RemoteInference] = None
    organization_pipes = {}
    words_tried: set = set()                # namuna yozuvi o'qib ko'rilgan (katalog, SKU) juftlari
    images_dir: Path = ROOT / "data" / "catalog_images"
    archive_dir: Optional[Path] = None      # tahlil qilingan rasmlar arxivi (None — o'chiq)
    public = False      # server tarmoqqa ochiq (0.0.0.0) — parolsiz admin yopiladi


async def analysis_slot():
    if not State.analysis_lock.acquire(blocking=False):
        raise HTTPException(503, "Hozir boshqa javon rasmi tahlil qilinmoqda. Birozdan keyin qayta urinib ko‘ring.",
                            headers={"Retry-After": "2"})
    try:
        yield
    finally:
        State.analysis_lock.release()


def _base_pipe():
    if State.pipe is None:
        with State.lock:
            if State.pipe is None:
                from rasta.pipeline import RastaPipeline
                State.cfg = State.cfg or Config.load()
                url = os.environ.get("RASTA_AI_URL")
                State.inference = RemoteInference(State.cfg, url) if url else None
                State.pipe = RastaPipeline(State.cfg, inference=State.inference)
                State.images_dir = resolve_path(State.cfg.get("web.images_dir",
                                                              "data/catalog_images"))
                if os.environ.get("RASTA_ARCHIVE", "1") != "0":
                    State.archive_dir = resolve_path(State.cfg.get("web.archive_dir", "data/archive"))
    return State.pipe


def get_pipe():
    pipe = _base_pipe()
    org_id = organization_scope.get()
    if org_id in (None, 1):
        return pipe
    with State.lock:
        if org_id not in State.organization_pipes:
            from rasta.catalog import build_catalog

            cfg = State.cfg.copy()
            path = cfg.get("catalog.path", "data/qdrant")
            if path != ":memory:":
                cfg.set("catalog.path", str(resolve_path(path) / "organizations" / str(org_id)))
            cfg.set("catalog.collection", f"{cfg.get('catalog.collection', 'rasta_skus')}_org_{org_id}")
            cfg.set("recognition.negatives", None)
            scoped = copy(pipe)
            scoped.catalog = build_catalog(cfg, pipe.catalog.dim)
            scoped.catalog.negatives = pipe.catalog.negatives
            State.organization_pipes[org_id] = scoped
        return State.organization_pipes[org_id]


def images_root():
    org_id = organization_scope.get()
    if org_id in (None, 1):
        return State.images_dir
    return State.images_dir / ".organizations" / str(org_id)


ARCHIVE_MIN_FREE = 5 * 1024 ** 3     # diskda shundan kam joy qolsa arxivlash to'xtaydi


def archive_analysis(upload: bytes, img: np.ndarray, result, image_jpeg: bytes | None = None) -> None:
    """Tahlil qilingan rasmni va detektor qutilarini arxivga yozadi — keyinchalik o'z
    rasmlarimizda kuchliroq (va tijoriy litsenziyali) detektor o'qitish uchun.

    Format mavjud asboblarga mos: <xesh>.jpg + <xesh>.det.json (status "draft") —
    tools/box_editor.html da tuzatiladi, scripts/export_yolo.py bilan o'qitishga chiqariladi.
    Fayl nomi yuklangan rasm xeshi: qayta tahlil qilingan rasm takrorlanmaydi.
    Arxivlashdagi xato tahlilni buzmaydi.
    """
    root = State.archive_dir
    if root is None:
        return
    try:
        org_id = organization_scope.get()
        folder = root / (f"org_{org_id}" if org_id is not None else "default") / time.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(folder).free < ARCHIVE_MIN_FREE:
            return
        name = hashlib.sha1(upload).hexdigest()[:16]
        photo = folder / f"{name}.jpg"
        if not photo.exists():
            if image_jpeg is None:
                Image.fromarray(img).save(photo, "JPEG", quality=90)
            else:
                photo.write_bytes(image_jpeg)
        detections = list(result.detections) + list(result.background)
        try:
            image_ref = str(photo.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            image_ref = str(photo)
        (folder / f"{name}.det.json").write_text(json.dumps({
            "image": image_ref,
            "status": "draft",
            "size": [int(img.shape[1]), int(img.shape[0])],
            "boxes": [[round(float(v), 1) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)] for d in detections],
            "ignore": [],
            "scores": [round(float(d.box.score), 3) for d in detections],
            "labels": [d.match.sku_id if d.match is not None and d.match.is_known else None for d in detections],
            "source": "analyze",
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def check_legacy_admin(token: Optional[str]):
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


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Rasm juda katta (20 MB dan ko'p)")
    with io.BytesIO(data) as stream:
        try:
            im = Image.open(stream)
        except Exception:
            raise HTTPException(400, "Rasmni o'qib bo'lmadi")
        if im.width * im.height > MAX_PIXELS:
            raise HTTPException(413, "Rasm o'lchami juda katta")
        try:
            ImageOps.exif_transpose(im, in_place=True)
            return im if im.mode == "RGB" else im.convert("RGB")
        except Exception:
            raise HTTPException(400, "Rasmni o'qib bo'lmadi")


def image_pixels(im: Image.Image, max_side: int) -> np.ndarray:
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.array(im)


def read_image(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    with decode_image(data) as im:
        return image_pixels(im, max_side)


def slug(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", s.strip().lower()).strip("_")
    return s[:48] or f"sku_{int(time.time())}"


def thumb_url(sku: str) -> Optional[str]:
    d = images_root() / sku
    if d.is_dir():
        fs = sorted(d.glob("*.jpg"))
        if fs:
            return f"/media/{sku}/{fs[0].name}"
    return None


def image_files(sku: str) -> list[Path]:
    d = images_root() / sku
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def catalog_metadata(info: dict) -> dict:
    return {key: info.get(key) for key in ("category", "barcode")}


async def catalog_metadata_patch(request: Request) -> dict:
    form = await request.form()
    return {key: form[key].strip() or None for key in ("category", "barcode")
            if isinstance(form.get(key), str)}


def validate_catalog_metadata(category, barcode, products, sku):
    if category is not None and len(category) > 80:
        raise HTTPException(400, "Kategoriya 80 belgidan oshmasligi kerak.")
    if barcode and not re.fullmatch(r"[0-9]{4,32}", barcode):
        raise HTTPException(400, "Shtrix-kod 4–32 ta raqamdan iborat bo‘lishi kerak.")
    if barcode and any(key != sku and info.get("barcode") == barcode for key, info in products.items()):
        raise HTTPException(409, "Bu shtrix-kod katalogdagi boshqa mahsulotga biriktirilgan.")


def save_images(sku: str, imgs: list, prefix: str = "", max_side: Optional[int] = 800) -> list[Path]:
    """Mahsulot rasmlarini saqlash. prefix "c" — javondan o'rgatilgan crop (qayta
    indekslashda augmentatsiyasiz, "crop" sifatida qo'shiladi)."""
    d = images_root() / sku
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


def box_file(image_path: Path) -> Path:
    """Namuna rasmidagi mahsulot qutisi saqlanadigan yondosh fayl (000.jpg -> 000.box.json)."""
    return image_path.with_suffix(".box.json")


def valid_box(box) -> Optional[list[float]]:
    """0..1 normallashgan [x1,y1,x2,y2]; noto'g'ri bo'lsa ValueError. None — butun rasm."""
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(np.isfinite([x1, y1, x2, y2])):
        raise ValueError
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if x2 - x1 < 0.03 or y2 - y1 < 0.03:
        raise ValueError
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def read_box(image_path: Path) -> Optional[dict]:
    """{"box": [...]|None, "auto": bool} yoki None (rasm hali tekshirilmagan / fayl buzilgan)."""
    try:
        data = json.loads(box_file(image_path).read_text(encoding="utf-8"))
        return {"box": valid_box(data.get("box")), "auto": bool(data.get("auto", True))}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def write_box(image_path: Path, box: Optional[list[float]], auto: bool) -> None:
    box_file(image_path).write_text(json.dumps({"box": box, "auto": auto}), encoding="utf-8")


def add_with_images(pipe, sku: str, name: str, imgs: list, brand: Optional[str],
                    prefix: str = "", max_side: Optional[int] = 800, **kw) -> int:
    """Avval rasmlar diskka, keyin vektorlar katalogga. Vektor qo'shish xato bersa —
    saqlangan rasmlar o'chiriladi: rasm va katalog bir-biridan ajralib qolmaydi
    (qayta indekslash aynan shu rasmlardan qiladi).

    Katalog rasmlarida (prefix bo'sh) mahsulot kadrdan ajratib olinadi: quti yondosh
    faylga yoziladi (katalogda ko'rsatiladi, operator to'g'rilashi mumkin) va embedding
    shu qutidan olinadi. Javon crop'lari (prefix "c") allaqachon kesilgan.
    """
    boxes = None if prefix else [pipe.locate_product(im) for im in imgs]
    saved = save_images(sku, imgs, prefix=prefix, max_side=max_side)
    try:
        if boxes is not None:
            for p, b in zip(saved, boxes):
                write_box(p, b, auto=True)
        return pipe.add_product(sku, name, imgs, brand=brand, boxes=boxes, **kw)
    except Exception:
        for p in saved:
            p.unlink(missing_ok=True)
            box_file(p).unlink(missing_ok=True)
        raise


def sample_label_words(pipe, sku: str) -> list:
    """Mahsulotning namuna yozuvi katalogda yo'q (OCR ulanishidan oldin qo'shilgan) bo'lsa —
    saqlangan rasmlardan o'qib katalogga yozadi. Xizmat javob bermasa keyinroq qayta uringan
    bo'ladi; yozuv topilmasa (rasmda matn yo'q) takror o'qilmaydi."""
    key = (id(pipe.catalog), sku)
    if key in State.words_tried:
        return []
    from rasta.augment import crop_normalized

    images = []
    for path in image_files(sku)[:6]:
        try:
            with Image.open(path) as image:
                pixels = np.array(image.convert("RGB"))
        except (OSError, ValueError):
            continue
        located = None if path.stem.startswith("c") else read_box(path)
        images.append(crop_normalized(pixels, located["box"]) if located and located["box"] else pixels)
    words = pipe._label_words(images)
    if words is None:
        return []
    State.words_tried.add(key)
    if words:
        pipe.catalog.update_sku(sku, words=words)
    return words


def rebuild_sku(pipe, sku: str, name: str, brand: Optional[str], extra: Optional[dict] = None) -> int:
    """SKU vektorlarini saqlangan rasmlardan qaytadan yaratish (rasm o'chirilganda,
    onboarding mantiqi o'zgarganda). Javon crop'lari va katalog rasmlari alohida."""
    files = image_files(sku)
    def load(paths):
        for path in paths:
            with Image.open(path) as image:
                yield np.array(image.convert("RGB"))

    crops = [f for f in files if f.stem.startswith("c")]
    shots = [f for f in files if not f.stem.startswith("c")]
    shot_images = list(load(shots))
    boxes = []
    for path, image in zip(shots, shot_images):
        saved = read_box(path)
        if saved is None:                    # eski rasm: quti hali aniqlanmagan
            saved = {"box": pipe.locate_product(image), "auto": True}
            write_box(path, saved["box"], auto=True)
        boxes.append(saved["box"])
    n = 0
    with pipe.catalog.replace_sku(sku):
        if shots:
            n += pipe.add_product(sku, name, shot_images, brand=brand, extra=extra, boxes=boxes)
        if crops:
            n += pipe.add_product(sku, name, list(load(crops)), brand=brand, extra=extra,
                                  augment=False, source="crop")
    return n


def color_map(result) -> dict[str, str]:
    order = list(result.kpi["share_of_shelf"].keys())        # ulush bo'yicha kamayish tartibida
    return {sku: (PALETTE[i] if i < len(PALETTE) else OTHER) for i, sku in enumerate(order)}


def explain(d, params) -> dict:
    """Nega tanildi / tanilmadi — operator uchun qisqa izoh ma'lumoti.

    r: ok | context (qo'shnilar bo'yicha) | brand (faqat brend) | empty (katalog bo'sh) |
       low (o'xshashlik chegaradan past) | color (rang mos emas) | other (katalogda yo'q
       boshqa mahsulotga ko'proq o'xshaydi) | ambiguous (ikki SKU orasida) | size (o'lchami
       shu SKU'ning boshqa donalaridan farq qiladi) | text (brend ko'rinishidan, mahsulot
       yorliqdagi yozuvdan aniqlandi; w — o'qilgan so'z) | variant (o'xshaydi, lekin rangi farq
       qiladi — bir xil dizayndagi boshqa ta'm bo'lishi mumkin) | weak (ball chegaradan
       faqat rang/shakl hisobiga o'tgan, ko'rinishning o'zi mos emas).
    n — eng yaqin mahsulot, sc — o'xshashlik, tau — chegara, m — 2-nomzoddan farq.
    """
    m = d.match
    if m is not None and not m.is_known and m.level == "brand" and (m.checked or "").startswith("brand:"):
        # brend qadoqdagi yozuvdan topildi (katalogdan mos nomzod bo'lmasligi ham mumkin); w — brend
        return {"r": "brand_text", "w": m.brand or m.checked[6:]}
    top = m.candidates[0] if (m is not None and m.candidates) else None
    if top is None:
        return {"r": "empty"}
    tau, color_min, neg_margin = params.limits_for_source(top.get("src"))
    info = {"n": m.name if m.is_known else top.get("name"), "sc": round(float(m.score), 3),
            "tau": round(float(tau), 3), "m": round(float(m.margin), 3)}
    neg = top.get("neg")
    if m.is_known and m.text is not None:
        info["r"], info["w"] = "text", m.text
    elif m.is_known:
        info["r"] = ("family_context" if d.context_fixed else "family") if m.family else (
            "context" if d.context_fixed else "ok")
        if m.checked:                       # yorliq yozuvi bo'yicha tekshirilgan: support:<so'z> | silent
            info["lc"] = m.checked
    elif m.mismatch is not None:
        # ko'rinishi o'xshaydi, lekin qadoqdagi yozuv boshqa: label (butunlay boshqa mahsulot)
        # yoki label_brand (brend to'g'ri, mahsulot boshqa); w — qadoqda o'qilgan so'z
        info["r"], info["w"] = ("label_brand" if m.level == "brand" else "label"), m.mismatch
        if m.mismatch_kind in ("flavor", "volume"):
            info["r"] = f"label_{m.mismatch_kind}"
    elif m.level == "variant":
        info["r"], info["c"] = "variant", round(float(top.get("color", 0.0)), 3)
    elif m.level == "brand":
        info["r"] = "brand_context" if d.context_fixed else "brand"
        info["brand"] = m.brand
    elif m.score < tau:
        info["r"] = "low"
    elif (emin := params.embed_min_for(top.get("src"))) is not None and float(top.get("dino", 1.0)) < emin:
        info["r"], info["e"] = "weak", round(float(top.get("dino", 0.0)), 3)
    elif float(top.get("color", 1.0)) < color_min:
        info["r"], info["c"] = "color", round(float(top.get("color", 0.0)), 3)
    elif neg is not None and neg_margin is not None and m.score < neg - neg_margin:
        info["r"] = "other"
    elif m.margin < params.min_margin:
        info["r"] = "ambiguous"
    else:
        info["r"] = "size"
    return info


def geometry(result, params=None) -> dict:
    """Brauzer mahsulotlar, bo'sh joylar va zonani chizishi uchun koordinatalar.

    Koordinatalar tahlil qilingan rasm pikselida (javobdagi "size").
    """
    r1 = lambda v: round(float(v), 1)  # noqa: E731
    dets = []
    for d in result.detections:
        m = d.match
        item = {
            "b": [r1(d.box.x1), r1(d.box.y1), r1(d.box.x2), r1(d.box.y2)],
            "s": m.sku_id if (m and m.is_known) else None,
            "brand": m.brand if (m and m.level == "brand") else None,
        }
        if m and not m.is_known and m.mismatch_kind == "flavor":
            item["other_flavor"] = True
        if m and m.level == "variant" and m.candidates:
            item["v"] = m.candidates[0].get("sku_id")      # shunga o'xshaydi, rangi boshqa
        if item["s"] is None and d.cluster >= 0:
            item["g"] = int(d.cluster)                     # katalogda yo'q, bir-biriga o'xshash qutilar guruhi
        if params is not None:
            item["why"] = explain(d, params)
        dets.append(item)
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
            names[det.match.sku_id] = (det.match.brand, det.match.family)
    products = [{
        "sku_id": sku, "name": info["name"], "brand": names.get(sku, (None, False))[0],
        "family": names.get(sku, (None, False))[1],
        "facings": info["facings"], "share_pct": info["share_pct"],
        "color": colors.get(sku, OTHER), "thumb": thumb_url(sku)}
        for sku, info in k["share_of_shelf"].items()]
    brands = [{"brand": b, "facings": v["facings"], "share_pct": v["share_pct"],
               "unresolved_facings": v.get("unresolved_facings", 0),
               "other_flavor_facings": v.get("other_flavor_facings", 0),
               "products": [p for p in products if p["sku_id"] in v.get("sku_ids", ())]}
              for b, v in k.get("brand_share", {}).items() if b != "?"]
    known_share = sum(p["share_pct"] for p in products)
    oos = k["oos"]
    return {
        "num_products": k["num_products"],
        "num_known": k["num_known"],
        "num_unknown": k["num_unknown"] - k.get("num_brand_only", 0) - k.get("num_variant_no_brand", 0),
        "num_variant": k.get("num_variant", 0),
        "num_other_flavor": k.get("num_other_flavor", 0),
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


def create_app(cfg: Optional[Config] = None, public: bool = False, *,
               accounts: Accounts | None = None, secure_cookies: bool = False,
               serve_frontend: bool = True, workspaces=None) -> FastAPI:
    State.cfg = cfg
    State.public = public
    State.organization_pipes = {}
    app = FastAPI(title="Rasta AI backend", docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(session_router(accounts, secure_cookies, workspaces=workspaces))
    if workspaces is not None:
        app.include_router(workspace_router(workspaces))
        app.include_router(field_router(workspaces.field, get_pipe, State.lock))

    def check_admin(token):
        if accounts is None:
            check_legacy_admin(token)

    @app.middleware("http")
    async def access(request: Request, call_next):
        path = request.url.path
        request.state.user = None
        private = path.startswith(("/api/", "/media/"))
        if private and accounts is not None:
            request.state.user = await run_in_threadpool(accounts.session, request.cookies.get(SESSION_COOKIE))
            if workspaces is not None:
                request.state.user = await run_in_threadpool(workspaces.identity, request.state.user)
            if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("X-Rasta-Request") != "1":
                return JSONResponse({"detail": "So‘rov tasdiqlanmadi."}, status_code=403)
            public_api = path in {"/api/session", "/api/session/login", "/api/session/logout", "/api/health"}
            if not public_api and request.state.user is None:
                return JSONResponse({"detail": "Hisobingizga kiring."}, status_code=401)
        if path == "/api/team" or path.startswith("/api/team/"):
            if workspaces is not None:
                return JSONResponse({"detail": "Korxona jamoasi sahifasidan foydalaning."}, status_code=404)
            if not request.state.user or request.state.user["role"] != "super_admin":
                return JSONResponse({"detail": "Super admin ruxsati kerak."}, status_code=403)
        user = request.state.user or {}
        org_id = user.get("org_id")
        if workspaces is not None:
            catalog_read = (path == "/api/products" or path.startswith(("/api/products/", "/media/"))
                            or path == "/api/catalog/export")
            catalog_action = ((catalog_read and not path.startswith("/media/"))
                              or path in {"/api/teach", "/api/reindex", "/api/catalog/import"})
            analysis_action = path == "/api/analyze" or path.startswith(("/api/analyze/", "/media/"))
            selected = request.query_params.get("organization")
            platform_read = catalog_read and request.method in {"GET", "HEAD"} and user.get("role") == "super_admin"
            if platform_read:
                if not selected or not re.fullmatch(r"[0-9]{1,18}", selected):
                    return JSONResponse({"detail": "Katalog uchun korxonani tanlang."}, status_code=400)
                org_id = int(selected)
                if not await run_in_threadpool(workspaces.organization_exists, org_id):
                    return JSONResponse({"detail": "Korxona topilmadi."}, status_code=404)
            elif catalog_read and selected is not None:
                return JSONResponse({"detail": "Ushbu korxona katalogiga ruxsat yo‘q."}, status_code=403)
            elif (catalog_action and user.get("role") != "org_admin"
                    or analysis_action and user.get("role") not in {"org_admin", "operator"}):
                return JSONResponse({"detail": "Ushbu amal uchun ruxsat yetarli emas."}, status_code=403)
        scope = organization_scope.set(org_id)
        try:
            response = await call_next(request)
        finally:
            organization_scope.reset(scope)
        if private and accounts is not None:
            response.headers["Cache-Control"] = "no-store"
        # Xavfsizlik sarlavhalari (clickjacking, MIME-sniffing, referrer sizib chiqishi)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
        if secure_cookies:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains")
        return response

    @app.exception_handler(InferenceUnavailable)
    async def unavailable(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    if serve_frontend:
        for folder in ("assets", "static"):
            app.mount(f"/{folder}", StaticFiles(directory=STATIC / folder, check_dir=False), name=folder)

        def page():
            index = STATIC / "index.html"
            if not index.is_file():
                raise HTTPException(503, "Frontend build is missing. Run npm --prefix frontend ci and npm --prefix frontend run build.")
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

        for path in ("/", "/admin", "/login", "/team", "/account", "/dashboard", "/reports",
                     "/organizations", "/platform", "/members", "/stores", "/field"):
            app.add_api_route(path, page, methods=["GET"], include_in_schema=False)

    @app.get("/media/{sku}/{name}")
    def media(sku: str, name: str):
        root = images_root().resolve()
        p = (root / sku / name).resolve()
        if not p.is_relative_to(root) or len(p.relative_to(root).parts) != 2 or p.suffix != ".jpg" or not p.is_file():
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
                            **catalog_metadata(v),
                            "images": len(photos), "vectors": v["count"],
                            "thumb": f"/media/{sku}/{photos[0].name}" if photos else None})
        out.sort(key=lambda r: (r.get("brand") or "", r["name"]))
        return {"products": out}

    @app.post("/api/products")
    def add_product(name: str = Form(...), brand: str = Form(""), sku_id: str = Form(""),
                    category: str = Form(""), barcode: str = Form(""),
                    files: list[UploadFile] = File(...),
                    x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        name = name.strip()
        if not name:
            raise HTTPException(400, "Mahsulot nomini kiriting")
        sku = slug(sku_id or name)
        category, barcode = category.strip() or None, barcode.strip() or None
        imgs = [read_image(read_upload(f)) for f in files if f.filename]
        if not imgs:
            raise HTTPException(400, "Kamida bitta rasm yuklang")
        pipe = get_pipe()
        with State.lock:
            products = pipe.catalog.sku_info()
            if sku in products:
                raise HTTPException(409, f"Bu nomdagi mahsulot allaqachon bor ({sku}) — rasmni "
                                         "o'sha mahsulot sahifasidan qo'shing yoki boshqa nom bering")
            validate_catalog_metadata(category, barcode, products, sku)
            n = add_with_images(pipe, sku, name, imgs, brand.strip() or None,
                                extra={"category": category, "barcode": barcode})
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
            images = []
            for f in image_files(sku):
                shelf = f.stem.startswith("c")
                located = None if shelf else read_box(f)
                images.append({"name": f.name, "url": f"/media/{sku}/{f.name}", "shelf": shelf,
                               # located: mahsulot qutisi aniqlanganmi (eski rasmlarda yo'q —
                               # qayta indekslash kerak); box None — butun rasm ishlatiladi
                               "located": shelf or located is not None,
                               "box": located["box"] if located else None,
                               "auto": located["auto"] if located else True})
            return {"sku_id": sku, "name": v["name"], "brand": v.get("brand"), "vectors": v["count"],
                    **catalog_metadata(v), "images": images}

    @app.patch("/api/products/{sku}")
    def edit_product(sku: str, name: str = Form(...), brand: str = Form(""),
                     changes: dict = Depends(catalog_metadata_patch),
                     x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        name = name.strip()
        if not name:
            raise HTTPException(400, "Mahsulot nomini kiriting")
        with State.lock:
            products = pipe.catalog.sku_info()
            if sku not in products:
                raise HTTPException(404, "Mahsulot topilmadi")
            metadata = {**catalog_metadata(products[sku]), **changes}
            validate_catalog_metadata(metadata["category"], metadata["barcode"], products, sku)
            fields = {"name": name, "brand": brand.strip() or None, **metadata}
            pipe.catalog.update_sku(sku, **fields)
        return {"sku_id": sku, **fields}

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
            n = add_with_images(pipe, sku, v["name"], imgs, v.get("brand"), extra=catalog_metadata(v))
        return {"sku_id": sku, "added": len(imgs), "vectors": n}

    @app.delete("/api/products/{sku}/images/{name}")
    def delete_image(sku: str, name: str, x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            v = product_or_404(pipe, sku)
            f = images_root() / sku / name
            files = image_files(sku)
            if f not in files:
                raise HTTPException(404, "Rasm topilmadi")
            if len(files) == 1:
                raise HTTPException(400, "Oxirgi rasmni o'chirib bo'lmaydi — mahsulotni o'chiring")
            bak = f.with_suffix(".jpg.del")
            f.rename(bak)
            try:
                n = rebuild_sku(pipe, sku, v["name"], v.get("brand"), catalog_metadata(v))
            except Exception:
                bak.rename(f)
                raise
            bak.unlink()
            box_file(f).unlink(missing_ok=True)
        return {"sku_id": sku, "deleted": name, "vectors": n}

    @app.post("/api/products/{sku}/reindex")
    def reindex_product(sku: str, x_admin_token: Optional[str] = Header(None)):
        """Bitta mahsulotni saqlangan rasmlaridan qayta o'rganish. Mahsulot qutisi hali
        aniqlanmagan (eski) rasmlarda u avtomatik topiladi."""
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            v = product_or_404(pipe, sku)
            if not image_files(sku):
                raise HTTPException(400, "Mahsulotda rasm yo'q")
            n = rebuild_sku(pipe, sku, v["name"], v.get("brand"), catalog_metadata(v))
        return {"sku_id": sku, "vectors": n}

    @app.put("/api/products/{sku}/images/{name}/box")
    def set_image_box(sku: str, name: str, box: str = Form(...),
                      x_admin_token: Optional[str] = Header(None)):
        """Namuna rasmidagi mahsulot qutisini qo'lda to'g'rilash. box — JSON: 0..1
        normallashgan [x1,y1,x2,y2], "null" — butun rasm, "auto" — qayta avtomatik aniqlash.
        SKU vektorlari shu zahoti qayta quriladi."""
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            v = product_or_404(pipe, sku)
            f = images_root() / sku / name
            if f not in image_files(sku) or f.stem.startswith("c"):
                raise HTTPException(404, "Rasm topilmadi")
            sidecar = box_file(f)
            previous = sidecar.read_bytes() if sidecar.exists() else None
            try:
                if box.strip() == "auto":
                    sidecar.unlink(missing_ok=True)          # rebuild_sku qayta aniqlaydi
                else:
                    write_box(f, valid_box(json.loads(box)), auto=False)
            except (ValueError, TypeError):
                raise HTTPException(400, "Quti noto'g'ri")
            try:
                n = rebuild_sku(pipe, sku, v["name"], v.get("brand"), catalog_metadata(v))
            except Exception:
                if previous is None:
                    sidecar.unlink(missing_ok=True)
                else:
                    sidecar.write_bytes(previous)
                raise
            saved = read_box(f) or {"box": None, "auto": True}
        return {"sku_id": sku, "name": name, "vectors": n, **saved}

    @app.delete("/api/products/{sku}")
    def delete_product(sku: str, x_admin_token: Optional[str] = Header(None)):
        check_admin(x_admin_token)
        pipe = get_pipe()
        with State.lock:
            product_or_404(pipe, sku)
            pipe.catalog.delete_sku(sku)
            for f in image_files(sku):
                f.unlink()
                box_file(f).unlink(missing_ok=True)
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
            metadata = None
            if sku_id:
                existing = product_or_404(pipe, sku)
                name, brand = existing["name"], existing.get("brand") or ""
                metadata = catalog_metadata(existing)
            elif sku in pipe.catalog.sku_info():
                raise HTTPException(409, f"Bu nomdagi mahsulot allaqachon bor ({sku}) — "
                                         "ro'yxatdan mavjud mahsulotni tanlang")
            n = add_with_images(pipe, sku, name, crops, brand.strip() or None, prefix="c",
                                max_side=None, augment=False, source="crop", extra=metadata)
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
                n = rebuild_sku(pipe, sku, v["name"], v.get("brand"), catalog_metadata(v))
                done.append({"sku_id": sku, "images": len(files), "vectors": n})
            if not skipped:
                pipe.catalog.mark_current()    # hamma SKU joriy embedder bilan qayta qurildi
        return {"reindexed": done, "skipped": skipped}

    @app.get("/api/catalog/export")
    def export_catalog():
        """Katalogni bitta .zip to'plam qilib berish: catalog.json (nom, brend, kategoriya,
        shtrix-kod) + har mahsulotning namuna rasmlari va qutilari. Boshqa korxonada
        /api/catalog/import orqali ochiladi (vektorlar o'sha yerda qaytadan hisoblanadi)."""
        pipe = get_pipe()
        tmp = tempfile.NamedTemporaryFile(prefix="rasta-catalog-", suffix=".zip", delete=False)
        tmp.close()
        try:
            with State.lock, zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as z:
                products = []
                for sku, v in sorted(pipe.catalog.sku_info().items()):
                    files = image_files(sku)
                    for f in files:
                        z.write(f, f"{sku}/{f.name}")
                        if box_file(f).is_file():
                            z.write(box_file(f), f"{sku}/{box_file(f).name}")
                    products.append({"sku_id": sku, "name": v["name"], "brand": v.get("brand"),
                                     **catalog_metadata(v), "images": [f.name for f in files]})
                z.writestr("catalog.json", json.dumps(
                    {"format": "rasta-catalog", "version": 1, "products": products},
                    ensure_ascii=False, indent=1))
        except Exception:
            os.unlink(tmp.name)
            raise
        return FileResponse(tmp.name, media_type="application/zip", filename="rasta-katalog.zip",
                            background=BackgroundTask(os.unlink, tmp.name))

    @app.post("/api/catalog/import")
    def import_catalog(file: UploadFile = File(...), x_admin_token: Optional[str] = Header(None)):
        """Tayyor katalog to'plamini (.zip) shu korxona katalogiga qo'shish. Katalogda bor
        mahsulot (SKU bo'yicha) o'zgartirilmaydi — o'tkazib yuboriladi. Har rasm qayta
        o'qib tekshiriladi va qayta saqlanadi; vektorlar joriy model bilan hisoblanadi."""
        check_admin(x_admin_token)
        tmp = tempfile.TemporaryFile()        # diskda: katta to'plam xotirani to'ldirmaydi
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > BUNDLE_MAX_BYTES:
                raise HTTPException(413, "Katalog fayli juda katta (300 MB dan ko'p)")
            tmp.write(chunk)
        try:
            z = zipfile.ZipFile(tmp)
            infos = {i.filename: i for i in z.infolist()}
            if sum(i.file_size for i in infos.values()) > BUNDLE_MAX_UNPACKED:
                raise HTTPException(413, "Katalog fayli juda katta")
            if "catalog.json" not in infos or infos["catalog.json"].file_size > 8 * 1024 * 1024:
                raise ValueError
            meta = json.loads(z.read("catalog.json").decode("utf-8"))
            items = meta["products"]
            if meta.get("format") != "rasta-catalog" or not isinstance(items, list):
                raise ValueError
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Bu Rasta katalog fayli emas (.zip, ichida catalog.json bo'lishi kerak)")
        if len(items) > BUNDLE_MAX_PRODUCTS:
            raise HTTPException(413, "Katalogda mahsulotlar juda ko'p")

        def text(value, limit):
            return value.strip()[:limit] if isinstance(value, str) else ""

        pipe = get_pipe()
        added, skipped, failed = [], [], []
        for item in items:
            if not isinstance(item, dict):
                continue
            src = item.get("sku_id")
            name = text(item.get("name"), 200)
            names = item.get("images")
            if (not isinstance(src, str) or not BUNDLE_SKU.fullmatch(src) or not name
                    or not isinstance(names, list)):
                failed.append({"sku_id": str(src)[:64], "name": name, "reason": "Ma'lumot noto'g'ri"})
                continue
            sku = slug(src)
            names = [n for n in names if isinstance(n, str) and BUNDLE_IMAGE.fullmatch(n)
                     and f"{src}/{n}" in infos][:BUNDLE_MAX_IMAGES]
            brand = text(item.get("brand"), 120) or None
            category = text(item.get("category"), 80) or None
            barcode = text(item.get("barcode"), 32) or None
            with State.lock:
                products = pipe.catalog.sku_info()
                folder = images_root() / sku
                if sku in products or image_files(sku):
                    skipped.append({"sku_id": sku, "name": name})
                    continue
                if barcode and (not re.fullmatch(r"[0-9]{4,32}", barcode)
                                or any(v.get("barcode") == barcode for v in products.values())):
                    barcode = None                     # band yoki noto'g'ri shtrix-kod ko'chirilmaydi
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    for n in names:
                        if infos[f"{src}/{n}"].file_size > MAX_BYTES:
                            continue
                        img = read_image(z.read(f"{src}/{n}"))
                        crop = n.startswith("c")
                        im = Image.fromarray(img)
                        if not crop:
                            im.thumbnail((800, 800))
                        im.save(folder / n, quality=92 if crop else 88)
                        side = f"{src}/{n[:-4]}.box.json"
                        if not crop and side in infos and infos[side].file_size < 4096:
                            try:
                                saved = json.loads(z.read(side).decode("utf-8"))
                                write_box(folder / n, valid_box(saved.get("box")), bool(saved.get("auto", True)))
                            except (ValueError, TypeError, AttributeError):
                                pass                   # quti buzilgan — qayta avtomatik aniqlanadi
                    if not image_files(sku):
                        raise ValueError("Rasm yo'q")
                    vectors = rebuild_sku(pipe, sku, name, brand, {"category": category, "barcode": barcode})
                    count = len(image_files(sku))
                except Exception as error:
                    shutil.rmtree(folder, ignore_errors=True)
                    pipe.catalog.delete_sku(sku)
                    if isinstance(error, InferenceUnavailable):
                        raise
                    reason = error.detail if isinstance(error, HTTPException) else "Rasmlarni o'qib bo'lmadi"
                    failed.append({"sku_id": sku, "name": name, "reason": str(reason)})
                    continue
            added.append({"sku_id": sku, "name": name, "images": count, "vectors": vectors})
        return {"added": added, "skipped": skipped, "failed": failed}

    @app.get("/api/analyze/products")
    def analysis_products():
        """Tahlildan oldin "javonda qaysi mahsulotlar bor" deb tanlash uchun katalog ro'yxati.
        Agent ham ko'ra oladi (katalogni boshqarish huquqisiz): faqat nom, brend va rasm."""
        pipe = get_pipe()
        out = []
        with State.lock:
            for sku, v in pipe.catalog.sku_info().items():
                photos = image_files(sku)
                out.append({"sku_id": sku, "name": v["name"], "brand": v.get("brand"),
                            "thumb": f"/media/{sku}/{photos[0].name}" if photos else None})
        out.sort(key=lambda r: ((r.get("brand") or "").lower(), r["name"].lower()))
        return {"products": out}

    @app.post("/api/analyze", dependencies=[Depends(analysis_slot)])
    def analyze(request: Request, image: UploadFile = File(...), roi: str = Form(""),
                roi_normalized: bool = Form(False), visit_id: int | None = Form(None, gt=0),
                skus: str = Form("")):
        user = request.state.user
        visit = None
        if workspaces is not None and user["role"] == "operator":
            if visit_id is None:
                raise HTTPException(400, "Rasm tahlili uchun avval do‘konda tashrifni boshlang.")
            visit = workspaces.field.own_visit(user, visit_id)
            if visit is None:
                raise HTTPException(404, "Ochiq tashrif topilmadi.")
        upload = read_upload(image)
        with decode_image(upload) as decoded:
            roi_requested = bool(roi.strip())
            pipe = None if roi_requested else get_pipe()
            hires_needed = pipe is not None and getattr(pipe, "label_reader", None) is not None
            if hires_needed and max(decoded.size) > MAX_SIDE:
                with decoded.copy() as work_image:
                    img = image_pixels(work_image, MAX_SIDE)
            else:
                img = image_pixels(decoded, MAX_SIDE)
            region = None
            if roi_requested:
                from rasta.roi import make_region
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
            if pipe is None:
                pipe = get_pipe()
            extra = {}
            if hires_needed:
                extra["hires"] = lambda: image_pixels(decoded, OCR_MAX_SIDE)
                extra["words_for"] = lambda sku: sample_label_words(pipe, sku)
            scope = None
            with State.lock:
                if skus.strip():
                    # Qidiruv doirasi: javonda bor deb tanlangan mahsulotlar. Katalogdan o'chirilgan SKU'lar
                    # e'tiborga olinmaydi; birortasi ham qolmasa — xato (jimgina butun katalogga o'tmaymiz).
                    try:
                        wanted = json.loads(skus)
                        if (not isinstance(wanted, list) or len(wanted) > 5000
                                or not all(isinstance(item, str) and len(item) <= 200 for item in wanted)):
                            raise ValueError
                    except ValueError:
                        raise HTTPException(400, "Mahsulotlar ro'yxati noto'g'ri")
                    catalog_skus = set(pipe.catalog.sku_info())
                    scope = sorted(set(wanted) & catalog_skus)
                    if not scope:
                        raise HTTPException(400, "Tanlangan mahsulotlar katalogda topilmadi")
                    if len(scope) == len(catalog_skus):
                        scope = None                     # hammasi tanlangan — cheklov yo'q
                if scope is not None:
                    extra["skus"] = scope
                res = (pipe.analyze_region(img, region, **({"skus": scope} if scope is not None else {}))
                       if region is not None else pipe.analyze_image(img, **extra))
                summary = summarize(res, color_map(res))
        report_image = None
        if workspaces is not None:
            with io.BytesIO() as buf:
                Image.fromarray(img).save(buf, "JPEG", quality=90)
                report_image = buf.getvalue()
        archive_analysis(upload, img, res, report_image)
        data = {"summary": summary, **geometry(res, getattr(pipe, "params", None)),
                "size": [int(img.shape[1]), int(img.shape[0])]}
        if scope is not None:
            data["scope"] = len(scope)
        if res.timings_ms.get("ocr_failed"):
            data["label_check"] = "skipped"          # yozuv xizmati javob bermadi: tekshiruv o'tkazilmadi
        elif res.timings_ms.get("ocr_outdated"):
            data["label_check"] = "outdated"         # xizmat eski: harf o'lchamini qaytarmaydi, tekshiruv ishlamaydi
        elif res.timings_ms.get("ocr_disabled"):
            data["label_check"] = "disabled"
        if workspaces is not None:
            if visit is not None:
                data["visit_id"] = visit_id
                data["visit_store_id"] = visit["store_id"]
            try:
                data["report_id"] = workspaces.create_draft(user, data, report_image, visit_id if visit is not None else None)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return JSONResponse(data)

    @app.get("/api/health")
    def health():
        stale = getattr(State.pipe.catalog, "stale", None) if State.pipe is not None else None
        database_ready = accounts is None or accounts.database.ready()
        ready = database_ready and (State.inference is None or State.inference.ready())
        status = {"ok": ready, "loaded": State.pipe is not None and ready,
                  "catalog_stale": bool(stale), "database_ready": database_ready,
                  # "boshqa mahsulotlar" bazasi yuklanganmi (kichik katalogda soxta tanishni to'xtatadi)
                  "negatives": State.pipe is not None and getattr(State.pipe.catalog, "negatives", None) is not None,
                  # yorliq yozuvini o'qish xizmati ulanganmi (RASTA_OCR_URL)
                  "ocr": State.pipe is not None and getattr(State.pipe, "label_reader", None) is not None}
        return JSONResponse(status, status_code=200 if ready else 503)

    return app


app = create_app(serve_frontend=STATIC.is_dir())
