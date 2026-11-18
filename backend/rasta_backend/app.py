"""Rasta veb-ilovasi (FastAPI): admin — mahsulot katalogi, asosiy sahifa — javon tahlili.

Ishga tushirish:  python scripts/serve.py          ->  http://127.0.0.1:8000
Admin parol (ixtiyoriy):  RASTA_ADMIN_PASSWORD muhit o'zgaruvchisi.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import copy
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
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from rasta.config import ROOT, Config, resolve_path
from rasta.inference import InferenceUnavailable, RemoteInference
from .accounts import Accounts, SESSION_COOKIE
from .session_api import session_router
from .workspace_api import router as workspace_router
from .field_api import router as field_router

STATIC = ROOT / "frontend" / "dist"
MAX_SIDE = 2000                    # katta telefon rasmlari shu o'lchamgacha kichraytiriladi
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 50_000_000            # "dekompressiya bombasi": kichik fayl, ulkan rasm
# Bir-biridan aniq farqlanadigan ranglar: natijadagi mahsulotlarga ulush tartibida beriladi,
# shunda jadvaldagi rang belgisi rasmdagi ramka bilan bir qarashda mos keladi.
PALETTE = ["#2563eb", "#e11d48", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#ea580c",
           "#db2777", "#65a30d", "#0d9488", "#9333ea", "#ca8a04", "#4f46e5", "#dc2626"]
OTHER = "#94a3b8"
organization_scope: ContextVar[int | None] = ContextVar("organization_scope", default=None)


class State:
    pipe = None
    lock = threading.Lock()
    cfg: Optional[Config] = None
    inference: Optional[RemoteInference] = None
    organization_pipes = {}
    images_dir: Path = ROOT / "data" / "catalog_images"
    public = False      # server tarmoqqa ochiq (0.0.0.0) — parolsiz admin yopiladi


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
    n = 0
    with pipe.catalog.replace_sku(sku):
        if shots:
            n += pipe.add_product(sku, name, list(load(shots)), brand=brand, extra=extra)
        if crops:
            n += pipe.add_product(sku, name, list(load(crops)), brand=brand, extra=extra,
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
            catalog_read = path == "/api/products" or path.startswith("/api/products/") or path.startswith("/media/")
            catalog_action = (catalog_read and not path.startswith("/media/")) or path in {"/api/teach", "/api/reindex"}
            analysis_action = path == "/api/analyze" or path.startswith("/media/")
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
            return {"sku_id": sku, "name": v["name"], "brand": v.get("brand"), "vectors": v["count"],
                    **catalog_metadata(v),
                    "images": [{"name": f.name, "url": f"/media/{sku}/{f.name}",
                                "shelf": f.stem.startswith("c")} for f in image_files(sku)]}

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

    @app.post("/api/analyze")
    def analyze(request: Request, image: UploadFile = File(...), roi: str = Form(""),
                roi_normalized: bool = Form(False), visit_id: int | None = Form(None, gt=0)):
        user = request.state.user
        visit = None
        if workspaces is not None and user["role"] == "operator":
            if visit_id is None:
                raise HTTPException(400, "Rasm tahlili uchun avval do‘konda tashrifni boshlang.")
            visit = workspaces.field.own_visit(user, visit_id)
            if visit is None:
                raise HTTPException(404, "Ochiq tashrif topilmadi.")
        img = read_image(read_upload(image))
        region = None
        if roi.strip():
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
        pipe = get_pipe()
        with State.lock:
            res = (pipe.analyze_region(img, region) if region is not None
                   else pipe.analyze_image(img))
            summary = summarize(res, color_map(res))
        data = {"summary": summary, **geometry(res), "size": [int(img.shape[1]), int(img.shape[0])]}
        if workspaces is not None:
            with io.BytesIO() as buf:
                Image.fromarray(img).save(buf, "JPEG", quality=90)
                if visit is not None:
                    data["visit_id"] = visit_id
                    data["visit_store_id"] = visit["store_id"]
                try:
                    data["report_id"] = workspaces.create_draft(user, data, buf.getvalue(), visit_id if visit is not None else None)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
        return JSONResponse(data)

    @app.get("/api/health")
    def health():
        stale = getattr(State.pipe.catalog, "stale", None) if State.pipe is not None else None
        database_ready = accounts is None or accounts.database.ready()
        ready = database_ready and (State.inference is None or State.inference.ready())
        status = {"ok": ready, "loaded": State.pipe is not None and ready,
                  "catalog_stale": bool(stale), "database_ready": database_ready}
        return JSONResponse(status, status_code=200 if ready else 503)

    return app


app = create_app(serve_frontend=STATIC.is_dir())
