"""Veb-ilova API testi — og'ir modellarsiz (soxta pipeline + xotiradagi katalog)."""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from rasta.catalog import SkuCatalog  # noqa: E402
from rasta.kpi import compute_kpi  # noqa: E402
from rasta.types import AnalysisResult, Box, Detection, Gap, Match  # noqa: E402
from rasta_backend import app as web  # noqa: E402


class FakePipe:
    def __init__(self):
        self.catalog = SkuCatalog(dim=4, path=":memory:", collection="web_t")
        self.regions = []

    located = [0.25, 0.1, 0.75, 0.9]

    def locate_product(self, image):
        return self.located

    def add_product(self, sku, name, imgs, brand=None, **kw):
        self.boxes = kw.get("boxes")
        v = np.eye(4, dtype=np.float32)[: len(imgs) % 4 + 1]
        return self.catalog.add_sku(sku, name, v, extra={**(kw.get("extra") or {}), "brand": brand})

    def _result(self, img):
        dets = [Detection(Box(10, 10, 40, 90)), Detection(Box(50, 10, 80, 90))]
        dets[0].match = Match("cola", "Cola", 0.9, 0.2, True, brand="ColaCo", level="sku")
        dets[1].match = Match(None, None, 0.3, 0.0, False)
        for d in dets:
            d.row = 0
        return AnalysisResult("x", (img.shape[1], img.shape[0]), dets, [], 1,
                              compute_kpi(dets, [], 1, {"cola": "ColaCo"}))

    def analyze_image(self, img, **kw):
        return self._result(img)

    def analyze_region(self, img, region, *a, **kw):
        self.regions.append(region)
        r = self._result(img)
        r.roi = region
        r.kpi["roi_shelf_fill_pct"] = 50.0
        return r


def jpg(w=120, h=100):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 40)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("RASTA_ADMIN_PASSWORD", raising=False)
    app = web.create_app()
    web.State.pipe = FakePipe()
    web.State.images_dir = tmp_path
    yield TestClient(app)
    web.State.pipe = None


def test_pages_served(client):
    if not (web.STATIC / "index.html").is_file():
        pytest.skip("React distribution is built in the frontend job")
    for route in ("/", "/admin", "/login", "/dashboard", "/field"):
        page = client.get(route)
        assert page.status_code == 200
        assert '<div id="root"></div>' in page.text
    assert client.get("/static/login-shelf.webp").status_code == 200


def test_catalog_metadata_survives_photo_changes_and_reindex(client):
    photos = [("files", ("a.jpg", jpg(), "image/jpeg"))]
    sku = "/api/products/company_cola"
    assert client.post("/api/products", data={
        "name": "Cola", "sku_id": "company_cola", "category": "Ichimliklar", "barcode": "001234",
    }, files=photos).status_code == 200
    assert client.post("/api/products", data={"name": "Other", "barcode": "001234"},
                       files=photos).status_code == 409
    assert client.patch(sku, data={"name": "Cola 1L", "barcode": "abcd"}).status_code == 400
    assert client.post(sku + "/images", files=photos).status_code == 200
    assert client.delete(sku + "/images/000.jpg").status_code == 200
    assert client.post("/api/reindex").status_code == 200
    product = client.get(sku).json()
    assert (product["category"], product["barcode"]) == ("Ichimliklar", "001234")
    assert client.patch(sku, data={"name": "Cola 1L"}).json()["barcode"] == "001234"
    assert client.patch(sku, data={"name": "Cola 1L", "barcode": ""}).json()["barcode"] is None


def test_reference_photo_box_is_stored_shown_and_editable(client, tmp_path):
    sku = "/api/products/juice"
    assert client.post("/api/products", data={"name": "Juice"},
                       files=[("files", ("a.jpg", jpg(), "image/jpeg"))]).status_code == 200
    pipe = web.State.pipe
    assert pipe.boxes == [pipe.located]                       # embedding shu qutidan olinadi
    image = client.get(sku).json()["images"][0]
    assert (image["located"], image["box"], image["auto"]) == (True, pipe.located, True)

    box = sku + "/images/000.jpg/box"
    assert client.put(box, data={"box": "[0.1, 0.2, 0.6, 0.8]"}).json()["auto"] is False
    assert pipe.boxes == [[0.1, 0.2, 0.6, 0.8]]               # qo'lda to'g'rilangan quti qayta o'rganildi
    assert client.put(box, data={"box": "null"}).json()["box"] is None
    assert pipe.boxes == [None]
    for bad in ("[0.5, 0.5, 0.5, 0.9]", "[1, 2]", "oops"):
        assert client.put(box, data={"box": bad}).status_code == 400
    assert client.get(sku).json()["images"][0]["box"] is None  # xato so'rov holatni buzmaydi
    restored = client.put(box, data={"box": "auto"}).json()
    assert (restored["box"], restored["auto"]) == (pipe.located, True)
    assert client.put(sku + "/images/missing.jpg/box", data={"box": "null"}).status_code == 404

    # eski rasm (quti fayli yo'q): katalogda "ajratilmagan", qayta o'rganishda aniqlanadi
    (tmp_path / "juice" / "000.box.json").unlink()
    assert client.get(sku).json()["images"][0]["located"] is False
    assert client.post(sku + "/reindex").status_code == 200
    assert client.get(sku).json()["images"][0]["box"] == pipe.located
    assert client.delete(sku).status_code == 200
    assert not list((tmp_path / "juice").glob("*"))


def test_analyzed_photos_are_archived_once_with_draft_boxes(client, tmp_path):
    import json

    web.State.archive_dir = tmp_path / "archive"
    try:
        photo = [("image", ("a.jpg", jpg(), "image/jpeg"))]
        assert client.post("/api/analyze", files=photo).status_code == 200
        assert client.post("/api/analyze", files=photo).status_code == 200      # qayta tahlil — takrorlanmaydi
        images = list(web.State.archive_dir.rglob("*.jpg"))
        assert len(images) == 1
        draft = json.loads(images[0].with_suffix(".det.json").read_text(encoding="utf-8"))
        assert draft["status"] == "draft" and len(draft["boxes"]) == 2
        assert draft["labels"] == ["cola", None] and draft["size"] == [120, 100]
        web.State.archive_dir = None                                             # o'chirilgan: hech narsa yozilmaydi
        other = [("image", ("b.jpg", jpg(90, 80), "image/jpeg"))]
        assert client.post("/api/analyze", files=other).status_code == 200
        assert len(list((tmp_path / "archive").rglob("*.jpg"))) == 1
    finally:
        web.State.archive_dir = None


def test_explain_reports_why_a_box_was_or_was_not_recognised():
    from rasta.catalog import RecognitionParams

    params = RecognitionParams(sim_threshold=0.76, sim_threshold_packshot=0.65, min_margin=0.02)

    def det(score, known, margin=0.2, **top):
        d = Detection(Box(0, 0, 10, 20))
        d.match = Match("juice" if known else None, "Juice" if known else None, score, margin, known,
                        candidates=[{"sku_id": "juice", "name": "Juice", "score": score, "src": "packshot", **top}])
        return d

    assert web.explain(det(0.71, True), params) == {"n": "Juice", "sc": 0.71, "tau": 0.65, "m": 0.2, "r": "ok"}
    assert web.explain(det(0.60, False), params)["r"] == "low"
    assert web.explain(det(0.70, False, color=0.1), RecognitionParams(
        sim_threshold_packshot=0.65, color_min_packshot=0.5))["r"] == "color"
    assert web.explain(det(0.70, False, neg=0.80), RecognitionParams(
        sim_threshold_packshot=0.65, neg_margin_packshot=0.03))["r"] == "other"
    assert web.explain(det(0.70, False, margin=0.0), params)["r"] == "ambiguous"
    assert web.explain(det(0.70, False), params)["r"] == "size"                # barcha shartlar o'tgan, lekin bekor qilingan
    assert web.explain(Detection(Box(0, 0, 1, 1)), params) == {"r": "empty"}
    weak = RecognitionParams(sim_threshold_packshot=0.65, embed_min_packshot=0.6)
    assert web.explain(det(0.66, False, dino=0.58), weak)["r"] == "weak"
    variant = det(0.70, False, color=0.7)
    variant.match.level = "variant"
    assert web.explain(variant, params) == {"n": "Juice", "sc": 0.7, "tau": 0.65, "m": 0.2, "r": "variant", "c": 0.7}


def test_product_add_list_delete(client):
    r = client.post("/api/products", data={"name": "Coca-Cola 1.5L", "brand": "Coca-Cola"},
                    files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    assert r.status_code == 200 and r.json()["sku_id"] == "coca_cola_1_5l"
    ps = client.get("/api/products").json()["products"]
    assert ps[0]["name"] == "Coca-Cola 1.5L" and ps[0]["brand"] == "Coca-Cola"
    assert ps[0]["thumb"] and client.get(ps[0]["thumb"]).status_code == 200
    assert client.delete("/api/products/coca_cola_1_5l").status_code == 200
    assert client.get("/api/products").json()["products"] == []


def test_product_validation(client):
    r = client.post("/api/products", data={"name": "  "},
                    files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    assert r.status_code == 400
    r = client.post("/api/products", data={"name": "X"},
                    files=[("files", ("a.jpg", b"not an image", "image/jpeg"))])
    assert r.status_code == 400


def test_media_path_traversal_blocked(client):
    assert client.get("/media/..%2F..%2Fconfig.yaml/x").status_code == 404


def test_admin_password(client, monkeypatch):
    monkeypatch.setenv("RASTA_ADMIN_PASSWORD", "s3cret")
    files = [("files", ("a.jpg", jpg(), "image/jpeg"))]
    assert client.post("/api/products", data={"name": "A"}, files=files).status_code == 401
    ok = client.post("/api/products", data={"name": "A"}, files=files,
                     headers={"X-Admin-Token": "s3cret"})
    assert ok.status_code == 200


def test_analyze_whole_and_zone(client):
    r = client.post("/api/analyze", files={"image": ("s.jpg", jpg(), "image/jpeg")})
    d = r.json()
    assert r.status_code == 200 and d["size"] == [120, 100]
    assert len(d["detections"]) == 2 and d["detections"][0]["s"] == "cola"
    assert d["detections"][1]["s"] is None and d["roi"] is None
    s = d["summary"]
    assert s["num_products"] == 2 and s["num_known"] == 1 and s["num_unknown"] == 1
    assert d["suspect_gaps"] == [] and d["divider_gaps"] == []
    assert s["products"][0]["name"] == "Cola" and s["products"][0]["color"].startswith("#")
    assert 0 <= s["unknown_share_pct"] <= 100
    r2 = client.post("/api/analyze", files={"image": ("s.jpg", jpg(), "image/jpeg")},
                     data={"roi": "[[5,5],[100,5],[100,90],[5,90]]"})
    assert r2.json()["summary"]["fill_pct"] == 50.0
    assert r2.json()["roi"] == [[5, 5], [100, 5], [100, 90], [5, 90]]
    assert len(web.State.pipe.regions) == 1
    bad = client.post("/api/analyze", files={"image": ("s.jpg", jpg(), "image/jpeg")},
                      data={"roi": "nonsense"})
    assert bad.status_code == 400


def test_analysis_keeps_uncertain_gaps_out_of_oos_total(client, monkeypatch):
    pipe = web.State.pipe
    result = pipe._result(np.zeros((100, 120, 3), dtype=np.uint8))
    result.suspects = [Gap(0, 40, 10, 50, 90, 10, 1, status="suspect")]
    result.dividers = [Gap(0, 90, 10, 110, 90, 20, 1, status="divider")]
    result.kpi["oos"]["status"] = "limited_scene"
    monkeypatch.setattr(pipe, "analyze_image", lambda img: result)

    data = client.post("/api/analyze", files={"image": ("s.jpg", jpg(), "image/jpeg")}).json()
    assert data["suspect_gaps"] == [[40, 10, 50, 90]]
    assert data["divider_gaps"] == [[90, 10, 110, 90]]
    assert data["gaps"] == [] and data["summary"]["oos_facings"] == 0
    assert data["summary"]["oos_status"] == "limited_scene"


def test_parallel_teach_requests_create_only_one_sku(client):
    start = Barrier(2, timeout=5)

    def teach():
        start.wait()
        return client.post(
            "/api/teach", data={"name": "Cola", "boxes": "[[10,10,40,75]]"},
            files={"image": ("s.jpg", jpg(), "image/jpeg")},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(teach) for _ in range(2)]
        assert sorted(future.result().status_code for future in futures) == [200, 409]
    images = client.get("/api/products/cola").json()["images"]
    assert len(images) == 1 and images[0]["shelf"] is True


def test_reindex_endpoint(client):
    client.post("/api/products", data={"name": "Cola"}, files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    r = client.post("/api/reindex")
    assert r.status_code == 200 and r.json()["reindexed"][0]["sku_id"] == "cola"


def test_normalized_zone_uses_resized_image_coordinates(client):
    response = client.post(
        "/api/analyze",
        files={"image": ("s.jpg", jpg(4000, 2000), "image/jpeg")},
        data={"roi": "[[0.1,0.2],[0.9,0.2],[0.9,0.8],[0.1,0.8]]", "roi_normalized": "true"},
    )
    assert response.status_code == 200
    assert response.json()["size"] == [2000, 1000]
    assert response.json()["roi"] == [[200, 200], [1800, 200], [1800, 800], [200, 800]]


def test_reindex_keeps_old_vectors_if_reading_a_crop_fails(client):
    response = client.post("/api/products", data={"name": "Cola"},
                           files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    assert response.status_code == 200
    catalog = web.State.pipe.catalog
    ids_before = {point.id for point in catalog._scroll()}
    (web.State.images_dir / "cola" / "c001.jpg").write_bytes(b"invalid")

    with TestClient(web.create_app(), raise_server_exceptions=False) as api:
        assert api.post("/api/reindex").status_code == 500
    assert {point.id for point in catalog._scroll()} == ids_before
    assert client.get("/api/products").json()["products"][0]["name"] == "Cola"


def test_teach_from_shelf(client):
    files = {"image": ("s.jpg", jpg(200, 150), "image/jpeg")}
    r = client.post("/api/teach", files=files, data={"boxes": "[[10,10,60,120],[70,10,120,120]]",
                                                     "name": "Tarxun 1.5L", "brand": "CG"})
    assert r.status_code == 200 and r.json()["added"] == 2 and r.json()["sku_id"] == "тархун_1_5l".replace("тархун", "tarxun")
    ps = client.get("/api/products").json()["products"]
    assert ps[0]["images"] == 2 and ps[0]["brand"] == "CG"
    # mavjud mahsulotga qo'shish
    r = client.post("/api/teach", files={"image": ("s.jpg", jpg(200, 150), "image/jpeg")},
                    data={"boxes": "[[10,10,60,120]]", "sku_id": "tarxun_1_5l"})
    assert r.status_code == 200 and client.get("/api/products").json()["products"][0]["images"] == 3
    assert client.post("/api/teach", files={"image": ("s.jpg", jpg(), "image/jpeg")},
                       data={"boxes": "[]", "name": "X"}).status_code == 400
    assert client.post("/api/teach", files={"image": ("s.jpg", jpg(), "image/jpeg")},
                       data={"boxes": "[[1,1,50,50]]", "sku_id": "yoq"}).status_code == 404


def test_product_edit_and_images(client):
    files = [("files", (f"{i}.jpg", jpg(), "image/jpeg")) for i in range(2)]
    sku = client.post("/api/products", data={"name": "Tarxun", "brand": "CG"}, files=files).json()["sku_id"]
    # nom/brendni o'zgartirish — sku_id o'zgarmaydi
    r = client.patch(f"/api/products/{sku}", data={"name": "Тархун 1.5L", "brand": "Черноголовка"})
    assert r.status_code == 200
    p = client.get(f"/api/products/{sku}").json()
    assert (p["name"], p["brand"], len(p["images"])) == ("Тархун 1.5L", "Черноголовка", 2)
    # rasm qo'shish
    r = client.post(f"/api/products/{sku}/images", files=[("files", ("n.jpg", jpg(), "image/jpeg"))])
    assert r.json()["added"] == 1
    imgs = client.get(f"/api/products/{sku}").json()["images"]
    assert len(imgs) == 3 and len({i["name"] for i in imgs}) == 3
    # bitta rasmni o'chirish -> qolganlardan qayta o'rganadi
    r = client.delete(f"/api/products/{sku}/images/{imgs[0]['name']}")
    assert r.status_code == 200 and r.json()["vectors"] > 0
    assert len(client.get(f"/api/products/{sku}").json()["images"]) == 2
    assert client.get("/api/products").json()["products"][0]["name"] == "Тархун 1.5L"
    assert client.delete(f"/api/products/{sku}/images/../x.jpg").status_code == 404
    assert client.patch("/api/products/nope", data={"name": "x"}).status_code == 404


def test_catalog_delete_and_rename_refresh_cache():
    cat = SkuCatalog(dim=4, path=":memory:", collection="cache_t")
    cat.add_sku("a", "A", np.eye(4, dtype=np.float32)[:1])
    cat._load_cache()
    cat.update_sku("a", name="A2", brand="B")
    assert cat._load_cache()[1][0]["name"] == "A2"
    cat.delete_sku("a")
    assert len(cat._load_cache()[0]) == 0          # o'chirilgan SKU keshda qolmaydi


def test_public_server_without_password_blocks_admin(client):
    web.State.public = True
    try:
        files = [("files", ("a.jpg", jpg(), "image/jpeg"))]
        r = client.post("/api/products", data={"name": "A"}, files=files)
        assert r.status_code == 403 and "RASTA_ADMIN_PASSWORD" in r.json()["detail"]
        assert client.post("/api/reindex").status_code == 403
        # tahlil admin emas — ishlayveradi
        assert client.post("/api/analyze", files={"image": ("s.jpg", jpg(), "image/jpeg")}).status_code == 200
    finally:
        web.State.public = False


def test_public_server_admin_open_flag(client, monkeypatch):
    """RASTA_ADMIN_OPEN=1 — parolsiz ham tarmoqda admin ochiq (ataylab)."""
    monkeypatch.delenv("RASTA_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("RASTA_ADMIN_OPEN", "1")
    web.State.public = True
    try:
        files = [("files", ("a.jpg", jpg(), "image/jpeg"))]
        assert client.post("/api/products", data={"name": "A"}, files=files).status_code == 200
    finally:
        web.State.public = False


def test_duplicate_name_conflict(client):
    files = [("files", ("a.jpg", jpg(), "image/jpeg"))]
    assert client.post("/api/products", data={"name": "Cola 1L"}, files=files).status_code == 200
    r = client.post("/api/products", data={"name": "cola-1l"}, files=files)   # bir xil slug
    assert r.status_code == 409
    assert client.get("/api/products").json()["products"][0]["images"] == 1
    r = client.post("/api/teach", files={"image": ("s.jpg", jpg(200, 150), "image/jpeg")},
                    data={"boxes": "[[10,10,60,120]]", "name": "Cola 1L"})
    assert r.status_code == 409


def test_huge_image_rejected(client):
    buf = io.BytesIO()
    Image.new("L", (9000, 6000)).save(buf, "PNG")          # kichik fayl, 54 Mpx
    r = client.post("/api/analyze", files={"image": ("big.png", buf.getvalue(), "image/png")})
    assert r.status_code == 413


def test_failed_indexing_leaves_no_orphan_images(client, tmp_path):
    def boom(*a, **kw):
        raise RuntimeError("GPU xatosi")
    web.State.pipe.add_product = boom
    c = TestClient(client.app, raise_server_exceptions=False)
    r = c.post("/api/products", data={"name": "X"}, files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    assert r.status_code == 500
    assert list(tmp_path.rglob("*.jpg")) == []


def test_catalog_remembers_embedder(tmp_path):
    p = str(tmp_path / "q")
    cat = SkuCatalog(dim=4, path=p, collection="m", embedder="model=A")
    cat.add_sku("a", "A", np.eye(4, dtype=np.float32)[:1])
    cat._close()
    same = SkuCatalog(dim=4, path=p, collection="m", embedder="model=A")
    assert same.stale is None
    same._close()
    other = SkuCatalog(dim=4, path=p, collection="m", embedder="model=B")
    assert other.stale == "model=A" and other.count() == 1     # ochiladi, lekin eskirgan
    other.mark_current()
    other._close()
    assert SkuCatalog(dim=4, path=p, collection="m", embedder="model=B").stale is None


def bundle(products, files):
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("catalog.json", json.dumps({"format": "rasta-catalog", "version": 1, "products": products}))
        for name, data in files.items():
            z.writestr(name, data)
    return {"file": ("katalog.zip", buf.getvalue(), "application/zip")}


def test_catalog_bundle_round_trip(client):
    import zipfile
    client.post("/api/products", data={"name": "Cola", "brand": "ColaCo", "category": "Ichimlik", "barcode": "123456"},
                files=[("files", ("a.jpg", jpg(), "image/jpeg"))])
    client.post("/api/teach", files={"image": ("s.jpg", jpg(200, 150), "image/jpeg")},
                data={"boxes": "[[10,10,60,120],[70,10,120,120]]", "sku_id": "cola"})
    exported = client.get("/api/catalog/export")
    assert exported.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(exported.content)).namelist())
    assert {"catalog.json", "cola/000.jpg", "cola/000.box.json", "cola/c001.jpg", "cola/c002.jpg"} <= names

    again = {"file": ("k.zip", exported.content, "application/zip")}
    skipped = client.post("/api/catalog/import", files=again).json()
    assert skipped["added"] == [] and skipped["skipped"][0]["sku_id"] == "cola"

    assert client.delete("/api/products/cola").status_code == 200
    result = client.post("/api/catalog/import", files=again).json()
    assert result["failed"] == [] and result["added"][0]["images"] == 3
    product = client.get("/api/products").json()["products"][0]
    assert (product["name"], product["brand"], product["category"], product["barcode"], product["images"]) == (
        "Cola", "ColaCo", "Ichimlik", "123456", 3)
    saved = json.loads((web.State.images_dir / "cola" / "000.box.json").read_text())
    assert saved["box"] == FakePipe.located


def test_catalog_import_rejects_unsafe_or_broken_bundles(client, tmp_path):
    bad = client.post("/api/catalog/import", files={"file": ("k.zip", b"not a zip", "application/zip")})
    assert bad.status_code == 400
    result = client.post("/api/catalog/import", files=bundle(
        [{"sku_id": "../evil", "name": "Evil", "images": ["000.jpg"]},
         {"sku_id": "escape", "name": "Escape", "images": ["../../x.jpg", "..\\x.jpg", "000.exe"]},
         {"sku_id": "broken", "name": "Broken", "images": ["000.jpg"]},
         {"sku_id": "good", "name": "Good", "images": ["c000.jpg"]}],
        {"../evil/000.jpg": jpg(), "escape/../../x.jpg": jpg(), "broken/000.jpg": b"<html>", "good/c000.jpg": jpg()},
    )).json()
    assert [p["sku_id"] for p in result["added"]] == ["good"]
    assert {p["name"] for p in result["failed"]} == {"Evil", "Escape", "Broken"}
    assert sorted(p.name for p in web.State.images_dir.iterdir()) == ["good"]
    assert not (tmp_path.parent / "x.jpg").exists()
    assert [p["sku_id"] for p in client.get("/api/products").json()["products"]] == ["good"]
