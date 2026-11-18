"""Yangi komponentlar testi: rang/shakl deskriptorlari, gibrid qaror, qatorlar,
orqa fon, OOS (javon kengligi + rasm tekshiruvi), kontekst, klasterlash, kalibrlash.

Og'ir modellar (DINOv2/YOLO) kerak emas — Qdrant xotira rejimida ishlatiladi.
  python -m pytest tests -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasta.calibrate import Rec, fit_thresholds, open_accuracy, report
from rasta.catalog import RecognitionParams, SkuCatalog
from rasta.cluster import build_cluster_features, cluster_features, purity_and_coverage
from rasta.context import smooth_with_context
from rasta.detector import suppress_nested
from rasta.features import (COLOR_DIM, color_descriptor, color_similarity,
                            shape_similarity)
from rasta.kpi import compute_kpi
from rasta.oos import detect_gaps, verify_gaps
from rasta.shelf import assign_rows, mark_background_rows
from rasta.types import Box, Detection, Match


def solid(rgb, h=120, w=50):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


# ---- rang / shakl ----------------------------------------------------------

def test_color_descriptor_shape_and_norm():
    v = color_descriptor(solid((200, 30, 40)))
    assert v.shape == (COLOR_DIM,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_color_same_vs_different():
    red = color_descriptor(solid((200, 30, 40)))
    red2 = color_descriptor(solid((185, 40, 45)))       # biroz boshqa yorug'lik
    orange = color_descriptor(solid((240, 140, 20)))
    black = color_descriptor(solid((25, 25, 30)))
    assert color_similarity(red, red2) > 0.9
    assert color_similarity(red, orange) < color_similarity(red, red2)
    assert color_similarity(orange, black) < 0.3       # Fanta vs Chin8


def test_color_brightness_robust():
    a = color_descriptor(solid((40, 90, 200)))
    b = color_descriptor((solid((40, 90, 200)) * 0.75).astype(np.uint8))
    assert color_similarity(a, b) > 0.8


def test_color_spatial_layout():
    """Qopqoq rangi farqi (tepa tasma) o'xshashlikni pasaytiradi."""
    a = solid((200, 30, 40)); a[:30] = (20, 20, 20)
    b = solid((200, 30, 40)); b[:30] = (240, 240, 240)
    c = a.copy()
    sa, sb, sc = map(color_descriptor, (a, b, c))
    assert color_similarity(sa, sc) > color_similarity(sa, sb)


def test_shape_similarity():
    assert shape_similarity(2.5, 2.5) == pytest.approx(1.0)
    assert shape_similarity(2.5, 1.2) < 0.2      # shisha vs quti
    assert shape_similarity(0, 2.0) == 1.0       # ma'lumot yo'q


# ---- gibrid qaror ----------------------------------------------------------

def _hit(sku, score, color=None, aspect=None):
    p = {"sku_id": sku, "name": sku}
    if color is not None:
        p["color"] = color.tolist()
    if aspect is not None:
        p["aspect"] = aspect
    return SimpleNamespace(score=score, payload=p)


def test_color_rerank_fixes_dino_confusion():
    """DINO Fanta'ni yuqori qo'yadi, lekin so'rov qora — rang uni pastga tushiradi."""
    orange = color_descriptor(solid((240, 140, 20)))
    black = color_descriptor(solid((25, 25, 30)))
    hits = [_hit("fanta", 0.80, orange), _hit("chin8", 0.76, black)]
    q = black
    p0 = RecognitionParams(sim_threshold=0.5, min_margin=0.0)
    p1 = RecognitionParams(sim_threshold=0.5, min_margin=0.0, w_color=0.3)
    assert SkuCatalog.decide(hits, p0, q).sku_id == "fanta"
    assert SkuCatalog.decide(hits, p1, q).sku_id == "chin8"


def test_color_veto_rejects_unknown():
    orange = color_descriptor(solid((240, 140, 20)))
    black = color_descriptor(solid((25, 25, 30)))
    hits = [_hit("fanta", 0.80, orange)]
    p = RecognitionParams(sim_threshold=0.5, min_margin=0.0, w_color=0.0, color_min=0.5)
    m = SkuCatalog.decide(hits, p, black, None)
    assert not m.is_known and m.sku_id is None


def test_catalog_inmemory_roundtrip():
    cat = SkuCatalog(dim=4, path=":memory:", collection="t")
    rng = np.random.default_rng(0)
    va = rng.normal(size=(3, 4)); va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb = -va
    col = np.stack([color_descriptor(solid((200, 30, 40)))] * 3)
    cat.add_sku("a", "A", va, extra={"brand": "X"}, colors=col, aspects=[2.5] * 3)
    cat.add_sku("b", "B", vb, colors=col, aspects=[2.5] * 3)
    assert cat.list_skus() == {"a": 3, "b": 3}
    assert cat.brand_map() == {"a": "X"}
    m = cat.search(va[:1], RecognitionParams(sim_threshold=0.5, min_margin=0.0, w_color=0.2),
                   colors=col[:1], aspects=[2.5])
    assert m[0].sku_id == "a" and m[0].brand == "X"
    cat.delete_sku("b")
    assert cat.list_skus() == {"a": 3}
    with pytest.raises(ValueError):
        SkuCatalog(dim=8, client=cat.client, collection="t")   # o'lcham mos emas


# ---- detektsiya / qatorlar -------------------------------------------------

def test_suppress_nested():
    big = Box(0, 0, 50, 150)
    part = Box(5, 60, 45, 140)      # bankaning qismi
    other = Box(60, 0, 110, 150)
    out = suppress_nested([big, part, other])
    assert part not in out and big in out and other in out


def test_rows_bottom_anchor_mixed_heights():
    """Baland shisha va past banka bir javonda: y-markazlari farq qiladi, pastki chiziq bir xil."""
    dets = []
    for i, h in enumerate([280, 60, 280, 60, 280, 60]):
        dets.append(Detection(Box(i * 60, 300 - h, i * 60 + 50, 300)))
    for i in range(6):
        dets.append(Detection(Box(i * 60, 450, i * 60 + 50, 600)))
    assert assign_rows([Detection(d.box) for d in dets], 0.6, "bottom") == 2
    assert assign_rows([Detection(d.box) for d in dets], 0.6, "center") > 2


def test_background_row_detected():
    dets = [Detection(Box(x, 100, x + 50, 250)) for x in range(0, 600, 60)]
    necks = [Detection(Box(x, 40, x + 30, 105)) for x in (100, 300, 500)]   # orqa bo'yinlar
    allp = dets + necks
    n = assign_rows(allp, 0.6, "bottom")
    bg = mark_background_rows(allp, n)
    assert bg and all(d.row in bg for d in necks)
    assert not any(d.row in bg for d in dets)


# ---- OOS -------------------------------------------------------------------

def _grid(rows=3, cols=8, empties=()):
    dets = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in empties:
                continue
            dets.append(Detection(Box(c * 60, r * 200, c * 60 + 50, r * 200 + 150)))
    return dets


def test_trailing_gap_found_with_shelf_extent():
    dets = _grid(empties={(1, 6), (1, 7)})      # 1-qator OXIRI bo'sh
    n = assign_rows(dets)
    old = detect_gaps(dets, n, use_shelf_extent=False)
    new = detect_gaps(dets, n, use_shelf_extent=True)
    assert not any(g.row == 1 for g in old)
    assert any(g.row == 1 and g.est_missing_facings >= 2 for g in new)


def test_verify_gaps_marks_textured_gap_suspect():
    img = np.full((600, 500, 3), 200, dtype=np.uint8)
    dets = _grid(rows=1, cols=8, empties={(0, 3)})
    rng = np.random.default_rng(0)
    for d in dets:   # mahsulotlar — teksturali
        x1, y1, x2, y2 = d.box.as_xyxy()
        img[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3))
    n = assign_rows(dets)
    gaps = detect_gaps(dets, n)
    assert gaps
    oos, sus = verify_gaps(img, [g for g in gaps], dets)
    assert oos and not sus                     # tekis bo'shliq -> haqiqiy OOS
    # endi bo'shliqqa "o'tkazib yuborilgan mahsulot" qo'yamiz
    g = gaps[0]
    img[int(g.y1):int(g.y2), int(g.x1):int(g.x2)] = rng.integers(
        0, 255, (int(g.y2) - int(g.y1), int(g.x2) - int(g.x1), 3))
    oos2, sus2 = verify_gaps(img, detect_gaps(dets, n), dets)
    assert sus2 and not oos2


# ---- kontekst --------------------------------------------------------------

def _m(sku, known, cands=()):
    return Match(sku if known else None, sku if known else None, 0.7, 0.1, known,
                 [{"sku_id": c, "score": s, "color": 1.0} for c, s in cands])


def test_context_fixes_unknown_between_same_sku():
    dets = [Detection(Box(i * 60, 0, i * 60 + 50, 150)) for i in range(3)]
    dets[0].match = _m("cola", True)
    dets[1].match = _m(None, False, [("cola", 0.50), ("pepsi", 0.45)])
    dets[2].match = _m("cola", True)
    assign_rows(dets)
    assert smooth_with_context(dets, sim_threshold=0.55, relax=0.08) == 1
    assert dets[1].match.sku_id == "cola" and dets[1].context_fixed


def test_context_never_overrides_known_or_far_candidates():
    dets = [Detection(Box(i * 60, 0, i * 60 + 50, 150)) for i in range(3)]
    dets[0].match = _m("cola", True)
    dets[1].match = _m(None, False, [("cola", 0.30)])      # juda past ball
    dets[2].match = _m("cola", True)
    assign_rows(dets)
    assert smooth_with_context(dets, sim_threshold=0.55, relax=0.08) == 0


def test_context_preserves_brand_and_negative_veto():
    params = RecognitionParams(sim_threshold=0.55, neg_margin=0.03)
    dets = [Detection(Box(i * 60, 0, i * 60 + 50, 150)) for i in range(3)]
    dets[0].match = dets[2].match = _m("cola", True)
    match = dets[1].match = _m(None, False, [("cola", 0.62)])
    assign_rows(dets)
    match.level = "brand"
    assert smooth_with_context(dets, sim_threshold=0.55, params=params) == 0
    match.level = None
    match.negative_score = 0.8
    assert smooth_with_context(dets, sim_threshold=0.55, params=params) == 0
    match.negative_score = 0.63
    assert smooth_with_context(dets, sim_threshold=0.55, params=params) == 1
    assert match.level == "sku"


def test_no_detections_leave_oos_unestimated():
    from types import SimpleNamespace

    from rasta.config import Config
    from rasta.pipeline import RastaPipeline

    pipe = RastaPipeline.__new__(RastaPipeline)
    pipe.cfg = Config({})
    pipe.detector = SimpleNamespace(detect=lambda image: [])
    image = np.zeros((30, 50, 3), dtype=np.uint8)
    result = pipe.analyze_image(image, roi_mask=np.ones(image.shape[:2], dtype=bool))
    assert result.kpi["oos"]["status"] == "insufficient_products"
    assert result.kpi["roi_fill_pct"] == 0.0


# ---- klasterlash -----------------------------------------------------------

def test_clustering_separates_color_groups():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 32))
    dino = np.repeat(base, 12, axis=0) + 0.05 * rng.normal(size=(12, 32))  # DINO deyarli bir xil
    cols = np.stack([color_descriptor(solid((240, 140, 20)))] * 6
                    + [color_descriptor(solid((25, 25, 30)))] * 6)
    truth = ["orange"] * 6 + ["black"] * 6
    f_dino = build_cluster_features(dino, None, None)
    f_hyb = build_cluster_features(dino, cols, None, color_weight=0.6)
    r_dino = purity_and_coverage(cluster_features(f_dino, 3), truth)
    r_hyb = purity_and_coverage(cluster_features(f_hyb, 3), truth)
    assert r_hyb["ARI"] > r_dino["ARI"]
    assert r_hyb["purity"] == 1.0


# ---- kalibrlash ------------------------------------------------------------

def test_calibration_finds_separating_threshold():
    recs = [Rec("a", "a", 0.8, 0.1)] * 50 + [Rec("unknown", "a", 0.4, 0.1)] * 50
    t, m, res = fit_thresholds(recs)
    assert 0.4 < t <= 0.8 and res["open_acc"] == 1.0
    assert open_accuracy(recs, 0.95, 0.0)["open_acc"] == 0.5
    r = report(recs, [i % 2 for i in range(100)])
    assert r["auroc"] == 1.0 and r["holdout"]["open_acc"] == 1.0


# ---- augmentatsiya / qidiruv yo'llari / konfiguratsiya ------------------------

def test_augment_variants_deterministic():
    from rasta.augment import make_variants
    img = solid((200, 30, 40), h=200, w=80)
    a = make_variants(img, 6, seed=3)
    b = make_variants(img, 6, seed=3)
    assert len(a) == 6 and np.array_equal(a[0], img)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # rang saqlanadi (augmentatsiya mahsulotni boshqa rangga aylantirmaydi)
    base = color_descriptor(img)
    assert min(color_similarity(base, color_descriptor(v)) for v in a) > 0.6


def test_numpy_search_matches_qdrant():
    rng = np.random.default_rng(1)
    cat = SkuCatalog(dim=16, path=":memory:", collection="eq")
    for k in range(6):
        v = rng.normal(size=(4, 16)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        cols = np.stack([color_descriptor(solid(tuple(int(x) for x in rng.integers(0, 255, 3))))
                         for _ in range(4)])
        cat.add_sku(f"s{k}", f"S{k}", v, colors=cols, aspects=list(rng.uniform(1, 3, 4)))
    q = rng.normal(size=(20, 16)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    qc = np.stack([color_descriptor(solid((120, 60, 30)))] * 20)
    qa = list(rng.uniform(1, 3, 20))
    p = RecognitionParams(sim_threshold=0.1, min_margin=0.0, w_color=0.25, w_shape=0.05)
    cat.is_local = True
    m_np = cat.search(q, p, qc, qa)
    cat.is_local = False
    m_qd = cat.search(q, p, qc, qa)
    for a, b in zip(m_np, m_qd):
        assert a.sku_id == b.sku_id and abs(a.score - b.score) < 1e-4


def test_cache_invalidated_on_add():
    cat = SkuCatalog(dim=4, path=":memory:", collection="inv")
    v = np.eye(4, dtype=np.float32)
    cat.add_sku("a", "A", v[:1])
    p = RecognitionParams(sim_threshold=0.5, min_margin=0.0)
    assert cat.search(v[1:2], p)[0].sku_id is None      # 'b' hali yo'q
    cat.add_sku("b", "B", v[1:2])
    assert cat.search(v[1:2], p)[0].sku_id == "b"       # kesh yangilangan


def test_config_override_types():
    from rasta.config import Config
    c = Config({"a": {"b": 1}}).override(["a.b=0.25", "a.c=[crop, stretch]", "x=true", "y=null",
                                          "p=:memory:", "u=http://localhost:6333"])
    assert c.get("p") == ":memory:" and c.get("u") == "http://localhost:6333"
    assert c.get("a.b") == 0.25 and c.get("a.c") == ["crop", "stretch"]
    assert c.get("x") is True and c.get("y") is None


def test_default_config_is_consistent():
    """config.yaml: ansambl modellari soni = weights/resize_mode/pooling uzunligi."""
    from rasta.config import Config
    cfg = Config.load()
    models = [m for m in str(cfg.get("embedder.model")).split("+") if m.strip()]
    for key in ("embedder.weights", "embedder.resize_mode", "embedder.pooling"):
        v = cfg.get(key)
        if isinstance(v, list):
            assert len(v) == len(models), key
    w = cfg.get("recognition.w_color") + cfg.get("recognition.w_shape")
    assert 0 <= w < 1


def test_threshold_depends_on_source():
    """Packshot'dan qurilgan SKU uchun pastroq chegara qo'llanadi."""
    hits_pack = [SimpleNamespace(score=0.70, payload={"sku_id": "a", "name": "A", "src": "packshot"})]
    hits_crop = [SimpleNamespace(score=0.70, payload={"sku_id": "a", "name": "A", "src": "crop"})]
    p = RecognitionParams(sim_threshold=0.76, sim_threshold_packshot=0.65, min_margin=0.0)
    assert SkuCatalog.decide(hits_pack, p).is_known
    assert not SkuCatalog.decide(hits_crop, p).is_known
    p2 = RecognitionParams(sim_threshold=0.76, min_margin=0.0)     # packshot τ berilmagan
    assert not SkuCatalog.decide(hits_pack, p2).is_known


def test_brand_level_when_sku_ambiguous():
    """0.5L va 1L egizaklar: SKU noaniq, lekin brend aniq -> level='brand'."""
    hits = [SimpleNamespace(score=0.90, payload={"sku_id": "cola_05", "name": "c5", "brand": "Cola"}),
            SimpleNamespace(score=0.895, payload={"sku_id": "cola_10", "name": "c10", "brand": "Cola"})]
    p = RecognitionParams(sim_threshold=0.76, min_margin=0.01)
    m = SkuCatalog.decide(hits, p)
    assert not m.is_known and m.sku_id is None and m.level == "brand" and m.brand == "Cola"
    hits[1].payload["brand"] = "Pepsi"           # turli brend -> butunlay noma'lum
    m2 = SkuCatalog.decide(hits, p)
    assert m2.level is None and m2.brand is None
    # KPI: brend ulushiga kiradi
    d = Detection(Box(0, 0, 50, 100)); d.match = m
    k = compute_kpi([d], [], 1)
    assert k["num_brand_only"] == 1 and k["brand_share"]["Cola"]["facings"] == 1


def test_brand_margin_only_for_same_brand():
    """Bir brend ta'mlari: margin brand_margin dan kichik -> brend; boshqa brend -> SKU."""
    hits = [SimpleNamespace(score=0.90, payload={"sku_id": "ch_tarxun", "name": "t", "brand": "Ch"}),
            SimpleNamespace(score=0.87, payload={"sku_id": "ch_dyushes", "name": "d", "brand": "Ch"})]
    p = RecognitionParams(sim_threshold=0.76, min_margin=0.01, brand_margin=0.04)
    m = SkuCatalog.decide(hits, p)
    assert m.sku_id is None and m.level == "brand" and m.brand == "Ch"
    hits[1].payload["brand"] = "Other"           # turli brend -> oddiy min_margin
    assert SkuCatalog.decide(hits, p).sku_id == "ch_tarxun"
    hits[1].payload["brand"] = "Ch"
    hits[1].score = 0.85                         # margin 0.05 >= 0.04 -> SKU
    assert SkuCatalog.decide(hits, p).sku_id == "ch_tarxun"
    p.brand_margin = None                        # o'chiq -> eski xatti-harakat
    hits[1].score = 0.87
    assert SkuCatalog.decide(hits, p).sku_id == "ch_tarxun"


# ---- ROI (poligon) ------------------------------------------------------------

def test_roi_prepare_and_map_back():
    from rasta import roi as R
    from rasta.types import AnalysisResult, Gap
    img = np.zeros((800, 600, 3), dtype=np.uint8)
    img[100:500, 200:300] = 255                                  # "muzlatgich"
    reg = R.make_region([[200, 100], [300, 100], [300, 500], [200, 500]], "M1")
    work = R.prepare(img, reg, target_long=1280, max_scale=3.0, pad_frac=0.0)
    assert work.scale == pytest.approx(3.0)                       # 400px -> 1200 (max 3x)
    assert work.mask.mean() > 0.95                                # to'rtburchak zona
    # ish rasmidagi quti -> asl koordinata
    b = Box(30, 60, 90, 300)
    res = AnalysisResult("x", (10, 10), detections=[Detection(b)],
                         gaps=[Gap(0, 0, 0, 30, 30, 30, 1)])
    R.to_original(res, work, reg, (600, 800), "x.jpg")
    assert res.detections[0].box.x1 == pytest.approx(200 + 30 / 3)
    assert res.detections[0].box.y2 == pytest.approx(100 + 300 / 3)
    assert res.gaps[0].width == pytest.approx(10)
    assert res.roi is reg and res.to_dict()["roi"]["name"] == "M1"


def test_roi_detect_image_masks_outside():
    from rasta import roi as R
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    reg = R.make_region([[100, 100], [300, 100], [200, 300]], "uchburchak")
    work = R.prepare(img, reg, target_long=200, pad_frac=0.0)
    outside = ~work.mask
    assert outside.any() and (work.detect_image[outside] == 114).all()
    assert (work.image[outside] == 200).all()                     # embedding uchun asl piksel


def test_roi_inside_fraction_and_relative_points():
    from rasta import roi as R
    reg = R.make_region([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
                        image_size=(400, 200))
    assert reg.bbox == (100.0, 50.0, 300.0, 150.0)
    m = R.polygon_mask((200, 400), reg.points)
    assert R.inside_fraction(Box(120, 60, 180, 140), m) == pytest.approx(1.0)
    assert R.inside_fraction(Box(0, 0, 50, 40), m) == 0.0
    assert R.parse_points("1,2;3,4;5,6").shape == (3, 2)
    with pytest.raises(ValueError):
        R.make_region([[0, 0], [1, 1]])


def test_rectify_roundtrip():
    """4 nuqtali zona: to'g'rilangan rasmdagi quti asl rasmdagi joyiga qaytadi."""
    import cv2
    from rasta import roi as R
    from rasta.types import AnalysisResult
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    quad = np.float32([[100, 80], [500, 140], [480, 520], [120, 560]])   # qiya to'rtburchak
    reg = R.make_region(quad[[2, 0, 3, 1]], "q")                         # tartibsiz berilgan
    assert np.allclose(R.order_corners(reg.points), quad)
    work = R.prepare(img, reg, target_long=800, rectify=True)
    assert work.Hinv is not None and work.mask.all()
    Hh, Ww = work.image.shape[:2]
    # to'g'rilangan rasmning chap-yuqori burchagidagi kichik quti -> asl rasmda quad[0] atrofida
    res = AnalysisResult("x", (1, 1), detections=[Detection(Box(0, 0, 4, 4))])
    R.to_original(res, work, reg, (800, 600), "x")
    b = res.detections[0].box
    assert abs(b.x1 - 100) < 3 and abs(b.y1 - 80) < 3
    # va butun ish rasmi -> asl quad'ning o'rab turuvchi qutisi
    res2 = AnalysisResult("x", (1, 1), detections=[Detection(Box(0, 0, Ww - 1, Hh - 1))])
    R.to_original(res2, work, reg, (800, 600), "x")
    b2 = res2.detections[0].box
    assert abs(b2.x1 - 100) < 3 and abs(b2.y2 - 560) < 3 and abs(b2.x2 - 500) < 3


def test_rectify_ignored_for_polygons():
    from rasta import roi as R
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    reg = R.make_region([[10, 10], [200, 20], [250, 150], [150, 250], [20, 200]], "5 nuqta")
    work = R.prepare(img, reg, rectify=True)
    assert work.Hinv is None                 # 5 nuqtali poligon -> oddiy kesish


def test_catalog_build_kmeans_and_merge():
    from rasta.catalog_build import cluster_stats, merge_clusters, pick_examples, spherical_kmeans
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:3]                                   # 3 ta aniq "SKU"
    X = np.concatenate([c + 0.05 * rng.normal(size=(60, 8)) for c in centers]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lab, C = spherical_kmeans(X, 9, device="cpu")            # ataylab ko'p klaster
    merged = merge_clusters(C, lab, thr=0.9)
    assert len(np.unique(merged)) == 3                        # complete-linkage birlashtirdi
    outlets = np.arange(len(X)) % 20
    st = cluster_stats(X, merged, outlets, min_size=5)
    assert len(st) == 3 and all(s.size == 60 for s in st)
    ex = pick_examples(st[0], outlets, 12)
    assert len(ex) == 12 and len({outlets[i] for i in ex}) == 12   # turli do'konlardan
