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
from rasta.cluster import (build_cluster_features, cluster_features, cluster_unknown_features,
                           purity_and_coverage, recover_noise, split_label_colors, split_mixed_labels)
from rasta.config import Config
from rasta.context import recover_cluster_brands, recover_occluded_family, smooth_with_context
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


def test_variant_when_only_the_colour_differs():
    """Bir xil dizayndagi boshqa ta'm: o'xshashlik yetarli, rang "aynan shu" darajasida emas."""
    from rasta.kpi import compute_kpi
    from rasta.types import Box, Detection

    green = color_descriptor(solid((90, 200, 60)))
    lime = color_descriptor(solid((150, 200, 60)))
    hits = [_hit("kiwi", 0.80, green)]
    hits[0].payload["brand"] = "micco"
    base = RecognitionParams(sim_threshold=0.5, min_margin=0.0)
    c = SkuCatalog.decide(hits, base, lime).candidates[0]["color"]
    assert 0.05 < c < 0.99
    strict = RecognitionParams(sim_threshold=0.5, min_margin=0.0, color_min=c - 0.05, color_same=c + 0.01)
    m = SkuCatalog.decide(hits, strict, lime)
    assert not m.is_known and m.sku_id is None and m.level == "variant" and m.brand == "micco"
    assert m.candidates[0]["sku_id"] == "kiwi"                      # nimaga o'xshashi saqlanadi
    loose = RecognitionParams(sim_threshold=0.5, min_margin=0.0, color_min=c - 0.05, color_same=c - 0.01)
    assert SkuCatalog.decide(hits, loose, lime).level == "sku"
    assert SkuCatalog.decide(hits, strict, green).level == "sku"    # rangi mos — oddiy tanish
    veto = RecognitionParams(sim_threshold=0.5, min_margin=0.0, color_min=c + 0.01, color_same=c + 0.02)
    assert SkuCatalog.decide(hits, veto, lime).level is None        # rang butunlay boshqa — noma'lum
    assert SkuCatalog.decide(hits, strict).level == "sku"           # rang o'lchanmagan — qoida ishlamaydi
    # KPI: SKU sifatida sanalmaydi, brend ulushiga kiradi
    d = Detection(Box(0, 0, 50, 100)); d.match = m
    kpi = compute_kpi([d], [], 1)
    assert kpi["num_known"] == 0 and kpi["num_variant"] == 1 and kpi["num_variant_no_brand"] == 0
    assert kpi["brand_share"]["micco"]["facings"] == 1


def test_single_generic_entry_counts_a_variant_and_scope_cannot_hide_other_skus():
    cat = SkuCatalog(dim=4, path=":memory:", collection="family")
    vector = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    green = color_descriptor(solid((90, 200, 60)))
    lime = color_descriptor(solid((150, 200, 60)))
    color_similarity = float(green @ lime)
    params = RecognitionParams(sim_threshold=0.5, min_margin=0.0,
                               color_min=color_similarity - 0.05, color_same=color_similarity + 0.01)
    cat.add_sku("nash", "Nash Sad", vector, extra={"brand": "nash sad"}, colors=green[None, :])
    [match] = cat.search(vector, params, lime[None, :])
    assert match.is_known and match.family and match.sku_id == "nash"
    unrelated = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert not cat.search(unrelated, params, lime[None, :])[0].is_known

    cat.add_sku("pear", "Nash Sad Nok", vector, extra={"brand": "Nash Sad"}, colors=lime[None, :])
    [scoped] = cat.search(vector, params, lime[None, :], allowed={"nash"})
    assert not scoped.is_known and not scoped.family and scoped.level == "brand"
    assert scoped.candidates[0]["brand_reference"]


def test_brand_reference_groups_back_label_and_exact_flavors_use_separate_skus():
    from rasta.kpi import compute_kpi

    cat = SkuCatalog(dim=3, path=":memory:", collection="brand-first")
    vectors = np.eye(3, dtype=np.float32)
    cat.add_sku("micco", "Miccoco", vectors[0], extra={"brand": "Miccoco"})
    cat.add_sku("pear", "Miccoco Nok", vectors[1], extra={"brand": "miccoco"})
    cat.add_sku("mango", "Miccoco Mango", vectors[2], extra={"brand": "Miccoco"})
    params = RecognitionParams(sim_threshold=0.6, brand_margin=0.04, brand_rival_margin=0.03)
    for local in (True, False):
        cat.is_local = local
        matches = cat.search(vectors, params)
        assert [(m.level, m.sku_id) for m in matches] == [("brand", None), ("sku", "pear"), ("sku", "mango")]
        dets = [Detection(Box(i, 0, i + 1, 1), match=m) for i, m in enumerate(matches)]
        brand = next(iter(compute_kpi(dets, [], 1)["brand_share"].values()))
        assert brand["facings"] == 3 and brand["unresolved_facings"] == 1
        assert set(brand["sku_ids"]) == {"pear", "mango"}

    cat.add_sku("rival", "Other juice", vectors[0], extra={"brand": "Other"})
    assert cat.search(vectors[0], params)[0].level is None


def test_embedding_floor_stops_colour_carrying_a_weak_match():
    """Rang bonusi ko'rinishi mos kelmagan mahsulotni chegaradan o'tkazmasligi kerak."""
    yellow = color_descriptor(solid((230, 200, 40)))
    hits = [_hit("dena", 0.58, yellow)]
    hits[0].payload.update(src="packshot", brand="DENA")
    p = RecognitionParams(sim_threshold=0.76, sim_threshold_packshot=0.65, min_margin=0.0, w_color=0.25)
    m = SkuCatalog.decide(hits, p, yellow)
    assert m.is_known and m.score >= 0.65                           # 0.75*0.58 + 0.25*1.0
    p.embed_min_packshot = 0.60
    m = SkuCatalog.decide(hits, p, yellow)
    assert not m.is_known and m.level is None and m.score >= 0.65
    hits[0].score = 0.62
    assert SkuCatalog.decide(hits, p, yellow).is_known
    hits[0].payload["src"] = "crop"                                 # crop SKU'ga packshot chegarasi tegmaydi
    hits[0].score = 0.58
    p2 = RecognitionParams(sim_threshold=0.6, min_margin=0.0, w_color=0.25, embed_min_packshot=0.60)
    assert SkuCatalog.decide(hits, p2, yellow).is_known


def test_catalog_reuses_loaded_points_and_refreshes_metadata_after_writes(monkeypatch):
    catalog = SkuCatalog(dim=2, path=":memory:")
    vector = np.array([1, 0], dtype=np.float32)
    catalog.add_sku("juice", "Dena Moxito", vector, extra={"brand": "DENA", "words": ["moxito"]})
    expected = catalog.sku_info()
    assert catalog.search(vector)[0].sku_id == "juice"

    def unexpected_read(*args, **kwargs):
        pytest.fail("Warm catalog must not fetch the same points again")

    with monkeypatch.context() as patch:
        patch.setattr(catalog.client, "scroll", unexpected_read)
        patch.setattr(catalog.client, "count", unexpected_read)
        assert catalog.count() == 1
        actual = catalog.sku_info()
        assert actual == expected
        actual["juice"]["words"].append("external change")
        assert catalog.sku_info() == expected

    catalog.update_sku("juice", name="Dena Mojito", words=["mojito"])
    assert catalog.sku_info()["juice"]["words"] == ["mojito"]
    assert catalog.search(vector)[0].name == "Dena Mojito"
    catalog.delete_sku("juice")
    assert catalog.count() == 0 and catalog.sku_info() == {}


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


def test_occluded_family_requires_a_close_overlap_and_respects_negative_evidence():
    ref = Detection(Box(0, 0, 100, 250), row=2, vec=np.array([1.0, 0.0]))
    ref.match = _m("nash", True, [("nash", 0.84)])
    ref.match.family = True
    ref.match.score = 0.84
    partial = Detection(Box(80, 25, 150, 245), row=2, vec=np.array([0.8, 0.6]))
    partial.match = _m(None, False, [("nash", 0.74)])
    partial.match.score = 0.74
    other = Detection(Box(150, 25, 220, 245), row=2, vec=partial.vec)
    other.match = _m(None, False, [("nash", 0.74)])
    other.match.score = 0.74
    params = RecognitionParams(sim_threshold=0.76, neg_margin=0.03)
    colors = np.ones((3, 1))
    partial.match.negative_score = 0.80
    assert recover_occluded_family([ref, partial, other], colors, params) == 0
    partial.match.negative_score = 0.75
    assert recover_occluded_family([ref, partial, other], colors, params) == 1
    assert partial.match.sku_id == "nash" and partial.context_fixed and not other.match.is_known


def test_cluster_can_confirm_brand_without_guessing_sku_or_overriding_a_rival(monkeypatch):
    dets = [Detection(Box(i * 30, 0, i * 30 + 25, 100), row=0) for i in range(5)]
    for det in dets[:3]:
        det.match = Match("bliss-apple", "Bliss Olma", 0.88, 0.15, True, brand="Bliss", level="sku")
    for score, det in zip((0.62, 0.72), dets[3:]):
        det.match = Match(None, None, 0.74, 0.02, False, [
            {"sku_id": "bliss-apple", "brand": "Bliss", "score": 0.74, "color": 0.9},
            {"sku_id": "dena-apple", "brand": "DENA", "score": score, "color": 0.9},
        ])
    monkeypatch.setattr("rasta.cluster.cluster_unknown_features", lambda *args: np.zeros(5, dtype=int))
    vectors = colors = np.ones((5, 1))
    params = RecognitionParams(sim_threshold=0.76, neg_margin=0.03)
    assert recover_cluster_brands(dets, vectors, colors, [4.0] * 5, Config({}), params) == 1
    assert (dets[3].match.brand, dets[3].match.level, dets[3].match.sku_id) == ("Bliss", "brand", None)
    assert not dets[3].match.is_known and dets[4].match.level is None


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


def test_two_close_unknown_products_form_a_group():
    vectors = np.array([[1.0, 0.0], [0.98, 0.2]])
    cfg = Config({})
    labels = cluster_unknown_features(vectors, None, None, cfg)
    assert labels[0] == labels[1] >= 0

    cfg.set("clustering.recovery_max_distance", None)
    assert cluster_unknown_features(vectors, None, None, cfg).tolist() == [-1, -1]


def test_variants_inside_one_group_are_separated():
    from rasta.cluster import split_variants
    from rasta.features import BINS

    def carton(fruit):                 # bir xil dizayn: tepasi oq, pasti yashil; o'rtadagi meva rangi farq qiladi
        image = np.zeros((90, 30, 3), dtype=np.uint8)
        image[:30], image[30:60], image[60:] = (245, 245, 245), fruit, (20, 120, 40)
        return color_descriptor(image)

    colors = np.stack([carton((240, 140, 20))] * 3 + [carton((200, 20, 30))] * 2 + [carton((90, 30, 150))])
    labels = np.zeros(6, dtype=int)                                  # dastlabki guruh: hammasi bitta brend
    vectors = np.tile([1.0, 0.0], (6, 1))
    split = split_variants(labels, vectors, colors, min_size=2)
    assert split[:3].tolist() == [split[0]] * 3 and split[3] == split[4] >= 0 and split[0] != split[3]
    assert split[5] == -1                                             # yakka ta'm guruh emas — alohida nomlanadi
    assert sorted(set(split[split >= 0].tolist())) == [0, 1]         # bo'laklar ketma-ket raqamlanadi

    # Rangi bir xil, lekin rasm bo'yicha markazlangan embeddingi qarama-qarshi qutilar ham ajraladi
    same = np.stack([carton((240, 140, 20))] * 4)
    vectors = np.array([[1.0, 0.2], [1.0, 0.25], [1.0, -0.2], [1.0, -0.25]])
    assert len(set(split_variants(np.zeros(4, dtype=int), vectors, same, 2).tolist())) == 1      # markazsiz: faqat rang
    by_vector = split_variants(np.zeros(4, dtype=int), vectors, same, 2, center=vectors.mean(axis=0))
    assert by_vector[0] == by_vector[1] != by_vector[2] == by_vector[3]
    assert split_variants(np.array([-1, -1]), vectors[:2], same[:2], 2).tolist() == [-1, -1]
    assert colors.shape[1] == 4 * BINS


def test_group_members_with_conflicting_flavour_words_are_separated():
    from rasta.flavour import split_by_words

    red, pale = color_descriptor(solid((200, 20, 30))), color_descriptor(solid((230, 200, 190)))
    colors = np.stack([red, red, red, pale, pale, red])
    labels = np.array([0, 0, 0, 0, 0, -1])
    apple, tomato, none = frozenset({"olma"}), frozenset({"pomidor"}), frozenset()
    # rangi bir xil qizil ta'mlar: yozuvi zid -> ikki guruh; so'zi o'qilmagan quti rangi yaqin bo'lakka qo'shiladi
    split = split_by_words(labels, [apple, apple, none, tomato, none, none], colors)
    assert split[0] == split[1] == split[2] != split[3] == split[4] and split[5] == -1
    # zid so'z yo'q (bitta ta'm yoki umuman o'qilmagan) -> guruh o'zgarmaydi
    assert split_by_words(labels, [apple, none, none, none, apple, none], colors).tolist() == labels.tolist()
    assert split_by_words(labels, [none] * 6, colors).tolist() == labels.tolist()
    # yakka qolgan ta'm guruh bo'lmaydi
    alone = split_by_words(np.array([0, 0, 0]), [apple, apple, tomato], colors[:3])
    assert alone[0] == alone[1] >= 0 and alone[2] == -1


def test_noise_recovery_does_not_merge_a_chain_or_extend_through_new_members():
    angles = np.deg2rad([0, 2, 30, 60, 90, 120, 180])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    initial = np.array([0, 0, -1, -1, -1, -1, -1])
    labels = recover_noise(vectors, initial, min_size=2, max_distance=0.20)

    assert labels[:3].tolist() == [0, 0, 0]
    assert (labels[3:6] == -1).sum() == 1
    assert (labels[3:6] > 0).sum() == 2
    assert labels[-1] == -1
    assert initial.tolist() == [0, 0, -1, -1, -1, -1, -1]


def test_tall_packages_split_when_lower_label_colors_form_two_groups():
    crops = []
    for rgb in [(220, 40, 115), (220, 110, 28)] * 2:
        crop = solid((30, 145, 50))
        crop[80:] = rgb
        crops.append(crop)
    colors = np.stack([color_descriptor(crop) for crop in crops])
    labels = np.zeros(4, dtype=int)
    refined = split_label_colors(labels, colors, np.full(4, 2.4), min_size=2)
    assert refined[0] == refined[2] != refined[1] == refined[3]
    assert np.array_equal(split_label_colors(labels, colors, np.full(4, 1.5), 2), labels)


def test_background_filter_excludes_small_reflections_between_tall_products():
    tall = [Detection(Box(x, 10, x + 40, 150, 0.9), row=1) for x in (10, 160)]
    reflections = [Detection(Box(x, y, x + 12, y + 20, 0.2), row=0)
                   for x, y in ((70, 50), (85, 50), (100, 80), (115, 80))]
    assert mark_background_rows(reflections + tall, 2) == {0}
    assert mark_background_rows(reflections + tall[:1], 2) == set()
    reflections[0].box.score = 0.8
    assert mark_background_rows(reflections + tall, 2) == set()


def test_clear_middle_label_conflict_keeps_repeated_variant_and_leaves_singletons_unassigned():
    from rasta.features import BINS

    colors = np.zeros((4, COLOR_DIM))
    colors[np.arange(4), 2 * BINS + np.array([0, 0, 1, 2])] = 1.0
    vectors = np.tile([1.0, 0.0], (4, 1))
    labels = np.zeros(4, dtype=int)
    assert split_mixed_labels(labels, vectors, colors, np.full(4, 3.0), 2).tolist() == [0, 0, -1, -1]


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


def test_search_can_be_limited_to_the_products_on_the_shelf():
    rng = np.random.default_rng(3)
    cat = SkuCatalog(dim=16, path=":memory:", collection="scope")
    base = rng.normal(size=16); base /= np.linalg.norm(base)
    for k, name in enumerate(("apple", "berry", "orange")):
        v = base + rng.normal(size=(3, 16)) * 0.05             # uchala ta'm bir-biriga juda o'xshash
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        cat.add_sku(name, name.title(), v)
    q = (base + rng.normal(size=(5, 16)) * 0.05)[:2]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = RecognitionParams(sim_threshold=0.1, min_margin=0.02, w_color=0.0, w_shape=0.0)
    for local in (True, False):                                # numpy va Qdrant yo'llari bir xil
        cat.is_local = local
        whole = cat.search(q, p)
        assert all(not m.is_known for m in whole)              # uchovi teng — noaniq
        only = cat.search(q, p, allowed={"berry"})
        assert all(m.sku_id == "berry" and m.is_known for m in only)      # raqobatchi yo'q — aniq
        assert all({c["sku_id"] for c in m.candidates} == {"berry"} for m in only)
        two = cat.search(q, p, allowed={"apple", "orange"})
        assert all(m.sku_id != "berry" and "berry" not in {c["sku_id"] for c in m.candidates} for m in two)
        assert all(m.sku_id is None and not m.is_known for m in cat.search(q, p, allowed=set()))
        assert all(not m.is_known for m in cat.search(q, p, allowed={"nothing-like-it"}))


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


def test_flavours_of_one_brand_are_grouped_by_colour_and_label_word():
    from rasta.flavour import assign_flavours, flavour_key, group_flavours, flavour_name
    from rasta.types import Detection, Match

    # ta'm so'zi: kirill so'zi lotin harflariga o'xshab o'qilsa ham («AHAHAC» = ананас), chetga tegsa — tashlanadi
    line = lambda text, x1=0.2, x2=0.8, size=0.08, conf=0.95: [text, conf, x1, x2, size]  # noqa: E731
    assert flavour_key([line("AHAHAC"), line("TOMAT")]) == frozenset({"ananas", "pomidor"})
    assert flavour_key([line("TOMAT", x1=0.0)]) == frozenset()
    assert flavour_key([line("TOMAT", conf=0.5), line("Bliss")]) == frozenset()
    assert flavour_name([frozenset({"ananas"}), frozenset({"ananas"}), frozenset({"anor"}), frozenset()]) == "ananas"

    red, yellow = (220, 40, 60), (240, 210, 40)
    colors = np.stack([color_descriptor(solid(c)) for c in (red, red, red, yellow, yellow)])
    # bir xil rangdagi ikki ta'm so'zi bilan ajraladi, so'zsiz quti rang bo'yicha mos guruhga qo'shiladi
    keys = [frozenset({"anor"}), frozenset({"pomidor"}), frozenset(), frozenset(), frozenset({"ananas"})]
    labels = group_flavours(colors, keys)
    assert labels[0] != labels[1] and labels[3] == labels[4]
    assert labels[2] in (labels[0], labels[1]) and labels[0] not in (labels[3], labels[4])
    assert group_flavours(colors, [frozenset()] * 5) == group_flavours(colors, [frozenset()] * 5)
    from rasta.flavour import colour_groups
    assert np.isfinite(colour_groups(np.stack([colors[0], colors[0]]))).all()     # bir xil ikki quti: NaN bo'lmaydi

    def det(i, fam, color):
        match = Match("dena" if fam else None, "DENA" if fam else None, 0.8, 0.5, fam, brand="DENA",
                      level="sku" if fam else "brand", family=fam)
        return Detection(box=Box(i * 10, 0, i * 10 + 8, 30, 1.0), match=match)

    boxes = [det(i, i < 3, None) for i in range(5)]
    other = Detection(box=Box(100, 0, 108, 30, 1.0), match=Match("x", "X", 0.9, 0.5, True, brand="Other", level="sku"))
    found = assign_flavours(boxes + [other], np.concatenate([colors, colors[:1]]))
    shown = [b.flavour for b in boxes]
    assert found == 2 and shown[0] == shown[1] == shown[2] != shown[3] == shown[4] >= 0   # rang bo'yicha ikki guruh (OCR yo'q)
    assert other.flavour == -1                                     # bitta quti — brend ichida ajratadigan narsa yo'q
    assert assign_flavours(boxes[:1], colors[:1]) == 0
