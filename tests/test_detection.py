"""Detektor kuchaytirishlari (WBF, bo'laklash) va detektor baholash — og'ir modellarsiz."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasta.det_eval import average_precision, evaluate, match_image
from rasta.detector import cut_by_tile, nms_array, tile_windows, wbf


def test_wbf_averages_and_votes():
    a = np.array([[10, 10, 50, 90, 0.9]], dtype=float)
    b = np.array([[14, 10, 54, 90, 0.9]], dtype=float)
    lone = np.array([[200, 10, 240, 90, 0.8]], dtype=float)
    out = wbf([a, np.concatenate([b, lone])], iou_thr=0.5)
    out = out[np.argsort(out[:, 0])]
    assert len(out) == 2
    assert np.allclose(out[0, :4], [12, 10, 52, 90])     # ikkala o'tish o'rtachasi
    assert np.isclose(out[0, 4], 0.9)                    # 2/2 ovoz
    assert np.isclose(out[1, 4], 0.4)                    # 1/2 ovoz -> ball yarmi
    assert len(wbf([a, np.concatenate([b, lone])], 0.5, min_votes=2)) == 1


def test_wbf_empty():
    assert wbf([np.zeros((0, 5)), np.zeros((0, 5))]).shape == (0, 5)


def test_tile_windows_cover_image_with_overlap():
    w, h = 1000, 800
    wins = tile_windows(w, h, 2, 0.25)
    assert len(wins) == 4
    assert min(x1 for x1, _, _, _ in wins) == 0 and max(x2 for _, _, x2, _ in wins) == w
    assert max(y2 for _, _, _, y2 in wins) == h
    (ax1, _, ax2, _), (bx1, _, bx2, _) = wins[0], wins[1]
    assert ax2 > bx1                                     # qo'shni bo'laklar qoplanadi


def test_cut_by_tile_keeps_real_image_edges():
    p = np.array([[0, 10, 30, 50, 1], [80, 10, 100, 50, 1], [40, 10, 60, 50, 1]], dtype=float)
    # chap — rasmning haqiqiy cheti, o'ng — ichki chegara
    m = cut_by_tile(p, 100, 100, left=False, top=False, right=True, bottom=False)
    assert m.tolist() == [False, True, False]


def test_nms_array():
    a = np.array([[0, 0, 10, 10, 0.9], [1, 0, 11, 10, 0.8], [50, 0, 60, 10, 0.7]], dtype=float)
    assert len(nms_array(a, 0.5)) == 2


def test_match_and_ap():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
    pred = np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 10, 0.8], [21, 0, 31, 10, 0.7],
                     [100, 0, 110, 10, 0.6]], dtype=float)
    ign = np.array([[95, 0, 115, 10]], dtype=float)
    s, f = match_image(pred, gt, ign)
    assert f.tolist() == [1, 0, 1, -1]                   # dublikat = FP, ignore zonasi = -1
    ap = average_precision(s, f, len(gt))
    assert 0.8 < ap <= 1.0
    r = evaluate([(pred, gt, ign)])
    assert (r.tp, r.fp, r.gt) == (2, 1, 2)
    assert np.isclose(r.recall, 1.0) and r.count_mae == 1


def test_drop_group_boxes_removes_group_keeps_parts():
    from rasta.detector import drop_group_boxes
    # guruh: ikki banka ustma-ust (ular qutini to'liq qoplaydi) -> guruh olib tashlanadi
    group = np.array([[0, 0, 40, 200, 0.5], [0, 0, 40, 100, 0.9], [0, 100, 40, 200, 0.9]], dtype=float)
    out = drop_group_boxes(group)
    assert len(out) == 2 and not np.any((out[:, 1] == 0) & (out[:, 3] == 200))
    # shisha + kichik yorliq + qopqoq (qoplash ~30%) -> shisha qoladi
    bottle = np.array([[0, 0, 40, 200, 0.9], [5, 0, 35, 20, 0.5], [0, 90, 40, 130, 0.5]], dtype=float)
    assert len(drop_group_boxes(bottle)) == 3


def test_app_logo_filter():
    from rasta.detector import is_app_logo
    from rasta.types import Box
    img = np.full((800, 600, 3), 120, dtype=np.uint8)
    img[20:90, 15:85] = (40, 200, 110)                      # yashil kvadrat logotip
    img[300:500, 200:260] = (40, 200, 110)                  # yashil shisha (markazda, baland)
    assert is_app_logo(img, Box(15, 20, 85, 90))
    assert not is_app_logo(img, Box(200, 300, 260, 500))    # burchakda emas
    img[10:120, 100:120] = (40, 200, 110)
    assert not is_app_logo(img, Box(100, 10, 120, 120))     # burchakda, lekin baland — mahsulot
    assert not is_app_logo(img, Box(15, 100, 85, 120))      # rangi yashil emas


def test_merge_split_boxes_joins_carton_halves_but_not_overlapping_products():
    from rasta.detector import merge_split_boxes
    from rasta.types import Box

    whole = [Box(x, 100, x + 90, 450, .5) for x in (0, 100, 200, 500, 600)]
    # bitta karton ikki bo'lakda: tepa (100..360) va past (190..450) — birga to'liq balandlik
    halves = [Box(300, 100, 395, 360, .4), Box(302, 190, 392, 450, .6)]
    merged = merge_split_boxes(whole + halves)
    assert len(merged) == len(whole) + 1
    joined = max(merged, key=lambda b: b.score if 290 < b.x1 < 310 else -1)
    assert (round(joined.y1), round(joined.y2), joined.score) == (100, 450, .6)

    # orqa qatordagi mahsulot oldingisi bilan ustma-ust: har biri TO'LIQ balandlikda — tegilmaydi
    behind = [Box(300, 100, 395, 450, .5), Box(302, 20, 392, 370, .5)]
    assert len(merge_split_boxes(whole + behind)) == len(whole) + 2
    # o'lchov uchun toza qo'shni yo'q — taxmin qilinmaydi
    assert len(merge_split_boxes(halves)) == 2


def test_pick_product_box_ignores_background_touching_the_frame():
    from rasta.augment import pick_product_box
    from rasta.types import Box

    table = Box(0, 520, 600, 800, .55)            # tagidagi quti: uch chetga tegadi, yirik
    product = Box(210, 120, 370, 660, .45)
    poster = Box(50, 300, 160, 420, .4)
    x1, y1, x2, y2 = pick_product_box([table, product, poster], 600, 800)
    assert x1 <= 210 and y1 <= 120 and x2 >= 370 and y2 >= 660 and x2 - x1 < 200
    assert pick_product_box([table], 600, 800) is None


def test_drop_group_boxes_keeps_whole_product_over_its_fragments():
    from rasta.detector import drop_group_boxes

    # baland karton: kichik masshtab to'liq topgan (ishonchli), katta masshtab ikki bo'lakka bo'lgan
    whole = [100, 100, 200, 450, .45]
    fragments = [[102, 104, 198, 290, .15], [103, 270, 199, 446, .18]]
    kept = drop_group_boxes(np.array([whole] + fragments, dtype=np.float32))
    assert any(np.allclose(row, whole) for row in kept)
    # ikki banka ustidan chizilgan guruh qutisi: bankalar ishonchliroq — guruh olib tashlanadi
    group = [100, 100, 200, 450, .25]
    cans = [[102, 104, 198, 272, .8], [103, 278, 199, 446, .7]]
    kept = drop_group_boxes(np.array([group] + cans, dtype=np.float32))
    assert len(kept) == 2 and not any(np.allclose(row, group) for row in kept)
