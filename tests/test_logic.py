"""Mantiq modullari testi (og'ir ML kutubxonalarisiz — faqat numpy).

Ishga tushirish:
  python tests/test_logic.py
  (yoki: pytest tests/)
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasta.types import Box, Detection, Match
from rasta.shelf import assign_rows
from rasta.oos import detect_gaps
from rasta.kpi import compute_kpi


def _build():
    dets = []
    for x in [0, 60, 200]:            # qator 0: 120-200 orasida bo'sh joy
        dets.append(Detection(Box(x, 40, x + 50, 140, 0.9)))
    for x in [0, 60, 120, 180]:       # qator 1: to'liq
        dets.append(Detection(Box(x, 240, x + 50, 340, 0.9)))
    labels = ["cola", "cola", "water", "cola", "water", "water", "cola"]
    for d, l in zip(dets, labels):
        d.match = Match(l, l.title(), 0.8, 0.1, True)
    return dets


def test_rows():
    dets = _build()
    assert assign_rows(dets, 0.6) == 2


def test_gaps():
    dets = _build()
    nrows = assign_rows(dets, 0.6)
    gaps = detect_gaps(dets, nrows, min_gap_ratio=0.8, edge_margin=0.02)
    assert len(gaps) >= 1
    assert any(g.row == 0 for g in gaps)


def test_divider_filtered():
    """Barcha qatorlarda bir x-da takrorlanuvchi bo'shliq (bo'linma) OOS emas."""
    dets = []
    # 4 qator; har birida x=200-300 oralig'ida bir xil bo'shliq (vertikal bo'linma)
    for r, y in enumerate([40, 140, 240, 340]):
        for x in [0, 60, 120, 320, 380, 440]:  # 180-320 orasi bo'sh (bo'linma)
            dets.append(Detection(Box(x, y, x + 50, y + 90, 0.9)))
    nrows = assign_rows(dets, 0.6)
    assert nrows == 4
    gaps = detect_gaps(dets, nrows, min_gap_ratio=0.8, divider_row_frac=0.6)
    # Bo'linma barcha qatorlarda -> filtrlangan -> haqiqiy OOS yo'q
    assert len(gaps) == 0, f"bo'linma OOS deb belgilandi: {[g.to_dict() for g in gaps]}"


def test_real_gap_kept():
    """Bitta qatordagi haqiqiy bo'sh joy saqlanib qoladi (bo'linma emas)."""
    dets = []
    for r, y in enumerate([40, 140, 240, 340]):
        xs = [0, 60, 120, 180, 240, 300]
        if r == 1:
            xs = [0, 60, 300]  # qator 1 da 120-300 orasida haqiqiy bo'shliq
        for x in xs:
            dets.append(Detection(Box(x, y, x + 50, y + 90, 0.9)))
    nrows = assign_rows(dets, 0.6)
    gaps = detect_gaps(dets, nrows, min_gap_ratio=0.8, divider_row_frac=0.6)
    assert any(g.row == 1 for g in gaps), "haqiqiy bo'sh joy o'chirib yuborildi"


def _shelf(rows):
    """rows: [(y, [x, ...])] — 50x90 qutilar; qatorlar tayinlangan Detection ro'yxati."""
    dets = [Detection(Box(x, y, x + 50, y + 90, 0.9)) for y, xs in rows for x in xs]
    return dets, assign_rows(dets, 0.6)


def test_wide_gap_does_not_turn_other_gaps_into_dividers():
    """Bitta qatordagi keng bo'shliq boshqa qatorlardagi tor bo'shliqlar bilan "moslashib",
    hammasini bo'linma qilib yubormasligi kerak (real javonda 16 ta soxta bo'linma)."""
    full = [0, 60, 120, 180, 240, 300, 360, 420]
    dets, nrows = _shelf([
        (40, [0, 420]),                                  # deyarli bo'sh qator: 50..420 keng bo'shliq
        (140, [x for x in full if x != 120]),            # har xil joyda bittadan bo'sh katak
        (240, [x for x in full if x != 240]),
        (340, [x for x in full if x != 360]),
    ])
    gaps, dividers = detect_gaps(dets, nrows, return_dividers=True)
    assert dividers == []
    assert sorted(g.row for g in gaps) == [0, 1, 2, 3]


def test_taller_neighbour_row_occupies_the_gap():
    """Qo'shni qatorning baland mahsuloti shu qator balandligida turgan joy bo'sh emas."""
    dets, nrows = _shelf([(40, [0, 60, 240, 300]), (240, [0, 60, 120, 180, 240, 300])])
    assert len(detect_gaps(dets, nrows)) == 1
    dets.append(Detection(Box(120, 20, 230, 190, 0.9)))   # 120..230 ni to'ldiradigan baland quti
    nrows = assign_rows(dets, 0.6)
    assert [g for g in detect_gaps(dets, nrows) if g.row == 0] == []


def test_gap_beside_a_frame_cut_product_is_not_reported():
    """Kadr cheti kesgan mahsulot (qo'shni seksiya) bilan birinchi to'liq mahsulot orasi —
    javon ramkasi; u bo'sh joy emas va javon kengligini ham belgilamaydi."""
    rows = [(y, [100, 160, 220, 280, 340]) for y in (40, 140, 240, 340, 440)]
    dets, _ = _shelf(rows)
    dets.append(Detection(Box(0, 40, 14, 130, 0.9)))          # chapdan kesilgan mahsulot bo'lagi
    nrows = assign_rows(dets, 0.6)
    assert len(detect_gaps(dets, nrows)) == 1                 # rasm kengligi noma'lum: soxta bo'shliq
    assert detect_gaps(dets, nrows, image_width=420) == []


def test_cut_row_is_marked_only_when_much_shorter_than_its_neighbour():
    from rasta.shelf import mark_cut_rows

    stubs = [Detection(Box(x, 0, x + 50, 30, 0.9)) for x in (0, 60, 120, 180)]       # tepadan kesilgan
    full = [Detection(Box(x, 100, x + 50, 220, 0.9)) for x in (0, 60, 120, 180)]
    dets = stubs + full
    nrows = assign_rows(dets, 0.6)
    assert mark_cut_rows(dets, nrows, image_height=600) == {0}
    tall = [Detection(Box(x, 0, x + 50, 110, 0.9)) for x in (0, 60, 120, 180)]       # chetga tegadi, lekin to'liq
    dets = tall + [Detection(Box(x, 200, x + 50, 320, 0.9)) for x in (0, 60, 120, 180)]
    assert mark_cut_rows(dets, assign_rows(dets, 0.6), image_height=600) == set()


def test_size_veto_demotes_weak_matches_of_the_wrong_size():
    from rasta.context import veto_size_outliers

    def det(x, h, score):
        d = Detection(Box(x, 0, x + 40, h, 0.9))
        d.match = Match("juice", "Juice", score, 0.1, True, brand="J", level="sku")
        return d

    big = [det(x, 200, 0.75) for x in (0, 50, 100, 150)]
    weak_small, strong_small = det(300, 95, 0.67), det(350, 95, 0.80)
    assert veto_size_outliers(big + [weak_small, strong_small], sim_threshold=0.65) == 1
    assert not weak_small.match.is_known and weak_small.match.sku_id is None
    assert strong_small.match.is_known and all(d.match.is_known for d in big)
    few = [det(0, 200, 0.7), det(50, 200, 0.7), det(100, 95, 0.66)]
    assert veto_size_outliers(few, sim_threshold=0.65) == 0          # ko'pchilik yo'q — taxmin qilinmaydi


def test_kpi():
    dets = _build()
    nrows = assign_rows(dets, 0.6)
    gaps = detect_gaps(dets, nrows)
    kpi = compute_kpi(dets, gaps, nrows,
                      brand_of={"cola": "ColaCo", "water": "Aqua"})
    assert kpi["num_products"] == 7
    assert kpi["facings"]["cola"] == 4
    assert kpi["facings"]["water"] == 3
    assert kpi["distinct_skus"] == 2
    assert kpi["oos"]["num_gaps"] >= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[✓] {name}")
    print("Barcha mantiq testlari o'tdi.")
