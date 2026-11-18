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
