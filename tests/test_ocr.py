"""Yorliq yozuvi bo'yicha aniqlashtirish: matnni solishtirish, tahlilga qo'llash, OCR xizmati."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rasta import ocr
from rasta.types import Box, Detection, Match
from rasta_ocr.service import create_ocr_app

AYS = {
    "peach": ("Ays Tea Shaftoli 1.25L", "Ays Tea"),
    "berry": ("Ays Tea O'rmon mevalari 1.25L", "Ays Tea"),
    "berry_small": ("Ays Tea O'rmon mevalari 0.5L", "Ays Tea"),
}
MICCO = {
    "kiwi": ("Micco Kiwi", "Micco"),
    "orange": ("Micco Orange & Mango", "Micco"),
    "pome": ("Micco Pomegranate", "Micco"),
    "pear": ("Micco Pear", "Micco"),
}


def line(text, conf=0.95, x1=0.3, x2=0.7):
    return [text, conf, x1, x2]


def test_keywords_drop_brand_volume_and_helper_words():
    assert ocr.keywords("Ays Tea Shaftoli 1.25L", "Ays Tea") == ["shaftoli"]
    assert ocr.keywords("Ays Tea Oʻrmon mevalari 0,5 л", "Ays Tea") == ["o'rmon", "mevalari"]
    assert ocr.keywords("Micco Orange & Mango", "Micco") == ["orange", "mango"]
    assert ocr.keywords("Наш сад Банан-клубника 1 л", "Наш сад") == ["банан", "клубника"]


def test_volume_in_names_and_in_noisy_label_text():
    assert ocr.name_volume("Ays Tea Shaftoli 1.25L") == 1250
    assert ocr.name_volume("Cola 0,5 л") == 500
    assert ocr.name_volume("Pepsi 330 ml") == 330
    assert ocr.name_volume("Micco Kiwi") is None
    for seen in ("1,25L", "1,25t", "1,251", "1.25l"):
        assert ocr.read_volume(seen + " ") == 1250
    assert ocr.read_volume("0.S1 ") == 500
    assert ocr.read_volume("330 ml") == 330
    assert ocr.read_volume("315 kkal") is None and ocr.read_volume("1.25% ") is None


def test_choose_by_flavour_word_whole_or_cut_off():
    assert ocr.choose([line("Tea"), line("shaftoli"), line("qora choy")], AYS) == ("peach", "shaftoli")
    assert ocr.choose([line("shaftc", 0.9)], AYS) == ("peach", "shaftoli")          # burilgan shisha
    assert ocr.choose([line("RANATE")], MICCO) == ("pome", "pomegranate")
    assert ocr.choose([line("ORANGE&MANGO")], MICCO) == ("orange", "orange")
    assert ocr.choose([line("PEAR")], MICCO) == ("pear", "pear")


def test_choose_needs_readable_unambiguous_text():
    assert ocr.choose([], AYS) is None
    assert ocr.choose([line("Tea"), line("qora choy"), line("CERTIFIED")], AYS) is None
    assert ocr.choose([line("shaftoli", 0.3)], AYS) is None                         # ishonch past
    assert ocr.choose([line("shaftoli"), line("o'rmon mevalari")], AYS) is None     # ikki ta'm
    assert ocr.choose([line("shaftoli")], {"peach": AYS["peach"]}) is None          # tanlov yo'q


def test_text_touching_the_crop_edge_belongs_to_the_neighbour():
    texts = [line("ANGO", 1.0, 0.0, 0.2), line("RANATE", 1.0, 0.35, 0.9)]
    assert ocr.choose(texts, MICCO) == ("pome", "pomegranate")
    assert ocr.choose([line("ANGO", 1.0, 0.0, 0.2)], MICCO) is None


def test_label_language_may_differ_from_the_catalog_name():
    nash = {"pear": ("Наш сад Груша", "Наш сад"), "rasp": ("Наш сад Малина", "Наш сад")}
    assert ocr.choose([line("nok")], nash) == ("pear", "груша")                     # o'zbekcha yorliq
    assert ocr.choose([line("malina")], nash) == ("rasp", "малина")
    assert ocr.choose([line("Кpушa")], nash) == ("pear", "груша")                   # lotin "p" aralash
    assert ocr.choose([line("лесные ягоды")], AYS) is None                          # ikki hajm, hajm o'qilmadi


def test_same_flavour_different_size_is_settled_by_the_volume():
    assert ocr.choose([line("o'rmon mevalari"), line("0,5l")], AYS) == ("berry_small", "500 ml")
    assert ocr.choose([line("o'rmon mevalari"), line("1,25L")], AYS) == ("berry", "1.25 L")
    assert ocr.choose([line("o'rmon mevalari")], AYS) is None
    twins = {"big": ("Cola 1.5L", "Cola"), "small": ("Cola 0.5L", "Cola")}
    assert ocr.choose([line("0.5l")], twins) == ("small", "500 ml")
    assert ocr.choose([line("Cola")], twins) is None


def test_flavour_matches_but_volume_contradicts():
    assert ocr.choose([line("shaftoli"), line("0,5l")], AYS) is None


class Catalog:
    def sku_info(self):
        return {sku: {"name": name, "brand": brand} for sku, (name, brand) in {**AYS, **MICCO}.items()}


class Reader:
    def __init__(self, results):
        self.results, self.crops = results, None

    def read(self, crops):
        self.crops = crops
        return self.results


def detection(level, cands, known=None, x=100):
    top = cands[0]
    match = Match(sku_id=known, name=known and AYS[known][0], score=top[1], margin=0.01,
                  is_known=known is not None, level=level, brand="Ays Tea" if level else None,
                  candidates=[{"sku_id": s, "score": sc, "name": AYS[s][0], "brand": "Ays Tea"} for s, sc in cands])
    return Detection(box=Box(x, 100, x + 100, 400), match=match)


def test_refine_promotes_only_brand_level_and_variant_boxes():
    image = np.zeros((500, 1000, 3), np.uint8)
    brand = detection("brand", [("berry", 0.86), ("peach", 0.85)])
    variant = detection("variant", [("peach", 0.8), ("berry", 0.7)], x=250)
    known = detection("sku", [("peach", 0.95), ("berry", 0.8)], known="peach", x=400)
    unknown = detection(None, [("peach", 0.5), ("berry", 0.45)], x=550)
    reader = Reader([[line("shaftoli")], [line("qora choy")]])
    fixed = ocr.refine_with_text([brand, variant, known, unknown], image, reader, Catalog())
    assert fixed == 1 and len(reader.crops) == 2
    assert (brand.match.sku_id, brand.match.is_known, brand.match.level) == ("peach", True, "sku")
    assert brand.match.name == AYS["peach"][0] and brand.match.text == "shaftoli"
    assert brand.match.score == 0.85
    assert brand.to_dict()["label_text"] == "shaftoli"
    assert variant.match.level == "variant" and not variant.match.is_known
    assert known.match.text is None and unknown.match.sku_id is None


def test_refine_reads_from_the_original_image_only_when_needed():
    image = np.zeros((500, 1000, 3), np.uint8)
    calls = []

    def hires():
        calls.append(1)
        return np.zeros((1500, 3000, 3), np.uint8)

    reader = Reader([[line("shaftoli")]])
    assert ocr.refine_with_text([detection("sku", [("peach", 0.9), ("berry", 0.8)], known="peach")],
                                image, reader, Catalog(), hires=hires) == 0
    assert not calls and reader.crops is None
    assert ocr.refine_with_text([detection("brand", [("berry", 0.86), ("peach", 0.85)])],
                                image, reader, Catalog(), hires=hires) == 1
    assert calls == [1]
    height, width = reader.crops[0].shape[:2]       # 100x300 quti asl rasmda 300x900, 4% chet bilan
    assert 320 <= width <= 560 and 960 <= height <= 1650


def test_refine_survives_a_silent_service_and_single_candidates():
    image = np.zeros((500, 1000, 3), np.uint8)
    brand = detection("brand", [("berry", 0.86), ("peach", 0.85)])
    assert ocr.refine_with_text([brand], image, Reader(None), Catalog()) == 0
    assert brand.match.level == "brand"
    lonely = detection("brand", [("peach", 0.85)])
    reader = Reader([[line("shaftoli")]])
    assert ocr.refine_with_text([lonely], image, reader, Catalog()) == 0 and reader.crops is None


def test_label_crop_scales_small_boxes_up_and_large_ones_down():
    small = ocr.label_crop(np.zeros((600, 400, 3), np.uint8), Box(100, 100, 250, 500))
    assert max(small.shape[:2]) == pytest.approx(1296, abs=12)       # 3 baravardan oshmaydi
    large = ocr.label_crop(np.zeros((8000, 6000, 3), np.uint8), Box(1000, 1000, 2000, 5000))
    assert max(large.shape[:2]) == pytest.approx(2600, abs=4)


def test_reader_is_built_only_when_a_url_is_configured(monkeypatch):
    from rasta.config import Config
    cfg = Config.load()
    monkeypatch.delenv("RASTA_OCR_URL", raising=False)
    assert ocr.build_label_reader(cfg) is None
    monkeypatch.setenv("RASTA_OCR_URL", "http://127.0.0.1:9/")
    reader = ocr.build_label_reader(cfg)
    assert reader.url == "http://127.0.0.1:9"
    assert reader.read([np.zeros((10, 10, 3), np.uint8)]) is None   # ulanmadi — xato emas


def jpeg(width=40, height=60):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, "JPEG")
    return buf.getvalue()


def test_ocr_service_requires_the_token_and_limits_the_batch():
    class Fake:
        def read(self, pixels):
            return [["shaftoli", 0.99, 0.2, 0.8], [f"{pixels.shape[1]}x{pixels.shape[0]}", 1.0, 0.1, 0.9]]

    token = "t" * 32
    client = TestClient(create_ocr_app(token, reader=Fake()))
    assert client.get("/ready").json() == {"ready": True}
    files = [("images", ("a.jpg", jpeg(), "image/jpeg")), ("images", ("b.jpg", jpeg(30, 20), "image/jpeg"))]
    assert client.post("/read", files=files).status_code == 401
    headers = {"X-Rasta-AI-Token": token}
    results = client.post("/read", files=files, headers=headers).json()["results"]
    assert [r[1][0] for r in results] == ["40x60", "30x20"] and results[0][0][0] == "shaftoli"
    assert client.post("/read", files=files * 9, headers=headers).status_code == 400
    bad = [("images", ("a.jpg", b"not an image", "image/jpeg"))]
    assert client.post("/read", files=bad, headers=headers).status_code == 400
    with pytest.raises(ValueError):
        create_ocr_app("short", reader=Fake())
