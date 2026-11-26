"""Yorliqdagi yozuv bo'yicha aniqlashtirish (OCR).

Vektor qidiruv brendni topib, lekin ikki ta'm/hajm orasida ikkilanganda ("brend darajasida")
yoki "o'xshaydi, rangi farq qiladi" (variant) deganda, qutidagi yozuv o'qiladi va katalog
NOMIdagi so'zlar bilan solishtiriladi: "shaftoli", "ORANGE & MANGO", "0,5L".

Qoida ataylab ehtiyotkor: noma'lum quti yozuvning o'zi bilan tanilmaydi, ikki mahsulotga
birdek mos kelgan yozuv e'tiborga olinmaydi. O'lchovlar: BENCHMARK.md 11f.

Ikkinchi vazifa — tekshirish: tanilgan qutining yozuvi shu mahsulotning katalogdagi namuna
rasmida o'qilgan yozuv bilan solishtiriladi. Qutida yirik va aniq boshqa so'z turgan,
mahsulotning o'z so'zlaridan esa birortasi ham o'qilmagan bo'lsa («Florida» qadog'i
«Cookies» deb tanilgan), moslik bekor qilinadi. Nom bilan emas, namunadagi yozuv bilan
solishtiriladi: katalog nomi qadoqdagi yozuvdan farq qilishi mumkin, OCR'ning bir xil
xatosi esa ikkala tomonda takrorlanadi. O'lchovlar: BENCHMARK.md 11g.

Matnni alohida xizmat o'qiydi (ocr/ — PaddleOCR o'z konteynerida); manzili `RASTA_OCR_URL`
yoki `ocr.url`. Manzil berilmagan bo'lsa yoki xizmat javob bermasa, tahlil OCR'siz davom etadi.
"""
from __future__ import annotations

import io
import os
import re
import sys
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

# ---- Matnni solishtirish -----------------------------------------------------

_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                ["a", "b", "v", "g", "d", "e", "e", "j", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                 "t", "u", "f", "x", "ts", "ch", "sh", "sh", "", "i", "", "e", "yu", "ya"]))
# Ko'rinishi bir xil harflar: OCR kirillcha so'zni ko'pincha lotin harflari bilan qaytaradi
_LOOK = dict(zip("авекмнорстух", "abekmhopctyx"))
_APOS = str.maketrans({"ʻ": "'", "ʼ": "'", "’": "'", "‘": "'", "`": "'"})


def _translit(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", "".join(_CYR.get(ch, ch) for ch in s.lower())).strip()


def _skeleton(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", " ", "".join(_LOOK.get(ch, ch) for ch in s.lower())).strip()


# Bir ma'noli so'zlar (o'zbekcha / ruscha / inglizcha): yorliq tili katalog nomidan farq qilishi mumkin
_SYNONYMS = [
    ("shaftoli", "персик", "peach"), ("olma", "яблоко", "apple"), ("nok", "груша", "pear"),
    ("malina", "малина", "raspberry"), ("qulupnay", "клубника", "strawberry"),
    ("banan", "банан", "banana"), ("kivi", "kiwi", "киви"), ("apelsin", "апельсин", "orange"),
    ("mango", "манго"), ("anor", "гранат", "pomegranate"), ("limon", "лимон", "lemon"),
    ("laym", "лайм", "lime"), ("olcha", "вишня", "cherry"), ("gilos", "черешня"),
    ("uzum", "виноград", "grape"), ("o'rik", "абрикос", "apricot"), ("ananas", "ананас", "pineapple"),
    ("tarvuz", "арбуз", "watermelon"), ("qovun", "дыня", "melon"), ("greypfrut", "грейпфрут", "grapefruit"),
    ("mandarin", "мандарин", "tangerine"), ("smorodina", "смородина", "currant"),
    ("pomidor", "томат", "tomato"), ("sabzi", "морковь", "carrot"), ("yalpiz", "мята", "mint"),
    ("shokolad", "шоколад", "chocolate"), ("vanil", "ваниль", "vanilla"), ("karamel", "карамель", "caramel"),
    ("yong'oq", "орех", "nut"), ("asal", "мёд", "мед", "honey"), ("sut", "молоко", "milk"),
    ("o'rmon", "лесные", "лесная", "forest"), ("mevalari", "ягоды", "berries"),
    ("multifrukt", "мультифрукт", "multifruit"), ("tropik", "тропик", "tropic", "tropical"),
    ("gazsiz", "негазированная", "still"), ("gazli", "газированная", "sparkling"),
]
_SYN = {w: group for group in _SYNONYMS for w in group}
_STOP = {"va", "and", "bilan", "the", "new", "pet", "ta'mi", "ta'mli", "vkus", "вкус", "со", "вкусом", "dona", "шт"}
_UNIT = re.compile(r"(?<![0-9a-zа-я])(\d+(?:[.,]\d+)?)\s*(ml|мл|l|л|g|gr|г|гр|kg|кг)(?![a-zа-я])")


def name_volume(name: str) -> Optional[int]:
    """Katalog nomidagi hajm, millilitrda: "1.25L" -> 1250, "0,5 л" -> 500, "330 ml" -> 330."""
    for value, unit in _UNIT.findall(name.lower()):
        if unit in ("l", "л"):
            return int(round(float(value.replace(",", ".")) * 1000))
        if unit in ("ml", "мл"):
            return int(round(float(value.replace(",", "."))))
    return None


def read_volume(text: str) -> Optional[int]:
    """Yorliqdan o'qilgan hajm (ml). OCR chalkashliklariga chidamli: "1,25t", "1,251", "0.S1"."""
    s = text.lower().replace("о", "0")
    m = re.search(r"(?<![0-9])([0-9])\s?[.,]\s?([0-9s]{1,2})\s?[lл1t:|]", s)
    if m:
        return int(round(float(f"{m.group(1)}.{m.group(2).replace('s', '5')}") * 1000))
    m = re.search(r"(?<![0-9])([0-9]{3,4})\s?(?:ml|мл)", s)
    return int(m.group(1)) if m else None


def keywords(name: str, brand: Optional[str] = None) -> list[str]:
    """Katalog nomidagi ajratuvchi so'zlar: brend, hajm va yordamchi so'zlarsiz."""
    text = _UNIT.sub(" ", name.lower().translate(_APOS))
    skip = set(re.findall(r"[a-zа-яё']+", (brand or "").lower().translate(_APOS))) | _STOP
    out = []
    for w in re.findall(r"[a-zа-яё']+", text):
        w = w.strip("'")
        if len(w) >= 3 and w not in skip and w not in out:
            out.append(w)
    return out


def _hit(word: str, tokens: list[str]) -> bool:
    """So'z o'qilgan matnda bormi: to'liq (taxminiy) yoki kesilgan ("shaft…", "…ranate")."""
    for norm in (_translit, _skeleton):
        k = norm(word).replace(" ", "")
        if len(k) < 3:
            continue
        for raw in tokens:
            for t in norm(raw).split():
                if len(t) < 3:
                    continue
                if abs(len(t) - len(k)) <= 1 and SequenceMatcher(None, k, t).ratio() >= 0.75:
                    return True
                if len(t) >= 5 and len(k) > len(t) and (
                        (k.startswith(t[:5]) and SequenceMatcher(None, k[:len(t)], t).ratio() >= 0.8)
                        or k.endswith(t)):
                    return True
    return False


def choose(texts, options: dict[str, tuple[str, Optional[str]]], *, min_conf: float = 0.5,
           edge: float = 0.03) -> Optional[tuple[str, str]]:
    """O'qilgan yozuv bo'yicha nomzodlardan bittasini tanlash.

    texts: [(matn, ishonch, x1, x2)] — x1, x2 matnning chap/o'ng cheti, quti eniga nisbatan;
    options: {sku_id: (nom, brend)} — bitta brend ichidagi nomzodlar.
    -> (sku_id, asos bo'lgan so'z) yoki None (yozuv o'qilmadi yoki bir ma'noli emas).
    """
    # Quti chetiga tegib turgan matn qo'shni mahsulotniki bo'lishi mumkin ("…ANGO") — tashlanadi
    tokens = [str(t[0]) for t in texts
              if float(t[1]) >= min_conf and (len(t) < 4 or (float(t[2]) > edge and float(t[3]) < 1 - edge))]
    if not tokens or len(options) < 2:
        return None
    words = {sku: keywords(name, brand) for sku, (name, brand) in options.items()}
    hits = {sku: [w for w in ws if any(_hit(v, tokens) for v in _SYN.get(w, (w,)))]
            for sku, ws in words.items()}
    best = max(len(h) for h in hits.values())
    seen = None
    for t in tokens:
        seen = seen or read_volume(t + " ")
    volumes = {sku: name_volume(options[sku][0]) for sku in options}
    if best:
        top = [sku for sku, h in hits.items() if len(h) == best]
        word = hits[top[0]][0]
    elif len({tuple(sorted(ws)) for ws in words.values()}) == 1:
        top, word = list(options), ""           # faqat hajmi bilan farq qiladigan egizaklar
    else:
        return None
    if len(top) > 1:
        # Bir xil ta'm, har xil hajm: yorliqdagi hajm hal qiladi
        if seen is None or len({tuple(sorted(words[s])) for s in top}) != 1:
            return None
        top = [s for s in top if volumes[s] == seen]
        if len(top) != 1:
            return None
        return top[0], f"{seen / 1000:g} L" if seen >= 1000 else f"{seen} ml"
    sku = top[0]
    if seen is not None and volumes[sku] is not None and volumes[sku] != seen:
        return None                             # ta'm mos, hajm boshqa — bu boshqa mahsulot
    return sku, word


# ---- Tanilgan qutini tekshirish ----------------------------------------------

def _words(text: str) -> list[str]:
    return [w.strip("'") for w in re.findall(r"[a-zа-яё']+", str(text).lower().translate(_APOS))]


def label_words(readings, *, min_conf: float = 0.8, limit: int = 60) -> list[str]:
    """Namuna rasmlarida o'qilgan so'zlar — mahsulotning "o'z yozuvi" (katalogda saqlanadi).
    readings: har rasm uchun [(matn, ishonch, ...)]."""
    out: list[str] = []
    for texts in readings or []:
        for t in texts or []:
            if float(t[1]) < min_conf:
                continue
            for w in _words(t[0]):
                if len(w) >= 3 and w not in _STOP and w not in out:
                    out.append(w)
    return out[:limit]


def _near(word: str, token: str, ratio: float = 0.7) -> bool:
    """Token mahsulotning o'z so'ziga yaqinmi (OCR xatosi bilan o'qilgan bo'lishi mumkin)."""
    for norm in (_translit, _skeleton):
        k = norm(word).replace(" ", "")
        if len(k) >= 3 and any(SequenceMatcher(None, k, t).ratio() >= ratio
                               for t in norm(token).split() if len(t) >= 3):
            return True
    return False


def judge(texts, own: list[str], brand: Optional[str] = None, *, min_conf: float = 0.5,
          strong: float = 0.95, big: float = 0.05) -> tuple[str, str]:
    """Qutidagi yozuv mahsulotning o'z so'zlariga mos keladimi.

    texts: [(matn, ishonch, x1, x2, harf o'lchami)]; own: namunadagi so'zlar + nom + brend.
    -> ("support", so'z)     o'z so'zi o'qildi
       ("brand", so'z)       faqat brend so'zi o'qildi, yonida yirik begona so'z bor
       ("contradict", so'z)  o'z so'zi yo'q, yirik va aniq begona so'z bor
       ("silent", "")        aniq narsa o'qilmadi — qaror o'zgarmaydi
    Begona so'z: ishonchi `strong` dan, harfi quti kichik tomonining `big` ulushidan kam emas
    (brend nomi yirik yoziladi, tarkib va belgilar mayda), kamida 4 harf.
    """
    tokens = [str(t[0]) for t in texts if float(t[1]) >= min_conf]
    if not tokens or not own:
        return "silent", ""
    spoken = lambda w: _SYN.get(w, (w,))                      # noqa: E731
    hit = [w for w in own if any(_hit(v, tokens) for v in spoken(w))]
    alien = []
    for t in texts:
        size = float(t[4]) if len(t) > 4 else 0.0
        letters = re.sub(r"[^a-zа-яё]", "", str(t[0]).lower())
        if float(t[1]) >= strong and size >= big and len(letters) >= 4 and not any(
                _near(v, str(t[0])) for w in own for v in spoken(w)):
            alien.append((size, str(t[0])))
    word = max(alien)[1] if alien else ""
    if hit:
        marks = set(_words(brand or ""))
        if marks and alien and all(w in marks for w in hit):
            return "brand", word
        return "support", hit[0]
    return ("contradict", word) if alien else ("silent", "")


# ---- Xizmat mijozi -----------------------------------------------------------

class RemoteLabelReader:
    """OCR xizmati (ocr/) mijozi. Xatoda None qaytaradi — tahlil OCR'siz davom etadi.

    Rasmlar kichik to'plamlarda yuboriladi: band protsessorda bitta yirik qutini o'qish
    10 soniyagacha cho'ziladi, 12 tasi bitta so'rovda vaqt chegarasidan oshib, hammasi
    yo'qolardi. `budget` — bir chaqiruvga ajratilgan umumiy vaqt; tugagach qolgan rasmlar
    o'qilmagan ([]) deb qaytadi."""

    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 60.0,
                 batch: int = 4, budget: float = 60.0):
        import requests
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.batch = max(1, int(batch))
        self.budget = budget
        self.session = requests.Session()
        self.session.headers["X-Rasta-AI-Token"] = token or os.environ.get("RASTA_AI_TOKEN", "")
        self._warned = False

    def ready(self) -> bool:
        try:
            return self.session.get(self.url + "/ready", timeout=(2, 3)).ok
        except Exception:
            return False

    def read(self, crops: list[np.ndarray]) -> Optional[list[list]]:
        import time
        from PIL import Image
        results: list[list] = []
        started = time.monotonic()
        for k in range(0, len(crops), self.batch):
            left = self.budget - (time.monotonic() - started)
            if left <= 1:
                break
            files = []
            for i, crop in enumerate(crops[k:k + self.batch]):
                buf = io.BytesIO()
                Image.fromarray(crop).save(buf, "JPEG", quality=92)
                files.append(("images", (f"{i}.jpg", buf.getvalue(), "image/jpeg")))
            try:
                r = self.session.post(self.url + "/read", files=files, timeout=(2, min(self.timeout, left)))
                r.raise_for_status()
                part = r.json()["results"]
                if len(part) != len(files):
                    raise ValueError("javob uzunligi mos emas")
                results.extend(part)
                self._warned = False
            except Exception as error:               # tarmoq, vaqt tugashi, noto'g'ri javob
                if not self._warned:
                    print(f"[rasta] OCR xizmati javob bermadi ({type(error).__name__}) — yozuvsiz davom etiladi",
                          file=sys.stderr)
                    self._warned = True
                if not results:
                    return None
                break
        return results + [[] for _ in range(len(crops) - len(results))]


def build_label_reader(cfg):
    url = os.environ.get("RASTA_OCR_URL") or cfg.get("ocr.url")
    if not url:
        return None
    return RemoteLabelReader(str(url), timeout=float(cfg.get("ocr.timeout", 60)),
                             budget=float(cfg.get("ocr.budget", 60)))


# ---- Tahlilga qo'llash ---------------------------------------------------------

def label_crop(image: np.ndarray, box, scale: float = 1.0, pad: float = 0.04,
               min_long: int = 1500, max_long: int = 2600) -> np.ndarray:
    """Quti atrofidan (4% chet bilan) kesilgan, o'qish uchun qulay o'lchamga keltirilgan rasm.
    Kichik qutilar kattalashtiriladi (2000 px li ish rasmida matn 3 baravar kattalashtirilganda
    yaxshiroq o'qiladi: 7/34 -> 12/34), juda kattalari kichraytiriladi."""
    from PIL import Image
    h, w = image.shape[:2]
    px, py = pad * box.width * scale, pad * box.height * scale
    x1, y1 = max(0, int(box.x1 * scale - px)), max(0, int(box.y1 * scale - py))
    x2, y2 = min(w, int(box.x2 * scale + px) + 1), min(h, int(box.y2 * scale + py) + 1)
    crop = image[y1:y2, x1:x2]
    long = max(crop.shape[:2])
    k = min(3.0, min_long / long) if long < min_long else (max_long / long if long > max_long else 1.0)
    if abs(k - 1.0) > 0.05 and min(crop.shape[:2]) > 0:
        size = (max(1, round(crop.shape[1] * k)), max(1, round(crop.shape[0] * k)))
        crop = np.asarray(Image.fromarray(crop).resize(size, Image.LANCZOS))
    return np.ascontiguousarray(crop)


def refine_with_text(detections, image: np.ndarray, reader, catalog, *, hires=None,
                     max_boxes: int = 12, min_conf: float = 0.5) -> int:
    """Brend darajasida / variant deb qolgan qutilarni yorliq yozuvi bo'yicha SKU'ga ko'tarish.
    hires — asl o'lchamdagi rasm yoki uni qaytaradigan funksiya (faqat kerak bo'lganda chaqiriladi).
    -> nechta quti aniqlashtirildi."""
    todo = [d for d in detections
            if d.match is not None and not d.match.is_known and d.match.level in ("brand", "variant")
            and d.match.candidates]
    if not todo:
        return 0
    info = catalog.sku_info()
    jobs = []
    for d in sorted(todo, key=lambda d: -d.box.area):
        m = d.match
        brand = m.brand or (info.get(m.candidates[0].get("sku_id")) or {}).get("brand")
        near = [c.get("sku_id") for c in m.candidates]
        # Nomzodlar: shu brendning katalogdagi mahsulotlari, lekin faqat ko'rinishi bo'yicha
        # ham yaqin bo'lganlari (top-k ichida) — yozuvning o'zi yetarli dalil emas.
        options = {s: (info[s]["name"], info[s].get("brand")) for s in near
                   if s in info and (brand is None or info[s].get("brand") == brand)}
        if len(options) >= 2:
            jobs.append((d, options))
    jobs = jobs[:max_boxes]
    if not jobs:
        return 0
    source, scale = image, 1.0
    big = hires() if callable(hires) else hires
    if big is not None and big.shape[1] > image.shape[1]:
        source, scale = big, big.shape[1] / image.shape[1]
    results = reader.read([label_crop(source, d.box, scale) for d, _ in jobs])
    if results is None:
        return 0
    fixed = 0
    for (d, options), texts in zip(jobs, results):
        picked = choose(texts, options, min_conf=min_conf)
        if picked is None:
            continue
        sku, word = picked
        m = d.match
        cand = next(c for c in m.candidates if c.get("sku_id") == sku)
        d.match = replace(m, sku_id=sku, name=options[sku][0], is_known=True, level="sku",
                          brand=options[sku][1] or m.brand, score=float(cand.get("score", m.score)),
                          text=word)
        fixed += 1
    return fixed


def verify_with_text(detections, image: np.ndarray, reader, catalog, *, hires=None,
                     max_boxes: int = 12, min_conf: float = 0.5, strong: float = 0.95,
                     big: float = 0.05, risky=None) -> int:
    """Tanilgan qutilarni yorliq yozuvi bilan tekshirish: qadoqda aniq boshqa so'z turgan
    bo'lsa, moslik bekor qilinadi (yoki faqat brend darajasida qoldiriladi).

    Eng past balli qutilardan boshlanadi (noto'g'ri mosliklar o'sha yerda to'planadi) va
    `max_boxes` bilan cheklanadi; `risky(match)` berilsa — faqat u True qaytargan (chegaraga
    yaqin balli) qutilar o'qiladi: har qutini o'qish ~2 s turadi, ishonchli mosliklarda esa
    xato deyarli uchramaydi. Katalogda namuna yozuvi saqlanmagan mahsulotlar tekshirilmaydi —
    nomning o'zi bilan solishtirish to'g'ri qutilarni ham rad etardi.
    -> nechta qutining qarori o'zgardi."""
    info = None
    jobs = []
    for d in sorted((d for d in detections if d.match is not None and d.match.is_known
                     and d.match.text is None and d.match.sku_id
                     and (risky is None or risky(d.match))), key=lambda d: d.match.score):
        if info is None:
            info = catalog.sku_info()
        entry = info.get(d.match.sku_id) or {}
        if entry.get("words"):
            own = list(dict.fromkeys(list(entry["words"]) + keywords(entry.get("name") or "", None)
                                     + [w for w in _words(entry.get("brand") or "") if len(w) >= 3]))
            jobs.append((d, own, entry.get("brand")))
            if len(jobs) >= max_boxes:
                break
    if not jobs:
        return 0
    source, scale = image, 1.0
    big_image = hires() if callable(hires) else hires
    if big_image is not None and big_image.shape[1] > image.shape[1]:
        source, scale = big_image, big_image.shape[1] / image.shape[1]
    # Brend va mahsulot nomi yirik yoziladi: kichikroq rasm yetarli va tezroq o'qiladi
    results = reader.read([label_crop(source, d.box, scale, min_long=700, max_long=900) for d, _, _ in jobs])
    if results is None:
        return 0
    changed = 0
    for (d, own, brand), texts in zip(jobs, results):
        verdict, word = judge(texts, own, brand, min_conf=min_conf, strong=strong, big=big)
        if verdict == "contradict":
            d.match = replace(d.match, is_known=False, level=None, mismatch=word)
        elif verdict == "brand":
            d.match = replace(d.match, is_known=False, level="brand", brand=brand, mismatch=word)
        else:
            continue
        changed += 1
    return changed
