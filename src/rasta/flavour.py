"""Brend ichida ta'm guruhlari: avval brend bo'yicha sanaladi, keyin brendning ta'mlari ajratiladi.

Brendi aniq, lekin ta'mi noma'lum qutilar («DENA» deb bitta yozuvga qo'shilgan hamma ta'mlar, brend
darajasida qolgan qutilar) bir brend ichida ta'mi bo'yicha guruhlanadi, shunda operator har ta'mga bir marta
nom beradi. Ikki dalil birlashtiriladi:

* qadoqdagi ta'm so'zi (OCR: «МАНГО», «ГРАНАТ», «ТОМАТ») — qizil ta'mlarni (anor, olma, pomidor) faqat u ajratadi;
* mevaning rangi (rang tasmalari, ta'mni ajratadigan tasmaga ko'proq og'irlik) — yozuv o'qilmagan qutilar uchun.

Yozuvlari bir-biriga zid ikki quti (ta'm so'zlari kesishmaydi) hech qachon bir guruhga tushmaydi.
"""
from __future__ import annotations

import numpy as np

from .features import BINS, STRIPES
from .ocr import _FLAVOR_WORDS, _SYNONYMS, _hit

# Ta'm so'zi -> asosiy (kanonik) shakl: «гранат» / «pomegranate» / «anor» hammasi bitta ta'm
_CANON = {word: group[0] for group in _SYNONYMS if group[0] not in ("gazli", "gazsiz") for word in group}


def flavour_key(texts, *, min_conf: float = 0.8, min_size: float = 0.035, edge: float = 0.03) -> frozenset:
    """Qutidagi o'qilgan matndan ta'm so'zlari to'plami (kanonik shaklda); bo'sh to'plam — ta'm so'zi o'qilmadi.
    texts: [(matn, ishonch, x1, x2, harf o'lchami)]; quti chetiga tegib turgan matn (qo'shni qadoq) tashlanadi."""
    found = set()
    for t in texts or []:
        if float(t[1]) < min_conf or (len(t) > 4 and float(t[4]) < min_size):
            continue
        if len(t) > 3 and (float(t[2]) <= edge or float(t[3]) >= 1 - edge):
            continue
        for word in _FLAVOR_WORDS:
            if len(word) >= 4 and _hit(word, [str(t[0])]):
                found.add(_CANON[word])
    return frozenset(found)


def _stripes(colors: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(STRIPES):
        s = colors[:, BINS * (k + 1): BINS * (k + 2)]
        out.append(s / np.linalg.norm(s, axis=1, keepdims=True).clip(1e-6))
    return out


def colour_groups(colors: np.ndarray, threshold: float = 0.06) -> np.ndarray:
    """Rang bo'yicha guruhlar. Brend ichida hamma qadoqning dizayni umumiy, farq — meva rasmi: shuning uchun rang
    tasmalaridan guruh ichida eng ko'p farq qiladiganiga ko'proq og'irlik beriladi."""
    n = len(colors)
    if n < 2:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    stripes = _stripes(colors)
    # Bir xil tasmali ikki quti: yaxlitlash xatosi tarqoqlikni -1e-7 qiladi, ildizi esa NaN beradi
    spread = np.array([max(0.0, float(np.mean(1 - (s @ s.T)[np.triu_indices(n, 1)]))) for s in stripes])
    weights = spread / spread.sum() if spread.sum() > 0 else np.full(STRIPES, 1.0 / STRIPES)
    feats = np.concatenate([np.sqrt(w) * s for w, s in zip(weights, stripes)], axis=1)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average",
                                   metric="cosine").fit_predict(feats)


def group_flavours(colors: np.ndarray, keys: list, threshold: float = 0.06) -> list[int]:
    """Bir brendning qutilari uchun ta'm guruhi raqamlari. colors — rang xususiyatlari, keys — har quti uchun
    ta'm so'zlari to'plami (bo'sh bo'lishi mumkin). Rang guruhi ichida ta'm so'zlari kesishmaydigan qutilar
    ajratiladi; so'zi o'qilmagan quti rang bo'yicha eng mos pastki guruhga qo'shiladi."""
    colour = colour_groups(np.asarray(colors), threshold)
    labels = [-1] * len(keys)
    nxt = 0
    for g in sorted(set(colour.tolist())):
        members = [i for i in range(len(keys)) if colour[i] == g]
        subgroups: list[tuple[int, set]] = []          # (raqam, shu guruhdagi ta'm so'zlari)
        for i in sorted(members, key=lambda i: (not keys[i], i)):     # so'zi o'qilganlar birinchi
            for k, (number, words) in enumerate(subgroups):
                if not keys[i] or not words or keys[i] & words:
                    labels[i] = number
                    subgroups[k] = (number, words | set(keys[i]))
                    break
            else:
                labels[i] = nxt
                subgroups.append((nxt, set(keys[i])))
                nxt += 1
    return labels


def flavour_name(keys: list) -> str | None:
    """Guruhning nomi: a'zolarida eng ko'p uchragan ta'm so'zi (bo'lmasa None)."""
    counts: dict[str, int] = {}
    for key in keys:
        for word in key:
            counts[word] = counts.get(word, 0) + 1
    return max(counts, key=lambda w: (counts[w], w)) if counts else None


def split_by_words(labels, keys: list, colors: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Guruh ichida qadoqdagi ta'm so'zlari zid qutilarni ajratadi: rangi bir xil qizil ta'mlar (olma, anor,
    pomidor) faqat yozuvi bilan farq qiladi. labels — guruh raqamlari (-1 — guruhsiz), keys — har quti uchun
    ta'm so'zlari to'plami (bo'sh — o'qilmadi). So'zi o'qilmagan quti rangi eng yaqin bo'lakka qo'shiladi.
    Yozuvlar zid bo'lmagan guruh o'zgarmaydi. Yakka qolgan quti guruhsiz (-1) bo'ladi."""
    labels = np.asarray(labels)
    refined = labels.copy()
    next_id = int(labels.max()) + 1 if len(labels) else 0
    unit = colors / np.linalg.norm(colors, axis=1, keepdims=True).clip(1e-6)
    for group in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == group)
        parts: list[tuple[list[int], set]] = []                   # (a'zolar, shu bo'lakdagi ta'm so'zlari)
        for i in members:
            if not keys[i]:
                continue
            for chosen, words in parts:
                if keys[i] & words:
                    chosen.append(int(i))
                    words |= set(keys[i])
                    break
            else:
                parts.append(([int(i)], set(keys[i])))
        if len(parts) < 2:
            continue
        cores = [list(chosen) for chosen, _ in parts]            # so'zsizlar faqat so'zi o'qilganlarga qiyoslanadi
        for i in members:
            if not keys[i]:
                best = max(range(len(parts)), key=lambda k: float((unit[cores[k]] @ unit[i]).mean()))
                parts[best][0].append(int(i))
        for number, (chosen, _) in enumerate(sorted(parts, key=lambda part: -len(part[0]))):
            if len(chosen) < min_size:
                refined[chosen] = -1
            elif number:
                refined[chosen] = next_id
                next_id += 1
    return refined


def read_flavour_keys(detections, indices, reader, image, hires=None, max_boxes: int = 60) -> dict:
    """indices dagi qutilarning ta'm so'zlari (yirik qutilar birinchi, max_boxes tagacha): {indeks: to'plam}."""
    from .ocr import label_crop

    order = sorted(indices, key=lambda i: -detections[i].box.area)[:max_boxes]
    if reader is None or image is None or not order:
        return {}
    source, scale = image, 1.0
    big_image = hires() if callable(hires) else hires
    if big_image is not None and big_image.shape[1] > image.shape[1]:
        source, scale = big_image, big_image.shape[1] / image.shape[1]
    texts = reader.read([label_crop(source, detections[i].box, scale, min_long=1400, max_long=1800) for i in order])
    return {i: flavour_key(lines) for i, lines in zip(order, texts or [])}


def assign_flavours(detections, colors, *, reader=None, image=None, hires=None, threshold: float = 0.06,
                    max_boxes: int = 60) -> int:
    """Brendi aniq, ta'mi noma'lum qutilarni (umumiy brend yozuviga qo'shilganlar va brend darajasida qolganlar)
    brend ichida ta'mi bo'yicha guruhlaydi: `Detection.flavour` (guruh raqami) va `flavour_label` (ta'm so'zi).
    reader berilsa, qadoqdagi ta'm so'zi ham o'qiladi (GPU'da tez). Bitta guruhga tushgan brend (ajratadigan narsa
    yo'q) belgilanmaydi. -> guruhlar soni."""
    eligible: dict[str, list[int]] = {}
    for i, d in enumerate(detections):
        m = d.match
        if m is None or not m.brand:
            continue
        if (m.is_known and m.family) or (not m.is_known and m.level == "brand"):
            eligible.setdefault(" ".join(m.brand.casefold().split()), []).append(i)
    eligible = {brand: ids for brand, ids in eligible.items() if len(ids) >= 2}
    if not eligible or colors is None:
        return 0
    keys = {i: frozenset() for ids in eligible.values() for i in ids}
    keys.update(read_flavour_keys(detections, list(keys), reader, image, hires, max_boxes))
    number = 0
    for ids in eligible.values():
        labels = group_flavours(np.asarray(colors)[ids], [keys[i] for i in ids], threshold)
        if len(set(labels)) < 2:
            continue
        for group in sorted(set(labels)):
            members = [i for i, label in zip(ids, labels) if label == group]
            name = flavour_name([keys[i] for i in members])
            for i in members:
                detections[i].flavour = number
                detections[i].flavour_label = name
            number += 1
    return number

