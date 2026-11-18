"""Noma'lum mahsulotlarni klasterlash (HDBSCAN) — operator paneli uchun.

Katalogda yo'q mahsulotlar "?" bo'lib qoladi. Ularni bittalab ko'rib chiqish
o'rniga, o'xshashlarini guruhlaymiz: operator bitta guruhga bitta nom beradi
va guruhdagi barcha crop'lar katalogga qo'shiladi (few-shot onboarding).

Klasterlash RANG bo'yichami yoki SHAKL bo'yichami?
  Hech biri yolg'iz emas. Faqat rang: Coca-Cola va Coca-Cola Zero (qizil-qora),
  qizil Pepsi Max va Coca-Cola aralashadi. Faqat shakl: barcha 0.33L bankalar
  bitta guruhga tushadi. Shuning uchun xususiyat vektori birlashtirilgan:
      f = [ DINOv2 (tekstura+naqsh+shakl) ,  alpha * rang ,  beta * tomonlar nisbati ]
  DINOv2 asosiy, rang — brendni ajratuvchi kuchli qo'shimcha, nisbat — hajmni
  (0.33L / 0.5L / 1.5L) ajratadi. HDBSCAN klasterlar sonini o'zi topadi va
  yolg'iz/tasodifiy crop'larni shovqin (-1) deb qoldiradi.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def build_cluster_features(
    dino: np.ndarray,
    colors: Optional[np.ndarray] = None,
    aspects: Optional[np.ndarray] = None,
    color_weight: float = 0.6,
    shape_weight: float = 0.3,
) -> np.ndarray:
    parts = [dino / np.linalg.norm(dino, axis=1, keepdims=True).clip(1e-6)]
    if colors is not None and color_weight > 0:
        parts.append(color_weight * colors)
    if aspects is not None and shape_weight > 0:
        la = np.log(np.asarray(aspects, dtype=np.float32).clip(1e-3))[:, None]
        parts.append(shape_weight * la)
    f = np.concatenate(parts, axis=1).astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-6)


def cluster_features(
    feats: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: Optional[int] = 1,
    epsilon: float = 0.0,
) -> np.ndarray:
    """HDBSCAN klaster yorliqlari (-1 = shovqin)."""
    n = len(feats)
    if n < max(2, min_cluster_size):
        return -np.ones(n, dtype=int)
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric="euclidean",          # normallashgan vektorlarda ~ kosinus
        copy=True,
    )
    return model.fit_predict(feats).astype(int)


def summarize(labels: np.ndarray, indices: list[int]) -> dict:
    """Klaster -> detection indekslari xulosasi (JSON uchun)."""
    out: dict[str, dict] = {}
    for lab in sorted(set(labels.tolist())):
        members = [indices[i] for i in np.where(labels == lab)[0]]
        key = "noise" if lab < 0 else str(lab)
        out[key] = {"size": len(members), "detections": members}
    return out


def purity_and_coverage(labels: np.ndarray, truth: list[str]) -> dict:
    """Baholash: klaster tozaligi (purity), qamrov va ARI/NMI (truth bilan)."""
    from collections import Counter

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels = np.asarray(labels)
    clustered = labels >= 0
    pure = 0
    for lab in set(labels[clustered].tolist()):
        members = [truth[i] for i in np.where(labels == lab)[0]]
        pure += Counter(members).most_common(1)[0][1]
    n_c = int(clustered.sum())
    # shovqinni har biri alohida klaster deb hisoblab ARI/NMI
    lab2 = labels.copy()
    nxt = lab2.max() + 1 if len(lab2) else 0
    for i in np.where(lab2 < 0)[0]:
        lab2[i] = nxt
        nxt += 1
    return {
        "n": len(labels),
        "n_clusters": len(set(labels[clustered].tolist())),
        "coverage": round(n_c / len(labels), 4) if len(labels) else 0.0,
        "purity": round(pure / n_c, 4) if n_c else 0.0,
        "ARI": round(float(adjusted_rand_score(truth, lab2)), 4),
        "NMI": round(float(normalized_mutual_info_score(truth, lab2)), 4),
    }
