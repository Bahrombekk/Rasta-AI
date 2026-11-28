"""Noma'lum mahsulotlarni klasterlash (HDBSCAN) — operator paneli uchun.

Katalogda yo'q mahsulotlar "?" bo'lib qoladi. Ularni bittalab ko'rib chiqish
o'rniga, o'xshashlarini guruhlaymiz: operator bitta guruhga bitta nom beradi
va guruhdagi barcha crop'lar katalogga qo'shiladi (few-shot onboarding).

Klasterlash RANG bo'yichami yoki SHAKL bo'yichami?
  Hech biri yolg'iz emas. Faqat rang: Coca-Cola va Coca-Cola Zero (qizil-qora),
  qizil Pepsi Max va Coca-Cola aralashadi. Faqat shakl: barcha 0.33L bankalar
  bitta guruhga tushadi. Shuning uchun xususiyat vektori birlashtirilgan:
      f = [ DINOv2 (tekstura+naqsh+shakl) ,  alpha * rang ,  beta * tomonlar nisbati ]
  Embedding asosiy, rang va nisbat yordamchi. HDBSCAN dastlabki guruhlarni
  topadi. Guruhsiz crop'lar orasidan yaqin juftlar alohida tekshiriladi.

Bu guruhlar brend/dizayn darajasida chiqadi (bir brendning ta'mlari birga). Operator nom beradigan
guruhlar uchun `split_variants` har guruh ichida variantni (rang tasmalari + markazlangan embedding) ajratadi.
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
    selection_method: str = "eom",
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
        cluster_selection_method=selection_method,
        metric="euclidean",          # normallashgan vektorlarda ~ kosinus
        copy=True,
    )
    return model.fit_predict(feats).astype(int)


def cluster_unknown_features(embedding, colors, aspects, cfg) -> np.ndarray:
    """Javon tahlilida noma'lum mahsulotlarga klaster ID beradi."""
    feats = build_cluster_features(
        embedding, colors, aspects,
        color_weight=cfg.get("clustering.color_weight", 0.6),
        shape_weight=cfg.get("clustering.shape_weight", 0.3),
    )
    min_size = cfg.get("clustering.min_cluster_size", 2)
    max_distance = cfg.get("clustering.recovery_max_distance", 0.20)
    if max_distance is None:
        return cluster_features(feats, min_size)

    # Katta to'plamda zichroq boshlang'ich guruhlar mayda klasterlarga
    # parchalanishni kamaytiradi. Kichik SKU guruhlari quyida qayta tekshiriladi.
    if len(feats) >= 500:
        labels = cluster_features(feats, max(min_size, 5), selection_method="leaf")
    else:
        labels = cluster_features(feats, min_size)
    labels = recover_noise(feats, labels, min_size, max_distance)
    if colors is not None and aspects is not None and cfg.get("clustering.split_label_colors", True):
        aspects = np.asarray(aspects)
        labels = split_label_colors(labels, colors, aspects, min_size)
        labels = split_mixed_labels(labels, embedding, colors, aspects, min_size)
    return labels


def split_label_colors(labels, colors, aspects, min_size) -> np.ndarray:
    """Split tall packages when their lower color bands form two tight groups."""
    from scipy.cluster.hierarchy import cut_tree, linkage

    from .features import BINS

    refined = labels.copy()
    for group in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == group)
        if len(members) < 2 * min_size or np.any(aspects[members] < 2):
            continue
        bands = colors[members, -BINS:]
        norms = np.linalg.norm(bands, axis=1, keepdims=True)
        if np.any(norms == 0):
            continue
        bands = bands / norms
        distances = 1 - np.clip(bands @ bands.T, -1, 1)
        upper = np.triu_indices(len(members), k=1)
        tree = linkage(distances[upper], method="complete")
        sides = cut_tree(tree, n_clusters=2).ravel()
        left = np.flatnonzero(sides == 0)
        right = np.flatnonzero(sides == 1)
        if min(len(left), len(right)) < min_size:
            continue
        within = max(distances[np.ix_(left, left)].max(), distances[np.ix_(right, right)].max())
        between = distances[np.ix_(left, right)].min()
        if between >= max(0.28, 3 * within):
            refined[members[right]] = refined.max() + 1
    return refined


def split_mixed_labels(labels, embedding, colors, aspects, min_size) -> np.ndarray:
    """Keep compact subgroups of tall packages when their middle labels strongly disagree."""
    from scipy.cluster.hierarchy import fcluster, linkage

    from .features import BINS

    refined = labels.copy()
    next_id = int(labels.max()) + 1 if len(labels) else 0
    for group in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == group)
        if not max(4, 2 * min_size) <= len(members) <= 12:
            continue
        group_aspects = aspects[members]
        if group_aspects.min() < 2.5 or group_aspects.max() > 1.18 * group_aspects.min():
            continue
        band = colors[members, 2 * BINS:3 * BINS]
        vectors = embedding[members]
        band = band / np.linalg.norm(band, axis=1, keepdims=True).clip(1e-6)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-6)
        distances = 1 - np.clip(0.6 * (band @ band.T) + 0.4 * (vectors @ vectors.T), -1, 1)
        upper = np.triu_indices(len(members), k=1)
        tree = linkage(distances[upper], method="complete")
        parts = fcluster(tree, 0.13, criterion="distance")
        within = parts[upper[0]] == parts[upper[1]]
        counts = np.bincount(parts)
        if not within.any() or within.all() or counts.max() < min_size:
            continue
        pair_distances = distances[upper]
        if pair_distances[~within].min() < max(0.18, pair_distances[within].max() + 0.05):
            continue
        refined[members] = -1
        for part in np.flatnonzero(counts >= min_size):
            first = part == counts.argmax()
            refined[members[parts == part]] = group if first else next_id
            if not first:
                next_id += 1
    return refined


def split_variants(labels, embedding, colors, min_size, max_color=0.12, max_embedding=0.65,
                   center=None) -> np.ndarray:
    """Bir guruhga tushgan, lekin varianti (ta'mi) boshqa qadoqlarni ajratadi.

    Dastlabki guruhlar brend/dizayn bo'yicha tuziladi: bir brendning mango va olma sharbati bitta guruhga
    tushadi va operator bergan bitta nom ikkalasiga yoziladi. Guruh ichida ikki quti faqat (a) uchala rang
    tasmasining har biri o'xshash VA (b) rasm bo'yicha markazlangan embeddingi o'xshash bo'lsa birga qoladi.
    `center` — rasmdagi hamma qutilarning o'rtacha vektori (umumiy «javondagi qadoq» tarkibini olib tashlaydi);
    berilmasa (qutilar kam) faqat rang sharti ishlaydi. Guruhsiz qolgan a'zo -1 bo'ladi.
    Bir ota guruhning bo'laklari ketma-ket raqam oladi (ro'yxatda yonma-yon turadi)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    from .features import BINS, STRIPES

    labels = np.asarray(labels)
    if colors is None or not (labels >= 0).any():
        return labels

    def unit(a):
        return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-6)

    stripes = [unit(colors[:, BINS * (k + 1): BINS * (k + 2)]) for k in range(STRIPES)]
    vectors = unit(embedding - center) if center is not None and max_embedding else None
    refined = -np.ones(len(labels), dtype=int)
    next_id = 0
    for group in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == group)
        parts = np.ones(len(members), dtype=int)
        if len(members) > 1:
            distance = np.max([1 - np.clip(s[members] @ s[members].T, -1, 1) for s in stripes], axis=0) / max_color
            if vectors is not None:
                near = 1 - np.clip(vectors[members] @ vectors[members].T, -1, 1)
                distance = np.maximum(distance, near / max_embedding)
            upper = np.triu_indices(len(members), k=1)
            parts = fcluster(linkage(distance[upper].clip(0), method="average"), 1.0, criterion="distance")
        for part in np.unique(parts):
            chosen = members[parts == part]
            if len(chosen) >= min_size:
                refined[chosen] = next_id
                next_id += 1
    return refined


def recover_noise(feats, labels, min_size, max_distance) -> np.ndarray:
    """Guruhsiz mahsulotlarni dastlabki klasterlarga yoki yangi ixcham guruhga qo'shadi."""
    from scipy.cluster.hierarchy import fcluster, linkage

    noise = np.flatnonzero(labels < 0)
    if not len(noise):
        return labels

    seeds = np.flatnonzero(labels >= 0)
    refined = labels.copy()
    if len(seeds):
        # O'rtacha masofa faqat dastlabki a'zolardan olinadi. Yangi a'zolar
        # keyingi mahsulotlarni tortib, zanjir hosil qila olmaydi.
        group_ids = np.unique(labels[seeds])
        centroids = np.stack([feats[labels == group].mean(axis=0) for group in group_ids])
        means = 1 - np.clip(feats[noise] @ centroids.T, -1, 1)
        nearest = means.argmin(axis=1)
        close = means[np.arange(len(noise)), nearest] <= max_distance
        refined[noise[close]] = group_ids[nearest[close]]
        noise = noise[~close]

    if len(noise) >= max(2, min_size):
        # Complete linkage: yangi klasterdagi har bir juftlik chegaradan o'tadi.
        rows, cols = np.triu_indices(len(noise), k=1)
        distances = 1 - np.clip(feats[noise] @ feats[noise].T, -1, 1)
        tree = linkage(distances[rows, cols], method="complete")
        new_groups = fcluster(tree, max_distance, criterion="distance")
        large_enough = np.bincount(new_groups)[new_groups] >= min_size
        refined[noise[large_enough]] = new_groups[large_enough] + refined.max()
    return refined


def summarize(labels: np.ndarray, indices: list[int]) -> dict:
    """Klaster -> detection indekslari xulosasi (JSON uchun)."""
    out: dict[str, dict] = {}
    for lab in sorted(set(labels.tolist())):
        members = [indices[i] for i in np.where(labels == lab)[0]]
        key = "noise" if lab < 0 else str(lab)
        out[key] = {"size": len(members), "detections": members}
    return out


def purity_and_coverage(labels: np.ndarray, truth: list[str]) -> dict:
    """Klaster sifatini va guruhlangan mahsulotlar ulushini o'lchaydi."""
    from collections import Counter

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix

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
    counts = contingency_matrix(truth, lab2).astype(np.float64)
    bcubed_f1 = false_merge = 0.0
    if len(labels):
        predicted_sizes = counts.sum(axis=0)
        true_sizes = counts.sum(axis=1)
        squared = counts ** 2
        precision = np.sum(squared / predicted_sizes) / len(labels)
        recall = np.sum(squared / true_sizes[:, None]) / len(labels)
        bcubed_f1 = 2 * precision * recall / (precision + recall)
        predicted_pairs = np.sum(predicted_sizes * (predicted_sizes - 1))
        correct_pairs = np.sum(counts * (counts - 1))
        if predicted_pairs:
            false_merge = 1 - correct_pairs / predicted_pairs
    return {
        "n": len(labels),
        "n_clusters": len(set(labels[clustered].tolist())),
        "coverage": round(n_c / len(labels), 4) if len(labels) else 0.0,
        "purity": round(pure / n_c, 4) if n_c else 0.0,
        "ARI": round(float(adjusted_rand_score(truth, lab2)), 4),
        "NMI": round(float(normalized_mutual_info_score(truth, lab2)), 4),
        "bcubed_f1": round(float(bcubed_f1), 4),
        "false_merge": round(float(false_merge), 4),
    }
