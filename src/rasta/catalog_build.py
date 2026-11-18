"""Katta foto arxividan SKU katalogi nomzodlarini qurish (klasterlash + montaj).

Oqim (scripts/build_catalog_candidates.py):
  1. Muzlatgich/javon rasmlarida mahsulotlarni kesish (YOLO).
  2. Har crop -> embedding (DINOv2+SigLIP2) + rang + shakl  (cluster.build_cluster_features).
  3. Sferik k-means (GPU) — ko'p sonli mayda klasterlar ("over-clustering").
  4. Centroidlari juda yaqin klasterlar birlashtiriladi (bir SKU bir nechta
     k-means klasteriga bo'linib ketgan bo'lishi mumkin).
  5. Har klaster uchun: hajm, zichlik (centroidga o'rtacha kosinus) va nechta
     TURLI do'konda uchrashi. Haqiqiy SKU ko'p do'konda takrorlanadi; bitta
     do'kondagi tasodifiy obyekt (reklama, quti) — yo'q. Saralash shu bo'yicha.
  6. Operator montajni ko'rib, klasterga SKU nomini yozadi -> katalogga import.

HDBSCAN o'rniga k-means: yuz minglab crop'da HDBSCAN juda sekin; GPU k-means +
birlashtirish soniyalarda ishlaydi va natija operator tekshiruvidan o'tadi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def spherical_kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0,
                     device: str = "cuda", batch: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """L2-normallashgan X (N,D) uchun kosinus k-means. Qaytaradi: (labels, centroids)."""
    import torch

    if not torch.cuda.is_available():
        device = "cpu"
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.from_numpy(X.astype(np.float16 if device != "cpu" else np.float32)).to(device)
    n = Xt.shape[0]
    k = max(1, min(k, n))
    # k-means++ ga yaqin boshlash: tasodifiy + uzoq nuqtalar (tez varianti)
    idx = torch.randperm(n, generator=g)[:k].to(device)
    C = Xt[idx].float()
    labels = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(iters):
        for s in range(0, n, batch):
            sim = Xt[s:s + batch].float() @ C.T
            labels[s:s + batch] = sim.argmax(1)
        newC = torch.zeros_like(C)
        newC.index_add_(0, labels, Xt.float())
        counts = torch.bincount(labels, minlength=k).float()
        empty = counts == 0
        if empty.any():   # bo'sh klasterlarni tasodifiy nuqtalar bilan qayta to'ldirish
            ridx = torch.randint(0, n, (int(empty.sum()),), generator=g).to(device)
            newC[empty] = Xt[ridx].float()
        C = torch.nn.functional.normalize(newC, dim=1)
    for s in range(0, n, batch):
        labels[s:s + batch] = (Xt[s:s + batch].float() @ C.T).argmax(1)
    return labels.cpu().numpy(), C.cpu().numpy()


def merge_clusters(centroids: np.ndarray, labels: np.ndarray, thr: float = 0.94) -> np.ndarray:
    """Centroidlarni COMPLETE-linkage bilan birlashtiradi: guruhdagi har bir juftlik
    kosinusi >= thr bo'lishi shart.

    Oddiy (single-link / union-find) birlashtirish zanjir hosil qiladi: A~B, B~C ->
    A=C. Aptash'da centroidlar juda zich (eng yaqin qo'shni kosinusi median 0.956),
    shuning uchun union-find 2037 klasterning 80 mingdan 79 870 tasini bitta guruhga
    yopishtirgan edi. Complete-linkage bunday bo'lmaydi.
    """
    if len(centroids) < 2:
        return labels.copy()
    from sklearn.cluster import AgglomerativeClustering

    groups = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete",
                                     distance_threshold=1.0 - thr).fit_predict(centroids)
    return groups[labels]


@dataclass
class ClusterInfo:
    cid: int
    size: int
    n_outlets: int
    tightness: float          # a'zolarning centroidga o'rtacha kosinusi
    members: np.ndarray       # indekslar, centroidga yaqinlik bo'yicha saralangan
    score: float = 0.0


def cluster_stats(X: np.ndarray, labels: np.ndarray, outlets: np.ndarray,
                  min_size: int = 8) -> list[ClusterInfo]:
    out = []
    for c in np.unique(labels):
        m = np.where(labels == c)[0]
        if len(m) < min_size:
            continue
        cen = X[m].astype(np.float32).mean(0)
        cen /= max(np.linalg.norm(cen), 1e-6)
        sims = X[m].astype(np.float32) @ cen
        order = m[np.argsort(-sims)]
        n_out = len(np.unique(outlets[m]))
        info = ClusterInfo(int(c), len(m), n_out, float(sims.mean()), order)
        # Saralash: ko'p do'konda uchraydigan, zich klasterlar birinchi
        info.score = n_out * info.tightness ** 4
        out.append(info)
    out.sort(key=lambda c: -c.score)
    return out


def pick_examples(info: ClusterInfo, outlets: np.ndarray, n: int = 12) -> list[int]:
    """Montaj uchun namunalar: centroidga yaqin, lekin iloji boricha turli do'konlardan."""
    picked, seen = [], set()
    for i in info.members:
        if outlets[i] not in seen:
            picked.append(int(i))
            seen.add(outlets[i])
        if len(picked) >= n:
            return picked
    for i in info.members:           # do'konlar kam bo'lsa — qolganini to'ldirish
        if int(i) not in picked:
            picked.append(int(i))
        if len(picked) >= n:
            break
    return picked
