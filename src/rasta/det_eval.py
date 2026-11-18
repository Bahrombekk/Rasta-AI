"""Detektor sifatini o'lchash: qutilarni GT bilan solishtirish (AP50, P/R, sanash xatosi).

GT fayli (data/gt_det/*.det.json):
  {"image": "yo'l/rasm.jpg", "status": "done",
   "boxes":  [[x1,y1,x2,y2], ...],     # har bir ko'rinadigan mahsulot (facing)
   "ignore": [[x1,y1,x2,y2], ...]}     # baholanmaydi (orqa fon, juda qisman, reklama)
Ignore zonasiga tushgan bashorat na TP, na FP hisoblanadi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_image(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None,
                iou_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """pred (N,5) ishonch bo'yicha ochko'z moslash.

    Qaytaradi: (scores, flags) — flags: 1 = TP, 0 = FP, -1 = ignore (hisobga olinmaydi).
    """
    if len(pred) == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    order = np.argsort(-pred[:, 4])
    pred = pred[order]
    flags = np.zeros(len(pred), dtype=int)
    ious = iou_matrix(pred[:, :4], gt)
    used = np.zeros(len(gt), dtype=bool)
    ign = None
    if ignore is not None and len(ignore):
        # ignore zonasi: bashorat maydonining yarmidan ko'pi zona ichida bo'lsa
        ix1 = np.maximum(pred[:, None, 0], ignore[None, :, 0]); iy1 = np.maximum(pred[:, None, 1], ignore[None, :, 1])
        ix2 = np.minimum(pred[:, None, 2], ignore[None, :, 2]); iy2 = np.minimum(pred[:, None, 3], ignore[None, :, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ign = (inter / (area[:, None] + 1e-9)).max(1) >= 0.5
    for i in range(len(pred)):
        if len(gt):
            cand = np.where(~used)[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                if ious[i, j] >= iou_thr:
                    used[j] = True
                    flags[i] = 1
                    continue
        flags[i] = -1 if (ign is not None and ign[i]) else 0
    return pred[:, 4], flags


def average_precision(scores: np.ndarray, flags: np.ndarray, n_gt: int) -> float:
    """VOC uslubidagi (barcha nuqtali) AP."""
    keep = flags >= 0
    scores, flags = scores[keep], flags[keep]
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-scores)
    tp = np.cumsum(flags[o] == 1)
    fp = np.cumsum(flags[o] == 0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class DetReport:
    images: int
    gt: int
    pred: int
    tp: int
    fp: int
    ap50: float
    precision: float
    recall: float
    f1: float
    count_mae: float        # |bashorat soni - GT soni| o'rtachasi (facings sanash xatosi)
    count_mape: float       # nisbiy, %
    best_f1: float = 0.0    # ishonch chegarasi optimal tanlanganda
    best_thr: float = 0.0   # o'sha chegara (detector.conf ga qo'yish uchun)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def evaluate(items: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
             iou_thr: float = 0.5) -> DetReport:
    """items: har rasm uchun (pred (N,5), gt (M,4), ignore (K,4) | None)."""
    all_s, all_f = [], []
    n_gt = tp = fp = n_pred = 0
    errs, rel = [], []
    for pred, gt, ign in items:
        s, f = match_image(pred, gt, ign, iou_thr)
        all_s.append(s); all_f.append(f)
        n_gt += len(gt)
        tp += int((f == 1).sum()); fp += int((f == 0).sum())
        counted = int((f >= 0).sum())
        n_pred += counted
        errs.append(abs(counted - len(gt)))
        rel.append(abs(counted - len(gt)) / max(1, len(gt)))
    scores = np.concatenate(all_s) if all_s else np.zeros(0)
    flags = np.concatenate(all_f) if all_f else np.zeros(0, dtype=int)
    p = tp / max(1, tp + fp)
    r = tp / max(1, n_gt)
    best_f1 = best_thr = 0.0
    keep = flags >= 0
    if keep.any() and n_gt:
        s, f = scores[keep], flags[keep]
        o = np.argsort(-s)
        ctp = np.cumsum(f[o] == 1)
        cfp = np.cumsum(f[o] == 0)
        f1 = 2 * ctp / (ctp + cfp + n_gt)          # = 2PR/(P+R)
        k = int(np.argmax(f1))
        best_f1, best_thr = float(f1[k]), float(s[o][k])
    return DetReport(
        images=len(items), gt=n_gt, pred=n_pred, tp=tp, fp=fp,
        ap50=average_precision(scores, flags, n_gt),
        precision=p, recall=r, f1=2 * p * r / max(1e-9, p + r),
        count_mae=float(np.mean(errs)) if errs else 0.0,
        count_mape=100.0 * float(np.mean(rel)) if rel else 0.0,
        best_f1=best_f1, best_thr=best_thr,
    )
