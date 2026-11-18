"""Open-set chegaralarini (τ, margin) kalibrlash va baholash metrikalari.

Har bir so'rov (crop) uchun yozuv:
    truth  — haqiqiy SKU yoki "unknown" (katalogda yo'q)
    pred   — top-1 SKU (chegarasiz)
    score  — top-1 ball (fused)
    margin — top-1 − top-2
Bulardan:
    closed_acc  — faqat tanish mahsulotlarda top-1 to'g'riligi (chegarasiz)
    auroc       — tanish vs noma'lum ajratish sifati (top-1 ball bo'yicha)
    open_acc(τ,m) — chegara qo'llangandagi yakuniy aniqlik
fit_thresholds() — τ va margin'ni berilgan yozuvlarda maksimal open_acc
bo'yicha tanlaydi (16.4-bo'limdagi kalibrlash).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Rec:
    truth: str
    pred: str
    score: float
    margin: float


def decide(r: Rec, tau: float, min_margin: float) -> str:
    return r.pred if (r.score >= tau and r.margin >= min_margin) else "unknown"


def _arrays(recs: list[Rec]):
    unk = np.array([r.truth == "unknown" for r in recs], dtype=bool)
    corr = np.array([r.pred == r.truth for r in recs], dtype=bool)
    sc = np.array([r.score for r in recs], dtype=np.float32)
    mg = np.array([r.margin for r in recs], dtype=np.float32)
    return unk, corr, sc, mg


def _metrics(unk, corr, accept) -> dict:
    """accept: (..., n) bool. Vektorlashtirilgan metrikalar."""
    hit = np.where(accept, corr & ~unk, unk)
    kn, un = int((~unk).sum()), int(unk.sum())
    out = {"open_acc": hit.mean(axis=-1)}
    out["known_acc"] = hit[..., ~unk].sum(axis=-1) / kn if kn else None
    out["unknown_reject"] = hit[..., unk].sum(axis=-1) / un if un else None
    out["balanced"] = (0.5 * out["known_acc"] + 0.5 * out["unknown_reject"]
                       if kn and un else None)
    return out


def open_accuracy(recs: list[Rec], tau: float, min_margin: float) -> dict:
    unk, corr, sc, mg = _arrays(recs)
    m = _metrics(unk, corr, (sc >= tau) & (mg >= min_margin))
    return {k: (None if v is None else float(v)) for k, v in m.items()}


def closed_accuracy(recs: list[Rec]) -> float:
    kn = [r for r in recs if r.truth != "unknown"]
    return sum(r.pred == r.truth for r in kn) / len(kn) if kn else 0.0


def auroc(recs: list[Rec]) -> float | None:
    """Tanish (katalogda bor) vs noma'lum — top-1 ball bo'yicha ajratish sifati."""
    y = [0 if r.truth == "unknown" else 1 for r in recs]
    if len(set(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, [r.score for r in recs]))


def fit_thresholds(
    recs: list[Rec],
    taus: np.ndarray | None = None,
    margins: tuple = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08),
    objective: str = "open_acc",
) -> tuple[float, float, dict]:
    """Eng yaxshi (τ, margin) ni topadi. objective: open_acc | balanced."""
    if taus is None:
        taus = np.round(np.arange(0.30, 0.951, 0.01), 3)
    unk, corr, sc, mg = _arrays(recs)
    best = (-1.0, 0.5, 0.0)
    for m in margins:
        acc = (sc[None, :] >= np.asarray(taus)[:, None]) & (mg >= m)[None, :]
        met = _metrics(unk, corr, acc)
        val = met.get(objective)
        if val is None:
            val = met["open_acc"]
        i = int(np.argmax(val))
        if val[i] > best[0] + 1e-9:
            best = (float(val[i]), float(taus[i]), float(m))
    return best[1], best[2], open_accuracy(recs, best[1], best[2])


def holdout_eval(recs: list[Rec], groups: list[int], objective: str = "open_acc") -> dict:
    """Halol baholash: juft guruhlarda kalibrlab, toq guruhlarda tekshirish va aksincha."""
    a = [r for r, g in zip(recs, groups) if g % 2 == 0]
    b = [r for r, g in zip(recs, groups) if g % 2 == 1]
    out = []
    for fit, test in ((a, b), (b, a)):
        if not fit or not test:
            continue
        t, m, _ = fit_thresholds(fit, objective=objective)
        res = open_accuracy(test, t, m)
        res.update(tau=t, margin=m)
        out.append(res)
    if not out:
        return {}
    keys = ["open_acc", "known_acc", "unknown_reject", "balanced"]
    agg = {k: float(np.mean([o[k] for o in out if o[k] is not None]))
           for k in keys if any(o[k] is not None for o in out)}
    agg["tau"] = float(np.mean([o["tau"] for o in out]))
    agg["margin"] = float(np.mean([o["margin"] for o in out]))
    return agg


def report(recs: list[Rec], groups: list[int] | None = None) -> dict:
    out = {
        "n": len(recs),
        "closed_acc": round(closed_accuracy(recs), 4),
        "auroc": None if auroc(recs) is None else round(auroc(recs), 4),
    }
    t, m, best = fit_thresholds(recs)
    out["oracle"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}
    out["oracle"].update(tau=t, margin=m)
    if groups is not None:
        h = holdout_eval(recs, groups)
        out["holdout"] = {k: round(v, 4) for k, v in h.items()}
    return out
