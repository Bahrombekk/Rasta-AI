"""Umumiy ma'lumot tuzilmalari (pipeline bo'ylab uzatiladi)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Box:
    """Detektor topgan bitta obyekt (mahsulot nomzodi)."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class Match:
    """Vektor qidiruv natijasi — bitta crop uchun eng yaqin SKU'lar."""
    sku_id: Optional[str]           # None => noma'lum
    name: Optional[str]
    score: float                    # top-1 o'xshashlik
    margin: float                   # top-1 - top-2
    is_known: bool
    candidates: list[dict] = field(default_factory=list)  # top-k [{sku_id,name,score,...}]
    brand: Optional[str] = None
    # "sku"   — SKU aniq tanildi
    # "brand" — SKU noaniq (masalan 0.5L vs 1L egizaklar), lekin brend aniq
    # "variant" — eng yaqin SKU'ga o'xshaydi, lekin rangi farq qiladi (boshqa ta'm/tur)
    # None    — noma'lum
    level: Optional[str] = None
    negative_score: Optional[float] = None
    # Yorliqdagi yozuv bo'yicha aniqlashtirilgan bo'lsa — asos bo'lgan so'z ("shaftoli", "0.5 L")
    text: Optional[str] = None
    # Yorliqdagi yozuv moslikni rad etgan bo'lsa — qadoqda o'qilgan boshqa so'z ("Florida")
    mismatch: Optional[str] = None
    # Yozuv bo'yicha tekshiruv natijasi: "support:<so'z>" (mahsulotning o'z so'zi o'qildi) yoki
    # "silent" (aniq narsa o'qilmadi). None — tekshirilmadi (ball yuqori yoki namuna yozuvi yo'q).
    checked: Optional[str] = None
    mismatch_kind: Optional[str] = None     # "flavor" yoki "volume"
    family: bool = False                    # katalogdagi yagona umumiy brend yozuvi


@dataclass
class Detection:
    """Detektsiya + tanish natijasi birlashgan holda."""
    box: Box
    match: Optional[Match] = None
    row: int = -1                   # javon qatori indeksi (0 = tepa)
    cluster: int = -1               # noma'lumlar klasteri (-1 = yo'q / shovqin)
    context_fixed: bool = False     # qo'shnilar konteksti bilan tuzatilganmi
    vec: Any = field(default=None, repr=False, compare=False)  # crop embedding (transient)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "box": [round(self.box.x1, 1), round(self.box.y1, 1),
                    round(self.box.x2, 1), round(self.box.y2, 1)],
            "score": round(self.box.score, 3),
            "row": self.row,
        }
        if self.match is not None:
            d["sku_id"] = self.match.sku_id
            d["name"] = self.match.name
            d["similarity"] = round(self.match.score, 3)
            d["margin"] = round(self.match.margin, 3)
            d["is_known"] = self.match.is_known
            if self.match.brand:
                d["brand"] = self.match.brand
            if self.match.level:
                d["level"] = self.match.level
            if self.match.text is not None:
                d["label_text"] = self.match.text
            if self.match.mismatch is not None:
                d["label_mismatch"] = self.match.mismatch
            if self.match.mismatch_kind is not None:
                d["label_mismatch_kind"] = self.match.mismatch_kind
            if self.match.candidates:
                d["top"] = [{"sku_id": c["sku_id"], "score": round(c["score"], 3)}
                            for c in self.match.candidates[:3]]
        if self.cluster >= 0:
            d["cluster"] = self.cluster
        if self.context_fixed:
            d["context_fixed"] = True
        return d


@dataclass
class Gap:
    """Javondagi bo'sh joy (OOS nomzodi)."""
    row: int
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    est_missing_facings: int        # taxminiy nechta mahsulot sig'adi
    emptiness: float = 1.0          # 0..1, rasm bo'yicha "bo'shlik" ishonchi
    status: str = "oos"             # oos | suspect | divider

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "box": [round(self.x1, 1), round(self.y1, 1),
                    round(self.x2, 1), round(self.y2, 1)],
            "width": round(self.width, 1),
            "est_missing_facings": self.est_missing_facings,
            "emptiness": round(self.emptiness, 3),
            "status": self.status,
        }


@dataclass
class AnalysisResult:
    """Bitta javon rasmi uchun to'liq tahlil natijasi."""
    image_path: str
    image_size: tuple[int, int]     # (width, height)
    detections: list[Detection] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    rows: int = 0
    kpi: dict[str, Any] = field(default_factory=dict)
    suspects: list[Gap] = field(default_factory=list)      # tekshiruvdan o'tmagan bo'shliqlar
    dividers: list[Gap] = field(default_factory=list)      # barcha qatorlarda bir x'da: bo'linma?
    background: list[Detection] = field(default_factory=list)  # orqa fon (KPI'ga kirmaydi)
    clusters: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    roi: Any = None                                        # roi.Region (zona bo'yicha tahlilda)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image_path,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "roi": self.roi.to_dict() if self.roi is not None else None,
            "rows": self.rows,
            "num_detections": len(self.detections),
            "num_gaps": len(self.gaps),
            "kpi": self.kpi,
            "detections": [d.to_dict() for d in self.detections],
            "gaps": [g.to_dict() for g in self.gaps],
            "suspect_gaps": [g.to_dict() for g in self.suspects],
            "divider_gaps": [g.to_dict() for g in self.dividers],
            "num_background": len(self.background),
            "unknown_clusters": self.clusters,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }
