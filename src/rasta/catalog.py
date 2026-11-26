"""SKU katalogi — Qdrant vektor bazasi ustidagi qatlam.

Bu tizimning yuragi: yangi mahsulot qo'shish = katalogga vektor upsert qilish.
Model qayta o'qitilmaydi.

Tanish ikki bosqichli (retrieve -> re-rank):
  1. DINOv2 vektori bo'yicha Qdrant'dan eng yaqin N nuqta olinadi.
  2. Har nomzod qayta baholanadi:
        fused = w_dino * cos(DINO) + w_color * Bhattacharyya(rang) + w_shape * shape_sim
     va SKU bo'yicha eng yuqori ball olinadi.
  3. Open-set qaror: fused >= τ  VA  margin >= min_margin
                     VA  rang o'xshashligi >= color_min  (rang "veto"si).
     τ eng yaqin nuqtaning manbasiga bog'liq (payload "src"):
       "crop"     -> sim_threshold          (real javon crop'lari; RP2K: 0.74–0.77)
       "packshot" -> sim_threshold_packshot (toza katalog rasmi; Grocery: ~0.65)
     Packshot va javon rasmi orasidagi domen farqi o'xshashlik shkalasini pasaytiradi.

Rang/shakl ma'lumoti har nuqtaning payload'ida saqlanadi ("color", "aspect").
Ular yo'q bo'lgan eski kataloglarda faqat DINO ball ishlatiladi.
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .features import shape_similarity
from .types import Match


@dataclass
class RecognitionParams:
    sim_threshold: float = 0.55
    sim_threshold_packshot: float | None = None   # faqat packshot'dan qurilgan SKU uchun
    min_margin: float = 0.03
    # Top-2 bir brenddan bo'lsa (ta'mlar bir xil yorliqda) — kattaroq margin talab
    # qilinadi, aks holda faqat brend darajasida tanish. None = min_margin.
    brand_margin: float | None = None
    top_k: int = 5
    w_color: float = 0.0
    w_shape: float = 0.0
    color_min: float = 0.0
    color_min_packshot: float | None = None      # packshot'dan qurilgan SKU uchun rang veto'si
    # Rang shu chegaradan past (lekin color_min dan yuqori) bo'lsa — "variant": ko'rinishi shu
    # mahsulotniki, rangi boshqa (bir xil dizayndagi boshqa ta'm). SKU sifatida sanalmaydi,
    # brend ulushiga kiradi. None = o'chiq.
    color_same: float | None = None
    color_same_packshot: float | None = None
    # Vektor (ko'rinish) o'xshashligining o'zi uchun quyi chegara: rang va shakl bonusi
    # ko'rinishi mos kelmagan mahsulotni chegaradan o'tkazib yubormasligi kerak. None = o'chiq.
    embed_min: float | None = None
    embed_min_packshot: float | None = None
    shape_sigma: float = 0.25
    candidates: int = 48           # qayta reyting uchun olinadigan nuqtalar soni
    # "Boshqa mahsulotlar" bazasi: so'rov bazadagi biror mahsulotga katalogdagidan
    # neg_margin dan ko'proq o'xshasa -> noma'lum. None = o'chiq.
    neg_margin: float | None = None              # javon crop'idan qurilgan SKU
    neg_margin_packshot: float | None = None     # internet/packshot rasmidan qurilgan SKU
    neg_exclude: float = 0.85      # katalogga shunchalik o'xshash baza klasterlari chiqariladi

    def limits_for_source(self, source: str | None) -> tuple[float, float, float | None]:
        tau, color_min, neg_margin = self.sim_threshold, self.color_min, self.neg_margin
        if source == "packshot":
            if self.sim_threshold_packshot is not None:
                tau = self.sim_threshold_packshot
            if self.color_min_packshot is not None:
                color_min = self.color_min_packshot
            if self.neg_margin_packshot is not None:
                neg_margin = self.neg_margin_packshot
        return tau, color_min, neg_margin

    def embed_min_for(self, source: str | None) -> float | None:
        if source == "packshot" and self.embed_min_packshot is not None:
            return self.embed_min_packshot
        return self.embed_min

    def color_same_for(self, source: str | None) -> float | None:
        if source == "packshot" and self.color_same_packshot is not None:
            return self.color_same_packshot
        return self.color_same

    @classmethod
    def from_cfg(cls, cfg) -> "RecognitionParams":
        g = lambda k, d: cfg.get(f"recognition.{k}", d)  # noqa: E731
        return cls(
            sim_threshold=g("sim_threshold", 0.55),
            sim_threshold_packshot=g("sim_threshold_packshot", None),
            min_margin=g("min_margin", 0.03),
            brand_margin=g("brand_margin", None),
            top_k=g("top_k", 5),
            w_color=g("w_color", 0.0),
            w_shape=g("w_shape", 0.0),
            color_min=g("color_min", 0.0),
            color_min_packshot=g("color_min_packshot", None),
            color_same=g("color_same", None),
            color_same_packshot=g("color_same_packshot", None),
            embed_min=g("embed_min", None),
            embed_min_packshot=g("embed_min_packshot", None),
            shape_sigma=g("shape_sigma", 0.25),
            candidates=g("candidates", 48),
            neg_margin=g("neg_margin", None),
            neg_margin_packshot=g("neg_margin_packshot", None),
            neg_exclude=g("neg_exclude", 0.85),
        )


_DIST = {"cosine": "COSINE", "dot": "DOT", "euclid": "EUCLID"}


class NegativeBank:
    """Boshqa mahsulotlar bazasi (scripts/build_negative_bank.py).

    Kichik katalogda har qanday shisha eng yaqin SKU'ga "yopishadi". Bu baza —
    foto arxividan olingan minglab real mahsulot klasterlari. So'rov bazadagi
    biror mahsulotga katalogdagidan ancha ko'proq o'xshasa — u katalogdagi mahsulot
    emas. Katalogdagi SKU'ning o'ziga o'xshash klasterlar chiqariladi (neg_exclude).
    """

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.emb = z["emb"].astype(np.float32)
        self.color = z["color"].astype(np.float32)
        self.aspect = z["aspect"].astype(np.float32)
        self.embedder = str(z["embedder"])

    @staticmethod
    def fused(Q, QC, QA, E, C, A, params):
        """(len(Q), len(E)) gibrid o'xshashlik — katalogdagi bilan bir xil formula."""
        wc, ws = params.w_color, params.w_shape
        S = Q @ E.T
        if QC is not None and wc > 0:
            S = (1 - wc - ws) * S + wc * np.clip(QC @ C.T, 0, 1)
        else:
            S = (1 - ws) * S
        if ws > 0 and QA is not None:
            d = np.log(np.asarray(QA, np.float32)[:, None].clip(1e-3) / A[None, :].clip(1e-3))
            S = S + ws * np.exp(-(d * d) / (2 * params.shape_sigma ** 2))
        elif ws > 0:
            S = S + ws
        return S


class SkuCatalog:
    def __init__(
        self,
        dim: int,
        path: str = "data/qdrant",
        url: Optional[str] = None,
        collection: str = "rasta_skus",
        distance: str = "cosine",
        client=None,
        recreate_on_mismatch: bool = False,
        embedder: Optional[str] = None,
    ):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.dim = dim
        self.distance = distance
        self.is_local = url is None
        # Lokal katalog yonida: qaysi embedder bilan qurilgani (rasta_meta.json)
        self._meta_path = (Path(path) / "rasta_meta.json"
                           if url is None and client is None and path != ":memory:" else None)
        self.embedder = embedder
        self.stale: Optional[str] = None   # eski embedder nomi, agar katalog unga mos bo'lsa
        self._cache = None        # lokal rejim: (vektorlar, payloadlar) xotirada
        self._pending_replacement = None
        self.negatives = None     # NegativeBank (ixtiyoriy)
        self._neg_keep = None     # katalogga o'xshamagan baza klasterlari maskasi
        if client is not None:
            self.client = client          # mavjud klientni qayta ishlatish (sweep/test)
        elif url:
            self.client = QdrantClient(url=url)
        elif path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
            import atexit
            # interpreter yopilayotganda Qdrant lock'ini toza bo'shatish
            atexit.register(self._close)

        if not self.client.collection_exists(collection):
            self._create()
        else:
            existing_dim = self._existing_dim()
            if existing_dim is not None and existing_dim != dim and recreate_on_mismatch:
                self.reset()   # vaqtinchalik/demo kataloglar uchun
            elif existing_dim is not None and existing_dim != dim:
                raise ValueError(
                    f"Katalog '{collection}' {existing_dim}-o'lchamli vektorlar bilan "
                    f"yaratilgan, joriy embedder esa {dim}-o'lchamli. Embedder sozlamasi "
                    f"(model/pooling) o'zgargan. Katalogni qayta indekslang: "
                    f"scripts/add_product.py --reset ... yoki boshqa collection nomini bering."
                )
            self._check_embedder(recreate_on_mismatch)

    # O'lcham bir xil, lekin boshqa model/preprocessing bilan qurilgan vektorlar jim-jit
    # noto'g'ri tanish beradi — shuning uchun embedder imzosi katalog yonida saqlanadi.
    def _read_meta(self) -> dict:
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError):
            return {}

    def _write_meta(self) -> None:
        if self._meta_path is None or not self.embedder:
            return
        meta = self._read_meta()
        meta[self.collection] = {"embedder": self.embedder, "dim": self.dim}
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def _check_embedder(self, recreate_on_mismatch: bool) -> None:
        if self._meta_path is None or not self.embedder:
            return
        old = self._read_meta().get(self.collection, {}).get("embedder")
        if old is None:
            self._write_meta()          # eski katalog: joriy embedder bilan belgilanadi
        elif old != self.embedder:
            if recreate_on_mismatch:
                self.reset()
                return
            # Xato bermaymiz — aks holda qayta indekslash ham ishga tushmasdi
            self.stale = old
            print(f"[!] Katalog '{self.collection}' boshqa embedder bilan qurilgan ({old}). "
                  f"Tanish ishonchsiz — qayta indekslang: POST /api/reindex yoki "
                  f"scripts/reindex_catalog.py")

    def mark_current(self) -> None:
        """To'liq qayta indekslashdan keyin: katalog joriy embedder bilan mos."""
        self.stale = None
        self._write_meta()

    def _close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _existing_dim(self) -> Optional[int]:
        try:
            info = self.client.get_collection(self.collection)
            vec = info.config.params.vectors
            return int(vec.size)
        except Exception:
            return None

    def _create(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        dist = getattr(Distance, _DIST.get(self.distance, "COSINE"))
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=dist),
        )
        self._write_meta()

    def reset(self) -> None:
        """Kolleksiyani bo'shatadi (demo/qayta indekslash uchun)."""
        self._cache = None
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._create()

    # ---- Yozish (few-shot onboarding) ----------------------------------

    def add_sku(
        self,
        sku_id: str,
        name: str,
        vectors: np.ndarray,
        extra: Optional[dict] = None,
        colors: Optional[np.ndarray] = None,
        aspects: Optional[list[float]] = None,
    ) -> int:
        """Bitta SKU uchun bir nechta vektorni katalogga qo'shadi.

        Har bir vektor alohida nuqta bo'lib saqlanadi (bir SKU = ko'p ko'rinish:
        turli do'kon, yorug'lik, burchak, eski/yangi upakovka).
        colors/aspects — rang va shakl deskriptorlari (gibrid tanish uchun).
        """
        from qdrant_client.models import PointStruct

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        points = []
        for i, v in enumerate(vectors):
            payload = {"sku_id": sku_id, "name": name}
            if extra:
                payload.update(extra)
            if colors is not None:
                payload["color"] = [round(float(x), 5) for x in colors[i]]
            if aspects is not None:
                payload["aspect"] = round(float(aspects[i]), 4)
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=payload)
            )
        if self._pending_replacement is not None:
            pending_sku, pending_points = self._pending_replacement
            if sku_id != pending_sku:
                raise ValueError("Replacement can only write its own SKU")
            pending_points.extend(points)
        else:
            self._insert_points(points)
        return len(points)

    def _insert_points(self, points) -> None:
        try:
            for k in range(0, len(points), 256):
                self.client.upsert(collection_name=self.collection, points=points[k:k + 256])
        except Exception:
            self._delete_points([p.id for p in points])
            raise
        finally:
            self._cache = None

    def _delete_points(self, ids) -> None:
        if ids:
            self.client.delete(collection_name=self.collection, points_selector=ids)
            self._cache = None

    @contextmanager
    def replace_sku(self, sku_id: str):
        """Stage replacement points before changing the stored SKU."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if self._pending_replacement is not None:
            raise RuntimeError("A catalog replacement is already active")
        sku_filter = Filter(must=[FieldCondition(key="sku_id", match=MatchValue(value=sku_id))])
        old_ids = [p.id for p in self._scroll(sku_filter)]
        points = []
        self._pending_replacement = (sku_id, points)
        try:
            yield
            if not points:
                raise ValueError("A SKU replacement must contain at least one vector")
            self._insert_points(points)
            self._delete_points(old_ids)
        finally:
            self._pending_replacement = None

    def delete_sku(self, sku_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="sku_id", match=MatchValue(value=sku_id))])),
        )
        self._cache = None

    def update_sku(self, sku_id: str, **fields) -> None:
        """SKU payload maydonlarini (name, brand) vektorlarni qayta hisoblamasdan yangilash."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.set_payload(
            collection_name=self.collection, payload=fields,
            points=Filter(must=[FieldCondition(key="sku_id", match=MatchValue(value=sku_id))]),
        )
        self._cache = None

    def count(self) -> int:
        return self.client.count(self.collection).count

    def _scroll(self, scroll_filter=None):
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, limit=256, offset=offset, scroll_filter=scroll_filter,
                with_payload=True, with_vectors=False,
            )
            yield from points
            if offset is None:
                break

    def list_skus(self) -> dict[str, int]:
        """SKU_id -> saqlangan vektorlar soni."""
        from collections import Counter

        return dict(Counter(p.payload.get("sku_id", "?") for p in self._scroll()))

    def sku_info(self) -> dict[str, dict]:
        """Return names, catalog metadata, and vector counts by SKU."""
        out: dict[str, dict] = {}
        for p in self._scroll():
            pl = p.payload or {}
            sku = pl.get("sku_id", "?")
            r = out.setdefault(sku, {"name": pl.get("name", sku), "brand": pl.get("brand"),
                                     "category": pl.get("category"), "barcode": pl.get("barcode"),
                                     "count": 0, "words": []})
            r["count"] += 1
            for w in pl.get("words") or ():       # namuna rasmlarida o'qilgan yozuv (rasta.ocr)
                if w not in r["words"]:
                    r["words"].append(w)
        return out

    def brand_map(self) -> dict[str, str]:
        """SKU_id -> brend (payload'da 'brand' bo'lsa)."""
        out: dict[str, str] = {}
        for p in self._scroll():
            b = p.payload.get("brand")
            if b:
                out[p.payload.get("sku_id", "?")] = b
        return out

    # ---- O'qish (tanish) -----------------------------------------------

    def search(
        self,
        vectors: np.ndarray,
        params: Optional[RecognitionParams] = None,
        colors: Optional[np.ndarray] = None,
        aspects: Optional[list[float]] = None,
        allowed: Optional[set] = None,
        **legacy,
    ) -> list[Match]:
        """Har bir vektor uchun eng yaqin SKU'ni gibrid + open-set qoida bilan qaytaradi.

        allowed — qidiruv doirasi: faqat shu SKU'lar orasidan izlanadi (operator javonda
        borini tanlagan). None — butun katalog. Bo'sh to'plam — hech narsa tanilmaydi."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, QueryRequest

        if params is None:
            params = RecognitionParams(**legacy) if legacy else RecognitionParams()
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if len(vectors) == 0:
            return []

        limit = max(params.candidates, params.top_k, 2)
        Qn = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-6)
        neg = self._negative_scores(Qn, colors, aspects, params)
        if allowed is not None and not allowed:
            return [Match(None, None, 0.0, 0.0, False, []) for _ in range(len(vectors))]
        if self.is_local:
            return self._search_numpy(vectors, params, colors, aspects, limit, neg, allowed)
        scope = None if allowed is None else Filter(
            must=[FieldCondition(key="sku_id", match=MatchAny(any=sorted(allowed)))])
        responses = []
        for k in range(0, len(vectors), 64):
            reqs = [QueryRequest(query=v.tolist(), filter=scope, limit=limit, with_payload=True)
                    for v in vectors[k:k + 64]]
            responses.extend(self.client.query_batch_points(self.collection, requests=reqs))

        matches = []
        for i, resp in enumerate(responses):
            c = colors[i] if colors is not None else None
            a = aspects[i] if aspects is not None else None
            matches.append(self.decide(resp.points, params, c, a,
                                       None if neg is None else float(neg[i])))
        return matches

    def _load_cache(self):
        """Lokal rejim: butun katalogni bir marta xotiraga yuklash.

        Qdrant lokal rejimi har so'rovda payload'larni Python'da qayta tahlil
        qiladi (169 so'rov ~2 s). Matritsa ko'paytmasi bilan bu ~ms darajada.
        """
        if self._cache is None:
            vecs, pays = [], []
            offset = None
            while True:
                pts, offset = self.client.scroll(
                    collection_name=self.collection, limit=512, offset=offset,
                    with_payload=True, with_vectors=True)
                for p in pts:
                    vecs.append(np.asarray(p.vector, dtype=np.float32))
                    pays.append(p.payload or {})
                if offset is None:
                    break
            G = np.stack(vecs) if vecs else np.zeros((0, self.dim), np.float32)
            if len(G):
                G /= np.linalg.norm(G, axis=1, keepdims=True).clip(1e-6)
            self._cache = (G, pays)
            self._neg_keep = None
        return self._cache

    def _negative_scores(self, Q, colors, aspects, params) -> Optional[np.ndarray]:
        """Har so'rov uchun bazadagi eng o'xshash boshqa mahsulot balli."""
        if self.negatives is None or (params.neg_margin is None
                                      and params.neg_margin_packshot is None):
            return None
        nb = self.negatives
        G, pays = self._load_cache()
        if self._neg_keep is None:
            if len(G):
                zero = np.zeros(nb.color.shape[1], np.float32)
                GC = np.stack([np.asarray(p.get("color") or zero, np.float32) for p in pays])
                GA = np.array([float(p.get("aspect") or 1.0) for p in pays], np.float32)
                sim = NegativeBank.fused(nb.emb, nb.color, nb.aspect, G, GC, GA, params).max(1)
                self._neg_keep = sim < params.neg_exclude
            else:
                self._neg_keep = np.ones(len(nb.emb), bool)
        keep = self._neg_keep
        if not keep.any():
            return None
        return NegativeBank.fused(Q, colors, aspects, nb.emb[keep], nb.color[keep],
                                  nb.aspect[keep], params).max(1)

    def _search_numpy(self, vectors, params, colors, aspects, limit, neg=None, allowed=None) -> list[Match]:
        from types import SimpleNamespace

        G, pays = self._load_cache()
        if allowed is not None and len(G):
            keep = [j for j, p in enumerate(pays) if p.get("sku_id") in allowed]
            G, pays = G[keep], [pays[j] for j in keep]
        if len(G) == 0:
            return [Match(None, None, 0.0, 0.0, False, []) for _ in range(len(vectors))]
        Q = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-6)
        S = Q @ G.T
        k = min(limit, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        out = []
        for i in range(len(Q)):
            hits = [SimpleNamespace(score=float(S[i, j]), payload=pays[j]) for j in top[i]]
            c = colors[i] if colors is not None else None
            a = aspects[i] if aspects is not None else None
            out.append(self.decide(hits, params, c, a, None if neg is None else float(neg[i])))
        return out

    @staticmethod
    def score_hits(hits, params: RecognitionParams, color=None, aspect=None) -> list[dict]:
        """Nomzodlarni qayta baholaydi; SKU bo'yicha eng yaxshisini, kamayish tartibida qaytaradi."""
        best: dict[str, dict] = {}
        for h in hits:
            p = h.payload or {}
            sku = p.get("sku_id", "?")
            dino = float(h.score)
            wc, ws = params.w_color, params.w_shape
            csim = 1.0
            ssim = 1.0
            # rang o'xshashligi og'irlik 0 bo'lsa ham hisoblanadi (veto uchun kerak)
            if color is not None and p.get("color") is not None:
                csim = float(np.clip(np.dot(color, np.asarray(p["color"], dtype=np.float32)), 0, 1))
            else:
                wc = 0.0
            if aspect is not None and ws > 0 and p.get("aspect"):
                ssim = shape_similarity(aspect, float(p["aspect"]), params.shape_sigma)
            else:
                ws = 0.0
            fused = (1.0 - wc - ws) * dino + wc * csim + ws * ssim
            if sku not in best or fused > best[sku]["score"]:
                best[sku] = {
                    "sku_id": sku,
                    "name": p.get("name", sku),
                    "brand": p.get("brand"),
                    "score": fused,
                    "src": p.get("src", "crop"),
                    "dino": dino,
                    "color": csim,
                    "shape": ssim,
                }
        return sorted(best.values(), key=lambda d: d["score"], reverse=True)

    @classmethod
    def decide(cls, hits, params: RecognitionParams, color=None, aspect=None,
               neg_score: Optional[float] = None) -> Match:
        if not hits:
            return Match(None, None, 0.0, 0.0, False, [])
        ranked = cls.score_hits(hits, params, color, aspect)
        top1 = ranked[0]
        score1 = top1["score"]
        score2 = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = score1 - score2

        tau, cmin, dneg = params.limits_for_source(top1.get("src"))
        # bazadagi biror boshqa mahsulotga ancha ko'proq o'xshasa — noma'lum
        neg_ok = neg_score is None or dneg is None or score1 >= neg_score - dneg
        same_brand = (len(ranked) > 1 and top1.get("brand") is not None
                      and ranked[1].get("brand") == top1.get("brand"))
        need_margin = params.min_margin
        if same_brand and params.brand_margin is not None:
            need_margin = max(need_margin, params.brand_margin)
        csame = params.color_same_for(top1.get("src"))
        emin = params.embed_min_for(top1.get("src"))
        emb_ok = emin is None or top1["dino"] >= emin
        passes = (
            score1 >= tau
            and emb_ok
            and margin >= need_margin
            and top1["color"] >= cmin
            and neg_ok
        )
        # Hammasi mos, faqat rang "aynan shu mahsulot" darajasida emas: bir xil dizayndagi
        # boshqa ta'm/tur (yashil kivi o'rniga sariq nok). SKU deb sanalmaydi, lekin
        # operatorga "o'xshaydi, rangi farq qiladi" deb ko'rsatiladi.
        variant = passes and csame is not None and color is not None and top1["color"] < csame
        is_known = passes and not variant
        cands = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                 for r in ranked[: params.top_k]]
        if neg_score is not None and cands:
            cands[0]["neg"] = round(float(neg_score), 4)
        # Ierarxik qaror: SKU noaniq (margin kichik), lekin top-2 bir brenddan ->
        # brend darajasida tanildi (brend ulushi KPI'si uchun yetarli, SKU taxmin qilinmaydi).
        brand_only = (
            not is_known
            and score1 >= tau
            and emb_ok
            and top1["color"] >= cmin
            and neg_ok
            and same_brand
        )
        level = "sku" if is_known else ("variant" if variant else ("brand" if brand_only else None))
        return Match(
            sku_id=top1["sku_id"] if is_known else None,
            name=top1["name"] if is_known else None,
            score=score1,
            margin=margin,
            is_known=is_known,
            candidates=cands,
            brand=top1.get("brand") if (is_known or brand_only or variant) else None,
            level=level,
            negative_score=neg_score,
        )


def embedder_signature(cfg) -> str:
    """Vektor fazosini belgilaydigan barcha embedder sozlamalari bitta satrda."""
    keys = ("model", "weights", "image_size", "resize_mode", "pooling", "tta_flip")
    return "; ".join(f"{k}={cfg.get('embedder.' + k)}" for k in keys)


def build_catalog(cfg, dim: int) -> SkuCatalog:
    from .config import resolve_path

    url = cfg.get("catalog.url", None)
    raw = cfg.get("catalog.path", "data/qdrant")
    path = raw if raw == ":memory:" else str(resolve_path(raw))
    cat = SkuCatalog(
        dim=dim,
        path=path,
        url=url,
        collection=cfg.get("catalog.collection", "rasta_skus"),
        distance=cfg.get("catalog.distance", "cosine"),
        recreate_on_mismatch=cfg.get("catalog.recreate_on_mismatch", False),
        embedder=embedder_signature(cfg),
    )
    neg_path = cfg.get("recognition.negatives")
    if neg_path:
        p = resolve_path(neg_path)
        if p.exists():
            nb = NegativeBank(p)
            if nb.emb.shape[1] == dim and nb.embedder == str(cfg.get("embedder.model")):
                cat.negatives = nb
            else:
                print(f"[!] {p.name}: boshqa embedder bilan qurilgan — o'chirildi "
                      f"(scripts/build_negative_bank.py bilan qayta quring)")
    return cat
