# Rasta AI — sinov natijalari (benchmark hisoboti)

Sana: 2026-10-22. Uskuna: RTX 5090, Python 3.10, torch 2.11 (cu128), transformers 5.17.
Barcha raqamlar quyidagi skriptlar bilan **qayta hosil qilinadi** (bo'lim oxirida ko'rsatilgan).

**Qisqa xulosa.** Eski tizim (faqat DINOv2, markaziy kesish) o'rniga gibrid tanish
(DINOv2 + SigLIP2 ansambli + rang + shakl), kalibrlangan chegaralar, javon geometriyasi
va noma'lumlarni klasterlash joriy qilindi. Barcha 9 ta real sinov konfiguratsiyasida
(~680 000 so'rov-baholash) yangi tizim yaxshiroq: closed-set aniqlik **+9…+29 punkt**,
open-set (noma'lumni rad etish bilan) aniqlik **+5…+22 punkt**. Real javon rasmida
SKU aniqligi **94.5% → 98.6%**, butunlay bo'sh javon qatorlarini topish **0% → 94%**.

---

## 1. Savolga javob: klasterlash/tanish RANG bo'yichami yoki SHAKL bo'yichami?

**Javob: hech biri yolg'iz emas — eng kuchlisi embedding + rang + shakl birikmasi.**
Rang — eng kuchli qo'shimcha belgi; shakl yolg'iz deyarli foydasiz, lekin birikmada
qo'shimcha foyda beradi.

### 1a. Noma'lum mahsulotlarni klasterlash (HDBSCAN), real javon crop'lari (RP2K)
20 tajriba × 30 ta tasodifiy noma'lum SKU (~600 crop har tajribada).
`python scripts/benchmark_clustering.py --trials 20 --skus 30`

| Xususiyat | ARI ↑ | NMI ↑ | Tozalik (purity) ↑ |
|---|---|---|---|
| faqat shakl (tomonlar nisbati) | 0.045 | 0.540 | 0.404 |
| faqat rang (HSV gistogramma) | 0.396 | 0.772 | 0.829 |
| rang + shakl | 0.434 | 0.760 | 0.834 |
| faqat embedding (SigLIP2) | 0.388 | 0.776 | 0.828 |
| embedding + rang | 0.518 | 0.815 | 0.927 |
| **embedding + rang + shakl** (standart) | **0.535** | **0.821** | **0.933** |

Tozalik 0.93 — operator klasterga bitta nom berganda crop'larning 93% to'g'ri yorliq oladi.

### 1b. Tanishda rang va shakl hissasi (RP2K, k=5, DINOv2-base)
| | closed-set | open-set (kalibrlangan) |
|---|---|---|
| faqat DINOv2 | 0.699 | 0.575 |
| + rang 0.15 | 0.745 | 0.614 |
| + rang 0.15 + shakl 0.1 | 0.753 | 0.624 |
| + rang 0.3 + shakl 0.1 | **0.784** | **0.652** |

Real misol (cola javoni): eski tizim qora **Chin8** bankasini to'q sariq **Fanta** deb tanirdi —
DINOv2 uchun shakl va kompozitsiya bir xil. Rang deskriptori (qora vs to'q sariq,
Bhattacharyya ≈ 0.1) bu xatoni yo'qotdi.

---

## 2. Yakuniy jadval: ESKI vs YANGI tizim (real datasetlar)

Protokol (few-shot, open-set — Rasta ssenariysi): sinflarning 25% "katalogda yo'q"
(noma'lum) sifatida ushlab qolinadi; katalog = k ta rasm/SKU; 20 tasodifiy bo'linish.
*closed-set* — tanish mahsulotlarda top-1 to'g'riligi; *AUROC* — tanish vs noma'lumni
ajratish; *open-set* — chegara juft seed'larda tanlanib toq seed'larda tekshirilgan
(halol kalibrlash) yakuniy aniqlik.
`python scripts/benchmark_summary.py --seeds 20`

| Dataset | k | closed-set | AUROC | open-set | so'rovlar |
|---|---|---|---|---|---|
| RP2K — real javon crop'lari, 120 SKU | 1 | 0.483 → **0.632** | 0.639 → **0.725** | 0.424 → **0.556** | 48 000 |
| | 3 | 0.641 → **0.779** | 0.680 → **0.787** | 0.525 → **0.672** | 48 000 |
| | 5 | 0.699 → **0.837** | 0.699 → **0.812** | 0.575 → **0.721** | 48 000 |
| Grocery Store — qadoqlar, 31 SKU | 0 (faqat packshot) | 0.559 → **0.849** | 0.598 → **0.681** | 0.434 → **0.652** | 15 620 |
| | 3 | 0.535 → **0.771** | 0.632 → **0.727** | 0.428 → **0.616** | 15 620 |
| | 5 | 0.571 → **0.805** | 0.632 → **0.729** | 0.455 → **0.641** | 15 620 |
| Grocery Store — hammasi, 81 sinf | 0 | 0.658 → **0.744** | 0.655 → 0.652 | 0.526 → **0.578** | 49 700 |
| | 3 | 0.715 → **0.829** | 0.678 → **0.721** | 0.575 → **0.658** | 49 700 |
| | 5 | 0.737 → **0.849** | 0.685 → **0.733** | 0.594 → **0.678** | 49 700 |

ESKI = DINOv2-base, markaziy kesish, CLS, rangsiz.
YANGI = DINOv2-base (crop, CLS+mean) ×0.4 + SigLIP2-base (stretch) ×0.6, rang 0.25, shakl 0.05.

Datasetlar:
- **RP2K** (Peng va boshq., 2020) — do'kon javonlaridan kesilgan mahsulotlar, 2388 SKU;
  120 tasodifiy SKU × 25 rasm ZIP range-so'rovlari bilan olingan (`scripts/fetch_rp2k.py`).
- **Grocery Store Dataset** (Klasson va boshq., WACV 2019) — rasmiy packshot + do'konda
  olingan fotolar (`git clone https://github.com/marcusklasson/GroceryStoreDataset data/external/GroceryStoreDataset`).

### Backbone tanlovi (RP2K, k=5, rangsiz)
| Backbone | closed-set | AUROC | open-set |
|---|---|---|---|
| DINOv2-base (eski) | 0.699 | 0.699 | 0.575 |
| DINOv2-large | 0.712 | 0.685 | 0.576 |
| CLIP ViT-L/14 | 0.615 | 0.672 | 0.526 |
| SigLIP2-base | 0.707 | 0.715 | 0.602 |
| DINOv2-base + CLIP-L | 0.759 | 0.732 | 0.628 |
| **DINOv2-base + SigLIP2-base** | **0.775** | **0.759** | **0.641** |

Rang+shakl qo'shilganda (k=5): SigLIP2-base 0.782, -large 0.814, -so400m 0.826;
ansambl bilan: base **0.838**, large 0.845, so400m 0.854. Standart — base ansambl
(tez); maksimal aniqlik uchun so400m varianti `config.yaml` da izohda.

Muhim topilma — **preprocessing**: HuggingFace standarti baland banka/shishani markazdan
kesib, faqat o'rta qismini qoldiradi. SigLIP uchun "stretch" eng yaxshi, DINOv2 uchun
markaziy kesish — shuning uchun har model uchun alohida preprocessing.

---

## 3. Real javon rasmi — to'liq pipeline (haqiqiy YOLO detektori bilan)

`data/samples/cola_cans_shelf.jpg`, qo'lda belgilangan GT (`data/gt/cola_cans_shelf.gt.json`):
79 quti → 73 baholanadigan mahsulot (65 tanish + 8 katalogda yo'q: Oransoda, Chin8),
4 ta orqa fon bo'yni, 2 ta dublikat (bitta bankaning qisman qutisi).
`python scripts/eval_shelf.py --mode full`

| | Eski | Yangi |
|---|---|---|
| SKU aniqligi | 69/73 = 94.5% | **72/73 = 98.6%** |
| Tanish mahsulotlar | 100% | 100% |
| Noma'lumni rad etish | 50% (4/8) | **87.5% (7/8)** |
| AUROC (tanish vs noma'lum) | — | **1.000** |
| Ikki marta sanalgan qutilar | 2 | **0** |
| Orqa fon "mahsulotlari" (bo'yinlar) | 4 ta noma'lum | **0** (fon deb ajratildi) |
| Qolgan xato | Chin8→Fanta (2), →Coca-Cola, →San Pellegrino | Oransoda→Lemon Soda (1; bir ishlab chiqaruvchining o'xshash dizayni) |

KPI'lar (facings) GT bilan to'liq mos: Coca-Cola 18, Beverino 18, Pepsi 13,
San Pellegrino 8, Fanta 1 (eski tizim: San Pellegrino 3, Coca-Cola 16).

**Operator oqimi** (`scripts/review_unknowns.py`): noma'lumlar 3 ta toza klasterga
bo'lindi (Oransoda ×3, Chin8 old ×2, Chin8 yon ×2) → nom berildi → qayta tahlilda
tanish darajasi **90.4% → 98.6%**, model qayta o'qitilmadi.

**Begona rasmlarda noto'g'ri tanish** (cola katalogi bilan 6 ta boshqa do'kon rasmi,
~470 mahsulot): **0 ta**. cola_fridge'dagi 2 ta Coca-Cola (boshqa mamlakat qadog'i)
to'g'ri tanildi.

---

## 4. Sintetik stress-test (100 javon, 1 976 mahsulot)

Rang-egizaklar, shakl-egizaklar, ~15% katalogda yo'q mahsulot, yorug'lik/oq balans/
blur/shovqin/JPEG buzilishlari; katalog — toza packshot'lar.
`python scripts/benchmark_synthetic.py --n 100` · `--ablation`

| | closed-set | AUROC | open-set | OOS recall | OOS precision |
|---|---|---|---|---|---|
| Eski | 0.573 | 0.576 | 0.472 | 0.684 | 1.00 |
| **Yangi** | **0.886** | **0.951** | **0.812** | **0.875** | **1.00** |

Ablation (har komponent hissasi, DINOv2 asosida): stretch+CLS/mean 0.622 →
+rang 0.3: **0.850** → +shakl 0.1: **0.895** (closed-set).

---

## 5. OOS (bo'sh joy)

### Butunlay bo'sh javon qatorlari (yangi modul: javon chetlari bo'yicha)
40 ta sintetik 4 qavatli javon, har birida bitta qator butunlay bo'sh.
`python scripts/benchmark_oos.py --n 40`

| | Bo'sh qatorlar topildi | Bo'sh katak recall | Noto'g'ri OOS |
|---|---|---|---|
| Faqat qator-asosli (eski) | 0/40 | 0.157 | 0 |
| **+ javon chetlari** | **31/40** (2–4-qatorlar: 31/33) | **0.806** | **0** |

Real rasm (`shelf_partial_empty.jpg`): eski tizim bo'sh pastki javonlarni umuman ko'rmasdi;
yangi — 6 ta bo'sh javon zonasi (~27 facing).

### Boshqa OOS yaxshilanishlari
- Qator boshi/oxiridagi bo'sh kataklar (javon kengligi bo'yicha) — sintetikada recall 0.68 → 0.88.
- Rasm bo'yicha tekshiruv: bo'shliqda qirralar zich bo'lsa (detektor o'tkazib yuborgan
  mahsulot) — "shubhali", OOS emas.
- **Sahna filtri**: mahsulotlarning javon chetida turgan ulushi < 80% bo'lsa (yo'lak, ombor,
  qiya kadr) OOS "shubhali" deb belgilanadi. O'lchov: frontal javonlar 0.90–0.95,
  yo'lak/ombor 0.52–0.69. Natija: ombor va qiya yo'lak rasmlaridagi 9 ta noto'g'ri OOS → 0.

---

## 6. Kalibrlash (chegaralar)

Chegara τ katalog turiga bog'liq ekani aniqlandi (halol kalibrlash):

| Katalog | Kalibrlangan τ | config.yaml |
|---|---|---|
| Real javon crop'lari (RP2K k=1/3/5, cola GT) | 0.74 / 0.76 / 0.765, 0.77 | `sim_threshold: 0.76` |
| Faqat packshot (+16 augmentatsiya) | 0.65 (Grocery, ikkala to'plam) | `sim_threshold_packshot: 0.65` |

Har katalog nuqtasida manba (`src: crop|packshot`) saqlanadi va qaror eng yaqin
nuqtaning manbasiga mos τ bilan qabul qilinadi. O'z ma'lumotingizda:
`python scripts/eval_shelf.py --calibrate --gt data/gt/<rasm>.gt.json`

**Augmentatsiya**: 1 ta packshot → 16 variant: closed-set 0.849 → 0.876 (Grocery qadoqlar).
Lekin 2+ real crop'ni augmentatsiya qilish zararli (begona rasmda 29 ta noto'g'ri
tanish → 0) — shuning uchun faqat yolg'iz rasm uchun yoqiladi.

---

## 7. Tezlik (RTX 5090, ansambl, to'liq pipeline)

| Rasm | Mahsulotlar | Vaqt |
|---|---|---|
| cola_cans_shelf (1280×852) | 73 | ~0.4–0.5 s |
| grocery_shelves (1280×853) | 169 | ~0.7–1.0 s |
| shelf_partial_empty (1280×960) | 27 | ~0.35 s |

Optimallashtirish: Qdrant lokal qidiruvi numpy matritsa ko'paytmasiga almashtirildi
(169 so'rov: 2.2 s → ~10 ms, natijalar aynan bir xil — testda tekshiriladi);
normallashtirish GPU'ga, resize parallel thread'larga (embedding 872 → ~240 ms).

---

## 8. Cheklovlar va keyingi qadamlar

- **Hajm egizaklari** (bir dizayn, 0.5L vs 1L): rang/shakl ajrata olmaydi → tizim
  ularni "brend darajasida" belgilaydi (SKU taxmin qilmaydi, brend ulushi saqlanadi).
  Yechim: katalogga jismoniy balandlik (mm) qo'shib, qatordagi nisbiy o'lcham bilan solishtirish.
- **Eng yuqori javon** butunlay bo'sh va uning ustida javon cheti ko'rinmasa — bo'sh zona yopilmaydi.
- To'liq ustun barcha qavatlarda bo'sh bo'lsa, bo'linma ustunidan farqlab bo'lmaydi →
  "bo'linma?" (ko'k punktir) deb alohida ko'rsatiladi, operator tasdiqlaydi.
- Detektor alohida baholanmadi (SKU-110K test to'plami 13 GB); GT qutilari
  detektor chiqishidan olingan.
- Kalibrlash har yangi do'kon/kamera turi uchun 2–3 ta GT rasmda takrorlanishi tavsiya etiladi.

---

## 9. Detektor (mahsulotlarni ajratib olish) — 2026-10-27

### 9a. SKU-110K test qismi (155 rasm, ~22 000 quti, CPU)
`python scripts/import_sku110k.py --eval 150` → `python scripts/eval_detector.py --gt data/gt_det_sku110k ...`
Natijalar: `outputs/benchmark_detector_sku110k*.json`. *bestF1* — ishonch chegarasi optimal tanlanganda.

| Variant | AP50 | recall | precision | sanash xatosi (MAPE) | bestF1 (@chegara) | CPU ms/rasm |
|---|---|---|---|---|---|---|
| Joriy: YOLO11-S (SKU-110K) @1280 + COCO butilka | 0.867 | 0.895 | 0.836 | 14.6% | 0.873 (@0.34) | 1027 |
| YOLO11-S @640 | 0.867 | 0.892 | 0.853 | 12.5% | 0.879 (@0.34) | 79 |
| YOLO11-N @640 | 0.838 | 0.867 | 0.836 | 11.9% | — | 44 |
| **YOLOv8-L @640** (HF: sbfisher/yolov8l-sku110k) | 0.883 | 0.905 | 0.865 | 12.3% | **0.890** (@0.34) | 378 |
| YOLOv8-L @1280 | 0.890 | 0.921 | 0.829 | 17.1% | — | 1469 |
| Joriy + TTA (640/1280 × aks, WBF) | **0.897** | **0.922** | 0.798 | 20.8% | 0.885 (@0.32) | 1588 |

Xulosalar (SKU-110K domeni — joriy model aynan shu datasetda o'qitilgan; o'zbek
muzlatgichlari uchun yakuniy javon — qo'lda tekshirilgan `data/gt_det`):
- `detector.conf: 0.20` past: ~0.34 da F1 +1 punkt (ortiqcha qutilar kamayadi). Agent
  rasmlarida tekshirmasdan o'zgartirilmadi.
- Kattaroq model (YOLOv8-L) +1.6 AP50 / +1.7 bestF1. TTA eng ko'p mahsulot topadi (recall
  0.922), lekin chegara ko'tarilmasa sanash xatosi oshadi.
- COCO butilka detektori SKU-110K'da hech narsa qo'shmaydi (u muzlatgichdagi shaffof PET uchun).

### 9b. Internetdagi boshqa detektorlar (6 ta agent rasmida, ko'z bilan)
| Model | Natija |
|---|---|
| Grounding DINO base (zero-shot, Apache-2.0) | Yomon: butun javon qatorini bitta quti; zich muzlatgichda 0 ta |
| YOLOE-26L (zero-shot) | Ko'p shishani o'tkazib yuboradi |
| foduucom YOLOv8 shelf | Deyarli hech narsa topmaydi |
| YOLOv12-S SKU-110K (hatuankiet) | Ultralytics 8.4 bilan mos emas |
| D-FINE-XL / RT-DETRv2 SKU-110K (benjamintli) | Model kartasida mAP50 0.48 — sinalmadi |
| SAM 3 (Meta) | HF'da qulflangan (litsenziyani qabul qilish kerak) — sinalmadi |

Umumiy (open-vocabulary) detektorlar zich javonda ixtisoslashgan SKU-110K detektoridan
ancha yomon. Kuchaytirish yo'li — kattaroq modelni o'qitish va agent rasmlarida fine-tune.

### 9c. Agent ilovasi logotipi
SKU-110K detektorlari agent ilovasining chap yuqori burchakdagi yashil logotipini har rasmda
"mahsulot" deb topadi. Filtr (`detector.drop_app_logo`): 40 agent rasmida 18 quti olib
tashlandi — 18/18 haqiqatan logotip, burchakdagi mahsulotlarga tegmadi.

### Litsenziya eslatmasi
SKU-110K — faqat akademik/notijorat. Undan o'qitilgan og'irliklar (joriy model ham) tijoriy
mahsulotda xavfli; o'z agent rasmlarida (52 000 ta, `data/external/aptash`) o'qitish kerak.

---

## 10. Katalog namunasi va bo'lingan qutilar — 2026-11-24

### 10a. Namuna fotosidan mahsulotni ajratib olish
Muammo: katalogga fonli foto (stol ustidagi mahsulot) yuklansa, butun kadr embedding
qilinardi (faqat oq fonli packshot kesilardi). O'lchov — real do'kon javoni (78 mahsulot,
11 ta DENA old tomoni ko'rinadi) va 5 ta real namuna fotosi, "DENA qatorida to'g'ri /
boshqa joyda noto'g'ri tanilgan":

| Katalog | To'g'ri | Noto'g'ri |
|---|---|---|
| Butun kadr, 5 foto (eski) | 4/11 | 0 |
| Mahsulotgacha kesilgan, 5 foto | 9/11 | 4 |
| Kesilgan, faqat OLD tomon (2 foto) | **9/11** | 2 |
| Kesilgan, faqat YON/ORQA tomon (3 foto) | 0/11 | 2 |

Xulosa: (1) kesish tanishni ikki barobardan ko'p oshiradi; (2) yon/orqa tomon fotolari
foyda bermaydi, faqat noto'g'ri tanish qo'shadi — katalogga old tomon kerak; (3) bunday
fotolar packshot kabi qo'shiladi (augmentatsiya + `sim_threshold_packshot`): "crop" deb
(τ 0.76) qo'shilganda 0/11.

Mahsulotni topish (6 ta real namuna fotosi, yaqindan olingan bitta mahsulot):

| Usul | Topdi |
|---|---|
| Javon detektori (SKU-110K YOLO), 320–1280 px masshtablar | 1–2 / 6, beqaror; yon tomon 0 |
| Grounding DINO base | 6/6 (RTX 5090: ~3–7 s/rasm) |
| **Grounding DINO tiny** (`onboarding.locator`) | **6/6** (~1.2–1.8 s/rasm) |

Tanlash qoidasi (`augment.pick_product_box`): kadrning ikki chetiga tegib turgan quti — fon
(tagidagi quti, stol); qolganlaridan yirik, markaziy va ishonchlisi.

### 10b. Bitta mahsulotning ikki qutiga bo'linishi
Sok javonida baland kartonlarning 5 tasi "tepa" va "past" qutilarga bo'linib, ikki marta
sanalgan. 395 ta GT rasm (~27 000 quti), `detector.merge_split_boxes`:

| Qoida | gt_det R / P | f1 R / P | f2 R / P | sku110k R / P | Sok javonida birlashdi |
|---|---|---|---|---|---|
| O'chiq | 0.804 / 0.943 | 0.650 / 0.881 | 0.714 / 0.875 | 0.895 / 0.836 | 0/5 |
| Faqat ustma-ustlik (IoMin ≥ 0.5) | 0.800 / 0.944 | 0.644 / 0.883 | 0.711 / 0.877 | 0.894 / 0.837 | 5/5 |
| **Toza qo'shni o'lchovi bilan** (joriy) | 0.804 / 0.943 | 0.650 / 0.882 | 0.714 / 0.876 | 0.895 / 0.836 | **5/5** |

Faqat ustma-ustlik bo'yicha birlashtirish muzlatgichlarda zarar: orqa qatordagi shisha
oldingisi bilan xuddi shunday ustma-ust tushadi. Joriy qoida bo'lakni to'liq mahsulotdan
ajratadi: har ikki quti shu qatordagi toza (sherigi yo'q) qo'shnilar balandligidan past,
birlashgani esa shu balandlikka teng bo'lsagina birlashtiradi.

### 10c. Sinab ko'rilib, qabul qilinmaganlar
- SigLIP2-so400m ansambli: sintetik 100 javonda 0.553 → 0.534, 4 baravar sekin.
- Qo'shnilar bo'yicha yorliq tarqatish (`context.propagate_blocks`): sintetikada foydasiz
  (bloklar bir xil yoritilgan — "urug'" yo'q); standart holatda o'chiq.
