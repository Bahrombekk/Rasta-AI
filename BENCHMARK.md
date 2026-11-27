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

---

## 11. Real javonda qatorlar, bo'linmalar, detektor masshtabi — 2026-11-24/25

Boshlang'ich holat (real do'kon javoni, 3 qavat, sok kartonlari): tizim 8 ta "qator" va 16 ta
"ehtimoliy bo'linma" ko'rsatdi, 10 ta yirik kartondan 7 tasini topdi, 1 litrli karton
katalogi bilan undan ikki baravar kichik qutilarni "tanidi".

### 11a. Detektor: ikki masshtab (640 + 1280, WBF) — joriy standart
`python scripts/eval_detector.py --gt data/<to'plam> --all --variant "..."`; recall / precision / F1:

| Variant | gt_det (40, qo'lda) | gt_det_f1 (100) | gt_det_f2 (100) | SKU-110K (155) |
|---|---|---|---|---|
| Bitta masshtab 1280 (eski) | 0.804 / 0.943 / 0.868 | 0.650 / 0.882 / 0.749 | 0.714 / 0.876 / 0.787 | 0.895 / 0.836 / 0.864 |
| **640 + 1280, WBF** | **0.900 / 0.959 / 0.928** | **0.683 / 0.877 / 0.768** | **0.744 / 0.879 / 0.806** | 0.917 / 0.816 / 0.863 |
| 640 + 1280, "to'ldirish" (asosiy o'tish qutilariga tegmasdan) | 0.883 / 0.936 / 0.909 | 0.672 / 0.862 / 0.755 | 0.732 / 0.863 / 0.792 | 0.912 / 0.821 / 0.864 |
| Bitta masshtab 960 | 0.730 / 0.886 / 0.800 | 0.665 / 0.909 / 0.768 | 0.726 / 0.916 / 0.810 | 0.901 / 0.864 / 0.882 |
| YOLOv8-L SKU-110K (sbfisher), 640 + 1280 | 0.822 / 0.656 / 0.730 | 0.691 / 0.844 / 0.760 | — | 0.935 / 0.815 / 0.871 |

Yaqindan olingan yirik mahsulot 1280 da o'tkazib yuboriladi, 640 da topiladi. Narxi: detektsiya
vaqti ~1.4 baravar (RTX 5090). Cola bankalari javonida (GT 73) WBF 2 ta qutini yo'qotadi
(73 → 71) — zich mayda mahsulotda dag'al masshtab qutilarni siljitadi; agent rasmlaridagi
yutuq undan ancha katta. YOLOv8-L SKU-110K'da yaxshiroq (9a), lekin agent rasmlarida ancha yomon.

Zona rejimi (4 nuqtali zona kattalashtirilib tahlil qilinadi): 640-o'tish kartonlarni to'liq,
1280-o'tish esa bo'laklab topadi; `drop_group_boxes` to'liq qutini "ikki mahsulotli guruh" deb
olib tashlab, bo'laklarni qoldirardi (Bliss qatorida 11 tadan 8 tasi to'liq). Endi quti faqat
qismlari yetarlicha ishonchli bo'lsa (o'rtacha bal >= 0.5 x quti bali) guruh hisoblanadi:
zonada 11/11 to'liq; gt_det F1 0.928 -> 0.926, f1 0.768 -> 0.770, f2 0.806 -> 0.807.
Shu zonada: bo'sh o'rin 17 (eski prod) -> 0, bo'linma 0, tekshirish kerak 1.
Boshqa qat'iy masshtab to'plamlari (512+1280, 512+896+1280, 512+960, 384+640+1280) gt_det'da
640+1280 dan past chiqdi (F1 0.83-0.905).

### 11b. Qatorlar va bo'sh joy mantig'i
49 ta real rasm (8 namuna + foydalanuvchi javoni + 40 agent rasmi), eski → yangi:

| | Mahsulot | Fon (kesilgan) | OOS zonalari | Tekshirish kerak | Bo'linma |
|---|---|---|---|---|---|
| Jami | 1627 → 1738 | 9 → 47 | 17 → 11 | 85 → 102 | **255 → 30** |
| Foydalanuvchi javoni | 75 → 65 | 0 → 10 | 0 → 1 | 2 → 2 | 16 → 0 |
| shelf_partial_empty (haqiqiy bo'sh javonlar) | 28 → 36 | 0 | 10 → 8 | 0 → 2 | 5 → 3 |

O'zgarishlar: bo'linma — barcha qatorlarda joyi VA kengligi bir xil bo'shliq (x bo'yicha IoU);
kadr cheti kesgan qator — fon; javon chetida turmagan "qator"dan bo'sh joy hisoblanmaydi;
qo'shni qatorning baland mahsuloti joyni band qiladi; kadr cheti kesgan mahsulot yonidagi
oraliq (ramka) bo'sh joy emas; bo'sh javon zonasi o'sha joydagi mahsulot bo'yiga mos bo'lishi
kerak. Regressiya: `benchmark_oos --n 40` 31/40, recall 0.806, noto'g'ri 0 (o'zgarmadi);
`benchmark_synthetic --n 100` aniqlik 0.5531 (o'zgarmadi), OOS recall 0.875 → 0.882, precision 1.0.

### 11c. O'lcham bo'yicha veto va "boshqa mahsulotlar" bazasi
Bir rasmda bir SKU facing'lari bir bo'yda bo'ladi: ko'pchilikdan 0.6 baravardan kichik/katta
VA chegaraga yaqin mos kelgan quti noma'lumga qaytariladi. Cola GT (bitta masshtab): 72/73 →
73/73 (Oransoda → Lemon Soda xatosi yo'qoldi).

`models/negatives/aptash_v1.npz` prodda yo'q edi (git'ga kiritilmagan). Real javon, bitta SKU'li
katalog, DENA qatorida to'g'ri / boshqa joyda noto'g'ri tanilgan:

| Katalog | Bazasiz (eski prod) | Baza bilan |
|---|---|---|
| 2 ta old tomon fotosi | 10/11, noto'g'ri 2 | 10/11, noto'g'ri **0** |
| 5 foto (old + yon) | 10/11, noto'g'ri 6 | 10/11, noto'g'ri 2 |

### 11d. Embedding modeli nomzodlari (almashtirilmadi)
`DINOv2-base ×0.4 + X ×0.6`, rang 0.25; closed / AUROC / open-set, 20 bo'linish:

| X | RP2K k=1 | RP2K k=5 | Grocery k=0 | Grocery k=5 | O'rtacha open (6 sozlama) |
|---|---|---|---|---|---|
| SigLIP2-base-224 (joriy) | 0.632 / 0.725 / 0.556 | 0.837 / 0.812 / 0.721 | 0.849 / 0.681 / 0.652 | 0.805 / 0.729 / 0.641 | 0.643 |
| SigLIP2-base-384 | 0.619 / 0.724 / 0.549 | 0.836 / 0.815 / 0.722 | 0.904 / 0.717 / 0.694 | 0.849 / 0.758 / 0.676 | 0.660 |
| SigLIP2-base-512 | 0.621 / 0.727 / 0.549 | 0.836 / 0.818 / 0.723 | 0.894 / 0.718 / 0.704 | 0.856 / 0.767 / 0.682 | 0.665 |
| SigLIP2-large-256 | 0.651 / 0.732 / 0.567 | 0.843 / 0.813 / 0.727 | 0.858 / 0.701 / 0.666 | 0.832 / 0.741 / 0.653 | 0.656 |
| SigLIP2-so400m-224 | 0.666 / 0.736 / 0.583 | 0.853 / 0.816 / 0.731 | 0.876 / 0.727 / 0.693 | 0.865 / 0.779 / 0.701 | **0.679** |
| DINOv2-reg-base (DINOv2 o'rnida) | 0.630 / 0.725 / 0.550 | 0.835 / 0.809 / 0.722 | 0.800 / 0.656 / 0.608 | 0.792 / 0.720 / 0.621 | 0.629 |
| CLIP ViT-L/14 | 0.634 / 0.712 / 0.555 | 0.825 / 0.788 / 0.699 | 0.779 / 0.670 / 0.608 | 0.778 / 0.738 / 0.628 | 0.626 |

DINOv3 — HuggingFace'da yopiq (litsenziya tasdig'i kerak), sinalmadi. so400m real datasetlarda
+3.6 punkt (sintetikada −1.9, 10c). Foydalanuvchining real javonida esa farq yo'q: DENA va
qolgan qutilarni ajratish AUROC 0.995 (joriy) → 1.000. Almashtirish narxi: 3–5 baravar sekin
embedding, chegaralarni qayta kalibrlash, negativ bazani qayta qurish, barcha kataloglarni qayta
indekslash. Xulosa: hozir almashtirilmaydi; server GPU kuchaytirilganda birinchi nomzod — so400m,
packshot-katalog uchun — SigLIP2-base-384/512.

### 11d. Bir xil dizayndagi boshqa ta'm va rang hisobiga "tanish" — 2026-11-25
Real sinov: foydalanuvchining muzlatgichi (2 kadr), katalogda `micco KIWI` (2 foto, betonda)
va `DENA` (2 foto). Boshlang'ich holat: micco nok (PEAR) "KIWI" deb, "Наш сад" nok shishasi
"DENA" deb tanilgan.

| Quti | umumiy ball | vektor | rang | negativ farqi | Eski qaror | Yangi qaror |
|---|---|---|---|---|---|---|
| micco KIWI (orqasi bilan), 2 kadr | 0.727 / 0.710 | ~0.63 | 0.90 / 0.91 | 0.06 / 0.08 | tanildi | tanildi |
| micco PEAR | 0.709 | ~0.69 | 0.70 | 0.11 | **KIWI** | variant (rangi farq qiladi) |
| micco ORANGE & MANGO | 0.730 | ~0.76 | 0.58 | 0.09 | noma'lum (rang) | noma'lum (rang) |
| "Наш сад" nok -> DENA, 2 kadr | 0.653 / 0.652 | 0.576 / 0.580 | 0.81 | 0.11 / 0.10 | **DENA** | noma'lum (kuchsiz) |
| DENA javoni, haqiqiy 10 ta | 0.68–0.94 | 0.62–0.94 | 0.78–0.98 | <= 0.06 | tanildi | tanildi |

- `recognition.color_same_packshot: 0.76` — hammasi mos, lekin rang 0.60–0.76 oralig'ida bo'lsa
  "variant": SKU sifatida sanalmaydi, brend ulushiga kiradi, interfeysda sariq uzuq ramka va
  "o'xshaydi, lekin rangi farq qiladi" izohi. Qo'shniga qarab SKU'ga aylantirilmaydi.
- `recognition.embed_min_packshot: 0.60` — vektor o'xshashligining o'zi uchun quyi chegara:
  rang (0.25) va shakl (0.05) bonusi ko'rinishi mos kelmagan qutini chegaradan o'tkazmaydi.
- Regressiya: `benchmark_synthetic --n 100` aniqlik 0.5531 -> 0.5582 (noma'lumni rad etish
  0.574 -> 0.627, ma'lumni tanish 0.549 -> 0.544); cola javoni 71/73 (o'zgarmadi); DENA javoni
  10/11, xato 0 (o'zgarmadi).
- Rad etildi: birlashtirishdan keyingi minimal ball (`min_score`) — aks/yorliq bo'laklarini
  olib tashlaydi, lekin agent rasmlarida recall 0.897 -> 0.827 (0.12 da); bitta o'tishda past
  ishonch bilan topilgan qutilarning ko'pi haqiqiy mahsulot. Segmentatsiya (SAM niqobi):
  DENA javonida ajralish +0.01–0.03, tahlilga 3–5 s qo'shadi — qo'shilmadi.
- Chegara: 2 ta muzlatgich kadri + 1 javon; chegaralar (0.76, 0.60) shu ma'lumotga sozlangan.

### 11e. Katalogga moslashish va "chuqurroq solishtirish" (qo'shilmadi) — 2026-11-25
Ikki g'oya o'lchandi; ikkalasi ham hozircha tizimga qo'shilmadi.

**Kattalashtirib, qismma-qism solishtirish** (muzlatgich, Ays shaftoli namunasi, boshqa kadr):

| Usul | shaftoli (old) | shaftoli (burilgan) | o'rmon mevalari |
|---|---|---|---|
| butun shisha (joriy) | 0.88 | 0.81–0.85 | 0.87 |
| uzun o'q bo'yicha 3 kvadrat bo'lak, yorliq bo'lagi | 0.88 | 0.86–0.87 | 0.88 |
| DINOv2 patch-tokenlarini zich moslash (448 px) | 0.83 | 0.81–0.82 | 0.84 |

Bir xil shishaning burilishi ko'rinishni ta'm rasmidan ko'proq o'zgartiradi — ajralish yo'q.
Ishlagani: ikkala ta'm katalogda (har biridan 1 quti) — o'rmon mevalari 0.98 ga 0.87 bilan to'g'ri,
burilgan 3 shisha brend darajasida; shaftoliga 2 old + 2 burilgan quti berilganda hammasi SKU
darajasida (0.97–0.98 ga 0.87–0.89).

**Katalogga moslashish** — katalog namunalaridan SKU ichidagi og'ishlar (namuna − SKU o'rtachasi)
fazosi topiladi va uning vazni pasaytiriladi (W = (αS_w + (1−α)I)^−1/2, α = 0.9; hisoblash ~1 s
CPU, tahlilga +10 ms).

| RP2K (real crop, rang+shakl bilan, 20 urug') | yopiq: joriy -> moslashgan | ochiq (holdout) |
|---|---|---|
| 5 SKU × 3 namuna | 0.977 -> 0.984 | 0.870 -> 0.884 |
| 10 SKU × 3 | 0.938 -> 0.958 | 0.846 -> 0.867 |
| 20 SKU × 3 | 0.898 -> 0.936 | 0.791 -> 0.823 |
| 45 SKU × 5 | 0.880 -> 0.932 | 0.761 -> 0.813 |
| 90 SKU × 5 | 0.839 -> 0.902 | 0.720 -> 0.786 |

- "Umumiy" variant (kovariatsiya boshqa mahsulotlardan): shu to'plamning boshqa 40 SKU'sidan —
  k=1 da 0.594 -> 0.639; aptash arxivi klasterlaridan (85 000 crop) — RP2K'da **yomonlashtiradi**
  (k=1: 0.557 -> 0.49–0.53). Ya'ni domenlar orasida o'tmaydi.
- Chiziqli sinflagich (logistik regressiya): k=5 da ochiq 0.753 — moslashishdan yaxshi emas,
  kam namunada buziladi.
- To'liq tizimda (sinov bayrog'i bilan): cola javoni 73/73 va 71/73 — o'zgarmadi; DENA javoni
  (2 foto) — haqiqiy va begona ballar birga ko'tariladi, ajralish +0.007 -> +0.007; muzlatgich
  (2–3 SKU) — ballar ±0.01–0.03, qarorlar o'zgarmadi.
- `benchmark_synthetic --n 100` (packshot + augmentatsiya katalogi): yopiq 0.927 -> 0.971, lekin
  tanish/notanish AUROC 0.982 -> 0.927, kalibrlangan ochiq aniqlik 0.887 -> 0.868 (notanishni rad
  etish 0.85 -> 0.52).
- Xulosa: foyda katalog katta (20+ SKU) va har SKU'da 3+ real javon qutisi bo'lganda; hozirgi
  kichik, asosan foto-namunali kataloglarda foyda yo'q, notanishni rad etish esa yomonlashishi
  mumkin. Sharti bajarilganda qayta ko'riladi (faqat `src: crop` namunalardan o'rganish bilan).

### 11f. Qadoqdagi yozuvni o'qish — OCR (ixtiyoriy xizmat) — 2026-11-26
Savol: tizim ta'mni ajrata olmagan qutilarda yorliqdagi yozuv yordam beradimi. Ma'lumot: muzlatgichning
2 kadri, ko'z bilan nomlangan 34 quti (10 mahsulot: Ays Tea ×3, Micco ×4, Наш сад ×3 — bir xil
shishalar ikki kadrda, ya'ni ~18 ta jismoniy shisha). PaddleOCR 3.7, CPU: matn
detektori PP-OCRv5 (mobile yoki server), tanish modeli PP-OCRv6_medium_rec. Matn katalog nomidagi ta'm so'zi bilan solishtirildi (taxminiy moslik, kesilgan
so'zlar — "shaft…", "…ranate" — ham qabul qilindi; moslash qoidalari natijani ko'rib yozilgan, ya'ni
OCR foydasiga optimistik).

| OCR | rasm o'lchami | ta'm o'qildi | boshqa ta'm | hajm (15 ta Ays) | vaqt / quti (CPU) |
|---|---|---|---|---|---|
| mobile det | asl (6048×8064 dan kesilgan) | 18/34 | 0 | 10/15 | 3.2 s (median 2.5) |
| mobile det + tarjima sinonimlari | asl | 19/34 | 0 | 10/15 | — |
| mobile det | prod o'lchami (uzun tomon 2000) | 7/34 | 0 | 2/15 | 0.44 s |
| mobile det | prod o'lchami ×3 kattalashtirilgan | 11/34 (+tarjima 12) | 0 | 6/15 | 1.2 s |
| server det + tarjima | asl | 21/34 | 0 | 10/15 | 20–40 s |
| mobile det + kirill modeli (eslav_PP-OCRv5_mobile_rec) + tarjima | asl | 18/34 | 0 | 10/15 | 3.3 s |

Mahsulot bo'yicha (mobile, asl o'lcham): Ays shaftoli 8/8 (4 tasida so'z kesilgan), Micco
orange-mango 2/2, pomegranate 2/2 ("RANATE"), pear 3/4, Наш сад малина 2/4 (mayda lotincha
"malina"; qo'lyozma shriftdagi «Малина» o'qilmaydi), Micco kiwi 1/2, Ays o'rmon mevalari 1.25L 1/4,
0.5L 0/3 (yorlig'i ruscha), Наш сад груша 0/4 (server modeli 2 tasida o'zbekcha "nok"ni o'qidi),
банан-клубника 0/1.

Joriy tizim bilan birga (kadrlararo; OCR faqat SKU darajasida tanilmagan qutilarda, bir brend ichida):

| Katalog | OCR'siz: to'g'ri / noto'g'ri | OCR bilan (mobile) |
|---|---|---|
| har mahsulotdan 1 quti, 6864 → 6863 | 13/16, 1 | 14/16, 1 |
| har mahsulotdan 1 quti, 6863 → 6864 | 14/18, 2 | 15/18, 2 |
| kadrdagi hamma qutilar, 6864 → 6863 | 16/16, 0 | 16/16, 0 |
| kadrdagi hamma qutilar, 6863 → 6864 | 16/18, 2 | 16/18, 2 |

- Tuzatilgani ikkala holatda ham bitta shisha — old tomoni ko'rinib turgan, tizim "brend darajasida"
  qoldirgan shaftoli. Tizim NOTO'G'RI tanigan qutilar (orqasi bilan turgan o'rmon mevalari → shaftoli,
  банан-клубника → малина) o'zgarmadi: ularda yorliq yozuvi kadrda ko'rinmaydi. Server modeli hamma
  qutilarga qo'llanganda yana bittasini tuzatdi (0.5L Ays — "0,5l" hajmi bo'yicha): 16/18 → 17/18.
- Qo'shni shisha yozuvi: quti chetiga tegib turgan matn tashlanmasa, 2 ta pomegranate qutisida
  qo'shnining "…ANGO" yozuvi ham o'qiladi (birinchi, sodda moslashda bu ikkalasi orange-mango'ga
  o'zgarib ketgan).
- Narxi: asl o'lchamdagi rasm kerak (prod 2000 px ga kichraytiradi — 7/34); noaniq qutiga 2.5–3 s CPU;
  yangi bog'liqlik (paddlepaddle + paddlex + modellar); yorliq tili katalog nomidan farq qilsa
  (o'rmon mevalari / лесные ягоды / nok) sinonimlar kerak.
- Xulosa: yozuv o'qiladigan joyda (old tomoni ko'ringan yorliq) vektor ham odatda to'g'ri ishlaydi;
  xato bo'ladigan joyda (burilgan, orqa tomon, yaltirash) yozuv ham ko'rinmaydi — bu to'plamda foyda
  kichik. O'lchanmagan: GPU'da vaqt; VLM bilan o'qish (GPU band edi); yassi old tomonli qadoqlar
  (karton quti, tetrapak) — bu to'plamda faqat dumaloq shishalar.

**Tizimga qo'shilgan shakli** (foydalanuvchi qarori; standart holatda o'chiq): alohida `ocr`
konteyneri (`docker compose --profile ocr`, `RASTA_OCR_URL`), `rasta/ocr.py`. Faqat "brend
darajasida" va "variant" qutilar o'qiladi; nomzodlar — shu brendning ko'rinish bo'yicha ham yaqin
(top-5) mahsulotlari; yozuv bittasiga bir ma'noli mos kelsa, quti SKU deb sanaladi. SKU darajasidagi
qaror va noma'lum quti o'zgarmaydi; ta'm mos, hajm zid bo'lsa — rad. Quti chetidagi matn tashlanadi.
Backend yozuv uchun asl rasmni (6000 px gacha) ochadi, zona tahlilida — ish rasmi (kattalashtirib).

To'liq zanjir (pipeline CPU'da + jonli OCR xizmati, kadrlararo):

| Katalog | OCR'siz | OCR, faqat 2000 px rasm | OCR, asl rasm | OCR vaqti |
|---|---|---|---|---|
| har mahsulotdan 1 quti, 6864 → 6863 | 13/16, xato 1 | 14/16, xato 1 | 14/16, xato 1 | 2.3 / 3.1 s |
| har mahsulotdan 1 quti, 6863 → 6864 | 14/18, xato 2 | 15/18, xato 2 | 15/18, xato 2 | 2.1 / 3.1 s |
| kadrdagi hamma qutilar (ikki yo'nalish) | 16/16; 16/18 | o'zgarmadi | o'zgarmadi | 0 (noaniq quti yo'q) |

Har kadrda atigi 1 ta quti "brend darajasida" qolgan edi — OCR faqat shunda ishladi va uni to'g'ri
aniqladi. Noaniq quti bo'lmasa xizmat chaqirilmaydi.

Veb-ilova orqali (lokal, CPU): 8 MB li asl rasm yuklandi — 24 quti, 1 tasi yozuv bo'yicha aniqlandi,
izohda "Yorliqdagi yozuv bo'yicha aniqlandi … (o'qilgan: «shaftoli»)"; butun tahlil 15 s.

Konteyner (`ocr/Dockerfile`, 2.5 GB): internetsiz (`--network none`) ishga tushadi, 3 ta haqiqiy
qutini Windows'dagi o'lchov bilan aynan bir xil o'qidi, 3 quti 4.5–5 s. Ogohlantirish: sinov
kompyuteridagi Docker'da (WSL2) Python jarayonlari ishga tushishda tasodifiy yiqiladi — toza
`python:3.10-slim` da oddiy standart kutubxona kodi 25 tadan 6 tasida, OCR image'i (3.11) 20 tadan
2 tasida (`python:3.11-slim` va `3.12-slim` toza holda 0/25). Disk, xotira va tizim jurnalida xato
yo'q; sababi aniqlanmadi. Yiqilgan konteynerni `restart: unless-stopped` qayta ko'taradi, backend esa
javobsiz qolgan so'rovda tahlilni yozuvsiz yakunlaydi. Serverda bu tekshirilmagan.


### 11g. Tanilgan qutini yozuv bilan tekshirish, guruhlar, o'lcham va qamrov — 2026-11-26

Sabab — prodda ko'ringan ikki xato: «Florida» qadog'i "COOKIES" deb, DENA sharbati "Ays Tea Shaftoli"
deb tanildi. OCR ulangan edi, lekin u faqat "brend darajasida" qolgan qutilarni hal qilardi, tanilgan
qutini tekshirmasdi. Lokal takrorlash (katalog: ZARQAND "COOKIES", 3 ta packshot; javon rasmi
`photo_2026-11-25_09-14-14.jpg`, 147 quti): 10 quti "COOKIES" bo'ldi — 5 tasi haqiqiy, 2 tasi Florida,
1 tasi Crafers, 2 tasi ZARQAND Mini Mango. Ball bilan ajratib bo'lmaydi: haqiqiylari 0.705–0.847,
noto'g'rilari 0.683–0.730. Yozuvda esa farq aniq: «Florida» 1.00 ishonch bilan o'qiladi.

**Nima bilan solishtirish kerak.** Muzlatgichning to'g'ri javobi ma'lum 33–34 qutisida:

| Mahsulotning "o'z so'zlari" | To'g'ri qutilardan nohaq rad etilgani |
|---|---|
| katalog nomi + brend | 4/34 (asl o'lcham), 3/34 (ish rasmi ×3) |
| namuna rasmida o'qilgan so'zlar + nom + brend | **0/33** (yirik crop: 30 tasida o'z so'zi o'qildi, 3 tasida aniq narsa yo'q; kichik crop: 23 va 10) |

Nom bilan solishtirish ishlamaydi: burilgan shishada faqat mayda yozuv ko'rinadi («CERTIFIED»,
«Сахар»), «Груша» esa "Tpya" deb o'qiladi. Namuna rasmidagi yozuv bilan solishtirganda OCR'ning
xatosi ikkala tomonda bir xil takrorlanadi va katalog nomi qadoqdagi yozuvdan farq qilsa ham ishlaydi.

**Qoida** (`rasta.ocr.judge`): qutida o'z so'zlaridan birortasi o'qilmagan VA yirik (harfi quti kichik
tomonining ≥5%), aniq (ishonch ≥0.95), kamida 4 harfli begona so'z bor → moslik bekor. Faqat brend
so'zi o'qilgan va yonida shunday begona so'z bor → "brend darajasi". Aks holda qaror o'zgarmaydi.
Harf o'lchami uchun OCR xizmati endi har so'zning balandligini ham qaytaradi.

Boshqa mahsulot bilan ataylab noto'g'ri juftlanganda (kichik crop, yuqoridagi sozlama):

| Juftlik | Rad etildi | Brend darajasi | O'zgarmadi | Noto'g'ri "tasdiqlandi" |
|---|---|---|---|---|
| boshqa brend, muzlatgich (215) | 82 | — | 130 | 3 |
| shu brendning boshqa mahsuloti (74) | 9 | 13 | 24 | 28 |
| DENA/Bliss kartonlari → Ays Tea (24) | 18 | — | 6 | 0 |

Ya'ni tekshiruv xatolarning bir qismini ushlaydi (yozuv aniq o'qilgan joyda), bir ta'mni boshqasidan
ajratmaydi (so'zlari deyarli bir xil) — lekin to'g'ri moslikni buzmaydi.

**To'liq zanjir** (pipeline CPU'da + jonli OCR xizmati):

| Sinov | Tekshiruvsiz | Tekshiruv bilan |
|---|---|---|
| Pechenye javoni, "COOKIES" deb tanilgan | 10 (5 to'g'ri, 5 noto'g'ri) | 5 (hammasi to'g'ri); Florida ×2 va Crafers rad, Mini Mango ×2 → brend |
| Muzlatgich, to'liq katalog, 6864 → 6863 | 16/16, xato 0 | o'zgarmadi, rad 0 |
| Muzlatgich, to'liq katalog, 6863 → 6864 | 16/18, xato 2 | o'zgarmadi, rad 0 |
| Muzlatgich, 1 namuna/mahsulot (ikki yo'nalish) | 14/16 xato 1; 15/18 xato 2 | o'zgarmadi, rad 0 |

Muzlatgichdagi 1–2 xato (bir brendning boshqa ta'mi) ushlanmadi; birorta to'g'ri quti rad etilmadi.
Rad etilgan Florida qutilari notanishlar guruhiga (o'z "Florida" guruhiga) qo'shildi.

**Vaqt.** Band protsessorda (GPU'dagi boshqa ish + shu sinovlar) bir qutini o'qish 2–2.5 s, yirik
(2600 px) crop 9 s gacha; 12 qutini tekshirish tahlilga 22–31 s qo'shdi. Shuning uchun: (1) faqat bali
o'z chegarasidan (τ) 0.10 dan kam yuqori bo'lgan qutilar o'qiladi (`ocr.verify_margin`) — pechenye
javonida 5 ta noto'g'rining hammasi shu oraliqda; muzlatgichda (1 namuna/mahsulot) tekshiruv tahlilga
22–29 s o'rniga 2–9 s qo'shdi, natija o'zgarmadi; (2) tekshiruv uchun kichik crop (700–900 px) olinadi — brend va nom yirik yoziladi;
(3) OCR mijoz rasmlarni 4 tadan yuboradi va umumiy vaqt chegarasi bor (`ocr.budget`): avval 12 yirik
qutili bitta so'rov 60 s dan oshib, natijaning hammasi jim yo'qolardi.

**Cheklovlar.** Namuna yozuvi mahsulot qo'shilganda o'qiladi; eski mahsulotlar uchun birinchi tahlilda
saqlangan rasmlardan o'zi o'qiladi (bir tahlilda 3 tagacha mahsulot; katalogga yoziladi, qayta o'qilmaydi).
Ilk bor lokal sinovda tekshiruv ishlamadi: sinov serverida yozuv xizmati o'chiq edi va jimgina
o'tkazib yuborilgan — endi bunday holatda natijada ogohlantirish chiqadi. Xizmat yoqilgach, eski
"COOKIES" mahsuloti bilan tahlil: 5 ta to'g'ri quti qoldi, Florida ×2 va Crafers "label", Mini Mango ×2
"label_brand" deb belgilandi. Yozuvi o'qilmaydigan (burilgan,
yaltiragan) qutida tekshiruv jim qoladi. Server protsessorida vaqt o'lchanmagan.

**Notanishlar guruhlari.** HDBSCAN guruhlari ilgari ham hisoblanardi, lekin interfeysga chiqmasdi.
Endi tahlil sahifasida "Katalogda yo'q · N guruh" ro'yxati bor: guruhni bosish hamma qutisini
tanlaydi, nom berilsa katalogga qo'shiladi. Pechenye javonida 147 qutidan 129 tasi 32 guruhga
tushdi (Florida 7, Labubu 5, Velona 6+5 …), sharbat javonida 65 qutidan 60 tasi 14 guruhga. Guruhlar
asosan toza; 2–3 tasida rangi o'xshash boshqa mahsulot aralashgan — operator tanlovni tuzatishi mumkin.

**Rasmni kattalashtirish (qo'shilmadi).** Muzlatgich, kadrlararo, tahlil rasmi o'lchami:

| O'lcham | 1 namuna: 6863→6864 | 1 namuna: 6864→6863 | To'liq: 6863→6864 | To'liq: 6864→6863 | Belgilangan qutidan topildi |
|---|---|---|---|---|---|
| 2000 px (joriy) | 14/18, xato 2 | 13/16, xato 1 | 16/18, xato 2 | 16/16 | 21/21; 20/20 |
| 3000 px | 13/18, xato 1 | 13/16, xato 0 | 16/18, xato 1 | 16/16 | 19/21; 19/20 |
| 4000 px | 13/18, xato 2 | 13/16, xato 0 | 16/18, xato 1 | 16/16 | 19/21; 17/20 |
| asl (6048×8064) | 13/18, xato 2 | 13/16, xato 0 | 16/18, xato 1 | 16/16 | 19/21; 17/20 |
| 2000 px ×2 kattalashtirilgan | 14/18, xato 3 | 13/16, xato 1 | 16/18, xato 2 | 16/16 | 21/21; 18/20 |

To'g'ri tanish o'smaydi, detektor esa bir necha qutini yo'qotadi: detektor rasmni 640/1280 ga,
embedding modeli har qutini 224 ga keltiradi — qo'shimcha piksel ularga yetib bormaydi. Asl o'lcham
faqat OCR uchun foydali va u allaqachon asl rasmdan o'qiydi.

**Tahlilni katalogning bir qismi bilan cheklash (foydalanuvchi so'rovi bilan qo'shildi — pastga qarang).** RP2K, 120 SKU, 25%
mahsulot katalogda yo'q, xato (noto'g'ri SKU yoki yo'q mahsulotni tanish) 2% bo'ladigan chegarada
to'g'ri tanilgan ulush (embedding'ning o'zi, 10 ta tasodifiy bo'linish):

| Qamrov | 1 namuna/SKU | 3 namuna | 5 namuna |
|---|---|---|---|
| to'liq katalog (90 SKU) | 16.3% | 24.3% | 27.6% |
| "tur" filtri, ~18 SKU (o'xshashlar ichida qoladi) | 18.2% | 25.3% | 28.5% |
| "tur" filtri, ~10 SKU | 18.9% | 26.9% | 29.3% |
| javon assortimenti, 45 SKU | 21.6% | 31.8% | 35.7% |
| javon assortimenti, 20 SKU | 28.7% | 43.4% | 48.5% |
| javon assortimenti, 10 SKU | 36.2% | 52.9% | 57.7% |

Mahsulot turi bo'yicha filtr kam beradi (+1–3 punkt): chalkashadigan mahsulotlar bir turga
kiradi. Javonda aynan qaysi mahsulotlar turishi ro'yxati esa ancha ko'p beradi. Foydalanuvchi shuni
so'radi: shirinliklar javonida katalogdagi 5–6 tasi bor, qolganlari yo'q va buni ko'rib turibdi.
**Qo'shildi:** tahlil sahifasida "Qidiruv doirasi" — katalogdan javonda borlari belgilanadi
(brend bo'yicha, qidiruv bilan), tizim faqat ular orasidan izlaydi (`catalog.search(allowed=…)`,
`/api/analyze` ning `skus` maydoni, agent ham foydalana oladi). Hamma belgilansa cheklov yo'q;
belgilanganlar sariq ramka bilan ko'rsatiladi, natijada ham izoh chiqadi; yangi o'rgatilgan mahsulot
doiraga o'zi qo'shiladi. Rasm almashtirilganda tanlov saqlanadi; sahifa yangilanganda butun katalogga qaytadi.

Chegarani o'zgartirmasdan, o'sha bitta embedding (RP2K, hamma sinov crop'lari katalogda bor, ball
chegarasi butun katalogda 2% noto'g'ri moslik beradigan qilib tanlangan):

| Qamrov | Noto'g'ri moslik (1 / 3 / 5 namuna) | To'g'ri tanish |
|---|---|---|
| butun katalog (120 SKU) | 2.0% / 2.0% / 2.0% | 14.6% / 23.1% / 25.6% |
| javon assortimenti 60 SKU | 1.1% / 1.0% / 1.1% | 14.7% / 23.5% / 26.1% |
| 30 SKU | 0.5% / 0.5% / 0.5% | 14.8% / 23.6% / 26.3% |
| 10 SKU | 0.2% / 0.2% / 0.2% | 14.8% / 23.7% / 26.5% |

Ya'ni noto'g'ri moslik 2–10 baravar kamayadi (foydalanuvchi taxmini to'g'ri), to'g'ri tanilganlar
ulushi esa chegara o'zgarmasa deyarli oshmaydi (+0.1…+0.9 punkt). Chegarani pasaytirsa foyda katta
(yuqoridagi jadval: 10 SKU da 27.6% → 57.7%), lekin buni qo'shmadim: haqiqiy javonda katalogda
yo'q mahsulot ko'p (foydalanuvchining shirinlik javonida 118 qutidan 5 tasi katalogda), ularni
ham "tanish" deb qo'yish xavfi oshadi. Kerak bo'lsa alohida sozlama sifatida, haqiqiy
javonda o'lchab qo'shiladi. Mahsulot haqiqatda javonda bor, lekin ro'yxatga kiritilmagan bo'lsa,
u tanilmaydi (noto'g'ri moslik 0.2–2.0%, tanilmay qoladigani 98–99.8%) — shuning uchun doira
sahifada doim ko'rinib turadi.

Tekshiruv: muzlatgich rasmida faqat "Ays Tea" (3 mahsulot) belgilandi — 24 qutidan faqat Ays
mahsulotlari tanildi (29%), Micco va Наш сад qutilari "noma'lum"ga o'tdi; hammasi qayta belgilanganda
oddiy natija qaytdi.

**Qoida kuchaytirildi (2026-11-26, kechqurun).** Foydalanuvchi muzlatgich skrinshotini yubordi: tepa
qatordagi «Наш сад» shishalari "Ays Tea Shaftoli" deb tanilgan, tekshiruv ularni rad etmagan.
Sabab: OCR bukilgan shishadagi «Наш сад» ni «Hau» + «cag» + «new» deb 3 harfli parchalarga bo'lib
o'qiydi, qoida esa kamida 4 harfli bitta begona so'z talab qilardi. Muzlatgichda («Наш сад» qutisi
→ "Ays" mahsulotiga qarshi 27 juftlik; to'g'ri javobli 33 juftlikda nohaq rad kerak emas):

| Begona yozuv mezoni | crop 700–900 px | crop 1400–1800 px | crop 1500–2600 px | to'g'ri qutini nohaq rad |
|---|---|---|---|---|
| bitta so'z ≥4 harf, ishonch ≥0.95 (oldingi) | 0/27 | 3/27 | 3/27 | 0/33 |
| + yirik parchalarning jami ≥5 harf, ishonch ≥0.95 | 3/27 | 12/27 | 9/27 | 0/33 |
| + ishonch ≥0.90 (**joriy**) | 6/27 | **21/27** | 12/27 | 0/33 |

Shuning uchun: ishonch chegarasi 0.90, jami 5 harf (`ocr.verify_conf`, `ocr.verify_letters`),
tekshiruv uchun crop 1400–1800 px (kichik crop kam o'qiydi, juda katta esa 2 baravar sekin va
ham kam o'qiydi). Boshqa brend mahsulotiga qarshi umuman: 160/215 rad. Ball bo'yicha saralash
ham kengaytirildi (`ocr.verify_margin` 0.10 → 0.25): oldin muzlatgichda bali yuqori (0.86–0.97)
noto'g'ri moslik tekshirilmasdi.

Namuna so'zlari endi faqat yirik yozuvdan olinadi (harf o'lchami ≥5%): «CERTIFIED», tarkib va manzil
ko'p qadoqda bor va mahsulotni ajratmaydi — oldin ular "o'z so'zi" bo'lib tekshiruvni
sustlashtirardi.

Rad etilmagan va ataylab qo'shilmagan: bir brendning boshqa ta'mi («Ays Shaftoli» o'rniga «Ays
O'rmon mevalari»). Brend so'zi ikkalasida bir xil, ta'm so'zi esa tez-tez o'qilmaydi. "Nomdagi ta'm
so'zi qutida o'qilmasa — brend darajasiga tushirish" qoidasi o'lchandi: 78 ta shu brendning boshqa
mahsuloti qutisidan 24 tasini ushlaydi, lekin to'g'ri qutilarning 4/34 (12%) ini ham SKU'dan
brendga tushiradi. Foyda xarajatdan sezilarli ustun emas — qo'shilmadi.

Natija tekshiruvga ko'ra qutini bosganda ko'rinadi: "Yozuv tekshirildi: qadoqda «ays» o'qildi — mos"
yoki "yozuv bo'yicha tekshirildi, lekin aniq matn o'qilmadi". Hech narsa yozilmagan quti tekshirilmagan
(ball yuqori yoki namuna yozuvi yo'q).

### 11h. Noma'lumlarni klasterlashda joylashuvni hisobga olish (qo'shilmadi) — 2026-11-26

Foydalanuvchi g'oyasi: javonda bir xil mahsulot yonma-yon yoki ustma-ust turadi, shuning uchun qo'shni
o'xshash qutilarni bir guruhga yaqinlashtirish kerak. Sinov: muzlatgichning ikki kadri, katalog bo'sh
(hamma quti noma'lum), to'g'ri javobi ma'lum 34 quti (10 SKU, kadrda 53 quti); ko'rsatkichlar shu
34 quti bo'yicha. Qo'shnilik: bir qatorda gorizontal tirqish ≤ 0.5 kenglik yoki x bo'yicha ≥60% ustma-ust
va vertikal tirqish kichik.

| Usul | Qamrov | Tozalik | ARI |
|---|---|---|---|
| joriy (HDBSCAN, joylashuvsiz) | 0.94 | 0.59 | 0.39 |
| qo'shni juftlar masofasi ×(1−λ), λ=0.5 (yonma-yon + ustma-ust) | 0.97 | 0.70 | 0.52 |
| ...λ=0.65 yoki 0.8 | 1.00 | 0.47 | 0.24 |
| faqat yonma-yon, λ=0.5 / 0.65 / 0.8 | 1.00 | 0.68 / 0.65 / 0.59 | 0.48 / 0.44 / 0.38 |
| faqat ustma-ust, λ=0.5…0.8 | 0.88 | 0.60 | 0.38 |
| qo'shni orqali birlashuvchi aglomerativ (cheklovli) vs cheklovsiz, t=0.5…0.9 | — | har chegarada teng yoki yomon | har chegarada teng yoki yomon |

Joylashuv bitta sozlamada yaxshilaydi (λ=0.5), yaqin sozlamada (λ≥0.65) esa guruhlar bir-biriga qo'shilib
tozalik 0.47 ga tushadi; cheklovli klasterlash esa umuman yordam bermadi. Bu barqaror signal emas,
34 qutida farq 3–4 quti. Ustma-ust turganlar foyda bermadi. Qo'shilmadi.

Boshqa sozlamalar ham sinaldi: HDBSCAN `min_samples`/`min_cluster_size` — foyda yo'q (tozalik 0.47–0.59);
rang og'irligi 0.6 → 0.8 tozalikni 0.59 → 0.69 ga ko'taradi (ARI 0.44), qamrov o'sha-o'sha; aglomerativ
(average, t=0.6) tozalik 0.80, ARI 0.60 beradi, lekin qamrov 0.74 ga tushadi (ko'proq quti guruhsiz qoladi).
Hech biri bepul yaxshilanish emas. Hukm: klasterlash hozirgi holatida qoladi; ishonchli baholash uchun
ko'proq belgilangan javon (20+ SKU) kerak. Tanilgan qutilar tomonida qo'shni konteksti (ikki bir xil
tanilgan qo'shni orasidagi chegaradagi noma'lum quti — `smooth_with_context`) allaqachon bor;
kengroq tarqatish (`propagate_blocks`) ilgari o'lchanib rad etilgan.

### 11i. Guruhsiz mahsulotlarni qayta tekshirish

Javon tahlilida HDBSCAN'dan keyin har bir guruhsiz crop dastlabki klaster
a'zolarigacha bo'lgan o'rtacha kosinus masofasi bo'yicha tekshiriladi. Qolgan
crop'lardan complete-linkage yordamida ixcham guruhlar tuziladi. Standart
masofa chegarasi 0.20. 500 yoki undan ko'p noma'lum crop bo'lganda dastlabki
guruhlar HDBSCAN leaf rejimida kamida beshtadan tuziladi. Qayta tekshirishda
kichik guruhlar konfiguratsiyadagi `min_cluster_size` bo'yicha qabul qilinadi.

`python scripts/benchmark_clustering.py --compare-production` bu usulni
avvalgi HDBSCAN bilan bir xil DINOv2 + SigLIP2, rang va shakl vektorlarida
solishtiradi. RP2K test kesmalaridagi 120 SKU ikki mustaqil SKU to'plamiga
ajratildi. Quyida holdout natijalari keltirilgan. Noto'g'ri birlashish foizi
birga klasterlangan juftliklar orasida SKU'si har xil bo'lganlar ulushidir.

| Sinov | B-cubed F1 | Qamrov | Noto'g'ri birlashish |
|---|---:|---:|---:|
| 30 crop, 14 guruh | 0.841 → 0.843 | 89.1% → 89.5% | 13.74% → 13.71% |
| 120 crop, 14 guruh | 0.781 → 0.784 | 90.7% → 92.3% | 6.21% → 6.70% |
| 240 crop, 14 guruh | 0.690 → 0.712 | 84.9% → 90.7% | 5.53% → 5.82% |
| 600 crop va 30 SKU, 2 guruh | 0.607 → 0.724 | 81.5% → 89.1% | 6.10% → 3.11% |

RP2K guruhlari dataset kesmalaridan yig'ildi. Alohida real javon sinovida
(`cola_cans_shelf.jpg`, bo'sh katalog) 71 deteksiyaning 61 tasi SKU
annotatsiyalariga IoU ≥ 0.5 bilan mos tushdi. Shu 61 qutida ikkala usul ham
B-cubed F1 1.0 va qamrov 98.4% berdi.

### 11j. Yorliq rangi bo'yicha ajratish va oynadagi aks

Baland qadoqlar klasteri pastki rang tasmasi bo'yicha ikki ixcham guruhga
ajralganda, har ikkala guruhda yetarli a'zo bo'lsa, klaster ajratiladi.
Javon qatori filtri ikki tomondagi ancha baland va ishonchli mahsulotlar
orasida, turli sathlarda turgan juda kichik va past ishonchli qutilarni fon
sifatida belgilaydi. Kadr chetidagi qator qolgan tayanch qatorlar bilan
solishtiriladi.

`IMG_6863.JPG` tik holatda va bo'sh katalog bilan tahlil qilindi. Avvalgi
variantdagi 29 qutidan oynadagi 9 ta aks va yuqori chetdagi 2 ta qisman
ko'ringan quti hisobdan chiqdi. 18 ta qutidan 17 tasi 4 guruhga tushdi.
Nash Sad malina va nok variantlari ikkitadan alohida guruhga ajraldi.
Miccoco va Ays Tea guruhlarida turli ta'mlar aralashgan.

RP2K kesmalarida rang tekshiruvi 30, 120 va 600 kesmali holdout
natijalarini o'zgartirmadi. 240 kesmali holdoutda B-cubed F1
0.7119 dan 0.7115 ga tushdi, qamrov 90.74% bo'lib qoldi.

### 11k. Katalogda bitta variant bo'lganda OCR tekshiruvi

`IMG_6863.JPG` dagi 8 ta Nash Sad va Miccoco kesmasi PaddleOCR bilan o'qildi.
O'qilgan matnlar dastlabki vizual mosliklari shartli ravishda tasdiqlangan
qutilarda qayta tekshirildi. Sinov katalogida Nash Sad Malina va Miccoco
Orange Mango bor edi. «nok» va «PEAR» aniq o'qilgan ikkita quti shu SKU'lar
hisobidan chiqarildi. Bu tekshiruv OCR qarorini o'lchaydi. Vizual matching
va javondagi barcha ta'mlarni topish aniqligi bu tajribada o'lchanmadi.

### 11l. Qisman yopilgan mahsulot va aralash klaster

`IMG_6863.JPG` to'liq rasmini tahlil qilishda vaqtinchalik katalogga shu
rasmdan bitta Nash Sad malina namunasi kiritildi. OCR ishlatilmadi.
Qisman yopilgan nokning katalog balli 0.741 bo'lib, 0.760 chegaradan
past edi. Shu qatordagi tanilgan nok bilan qutilar qoplanishi, rang
(0.917) va embedding (0.788) tekshiruvidan so'ng brendning umumiy
hisobiga kirdi. Nash Sad hisobi 3 tadan 4 taga chiqdi.

Miccoco'da ikki shisha yorlig'ini qamrab olgan bitta kuchsiz quti
olib tashlandi. Qolgan noma'lum kesmalardan ikki nok bitta klasterga
tushdi. Apelsin-mango, anor va orqasi ko'ringan yashil shisha guruhsiz
qoldi. Ays Tea ta'mlari bir guruhda qoldi.

RP2K holdoutda yangi rang tekshiruvi 30, 240 va 600 kesmali sinovlarni
o'zgartirmadi. 120 kesmali sinovda B-cubed F1 0.7844 dan 0.7838 ga
tushdi. `photo_2026-10-30_17-38-56.jpg` sinovida 40 quti va klaster
a'zolari bir xil natija berdi.

### 11m. Avval brend, keyin SKU

`IMG_6863.JPG` tahlili vaqtinchalik Miccoco katalogi bilan, OCR ishlatmasdan
o'tkazildi. Shu kadrdan olingan uchta old yorliq namunasida 4 ta mahsulot
tanildi (nok 2, apelsin-mango 1, anor 1). Umumiy Miccoco yozuviga shu
kadrdagi yashil shishaning orqa yorlig'i qo'shilgach, Miccoco jami 5 ta
bo'ldi. Yashil shisha brend darajasida qoldi. Bu sinov oqim va hisobni
tekshiradi, namunalar boshqa kadrlardan olinadigan sinov alohida kerak.

### 11n. Bir klasterdagi qadoqlarni brendga bog'lash

`IMG_6914.JPG` vaqtinchalik Bliss va DENA katalogi bilan, OCR'siz qayta
tahlil qilindi. Bir qatordagi klaster ko'pchiligi bitta tasdiqlangan
brendga tegishli bo'lsa, ayrim chegaraga yaqin qutilar mahalliy rang va
embedding mosligi bilan brand darajasida tasdiqlandi. Raqib brend va
negativ katalog dalili ham tekshirildi.

Bliss hisobi 7 tadan 10 taga, DENA hisobi 10 tadan 11 taga chiqdi. Yangi
tasdiqlangan to'rtta qutiga SKU berilmadi. Bliss hisobida 3 ta ananas,
1 ta olma va 6 ta varianti noaniq quti bor. Qatorning chap tomonida yon
tomoni ko'ringan, Bliss yozuvi ko'rinmaydigan quti noma'lum qoldi.

### 11l. Yozuv o'qish xizmati serverning GPU'sida — 2026-11-27

OCR konteyneri endi serverning NVIDIA A2 kartasida ishlaydi (`RASTA_OCR_DEVICE=gpu:0`, PaddlePaddle CUDA 12.6),
modellar va chegaralar o'zgarmagan. Server ichida `python -m rasta_ocr --selftest` (sintetik 900×1400 rasm,
4 qator matn, isitishdan keyin 5 o'qishning o'rtachasi): **104 ms/crop**, o'qilgan matn CPU bilan bir xil
(`Ays Tea`, `Shaftoli`, `1,25 L`, `qora choy`). Taqqoslash: ishlab chiquvchi kompyuteri protsessorida (band, 8 oqim)
shu sinov 4311 ms/crop; haqiqiy javon qutilarida 2–9 s. Serverning protsessorida o'lchanmagan.

Qurish: GPU'li Paddle `import` paytidayoq `libcuda.so.1` ni talab qiladi, u esa image qurilayotganda yo'q —
modellar alohida CPU bosqichida yuklanadi (ocr/Dockerfile). GPU ko'rinmasa xizmat o'zi CPU'ga o'tadi.

### 11m. Brend bo'yicha sanash, umumiy brend yozuvi va yozuv tekshiruvidagi xatolar — 2026-11-27

Foydalanuvchi Bliss javonini ko'rsatdi: katalogda bitta "Bliss" yozuvi, javonda ta'mlari va seriyasi har xil
kartonlar; hammasi Bliss hisobiga kirishi kerak edi. Bo'sh va 1 mahsulotli kataloglarda sinab, yozuv tekshiruvi
(OCR) **to'g'ri qutilarni noto'g'ri rad etayotgani** topildi:

- Har kartonda ishlab chiqaruvchi nomi (AGROMIR) bosilgan; namuna kartonining so'zlari orasida u yo'q edi, brend so'zi
  («Bliss») o'qilmagan qutida u "begona yozuv" bo'lib, quti rad etilardi. Endi mahsulotga mos deb tasdiqlangan
  qutilarning kamida ikkitasida takrorlanadigan so'z shu mahsulot uchun neytral hisoblanadi (mos kelishning dalili ham,
  begona yozuv ham emas).
- Umumiy brend yozuvida (nomi brendga teng) ta'm so'zi («ананас» → «AHAHAC») begona emas, sarlavha qoidasi ham o'chadi
  («Bliss» logotipi «LSI» deb o'qiladi).
- Kirill nomli mahsulotlarda sarlavha qoidasi o'chiq («Наш сад» → «Hau»), «new», «bio» kabi umumiy so'zlar begona
  hisoblanmaydi, 3–4 harfli so'zda bitta harf xatosi («haw» ~ «Hau») yaqin sanaladi. Bu xatolarni 1 mahsulotli
  muzlatgich sinovi ko'rsatdi: to'g'ri quti rad etilib 14/16 → 13/16 bo'lgan, tuzatilgach yana 14/16.

Bliss javoni, katalogda "Bliss" (bitta yozuv): OCR bilan 5 ta quti rad etilardi, endi 0; 6 ta quti SKU, 3 ta brend
darajasi, 1 ta noma'lum (kesilgan quti). Muzlatgich, 4 holat, OCR yoqilgan: 16/16, 16/18, 14/16, 15/18 — OCR'siz bilan
bir xil. Eski natijalar saqlandi: to'g'ri qutini nohaq rad 0/33, «Наш сад» qutisi Ays'ga qarshi 21/27.

**Brend bo'yicha sanash (avval brend, keyin ta'm).** Tanilmagan qutida katalogdagi brend nomi yirik va aniq yozilgan
bo'lsa (ishonch ≥0.85, harf ≥4%, bitta brend), quti shu brendga sanaladi, SKU/ta'mi noma'lum qoladi
(`ocr.brand_text`). Muzlatgichning 34 qutisida: 9 tasida brend topildi, noto'g'ri brend 0, 25 tasida yozuv
o'qilmadi (egilgan shisha) — ya'ni aniq, lekin qamrovi kichik qo'shimcha. Bunday quti klaster orqali brendni
tasdiqlashda tayanch bo'ladi. Boshqa ta'm/hajm deb rad etilgan quti ham brendi o'qilsa brendga kiradi.

**Avval rang/shakl, keyin klaster (qo'shilmadi).** Foydalanuvchi taklifi: avval rang va shakl guruhiga bo'lib, keyin
guruh ichida embedding bo'yicha klasterlash. RP2K holdout (B-cubed F1; joriy usul 0.72/0.84/0.78/0.71 — 30 SKU, 30,
120, 240 crop): chegara 0.25 da 0.19/0.48/0.27/0.26, 0.4 da 0.37/0.59/0.43/0.44, 0.6 da 0.59/0.70/0.63/0.63. Hamma
chegarada yomonroq: bitta mahsulotning crop'lari yorug'lik va burchak tufayli turli rang guruhlariga tushib ketadi,
qamrov 0.89 dan 0.23–0.80 ga tushadi. Rang foydali — lekin klaster tuzilgandan KEYIN dizayni o'xshash guruhni ta'mga
ajratish uchun (meva rasmi qismining rangi DENA ta'mlarining ko'pini ajratdi), birinchi qadam sifatida emas.

### 11n. Brend ichida ta'm guruhlari (avval brend bo'yicha sanash, keyin ta'm) — 2026-11-27

Foydalanuvchi: "avval brend bo'yicha sanaymiz, keyin ta'mni ajratamiz". Brend aniq, ta'mi noma'lum qutilar
(umumiy brend yozuviga qo'shilganlar va brend darajasida qolganlar) endi brend ichida ta'mi bo'yicha guruhlanadi
(`rasta/flavour.py`, `flavour.*` sozlamalari) va tahlil sahifasida "Brend ta'mlari" kartasida ko'rsatiladi: ta'mni
bossangiz hamma qutisi tanlanadi, nom ("DENA Mango") va brend oldindan to'ldiriladi, bir bosishda alohida mahsulot bo'ladi.

Ikki dalil: (1) qadoqdagi ta'm so'zi — OCR kirillcha so'zni lotin harflariga o'xshatib o'qiydi («AHAHAC» = ананас),
shuning uchun uz/ru/en sinonimlar jadvali va "ko'rinishi bir xil harflar" solishtirish ishlatiladi; (2) mevaning rangi —
brend ichida hamma qadoqning dizayni umumiy, farq faqat meva rasmida, shuning uchun rang tasmalaridan guruh ichida
eng ko'p farq qiladiganiga ko'proq og'irlik beriladi. Yozuvlari kesishmaydigan ikki quti hech qachon bir guruhga
tushmaydi; yozuvi o'qilmagan quti rang bo'yicha mos guruhga qo'shiladi.

Real javon (Bliss tepa tokcha 11 quti, DENA o'rta tokcha 12 quti, `IMG_6914`):

| Brend | Faqat rang | Rang + yozuv | Izoh |
|---|---|---|---|
| DENA (10 ta'm) | 8 guruh | 9 guruh | anor va pomidor yozuv bilan ajraldi; mango bilan o'rik rangi bir xil bo'lgani uchun birga qoldi |
| Bliss | 5 guruh | 7 guruh | ananas ×3, pomidor, anor, mango yozuvdan; olma va pomidor rang bo'yicha qo'shilib qoladi (olma so'zi o'qilmagan) |

Yozuvdan ta'm so'zi qutilarning taxminan 40% ida o'qildi (bukilmagan kartonlarda ham: so'z kichik harfda bosilgan).
Natija taklif sifatida beriladi: guruh aralash chiqsa, operator rasmda qutini alohida tanlay oladi. Mahalliy
tekshiruvda (CPU OCR) butun tahlil bir necha daqiqa oldi; serverning GPU'sida ta'm so'zlarini o'qish bir necha
soniya (≤60 quti).

