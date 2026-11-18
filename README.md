# Rasta AI — javon rasmlarini tahlil qilish tizimi

Javon (rasta) rasmidan mahsulotlarni **tanib olib**, biznes xulosalar beradi:
Share-of-Shelf, facings, brend ulushi va **OOS (bo'sh joy)** — jumladan butunlay
bo'sh javonlar.

Asosiy prinsip: mahsulotni tanish — klassifikatsiya emas, **vektor qidiruv
(open-set retrieval)**. Yangi mahsulot qo'shish modelni qayta o'qitishni talab
qilmaydi — katalogga bir nechta rasm (hatto bitta packshot) qo'shiladi.

Natijalar va barcha sinovlar: **[BENCHMARK.md](BENCHMARK.md)** (≈680 000 so'rov,
3 ta real dataset + real javon GT + sintetik stress-test).

## Pipeline

```
Rasm → YOLO detektor (SKU-110K) → ichma-ich qutilarni tozalash
     → javon qatorlari (pastki chiziq bo'yicha) + orqa fonni ajratish
     → embedding: DINOv2 + SigLIP2 ansambli  ┐
     → rang deskriptori (HSV, fazoviy)       ├→ gibrid qidiruv (open-set, kalibrlangan τ)
     → shakl (tomonlar nisbati)              ┘
     → javon konteksti (qo'shnilar) → noma'lumlarni klasterlash (HDBSCAN)
     → OOS: qator ichida + javon chetlari bo'yicha bo'sh zonalar + rasm tekshiruvi
     → KPI → JSON + belgilangan rasm + noma'lumlar montaji
```

| Modul | Vazifa |
|---|---|
| `detector.py` | Class-agnostic YOLO; "butun + qism" qutilarini tozalash; TTA (masshtab, aks, bo'laklar) + WBF |
| `det_eval.py` | Detektor baholash: AP50, precision/recall, facings sanash xatosi |
| `shelf.py` | Qatorlar (pastki chiziq bo'yicha), orqa fon qatorlarini ajratish |
| `embedder.py` | DINOv2 / CLIP / SigLIP2 va ularning ansambli; har modelga alohida preprocessing |
| `features.py` | Rang (HSV, 4 fazoviy gistogramma, Hellinger) va shakl deskriptorlari |
| `catalog.py` | Qdrant katalog; gibrid qayta reyting; open-set qaror (rang veto'si, manbaga mos τ, brend darajasi) |
| `context.py` | Noma'lumni qo'shni SKU bloki bo'yicha ehtiyotkor tuzatish |
| `cluster.py` | Noma'lumlarni HDBSCAN bilan guruhlash (embedding + rang + shakl) |
| `oos.py` | Qator ichidagi bo'shliqlar, bo'linma filtri, rasm bo'yicha tekshiruv |
| `shelf_lines.py` | Javon chetlari (Hough) → butunlay bo'sh javon zonalari; sahna "frontal"ligi |
| `augment.py` | 1 packshot → realistik variantlar (onboarding) |
| `calibrate.py` | τ/margin kalibrlash, AUROC, halol (holdout) baholash |
| `roi.py` | Poligon zona: kesish, kattalashtirish, maska, koordinatalarni qaytarish |
| `kpi.py` | Share-of-Shelf, facings, brend ulushi, OOS xulosasi |
| `web/` | Veb-ilova (FastAPI): admin katalog + tahlil sahifasi |
| `pipeline.py` | Barchasini bog'laydi |

## O'rnatish

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# yoki aynan sinalgan versiyalar:
pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Birinchi ishga tushirishda `facebook/dinov2-base` va `google/siglip2-base-patch16-224`
HuggingFace'dan avtomatik yuklanadi (~1.2 GB).

## Tez boshlash

```bash
python scripts/demo.py                    # offline sintetik demo
python scripts/onboard_and_analyze.py     # real javon: few-shot onboarding + tahlil
python -m pytest tests -q                 # 58 ta test
```

## Veb-ilova (eng oson yo'l)

```bash
python scripts/serve.py
```
Brauzerda: **http://127.0.0.1:8000** — tahlil, **http://127.0.0.1:8000/admin** — mahsulotlar.

- **Admin** (`/admin`): mahsulot nomi, brendi va rasm(lar)ini yuklaysiz — katalogga qo'shiladi
  (bitta toza rasm yetarli; javondan olingan 3–5 rasm aniqroq). Ro'yxat va o'chirish shu yerda.
- **Tahlil** (`/`): javon/muzlatgich rasmini yuklaysiz; xohlasangiz "Zona belgilash" bilan
  kerakli muzlatgich burchaklarini bosasiz. Natija: rangli ramkali rasm (rang = jadvaldagi
  mahsulot), mahsulotlar soni, tanish foizi, to'ldirilganlik, bo'sh joylar va har mahsulotning
  ulushi.
- Admin paroli: `set RASTA_ADMIN_PASSWORD=parol` (Windows) server ishga tushishidan oldin.
  Parolsiz admin faqat lokal serverda (127.0.0.1) ochiq; tarmoqda (`--host 0.0.0.0`, Docker,
  tunnel) parol qo'yilmasa admin amallari **o'chiriladi** (403), tahlil esa ishlayveradi.
- Bir xil nomli mahsulot qayta qo'shilsa — 409 (jim-jit boshqa mahsulotga qo'shilib ketmaydi).
- Katalog qaysi embedder bilan qurilganini eslaydi (`data/qdrant/rasta_meta.json`). Embedder
  sozlamasi o'zgarsa ogohlantiradi (`/api/health` → `catalog_stale`) — qayta indekslang.

## Docker'da ishga tushirish

Talab: Docker Desktop (Windows: WSL2) va NVIDIA drayveri. GPU bo'lmasa ham ishlaydi
(CPU'da sekinroq) — `docker-compose.yml` dan `gpus: all` qatorini olib tashlang.
GitHub'dan yangi checkout qilinganda, `config.yaml` dagi detektor og'irliklarini
`models/weights/` ga joylashtiring. `models/negatives/aptash_v1.npz` ixtiyoriy.
`.env.example` asosida `.env` yarating va admin uchun parol kiriting.

```bash
docker build -f Dockerfile.base -t rasta-ai-base:1 .
```
```bash
docker compose up -d --build
```
Birinchi buyruq bir marta bajariladi (~20 daqiqa: PyTorch + modellar alohida bazaviy
image'ga). Ikkinchisi kod o'zgarganda — ~30 soniya. Brauzerda: **http://localhost:8090**
(admin: **/admin**).

- Port o'zgartirish: `.env` faylida `RASTA_PORT=9000`.
- Admin paroli: `.env` faylida `RASTA_ADMIN_PASSWORD=parol`.
- Katalog va mahsulot rasmlari `rasta-catalog` / `rasta-images` volume'larida saqlanadi —
  `docker compose down` dan keyin ham yo'qolmaydi (`down -v` — o'chiradi).
- Loglar: `docker compose logs -f rasta`.
- Embedding modellari image ichida, detektor og'irliklari esa `./models` papkasidan ulanadi.
  Shu fayllar tayyor bo'lgach, konteyner internetsiz ishlaydi.
- Standart port faqat lokal interfeysga bog'lanadi. Domen orqali chiqarishda HTTPS reverse
  proxy ishlating. Vaqtinchalik tashqi demo uchun `docker compose --profile demo up -d`.
- Katalogni saqlangan rasmlardan qayta indekslash uchun admin paroli bilan
  `POST /api/reindex` so'rovini yuboring. Parol `X-Admin-Token` sarlavhasida uzatiladi.

## Real ishlatish (buyruq qatori)

**1. Mahsulotni katalogga qo'shish** (model qayta o'qitilmaydi):
```bash
python scripts/add_product.py --sku cola_05 --name "Cola 0.5L" --brand ColaCo --images packshot.jpg
python scripts/add_product.py --sku cola_05 --name "Cola 0.5L" --images crops_dir/ --replace
python scripts/add_product.py --list
```
Bitta rasm berilsa — u packshot hisoblanadi va 16 ta realistik variant yaratiladi.
Eng yaxshisi: 1 packshot + 3–5 real javon crop.

**2. Tahlil** (bitta rasm yoki papka):
```bash
python scripts/analyze.py --image shelf.jpg
python scripts/analyze.py --image photos/ --out outputs/batch
```
Natija: `*_annotated.jpg`, `*_result.json`, noma'lumlar bo'lsa `*_unknown_clusters.jpg`.

Rasmdagi belgilar: rangli quti — tanilgan SKU; `Brend (SKU?)` — brend aniq, SKU noaniq
(masalan 0.5L/1L egizaklar); `?cN` — noma'lum, N-klaster; qizil — OOS;
to'q sariq punktir — shubhali bo'shliq; ko'k punktir — "bo'linma?".

**2a. Faqat kerakli muzlatgich/javonni tahlil qilish (zona/poligon):**
Rasmda 2–3 ta muzlatgich bo'lsa, keraklisini poligon bilan belgilang:
1. `tools/roi_editor.html` ni brauzerda oching (internet kerak emas), rasm(lar)ni yuklang.
2. Muzlatgich burchaklarini bosing, birinchi nuqtaga qaytib (yoki Enter) yoping, nom bering.
   Nuqtalarni sudrab tuzatish mumkin; ←/→ — keyingi rasm.
3. "Shu rasm JSON" (yoki ko'p rasm uchun "Hammasi (batch) JSON") ni saqlang va:
```bash
python scripts/analyze.py --image rasm.jpg --roi rasm.roi.json
python scripts/analyze.py --image papka/ --roi zonalar.roi.json --roi-only
python scripts/analyze.py --image rasm.jpg --roi-points "95,105;390,150;362,985;110,1040"
```
Har zona alohida tahlil qilinadi (`*_zones.jpg`, `*_zones.json`), tashqarisi hisobga
olinmaydi. Zona kesib olinib kattalashtiriladi — 800 px li agent rasmlarida mayda
mahsulotlar ko'proq topiladi (sinovda bir muzlatgichda 13 → 21). Har zona uchun
**to'ldirilganlik**: `roi_shelf_fill_pct` (javon qatorlari bo'yicha, to'la muzlatgich
≈ 85–90%) va har qator alohida (`roi_row_fill_pct`) — bo'sh/siyrak muzlatgichni
qator-asosli OOS'dan ishonchliroq ko'rsatadi. Namuna: `data/roi/*.roi.json`.

**Perspektivani to'g'rilash:** zona aynan **4 nuqta** bo'lsa (muzlatgich/javon
burchaklari), u homografiya bilan "old tomondan" ko'rinishga keltiriladi — qatorlar
gorizontal, mahsulotlar tik. Neyron tarmoq kerak emas: 4 burchak homografiyani aniq
beradi. Yon tomondan olingan rasmlarda SKU aniqligi o'rtacha **86.5% → 92.4%**
(`scripts/benchmark_rectify.py`). O'chirish: `--set roi.rectify=false`.
Maslahat: zonani reklama paneli/logotip **ostidan**, faqat mahsulot javonlari bo'yicha chizing.

**2b. Foto arxividan SKU katalogini qurish** (masalan agentlar fotolari):
```bash
python scripts/build_catalog_candidates.py --max-images 3000 --top 400
# data/catalog_build/review/review.html ni oching: har klasterga nom yozing -> labels.json
python scripts/import_catalog_labels.py --labels labels.json
```
Mahsulotlar kesiladi, sifat filtri (kichik/xira/blikli crop'lar chiqariladi), GPU
k-means + complete-linkage birlashtirish, klasterlar ko'p do'konda uchrashi bo'yicha
saralanadi; har SKU uchun turli do'konlardan 20 ta eng tipik crop katalogga qo'shiladi.

**2c. Detektorni o'lchash va kuchaytirish** (mahsulotlarni ajratib olish):
```bash
python scripts/prelabel_detector.py --n 40          # agent rasmlaridan qoralama GT (data/gt_det)
# tools/box_editor.html ni Chrome/Edge'da oching -> "Loyiha papkasini ochish" -> RASTA papkasi
python scripts/eval_detector.py --variant "tta: detector.tta.enabled=true" \
    --variant "tiles: detector.tta.enabled=true detector.tta.tiles=2"
```
Qoralamani kuchli (TTA) detektor tayyorlaydi; muharrirda noto'g'ri qutini o'chirasiz (Del),
o'tkazib yuborilganini chizasiz, orqa fon/reklamani "ignore" (I) qilasiz, "Tayyor" (D).
Natija: AP50, recall, precision va sanash xatosi (facings) har variant uchun.
Agent ilovasining burchakdagi yashil logotipi avtomatik olib tashlanadi (`detector.drop_app_logo`).
SKU-110K test qismida solishtirish: `python scripts/import_sku110k.py --eval 300`, keyin
`eval_detector.py --gt data/gt_det_sku110k`. Nomzod modellar: `models/candidates/`;
matn-so'rovli (zero-shot) detektorlar: `--set detector.backend=zeroshot`.
O'qitishsiz kuchaytirish — `detector.tta` (bir nechta masshtab + aks + bo'laklar, WBF bilan
birlashtirish, "guruh" qutilarini tozalash); standartda o'chiq, o'lchovdan keyin yoqiladi.

**3. Noma'lumlarni katalogga qo'shish (operator oqimi):**
```bash
python scripts/review_unknowns.py --image shelf.jpg                       # klasterlarni ko'rish
python scripts/review_unknowns.py --image shelf.jpg --assign "0=chin8:Chin8 Neri:Galvanina"
```

**4. Chegaralarni o'z do'koningizga kalibrlash** (tavsiya etiladi):
`data/gt/` ga GT fayl qo'shing (format: `data/gt/cola_cans_shelf.gt.json`), keyin:
```bash
python scripts/eval_shelf.py --calibrate --gt data/gt/<rasm>.gt.json
python scripts/eval_shelf.py --mode full --gt data/gt/<rasm>.gt.json      # aniqlikni o'lchash
```

Istalgan sozlamani vaqtincha o'zgartirish: `--set recognition.sim_threshold=0.7`.

## Konfiguratsiya (`config.yaml`)

| Bo'lim | Asosiy sozlamalar |
|---|---|
| `embedder` | ansambl modeli, og'irliklar, har model uchun `resize_mode`/`pooling` |
| `recognition` | `sim_threshold` (crop katalog, 0.76), `sim_threshold_packshot` (0.65), `w_color` 0.25, `w_shape` 0.05, `color_min` |
| `onboarding` | packshot augmentatsiyasi |
| `clustering` | HDBSCAN, rang/shakl og'irliklari |
| `shelf` | qator anchori, orqa fon filtri |
| `oos` | bo'shliq chegaralari, rasm tekshiruvi, bo'sh javon zonalari, sahna filtri |

**Diqqat:** `embedder.model` o'zgarsa katalogni qayta indekslang (vektor fazosi boshqa).

## Benchmarklar (qayta hosil qilish)

```bash
git clone https://github.com/marcusklasson/GroceryStoreDataset data/external/GroceryStoreDataset
python scripts/fetch_rp2k.py --classes 120 --train 5 --test 20
python scripts/benchmark_grocery.py --subset packages --seeds 20
python scripts/benchmark_rp2k.py --seeds 20
python scripts/benchmark_summary.py --seeds 20       # eski vs yangi yakuniy jadval
python scripts/benchmark_clustering.py               # rang vs shakl vs embedding
python scripts/benchmark_synthetic.py --n 100 [--ablation]
python scripts/benchmark_oos.py --n 40
```

## Loyiha tuzilishi

```
config.yaml            # barcha sozlamalar
BENCHMARK.md           # sinov natijalari
src/rasta/             # yadro kutubxona
scripts/               # CLI, onboarding, baholash va benchmarklar
tests/                 # unit testlar (og'ir modellarsiz)
data/gt/               # qo'lda belgilangan ground-truth
data/samples/          # namuna javon rasmlari
data/external/         # benchmark datasetlari (git'da emas)
outputs/               # tahlil va benchmark natijalari
Rasta_AI_hisobot_v2.html  # dastlabki texnik hisobot
```

## Keyingi bosqichlar

- Hajm egizaklari uchun jismoniy o'lcham (mm) bilan solishtirish
- Narx yorlig'i OCR (PaddleOCR-VL / Qwen3-VL) — SigLIP allaqachon matnni qisman "o'qiydi"
- Panorama tikish, planogram compliance
- Firibgarlik filtri (pHash, EXIF, recapture)
- Detektorni o'z rasmlaringizda fine-tune (`scripts/train_detector.py`)
