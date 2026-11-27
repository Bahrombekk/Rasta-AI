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
| `cluster.py` | Noma'lumlarni HDBSCAN bilan guruhlash va guruhsiz o'xshash mahsulotlarni qayta tekshirish |
| `oos.py` | Qator ichidagi bo'shliqlar, bo'linma filtri, rasm bo'yicha tekshiruv |
| `shelf_lines.py` | Javon chetlari (Hough) → butunlay bo'sh javon zonalari; sahna "frontal"ligi |
| `augment.py` | 1 packshot → realistik variantlar (onboarding) |
| `calibrate.py` | τ/margin kalibrlash, AUROC, halol (holdout) baholash |
| `roi.py` | Poligon zona: kesish, kattalashtirish, maska, koordinatalarni qaytarish |
| `kpi.py` | Share-of-Shelf, facings, brend ulushi, OOS xulosasi |
| `backend/rasta_backend/` | FastAPI, hisoblar, korxonalar, katalog va audit API |
| `ai/rasta_ai/` | GPU detektori va embeddinglar uchun ichki API |
| `pipeline.py` | Barchasini bog'laydi |

## O'rnatish

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# Veb-backend yoki testlar uchun:
pip install -r backend/requirements.txt
# yoki aynan sinalgan versiyalar:
pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Birinchi ishga tushirishda `facebook/dinov2-base` va `google/siglip2-base-patch16-224`
HuggingFace'dan avtomatik yuklanadi (~1.2 GB).

## Tez boshlash

```bash
python scripts/demo.py                    # offline sintetik demo
python scripts/onboard_and_analyze.py     # real javon: few-shot onboarding + tahlil
python -m pytest tests -q
```

## Veb-ilova

- **Super admin** platforma statistikasi (`/platform`) va korxonalarni (`/organizations`) boshqaradi. Platforma hamda har bir korxona uchun 7, 30 va 90 kunlik audit statistikasi, tekshiruv navbati va boshqaruv tarixi mavjud.
- **Korxona admini** (`/dashboard`) o‘z korxonasi statistikasini ko‘radi, katalog (`/admin`), do‘konlar (`/stores`) va operatorlar (`/members`)ni boshqaradi hamda auditlarni ko‘rib chiqadi.
- **Agent** (`/field`) biriktirilgan do‘konlar va topshiriqlarni ko‘radi. Ochiq tashrifda fotoaudit yuboradi, topshirilgan SKU va dona sonini qayd etadi, topshiriqlarni va tashrifni yakunlaydi. O‘z auditlari va admin izohlari `/reports` sahifasida. Mavjud `operator` hisoblari Agent sifatida ishlaydi.
- **Tashriflar** (`/field`) korxona adminiga agentga do‘kon biriktirish, fotoaudit/tovar/boshqa ish turidagi topshiriq berish va tashrif tafsilotini ko‘rish imkonini beradi. **Do‘konlar** (`/stores`) tashriflar, faol agentlar, auditlar, topshiriqlar va topshirilgan tovar bo‘yicha davr statistikasini ko‘rsatadi.
- **Hisobotlar** (`/reports`) holat, do‘kon va Toshkent vaqti bo‘yicha sana oralig‘ida filtrlanadi. Korxona admini yuborgan xodim bo‘yicha ham qidiradi va 10 000 tagacha yuborilgan auditni CSV qilib yuklaydi. Bosh paneldagi holat, do‘kon va xodim havolalari shu filtrlarga olib boradi.
- **Hisob** (`/account`) joriy parolni almashtiradi va eski sessiyalarni yakunlaydi.

**Frontend** React va Vite yordamida yig‘iladi, Nginx orqali uzatiladi. **Backend** FastAPI orqali hisoblar, katalog, javon geometriyasi va KPI’larni boshqaradi. **AI** xizmati ichki tarmoqda YOLO deteksiyasi va embeddinglarni hisoblaydi. Brauzer frontend portiga ulanadi. PostgreSQL hisoblar, korxonalar, auditlar va tashriflar uchun ishlaydi. Mahsulot vektorlari Qdrant’da saqlanadi.

Sessiya korxona katalogi, SKU rasmlari va auditlar doirasini belgilaydi. Oldingi Docker admin hisoblari birinchi ishga tushishda mavjud katalog bilan **Asosiy korxona**ga biriktiriladi. Mahsulot rejasi va joylashtirish chegaralari [PRODUCT.md](PRODUCT.md) faylida.

Katalogda mahsulotlar nom, SKU, brend va shtrix-kod bo‘yicha qidiriladi. Natijani kategoriya, brend yoki namuna rasmlari bo‘yicha filtrlash va CSV formatida yuklash mumkin. Korxona admini SKU, kategoriya, shtrix-kod va AI namunalarini boshqaradi. Super admin katalogni korxona kesimida ko‘radi va eksport qiladi. Agent tovar topshirish vaqtida mahsulotni nom, SKU yoki shtrix-kod bo‘yicha topadi.

## Docker’da ishga tushirish

1. `config.yaml` dagi detektor og‘irliklarini `models/weights/` ga joylashtiring. `models/negatives/aptash_v1.npz` ("boshqa mahsulotlar" bazasi, Git LFS) — kichik katalogda begona mahsulotlarning katalogdagi SKU deb tanilishini to‘xtatadi; yuklanganini `/api/health` dagi `negatives` ko‘rsatadi.
2. `.env.example` ni `.env` ga nusxalang. `python -c "import secrets; print(secrets.token_urlsafe(48))"` bilan token yarating va `RASTA_AI_TOKEN` sifatida kiriting.
3. NVIDIA GPU mavjud bo‘lsa Docker GPU integratsiyasini yoqing. CPU’da ishlatish uchun Compose faylidagi `gpus: all` qatorini olib tashlang.

```bash
docker build -f ai/Dockerfile.base -t rasta-ai-base:1 .
docker compose up -d --build
docker compose exec backend python scripts/manage_users.py superadmin --role super_admin
```

Brauzerda **http://127.0.0.1:8090/login** ni oching. Super admin **Korxonalar** sahifasida korxona va uning birinchi adminini yaratadi. Korxona admini birinchi do‘kon va operatorni qo‘shadi. Compose portini `RASTA_PORT` orqali sozlash mumkin.

- `rasta-catalog`, `rasta-images` va `rasta-accounts` volume’larini birga zaxiralang. `docker compose down -v` ularni o‘chiradi.
- Yangilash: `git pull`, so‘ng `docker compose up -d --build`. `ai/Dockerfile.base` o‘zgargan bo‘lsa (yangi model qo‘shilgan) avval `docker build -f ai/Dockerfile.base -t rasta-ai-base:1 .` ni qayta bajaring — faqat yangi qatlam yuklanadi. Tanish mantig‘i yoki embedding o‘zgarganda korxona admini katalogda **Qayta o‘rganish** ni bosadi.
- Tahlil qilingan har rasm va detektor qutilari `rasta-archive` volume’iga yoziladi (o‘z detektoringizni o‘qitish uchun: `tools/box_editor.html` → `scripts/export_yolo.py`). O‘chirish: `.env` da `RASTA_ARCHIVE=0`.
- Tayyor katalogni korxonaga berish: **Katalog** sahifasida **Katalog fayli** katalogni rasmlari bilan bitta `.zip` ga saqlaydi, **Katalog faylini yuklash** uni boshqa korxona katalogiga qo‘shadi (faqat korxona admini; katalogda bor mahsulotlar o‘zgarmaydi, vektorlar joyida qayta hisoblanadi).
- Qidiruv doirasi: tahlil sahifasida katalogdan javonda borlarini belgilash mumkin — tizim faqat ular orasidan izlaydi va boshqa mahsulotlar bilan adashmaydi (noto‘g‘ri moslik 2–10 baravar kamayadi, BENCHMARK.md 11g). Belgilanmagan mahsulot javonda bo‘lsa ham tanilmaydi, shuning uchun tanlov sahifada doim ko‘rinib turadi.
- Yorliq yozuvini o‘qish (ixtiyoriy, CPU): `.env` ga `RASTA_OCR_URL=http://ocr:8013` yozing va `docker compose --profile ocr up -d --build` bilan ishga tushiring (keyingi yangilashlarda ham `--profile ocr` bilan). Tizim brendni topib, ta’m yoki hajmda ikkilangan qutilarda yorliqdagi yozuvni o‘qiydi va katalog nomidagi so‘zlar bilan solishtiradi; shuning uchun mahsulot nomiga ta’m va hajmni yozing (“Ays Tea Shaftoli 1.25L”) va brendni to‘ldiring. Bundan tashqari, chegaraga yaqin ball bilan tanilgan qutilarning yozuvi shu mahsulotning katalogdagi namuna rasmida o‘qilgan yozuv bilan solishtiriladi: qadoqda yirik va aniq boshqa so‘z tursa («Florida» qadog‘i «Cookies» deb tanilgan), moslik bekor qilinadi. Namuna yozuvi mahsulot qo‘shilganda o‘qiladi; OCR ulanishidan oldin qo‘shilgan mahsulotlar uchun esa birinchi tahlilda o‘zi o‘qiladi va saqlanadi (bir tahlilda 3 tagacha mahsulot, har biri bir marta — birinchi tahlil sekinroq bo‘ladi). Yozuv xizmati javob bermasa, natijada tekshiruv o‘tkazilmagani aytiladi. Ulanganini `/api/health` dagi `ocr` ko‘rsatadi; o‘chirish — `RASTA_OCR_URL` ni bo‘sh qoldirish. O‘lchovlar: BENCHMARK.md 11f, 11g.
- Xizmat loglari uchun `docker compose logs -f frontend backend ai` ishlating. Tayyorlik holati `/api/health` da ko‘rinadi.
- Domen bilan chiqarishda frontendni HTTPS reverse proxy ortiga joylashtiring va `RASTA_SECURE_COOKIES=1` belgilang.
- Lokal frontend uchun Node.js 22.12+ bilan `npm --prefix frontend ci` va `npm --prefix frontend run dev` ni bajaring. Vite API so‘rovlarini odatda `127.0.0.1:8000` ga uzatadi. FastAPI’ni `python scripts/serve.py --api-only` orqali oching. Bitta portda ishlatish uchun frontendni `npm --prefix frontend run build` bilan yig‘ing va `python scripts/serve.py` ni ishga tushiring.
- Korxonali SQLite rejimi uchun `RASTA_ACCOUNTS_DB` va `RASTA_WORKSPACES=1` belgilang. PostgreSQL bilan ishga tushirish va mavjud bazani ko‘chirish tartibi [SERVER.md](SERVER.md#postgresql) faylida.

OCR yoqilganda, katalog nomida ta’m yoki hajm ko‘rsatilgan SKU uchun qadoqdagi
aniq zid yozuv alohida ko‘rsatiladi. Masalan, katalogda «Nash Sad malina 1 L»
bo‘lsa, qadoqda «nok» yoki «500 ml» o‘qilgan mahsulot shu SKU hisobiga kirmaydi.
Deteksiya tafsilotida o‘qilgan yozuv va katalogdagi eng yaqin mahsulot ko‘rinadi.

Brendning barcha ta’mlarini bitta hisobda yuritish uchun mahsulot nomi va
brend maydoniga bir xil nom kiriting (masalan, «Nash Sad»). Shu brend katalogda
bitta yozuv bilan yuritilsa va unda shtrix-kod bo‘lmasa, tanish chegaralaridan
o‘tgan variantlar umumiy hisobga qo‘shiladi. Brend yoniga ta’m va hajm
nomi bilan alohida SKU’lar qo‘shilsa, umumiy yozuv brend namunasi bo‘lib
xizmat qiladi. Orqa yorliq namunasini shu umumiy yozuvga qo‘shish mumkin.
Tahlilda brend jami, tanilgan variantlar va varianti aniqlanmagan mahsulotlar soni ko‘rinadi.

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
| `clustering` | HDBSCAN, rang/shakl og'irliklari, guruhsiz mahsulotlar uchun kosinus masofasi chegarasi |
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
frontend/src/pages/    # React sahifalari va umumiy navigatsiya
frontend/src/controllers/ # murakkab sahifalardagi mavjud amallar va canvas integratsiyasi
backend/rasta_backend/ # FastAPI, biznes qoidalari va SQL migratsiyalari
ai/rasta_ai/           # ichki AI inference servisi
ocr/rasta_ocr/         # ixtiyoriy: yorliq yozuvini o'qiydigan xizmat (PaddleOCR, CPU)
src/rasta/             # servislar va CLI ishlatadigan javon tahlili kutubxonasi
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
