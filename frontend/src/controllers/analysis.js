export default function mountPage() {
  const $ = id => document.getElementById(id);
  const cv = $("cv"), ctx = cv.getContext("2d");
  const preview = document.createElement("canvas");
  const previewContext = preview.getContext("2d");
  const previewMaxSide = 1800;
  let drawFrame = 0;
  let file = null, img = null, pts = [], drawing = false, closed = false;
  let R = null, showBoxes = true, hl = null, hlGroup = null, hlFlavour = null, analysis = null, pendingImage = null;
  let canTeach = false;
  let activeVisit = null;
  let agent = false;
  const sendForm = $("sendForm");
  rastaSession.then(async session => {
    canTeach = !session.workspaces || session.user?.role === "org_admin";
    $("historyLink").hidden = !session.workspaces;
    $("submitPhase").hidden = !session.workspaces;
    agent = session.workspaces && session.user?.role === "operator";
    if (!agent) return;
    $("start").hidden = true;
    document.querySelector(".analysis-flow").hidden = true;
    $("visitLink").hidden = false;
    try {
      const { visits } = await rastaJson("/api/field");
      const requested = new URLSearchParams(location.search).get("visit");
      const visit = visits.find(row => !row.finished && (!requested || String(row.id) === requested));
      if (visit) {
        activeVisit = visit;
        $("visitLink").href = `/field?id=${visit.id}`;
        $("start").hidden = false;
        document.querySelector(".analysis-flow").hidden = false;
        return;
      }
    } catch (e) {
      $("visitPromptText").textContent = e.message;
    }
    $("visitPrompt").hidden = false;
  });
  function phase(current) {
    for (const item of document.querySelectorAll("[data-phase]")) {
      if (item.dataset.phase === current) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    }
  }
  // ---------- qidiruv doirasi: javonda bor mahsulotlar ----------
  let catalogItems = [];          // tahlil uchun katalog ro'yxati
  let scope = null;               // null — butun katalog; Set — faqat shu SKU'lar orasidan izlash
  async function loadScope() {
    try {
      const r = await rastaFetch("/api/analyze/products");
      if (!r.ok) return;
      catalogItems = (await r.json()).products || [];
    } catch (e) { return; }
    if (scope) for (const sku of [...scope]) if (!catalogItems.some(p => p.sku_id === sku)) scope.delete(sku);
    renderScope();
  }
  function scopeSummary() {
    const total = catalogItems.length;
    $("scopeCard").classList.toggle("on", scope !== null);
    $("scopeSummary").textContent = scope === null
      ? `Butun katalog · ${total} ta mahsulot`
      : scope.size ? `Faqat tanlangan ${scope.size} ta mahsulot orasidan (katalogda ${total} ta)`
                   : "Hech narsa tanlanmagan — kamida bitta mahsulotni belgilang";
  }
  function renderScope() {
    $("scopeCard").hidden = catalogItems.length < 2;          // bitta mahsulot bo'lsa tanlovning ma'nosi yo'q
    scopeSummary();
    if ($("scopeBody").hidden) return;
    const query = $("scopeSearch").value.trim().toLowerCase();
    const shown = catalogItems.filter(p => !query || `${p.name} ${p.brand || ""}`.toLowerCase().includes(query));
    const brands = new Map();
    for (const p of shown) {
      const brand = p.brand || "Brendsiz";
      if (!brands.has(brand)) brands.set(brand, []);
      brands.get(brand).push(p);
    }
    const on = sku => scope === null || scope.has(sku);
    $("scopeList").innerHTML = shown.length ? [...brands].map(([brand, items]) => `
      <div class="scope-brand"><span>${esc(brand)}</span>
        <button class="btn ghost" type="button" data-brand="${esc(brand)}">${items.every(p => on(p.sku_id)) ? "Olib tashlash" : "Belgilash"}</button></div>
      ${items.map(p => `<label class="scope-row"><input type="checkbox" data-sku="${esc(p.sku_id)}"${on(p.sku_id) ? " checked" : ""}>
        ${p.thumb ? `<img class="th" src="${p.thumb}" alt="" loading="lazy">` : '<span class="th"></span>'}<span class="nm">${esc(p.name)}</span></label>`).join("")}`).join("")
      : '<p class="analysis-help">Hech narsa topilmadi.</p>';
    const apply = skus => {
      for (const sku of skus.keys) {
        if (scope === null) scope = new Set(catalogItems.map(p => p.sku_id));
        skus.value ? scope.add(sku) : scope.delete(sku);
      }
      if (scope && scope.size === catalogItems.length) scope = null;       // hammasi belgilangan — cheklov yo'q
      $("err").textContent = "";
      renderScope();
    };
    $("scopeList").querySelectorAll("input[data-sku]").forEach(box => {
      box.onchange = () => apply({ keys: [box.dataset.sku], value: box.checked });
    });
    $("scopeList").querySelectorAll("button[data-brand]").forEach(button => {
      button.onclick = () => {
        const items = brands.get(button.dataset.brand);
        apply({ keys: items.map(p => p.sku_id), value: !items.every(p => on(p.sku_id)) });
      };
    });
  }
  // Zich javonda rasm kichik bo'lib qoladi: bir bosishda butun kenglikka kattalashtirish
  $("wideBtn").onclick = () => {
    const wide = $("work").classList.toggle("wide");
    $("wideBtn").setAttribute("aria-pressed", String(wide));
    $("wideBtn").querySelector("span").textContent = wide ? "Kichraytirish" : "Kattalashtirish";
    $("wideBtn").querySelector("svg").innerHTML = rastaIconPath(wide ? "shrink" : "expand");
    requestAnimationFrame(() => { draw(); $("work").scrollIntoView({ block: "nearest" }); });
  };
  $("scopeToggle").onclick = () => {
    const open = $("scopeBody").hidden;
    $("scopeBody").hidden = !open;
    $("scopeToggle").setAttribute("aria-expanded", String(open));
    $("scopeToggle").textContent = open ? "Yopish" : "Tanlash";
    if (open) renderScope();
  };
  $("scopeSearch").oninput = renderScope;
  $("scopeAll").onclick = () => { scope = null; $("err").textContent = ""; renderScope(); };
  $("scopeNone").onclick = () => { scope = new Set(); renderScope(); };
  loadScope();

  const sel = new Set();          // o'rgatish uchun tanlangan qutilar
  let picked = -1;                // oxirgi bosilgan quti (nega tanildi/tanilmadi izohi uchun)
  function whyUi() {
    const why = picked >= 0 && R ? R.detections[picked]?.why : null;
    $("why").hidden = !why;
    if (!why) return;
    const pct = value => Math.round(value * 100) + "%";
    const near = why.n ? `eng yaqin mahsulot — ${why.n}, o‘xshashlik ${pct(why.sc)} (chegara ${pct(why.tau)})` : "";
    const checked = !why.lc ? "" : why.lc.startsWith("support:")
      ? ` Yozuv tekshirildi: qadoqda «${why.lc.slice(8)}» o‘qildi — mos.`
      : " Yozuv bo‘yicha tekshirildi, lekin aniq matn o‘qilmadi — ko‘rinishi bo‘yicha qoldirildi.";
    $("whyText").textContent = ({
      ok: `Tanildi: ${why.n}. O‘xshashlik ${pct(why.sc)}, chegara ${pct(why.tau)}.${checked}`,
      family: `${why.n} umumiy hisobiga qo‘shildi. Katalogda bu brend bitta yozuv bilan yuritiladi.${checked}`,
      family_context: `${why.n} umumiy hisobiga qo‘shildi. Yonidagi tanilgan mahsulotga o‘xshashligi ham hisobga olindi.${checked}`,
      text: `Yorliqdagi yozuv bo‘yicha aniqlandi: ${why.n}${why.w ? ` (o‘qilgan: «${why.w}»)` : ""}. Brend ko‘rinishidan tanilgan, o‘xshashlik ${pct(why.sc)}.`,
      context: `Qo‘shni mahsulotlarga qarab tanildi: ${why.n} (o‘xshashlik ${pct(why.sc)}, chegara ${pct(why.tau)}).${checked}`,
      variant: `${why.n} ga o‘xshaydi (o‘xshashlik ${pct(why.sc)}), lekin rangi farq qiladi (rang mosligi ${pct(why.c ?? 0)}). Bu shu mahsulotning boshqa ta’mi yoki turi bo‘lishi mumkin — alohida mahsulot sifatida qo‘shing.`,
      label: `Tanilmadi: ko‘rinishi ${why.n} ga o‘xshaydi (o‘xshashlik ${pct(why.sc)}), lekin qadoqda${why.w ? ` «${why.w}»` : " boshqa so‘z"} yozilgan — bu boshqa mahsulot.`,
      label_brand: `Faqat brend aniqlandi: brend yozuvi ${why.n} bilan bir xil, lekin qadoqda${why.w ? ` «${why.w}»` : " boshqa nom"} yozilgan — shu brendning boshqa mahsuloti.`,
      label_flavor: `Boshqa ta’m aniqlandi. Yorliqda «${why.w}» o‘qildi. Katalogdagi eng yaqin mahsulot — ${why.n}.`,
      label_volume: `Boshqa hajm aniqlandi. Yorliqda ${why.w} o‘qildi. Katalogdagi eng yaqin mahsulot — ${why.n}.`,
      brand_text: `Brend yozuv bo‘yicha aniqlandi: «${why.w}». Ta’mi yoki mahsulot turi aniqlanmagan — brend hisobiga qo‘shildi.`,
      brand: `Brend aniqlandi. Ta’m yoki hajmni ishonchli ajratish uchun dalil yetarli emas.`,
      brand_context: `${why.brand} deb tanildi. Shu klasterdagi tanilgan qadoqlar bilan mos. Ta’mi yoki hajmi noaniq.`,
      empty: "Katalog bo‘sh — taqqoslash uchun mahsulot yo‘q.",
      low: `Tanilmadi: ${near}. Bu mahsulot katalogda yo‘q yoki uning namunasi yetarli emas.`,
      weak: `Tanilmadi: ${near}, lekin bu asosan rangi va shakli o‘xshagani uchun — yorlig‘i va ko‘rinishi mos emas (${pct(why.e ?? 0)}).`,
      color: `Tanilmadi: ${near}, lekin rangi mos kelmadi (${pct(why.c ?? 0)}).`,
      other: `Tanilmadi: ${near}, lekin u katalogda yo‘q boshqa mahsulotga ko‘proq o‘xshaydi.`,
      ambiguous: `Tanilmadi: ${near}, lekin ikkinchi nomzoddan farqi juda kichik (${pct(why.m)}).`,
      size: `Tanilmadi: ${near}, lekin o‘lchami shu mahsulotning rasmdagi boshqa donalaridan keskin farq qiladi.`,
    })[why.r] || "";
  }
  const gapLayers = [
    { key: "gaps", label: "Bo‘sh joy", color: "#dc2626", dash: [6, 4] },
    { key: "suspect_gaps", label: "Tekshirish kerak", color: "#d97706", dash: [5, 3] },
    { key: "divider_gaps", label: "Ehtimoliy bo‘linma", color: "#64748b", dash: [3, 3] },
  ];

  // ---------- rasm ----------
  function pick(f) {
    if (!f || (agent && !activeVisit)) return;
    const error = $("fileErr");
    error.hidden = false;
    if (f.size > 20 * 1024 * 1024) { error.textContent = "Fayl 20 MB dan katta."; return; }
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      error.textContent = "JPG, PNG yoki WebP rasm tanlang."; return;
    }
    error.hidden = true;
    const next = new Image();
    pendingImage = next;
    const url = URL.createObjectURL(f);
    next.onload = () => {
      URL.revokeObjectURL(url);
      if (pendingImage !== next) return;
      if (next.naturalWidth * next.naturalHeight > 50_000_000) {
        error.hidden = false;
        error.textContent = "Rasm o‘lchami 50 megapikseldan katta.";
        return;
      }
      analysis?.abort();
      img = next; file = f; pts = []; closed = false; drawing = false;
      resetResult(); R = null; hl = null;
      const scale = Math.min(1, previewMaxSide / Math.max(img.naturalWidth, img.naturalHeight));
      cv.width = preview.width = Math.max(1, Math.round(img.naturalWidth * scale));
      cv.height = preview.height = Math.max(1, Math.round(img.naturalHeight * scale));
      previewContext.drawImage(img, 0, 0, preview.width, preview.height);
      $("photoName").textContent = f.name;
      $("photoInfo").textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
      phase("photo");
      $("start").hidden = true; $("work").hidden = false;
      $("pre").hidden = false; $("res").hidden = true; $("boxesBtn").hidden = true;
      $("err").textContent = ""; ui(); draw();
    };
    next.onerror = () => {
      URL.revokeObjectURL(url);
      if (pendingImage !== next) return;
      error.hidden = false;
      error.textContent = "Rasmni ochib bo‘lmadi. Boshqa fayl tanlang.";
    };
    next.src = url;
  }
  $("choose").onclick = () => $("file").click();
  $("cameraBtn").onclick = () => $("cameraFile").click();
  for (const input of [$("file"), $("cameraFile")]) {
    input.onchange = e => { pick(e.target.files[0]); e.target.value = ""; };
  }
  const drop = $("drop");
  ["dragover", "dragenter"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, () => drop.classList.remove("over")));
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", e => { e.preventDefault(); if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
  $("newBtn").onclick = () => { $("file").value = ""; $("file").click(); };

  // ---------- chizish ----------
  const k = () => cv.width / cv.getBoundingClientRect().width;      // ekran -> rasm
  function draw() {
    if (!img || drawFrame) return;
    drawFrame = requestAnimationFrame(() => {
      drawFrame = 0;
      if (cv.clientWidth) paint(ctx, preview);
    });
  }
  function paint(ctx, source) {
    const s = k(), lw = 2 * s;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(source, 0, 0, cv.width, cv.height);
    const zone = R ? R.roiPts : (closed ? pts : null);
    if (zone) {                                   // zona tashqarisini xiralashtirish
      ctx.save(); ctx.fillStyle = "rgba(15,23,42,.5)"; ctx.beginPath();
      ctx.rect(0, 0, cv.width, cv.height);
      zone.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath(); ctx.fill("evenodd"); ctx.restore();
    }
    if (R && showBoxes) {
      const [scaleX, scaleY] = R.scale;
      const scaledBox = box => box.map((v, i) => v * (i % 2 ? scaleY : scaleX));
      ctx.save();
      ctx.lineWidth = lw;
      for (const { key, color, dash } of gapLayers) {
        ctx.setLineDash(dash.map(v => v * s));
        ctx.strokeStyle = color;
        ctx.fillStyle = color + (hl ? "18" : "30");
        for (const b of R[key] || []) {
          const [x1, y1, x2, y2] = scaledBox(b);
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }
      }
      ctx.restore();
      R.detections.forEach((d, i) => {
        const [x1, y1, x2, y2] = scaledBox(d.b);
        const col = d.s ? R.colors[d.s] : null;
        if ((hlGroup !== null && R.groupOf[i] === hlGroup) || (hlFlavour !== null && R.flavourOf?.[i] === hlFlavour)) {      // ro'yxatda ustiga borilgan guruh
          ctx.lineWidth = lw * 1.8; ctx.strokeStyle = "#22d3ee"; ctx.fillStyle = "#22d3ee33";
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); return;
        }
        const on = !hl || d.s === hl;
        if (!col && (d.v || d.other_flavor)) {   // boshqa ta’m yoki rang: uzuq sariq ramka
          ctx.save(); ctx.globalAlpha = hl ? .3 : 1;
          ctx.setLineDash([lw * 3, lw * 2]); ctx.lineWidth = lw * 1.1; ctx.strokeStyle = "#f59e0b";
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.restore(); return;
        }
        if (!col) {
          ctx.lineWidth = Math.max(1, lw * .6); ctx.strokeStyle = on && !hl ? "rgba(255,255,255,.75)" : "rgba(255,255,255,.25)";
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); return;
        }
        ctx.globalAlpha = on ? 1 : .22;
        if (hl && on) { ctx.fillStyle = col + "33"; ctx.fillRect(x1, y1, x2 - x1, y2 - y1); }
        ctx.lineWidth = hl && on ? lw * 1.8 : lw * 1.2; ctx.strokeStyle = col;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.globalAlpha = 1;
      });
      if (picked >= 0 && !sel.has(picked) && R.detections[picked]) {
        const [x1, y1, x2, y2] = scaledBox(R.detections[picked].b);
        ctx.lineWidth = lw * 2; ctx.strokeStyle = "#f8fafc"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      sel.forEach(i => {
        const [x1, y1, x2, y2] = scaledBox(R.detections[i].b);
        ctx.fillStyle = "rgba(16,185,129,.28)"; ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.lineWidth = lw * 2; ctx.strokeStyle = "#10b981"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      });
    }
    if (zone || (drawing && pts.length)) {
      const z = zone || pts;
      ctx.beginPath();
      z.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      if (zone) ctx.closePath();
      ctx.lineWidth = lw * 1.3; ctx.strokeStyle = "#10b981"; ctx.stroke();
      if (!R) z.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, 5 * s, 0, 7);
        ctx.fillStyle = i ? "#10b981" : "#fff"; ctx.fill(); ctx.lineWidth = 2 * s; ctx.strokeStyle = "#10b981"; ctx.stroke(); });
    }
  }
  window.addEventListener("resize", draw);

  // ---------- zona ----------
  function ui() {
    cv.classList.toggle("draw", drawing);
    $("zoneBtn").hidden = drawing || closed;
    $("zoneClear").hidden = !(drawing || closed);
    $("zoneClear").lastChild.textContent = closed ? " Zonani olib tashlash" : " Bekor qilish";
    $("zoneHint").textContent = drawing
      ? (pts.length < 3 ? "Muzlatgich/javon burchaklarini bosing" : "Yopish uchun birinchi nuqtani bosing")
      : (closed ? "Faqat zona ichi tahlil qilinadi" : "");
  }
  function resetResult() {
    analysis?.abort();
    phase("photo");
    sel.clear(); picked = -1;
    $("teach").hidden = true; $("why").hidden = true;
    $("gapLegend").hidden = true;
    if (R) { R = null; hl = null; $("res").hidden = true; $("pre").hidden = false; $("boxesBtn").hidden = true; $("wideBtn").hidden = true; }
  }
  $("zoneBtn").onclick = () => { resetResult(); drawing = true; pts = []; closed = false; ui(); draw(); };
  $("zoneClear").onclick = () => { resetResult(); drawing = false; pts = []; closed = false; ui(); draw(); };
  cv.addEventListener("click", e => {
    const r = cv.getBoundingClientRect();
    const imageX = (e.clientX - r.left) / r.width, imageY = (e.clientY - r.top) / r.height;
    if (!drawing) {
      if (!R || !showBoxes) return;
      const x = imageX * R.data.size[0], y = imageY * R.data.size[1];
      let best = -1, area = Infinity;
      R.detections.forEach((d, i) => {
        const [x1, y1, x2, y2] = d.b, a = (x2 - x1) * (y2 - y1);
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && a < area) { best = i; area = a; }
      });
      if (best < 0) return;
      picked = picked === best && !canTeach ? -1 : best;
      whyUi();
      if (canTeach) { sel.has(best) ? sel.delete(best) : sel.add(best); teachUi(); }
      draw();
      return;
    }
    const s = k();
    const p = [
      Math.round(imageX * img.naturalWidth) / img.naturalWidth * cv.width,
      Math.round(imageY * img.naturalHeight) / img.naturalHeight * cv.height,
    ];
    if (pts.length >= 3 && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) < 14 * s) { closed = true; drawing = false; }
    else pts.push(p);
    ui(); draw();
  });
  $("boxesBtn").onclick = () => {
    showBoxes = !showBoxes;
    $("boxesBtn").classList.toggle("on", showBoxes);
    $("gapLegend").hidden = !showBoxes || !$("gapLegend").children.length;
    draw();
  };

  // ---------- tahlil ----------
  async function analyze(btn) {
    if (!file || btn.disabled || (analysis && !analysis.signal.aborted)) return;
    if (drawing) { $("err").textContent = "Zonani yoping yoki bekor qiling."; return; }
    if (scope !== null && !scope.size) {
      $("err").textContent = "Qidiruv doirasida kamida bitta mahsulot belgilang yoki “Hammasi”ni bosing.";
      $("pre").hidden = false; return;
    }
    const fd = new FormData(); fd.append("image", file);
    if (activeVisit) fd.append("visit_id", String(activeVisit.id));
    if (scope !== null) fd.append("skus", JSON.stringify([...scope]));
    if (closed) {
      fd.append("roi", JSON.stringify(pts.map(([x, y]) => [x / cv.width, y / cv.height])));
      fd.append("roi_normalized", "true");
    }
    analysis?.abort();
    const request = new AbortController();
    analysis = request;
    const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Tahlil qilinmoqda…';
    $("err").textContent = "";
    const loader = window.rastaLoader(cv.parentElement, [
      "Rasm tayyorlanmoqda", "Mahsulotlar aniqlanmoqda", "Qatorlar belgilanmoqda",
      "Katalog bilan solishtirilmoqda", "Sanalmoqda"], { scan: true });
    try {
      const r = await rastaFetch("/api/analyze", { method: "POST", body: fd, signal: request.signal });
      const d = await r.json().catch(() => null);
      if (!r.ok || !d) throw new Error(typeof d?.detail === "string" ? d.detail : "Tahlilni yakunlab bo‘lmadi.");
      if (!request.signal.aborted) show(d);
    } catch (e) {
      if (!request.signal.aborted) {
        $("err").textContent = e.message || "Server bilan bog‘lanib bo‘lmadi.";
        $("pre").hidden = false;
      }
    } finally {
      loader.stop();
      if (analysis === request) analysis = null;
      btn.disabled = false;
      btn.innerHTML = old;
    }
  }
  $("go").onclick = () => analyze($("go"));
  $("again").onclick = () => analyze($("again"));

  function download(blob, suffix) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (file?.name.replace(/\.[^.]+$/, "") || "rasta") + suffix;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  $("exportJson").onclick = () => {
    if (R) download(new Blob([JSON.stringify(R.data, null, 2)], { type: "application/json" }), "-rasta.json");
  };
  $("exportImage").onclick = () => {
    if (!R) return;
    const output = document.createElement("canvas");
    output.width = img.naturalWidth;
    output.height = img.naturalHeight;
    const outputContext = output.getContext("2d");
    outputContext.scale(output.width / cv.width, output.height / cv.height);
    paint(outputContext, img);
    output.toBlob(blob => {
      if (blob) download(blob, "-rasta.png");
      output.width = output.height = 0;
    });
  };

  function kpi(ic, label, value, extra = "", cls = "") {
    return `<div class="kpi ${cls}"><div class="t">${icon(ic)}${label}</div><div class="v">${value}</div>${extra}</div>`;
  }
  function donut(items, rest) {
    const C = 2 * Math.PI * 46; let off = 0, out = "";
    const segs = items.map(p => ({ v: p.share_pct, c: p.color, id: p.sku_id }));
    if (rest > 0.5) segs.push({ v: rest, c: "var(--border-strong)", id: "" });
    const tot = segs.reduce((a, s) => a + s.v, 0) || 1;
    segs.forEach(sg => {
      const len = C * sg.v / tot;
      out += `<circle cx="60" cy="60" r="46" fill="none" stroke="${sg.c}" stroke-width="16"
        stroke-dasharray="${Math.max(0, len - 1.2)} ${C}" stroke-dashoffset="${-off}" transform="rotate(-90 60 60)" data-sku="${sg.id}"/>`;
      off += len;
    });
    return out;
  }
  function show(d) {
    phase("result");
    const s = d.summary;
    const scale = [cv.width / d.size[0], cv.height / d.size[1]];
    R = { ...d, data: d, scale, roiPts: d.roi?.map(([x, y]) => [x * scale[0], y * scale[1]]),
          colors: Object.fromEntries(s.products.map(p => [p.sku_id, p.color])) };
    hl = null; hlGroup = null; showBoxes = true; $("boxesBtn").classList.add("on");
    $("wideBtn").hidden = false;
    $("scopeBody").hidden = true; $("scopeToggle").textContent = "Tanlash"; $("scopeToggle").setAttribute("aria-expanded", "false");
    groupsUi();
    flavourUi();
    $("gapLegend").innerHTML = gapLayers
      .filter(({ key }) => d[key]?.length)
      .map(({ key, color, label }) => `<li><span style="--gap-color:${color}"></span>${label} · ${d[key].length}</li>`)
      .join("");
    $("gapLegend").hidden = !$("gapLegend").children.length;
    sel.clear(); picked = -1; teachUi(); whyUi();
    setupSubmission(d.report_id, d.visit_store_id);

    const exactCount = s.products.reduce((count, product) => count + (product.family ? 0 : product.facings), 0);
    const exactRate = s.num_products ? Math.round(100 * exactCount / s.num_products) : 0;
    const tiles = [
      kpi("box", "Mahsulotlar", s.num_products, `<div class="meter"><i style="width:100%"></i></div>`),
      kpi("check", "SKU tanildi", `${exactRate}<small>%</small>`,
          `<div class="meter"><i style="width:${exactRate}%"></i></div>`, exactRate >= 80 ? "good" : ""),
    ];
    if (s.fill_pct !== null && s.fill_pct !== undefined)
      tiles.push(kpi("layers", "Qatorlar bandligi", `${Math.round(s.fill_pct)}<small>%</small>`,
        `<div class="meter"><i style="width:${s.fill_pct}%;${s.fill_pct < 50 ? "background:var(--danger)" : ""}"></i></div>`, s.fill_pct < 50 ? "bad" : ""));
    const oosAvailable = s.oos_status === "estimated" || s.oos_gaps > 0;
    tiles.push(kpi("empty", "Taxminiy bo‘sh o‘rin", oosAvailable ? s.oos_facings : "—", "", s.oos_facings ? "bad" : ""));
    $("kpis").innerHTML = tiles.join("");

    const ps = s.products;
    $("shareCard").hidden = !ps.length;
    $("donut").innerHTML = donut(ps, s.unknown_share_pct) +
      `<text x="60" y="57" text-anchor="middle" font-size="22" font-weight="700" fill="var(--ink)">${ps.length}</text>
       <text x="60" y="74" text-anchor="middle" font-size="10" fill="var(--muted)">mahsulot turi</text>`;
    const top = ps.slice(0, 5);
    $("legend").innerHTML = top.map(p => `<div class="prow" data-sku="${esc(p.sku_id)}" style="grid-template-columns:minmax(0,1fr) auto;padding:5px 8px">
        <div class="nm" style="font-weight:550;font-size:13.5px"><span class="dot" style="background:${p.color}"></span>${esc(p.name)}</div>
        <div class="r"><b style="font-size:13.5px">${Math.round(p.share_pct)}%</b></div></div>`).join("") +
      (ps.length > 5 ? `<div class="sub" style="padding:4px 8px;color:var(--muted);font-size:12.5px">+ yana ${ps.length - 5} ta</div>` : "") +
      (s.unknown_share_pct > 0.5 ? `<div class="prow" style="grid-template-columns:minmax(0,1fr) auto;padding:5px 8px"><div class="nm" style="font-weight:500;font-size:13.5px;color:var(--muted)"><span class="dot" style="background:var(--border-strong)"></span>SKU aniqlanmagan</div><div class="r" style="color:var(--muted);font-size:13.5px">${Math.round(s.unknown_share_pct)}%</div></div>` : "");

    const max = Math.max(1, ...ps.map(p => p.share_pct));
    const productRow = p => `
      <div class="prow" data-sku="${esc(p.sku_id)}">
        ${p.thumb ? `<img class="th" src="${p.thumb}" alt="" loading="lazy" decoding="async">` : '<span class="th"></span>'}
        <div style="min-width:0">
          <div class="nm"><span class="dot" style="background:${p.color}"></span>${esc(p.name)}</div>
          <div class="sub">${esc(p.brand || "—")}</div>
          <div class="bar"><i style="width:${100 * p.share_pct / max}%;background:${p.color}"></i></div>
        </div>
        <div class="r"><b>${p.facings} ta</b><div>${p.share_pct.toFixed(1)}%</div></div>
      </div>`;
    const brands = (s.brands || []).filter(brand => brand.brand !== "?").map(brand => ({
      ...brand,
      products: brand.products ?? ps.filter(p => p.brand === brand.brand && !p.family),
    }));
    const grouped = new Set(brands.flatMap(brand => brand.products.map(p => p.sku_id)));
    const otherProducts = ps.filter(p => !p.family && !grouped.has(p.sku_id));
    const otherFlavorCount = Math.max(0, (s.num_other_flavor || 0) -
      brands.reduce((count, brand) => count + (brand.other_flavor_facings || 0), 0));
    const countRow = (label, count) => count ? `<div class="prow" style="grid-template-columns:minmax(0,1fr) auto">
      <div class="sub">${label}</div><div class="r">${count} ta</div></div>` : "";
    $("plTitle").textContent = brands.length || otherFlavorCount ? "Brendlar va variantlar" : "Mahsulotlar";
    $("plist").innerHTML = brands.length || otherProducts.length || otherFlavorCount ? brands.map(brand => `
      <div class="prow" style="grid-template-columns:minmax(0,1fr) auto">
        <div class="nm"><strong>${esc(brand.brand)}</strong></div>
        <div class="r"><b>${brand.facings} ta jami</b></div>
      </div>
      ${brand.products.map(productRow).join("")}
      ${countRow("Boshqa ta’m", brand.other_flavor_facings)}
      ${countRow("Ta’mi yoki hajmi noaniq", Math.max(0, brand.unresolved_facings - (brand.other_flavor_facings || 0)))}
      `).join("") + otherProducts.map(productRow).join("") + countRow("Boshqa ta’m · brend noaniq", otherFlavorCount)
      : `<div class="empty">Katalogdagi mahsulotlar topilmadi.${canTeach ? '<br><a href="/admin">Mahsulot qo‘shish →</a>' : ''}</div>`;

    document.querySelectorAll("[data-sku]").forEach(el => {
      if (!el.dataset.sku) return;
      el.addEventListener("mouseenter", () => setHl(el.dataset.sku));
      el.addEventListener("mouseleave", () => setHl(null));
    });

    const notes = [];
    if (canTeach) notes.push(["poly", "Mahsulotni rasmda bosib tanlang va katalogga qo‘shing."]);
    if (d.label_check === "disabled") notes.push(["alert", "Yorliqni o‘qish xizmati ulanmagan. Qadoqdagi ta’m va hajm yozuvi tekshirilmadi.", "warn"]);
    if (d.label_check === "skipped") notes.push(["alert", "Yozuv o‘qish xizmati javob bermadi — qadoqdagi yozuv bo‘yicha tekshiruv bu tahlilda o‘tkazilmadi. Katalogga o‘xshash, lekin boshqa mahsulotlar tanilgan bo‘lishi mumkin.", "warn"]);
    if (d.label_check === "outdated") notes.push(["alert", "Yozuv o‘qish xizmati eski versiyada — qadoqdagi yozuv bo‘yicha tekshiruv ishlamayapti. Serverda “docker compose --profile ocr up -d --build” ni bajaring.", "warn"]);
    if (d.scope) notes.push(["info", `Qidiruv faqat tanlangan ${d.scope} ta mahsulot orasidan o‘tkazildi — katalogdagi boshqa mahsulotlar bu tahlilda tanilmaydi.`]);
    if (s.num_variant > 0) notes.push(["info", `${s.num_variant} ta mahsulot katalogdagiga o‘xshaydi, lekin rangi farq qiladi (sariq uzuq ramka) — boshqa ta’m bo‘lishi mumkin. Ustiga bosib ko‘ring.`]);
    if (s.num_unknown > 0) notes.push(["info", `${s.num_unknown} ta mahsulotning SKU’si aniqlanmadi.`]);
    if (s.oos_status === "limited_scene") notes.push(["alert", "Bo‘sh o‘rinlarni aniqroq baholash uchun javonni old tomondan suratga oling yoki zona belgilang.", "warn"]);
    if (s.oos_status === "insufficient_products") notes.push(["info", "Bo‘sh o‘rinlar bahosi uchun kamida bitta qatorda ikki mahsulot aniqlanishi kerak."]);
    if (s.area_fill_pct != null) notes.push(["layers", `Belgilangan maydonning ${s.area_fill_pct}% qismida mahsulot aniqlandi.`]);
    if (s.row_fill && s.row_fill.length) notes.push(["layers", "Qatorlar bo'yicha to'ldirilganlik (yuqoridan): " + s.row_fill.map(v => Math.round(v) + "%").join(" · ")]);
    $("notes").innerHTML = notes.map(([ic, t, c]) => `<div class="note ${c || ""}">${icon(ic)}<div>${t}</div></div>`).join("");

    $("pre").hidden = true; $("res").hidden = false; $("boxesBtn").hidden = false;
    draw();
  }

  async function setupSubmission(reportId, visitStoreId) {
    $("sendPanel").hidden = !reportId;
    if (!reportId) return;
    sendForm.reset();
    sendForm.hidden = false;
    $("sentMessage").hidden = true;
    $("sendBtn").disabled = true;
    $("sendError").textContent = "";
    try {
      const { stores } = await rastaJson("/api/stores");
      if (R?.report_id !== reportId) return;
      const available = visitStoreId ? stores.filter(store => store.id === visitStoreId) : stores;
      sendForm.elements.store.replaceChildren(new Option("Do‘konni tanlang", ""), ...available.map(store => new Option(store.name, store.id)));
      if (available.length === 1) sendForm.elements.store.value = String(available[0].id);
      $("sendBtn").disabled = available.length === 0;
      if (!available.length) $("sendError").textContent = "Do‘kon biriktirilishini korxona adminidan tekshirtiring.";
    } catch (error) {
      if (R?.report_id === reportId) $("sendError").textContent = error.message;
    }
  }
  sendForm.onsubmit = async event => {
    event.preventDefault();
    const reportId = R?.report_id;
    if (!reportId) return;
    $("sendBtn").disabled = true;
    $("sendError").textContent = "";
    try {
      await rastaJson(`/api/reports/${reportId}/submit`, "POST", {
        store_id: Number(sendForm.elements.store.value), note: sendForm.elements.note.value
      });
      if (R?.report_id !== reportId) return;
      sendForm.hidden = true;
      $("sentMessage").hidden = false;
      phase("submit");
      $("sentMessage").querySelector("a").href = `/reports?id=${encodeURIComponent(reportId)}`;
    } catch (error) {
      if (R?.report_id === reportId) {
        $("sendError").textContent = error.message;
        $("sendBtn").disabled = false;
      }
    }
  };
  // ---------- javondan o'rgatish ----------
  const token = () => { try { return sessionStorage.getItem("rasta-admin") || ""; } catch (e) { return ""; } };
  let products = [];
  let productsLoading = null;       // bir vaqtda bir necha chaqiruv ro'yxatni bir-birining ustiga yozmasin
  function loadProducts() {
    return productsLoading ||= (async () => {
      try {
        const response = await rastaFetch("/api/products");
        products = response.ok ? (await response.json()).products : [];
      } catch (e) { products = []; }
      $("teachSku").innerHTML = products.map(p => `<option value="${esc(p.sku_id)}">${esc(p.name)}${p.brand ? " · " + esc(p.brand) : ""}</option>`).join("")
        + `<option value="__new">+ Yangi mahsulot</option>`;
      $("teachNew").hidden = $("teachSku").value !== "__new";
    })().finally(() => { productsLoading = null; });
  }
  // Guruhni tanlagach: «Yangi mahsulot» rejimi (ro'yxat yuklanmagan bo'lsa avval yuklanadi), nom va brend to'ldiriladi
  async function startNewProduct(name = "", brand = "") {
    if (!$("teachSku").querySelector('option[value="__new"]')) await loadProducts();
    $("teachSku").value = "__new"; $("teachNew").hidden = false;
    if (name) $("teachName").value = name;
    if (brand) $("teachBrand").value = brand;
    $("teach").scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  $("teachSku").onchange = () => { $("teachNew").hidden = $("teachSku").value !== "__new"; };
  // Katalogda yo'q, lekin bir-biriga o'xshash qutilar guruhlari: bitta bosishda hammasi tanlanadi
  function groupsUi() {
    const members = new Map();
    R.detections.forEach((d, i) => { if (d.g != null && !d.s) members.set(d.g, [...(members.get(d.g) || []), i]); });
    R.groups = [...members.values()].filter(list => list.length > 1).sort((a, b) => b.length - a.length);
    R.groupOf = {};
    R.groups.forEach((list, g) => list.forEach(i => { R.groupOf[i] = g; }));
    $("groupsCard").hidden = !R.groups.length;
    if (!R.groups.length) return;
    const shown = R.groups.slice(0, 12), boxes = R.groups.reduce((sum, list) => sum + list.length, 0);
    $("groupsTitle").textContent = `Katalogda yo‘q · ${R.groups.length} guruh`;
    $("groupsHelp").textContent = `Bir-biriga o‘xshash ${boxes} ta quti guruhlandi.`
      + (canTeach ? " Guruhni bosing — hamma qutisi tanlanadi, nom bersangiz katalogga qo‘shiladi." : "");
    $("groups").innerHTML = shown.map((list, g) => `
      <button class="group-tile" type="button" data-group="${g}" aria-pressed="false"
        aria-label="O‘xshash mahsulotlar guruhi: ${list.length} ta quti" title="${list.length} ta o‘xshash quti">
        <canvas width="112" height="112" aria-hidden="true"></canvas><b>${list.length}</b>
      </button>`).join("")
      + (R.groups.length > shown.length ? `<p class="group-more">+ yana ${R.groups.length - shown.length} ta kichik guruh</p>` : "");
    const imageScale = [img.naturalWidth / R.data.size[0], img.naturalHeight / R.data.size[1]];
    $("groups").querySelectorAll("[data-group]").forEach(row => {
      const g = Number(row.dataset.group), list = R.groups[g];
      const area = i => { const [x1, y1, x2, y2] = R.detections[i].b; return (x2 - x1) * (y2 - y1); };
      const [x1, y1, x2, y2] = R.detections[list.reduce((a, b) => area(b) > area(a) ? b : a)].b.map((value, i) => value * imageScale[i % 2]);
      const thumb = row.querySelector("canvas"), side = Math.max(x2 - x1, y2 - y1), fit = thumb.width / side;
      const paint = thumb.getContext("2d"); paint.fillStyle = "#e8eee9"; paint.fillRect(0, 0, thumb.width, thumb.height);
      const w = (x2 - x1) * fit, h = (y2 - y1) * fit;
      thumb.getContext("2d").drawImage(img, x1, y1, x2 - x1, y2 - y1, (thumb.width - w) / 2, (thumb.height - h) / 2, w, h);
      row.onmouseenter = () => { hlGroup = g; draw(); };
      row.onmouseleave = () => { hlGroup = null; draw(); };
      row.onclick = () => {
        if (!canTeach) return;
        const all = list.every(i => sel.has(i));
        list.forEach(i => all ? sel.delete(i) : sel.add(i));
        picked = -1; whyUi(); teachUi(); draw();
        if (!all) startNewProduct();
      };
    });
  }
  // Brend ichida ta'm guruhlari: avval brend bo'yicha sanaladi, keyin ta'mi ajratiladi
  function flavourUi() {
    const flavours = R.data.flavours || [];
    R.flavourOf = {};
    flavours.forEach(group => group.boxes.forEach(i => { R.flavourOf[i] = group.id; }));
    $("flavourCard").hidden = !flavours.length;
    if (!flavours.length) return;
    const byBrand = new Map();
    flavours.forEach(group => byBrand.set(group.brand, [...(byBrand.get(group.brand) || []), group]));
    $("flavourTitle").textContent = `Brend ta‘mlari · ${byBrand.size} brend`;
    $("flavourHelp").textContent = "Brendi aniq, lekin ta’mi aniqlanmagan qutilar ta’mi bo‘yicha guruhlandi."
      + (canTeach ? " Ta’mni bossangiz hamma qutisi tanlanadi, nom bersangiz alohida mahsulot bo‘ladi." : "");
    const cap = text => text ? text[0].toUpperCase() + text.slice(1) : "";
    $("flavours").innerHTML = [...byBrand].map(([brand, groups]) => `
      <div class="flavour-brand"><div class="scope-brand"><span>${esc(brand)}</span><span>${groups.reduce((sum, g) => sum + g.boxes.length, 0)} ta · ${groups.length} ta‘m</span></div>
        <div class="group-grid">${groups.map(group => `
          <button class="group-tile" type="button" data-flavour="${group.id}" aria-pressed="false"
            aria-label="${esc(brand)} ${esc(group.label || "ta‘mi noma‘lum")}: ${group.boxes.length} ta quti"
            title="${esc(group.label || "ta‘mi o‘qilmagan")} · ${group.boxes.length} ta quti">
            <canvas width="112" height="112" aria-hidden="true"></canvas><b>${group.boxes.length}</b>
            ${group.label ? `<i class="tile-label">${esc(cap(group.label))}</i>` : ""}
          </button>`).join("")}</div></div>`).join("");
    const imageScale = [img.naturalWidth / R.data.size[0], img.naturalHeight / R.data.size[1]];
    const byId = new Map(flavours.map(group => [group.id, group]));
    $("flavours").querySelectorAll("[data-flavour]").forEach(tile => {
      const group = byId.get(Number(tile.dataset.flavour)), list = group.boxes;
      const area = i => { const [x1, y1, x2, y2] = R.detections[i].b; return (x2 - x1) * (y2 - y1); };
      const [x1, y1, x2, y2] = R.detections[list.reduce((a, b) => area(b) > area(a) ? b : a)].b.map((value, i) => value * imageScale[i % 2]);
      const thumb = tile.querySelector("canvas"), side = Math.max(x2 - x1, y2 - y1), fit = thumb.width / side;
      const paint = thumb.getContext("2d"); paint.fillStyle = "#e8eee9"; paint.fillRect(0, 0, thumb.width, thumb.height);
      const w = (x2 - x1) * fit, h = (y2 - y1) * fit;
      paint.drawImage(img, x1, y1, x2 - x1, y2 - y1, (thumb.width - w) / 2, (thumb.height - h) / 2, w, h);
      tile.onmouseenter = () => { hlFlavour = group.id; draw(); };
      tile.onmouseleave = () => { hlFlavour = null; draw(); };
      tile.onclick = () => {
        if (!canTeach) return;
        const all = list.every(i => sel.has(i));
        list.forEach(i => all ? sel.delete(i) : sel.add(i));
        picked = -1; whyUi(); teachUi(); draw();
        if (!all) startNewProduct(`${group.brand} ${cap(group.label)}`.trim(), group.brand);    // nom — «Brend Ta’m»
      };
    });
  }
  function teachUi() {
    $("teach").hidden = sel.size === 0;
    document.querySelectorAll("#groups .group-tile").forEach(tile => {   // tanlangan guruhlar belgilanadi
      const list = R?.groups?.[Number(tile.dataset.group)] || [];
      tile.setAttribute("aria-pressed", String(list.length > 0 && list.every(i => sel.has(i))));
    });
    document.querySelectorAll("#flavours .group-tile").forEach(tile => {
      const list = R?.data?.flavours?.find(g => g.id === Number(tile.dataset.flavour))?.boxes || [];
      tile.setAttribute("aria-pressed", String(list.length > 0 && list.every(i => sel.has(i))));
    });
    $("teachTitle").textContent = `Tanlangan: ${sel.size} ta mahsulot`;
    $("teachErr").textContent = "";
    if (sel.size >= 1 && !products.length) loadProducts();
    if (sel.size && R) {                       // tanlangan qutilar tanilgan bo'lsa — o'sha mahsulotni oldindan tanlash
      const s0 = R.detections[[...sel][0]].s;
      if (s0 && products.some(p => p.sku_id === s0)) $("teachSku").value = s0;
    }
  }
  $("teachCancel").onclick = () => { sel.clear(); picked = -1; teachUi(); whyUi(); draw(); };
  $("teachGo").onclick = async () => {
    const fd = new FormData(); fd.append("image", file);
    fd.append("boxes", JSON.stringify([...sel].map(i => R.detections[i].b)));
    if ($("teachSku").value === "__new") {
      if (!$("teachName").value.trim()) { $("teachErr").textContent = "Mahsulot nomini kiriting."; return; }
      fd.append("name", $("teachName").value); fd.append("brand", $("teachBrand").value);
    } else fd.append("sku_id", $("teachSku").value);
    const b = $("teachGo"); b.disabled = true; b.innerHTML = '<span class="spin"></span> Qo\'shilmoqda…';
    const loader = window.rastaLoader($("teach"), ["Namuna o‘qilmoqda", "Belgilar ajratilmoqda", "Katalogga qo‘shilmoqda"]);
    try {
      let r = await rastaFetch("/api/teach", { method: "POST", body: fd, headers: token() ? { "X-Admin-Token": token() } : {} });
      if (r.status === 401 && !(await rastaSession).enabled) {
        const pw = prompt("Admin paroli:"); if (pw === null) throw new Error("Parol kiritilmadi");
        try { sessionStorage.setItem("rasta-admin", pw); } catch (e) {}
        r = await rastaFetch("/api/teach", { method: "POST", body: fd, headers: { "X-Admin-Token": pw } });
      }
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Xatolik");
      toast(`"${d.name}" ga ${d.added} ta namuna qo'shildi — qayta tahlil…`);
      sel.clear(); $("teachName").value = ""; $("teachBrand").value = ""; products = [];
      if (scope && d.sku_id) scope.add(d.sku_id);            // yangi mahsulot qidiruv doirasiga o'zi qo'shiladi
      await loadScope(); await loadProducts(); teachUi(); analyze($("again"));
    } catch (e) { $("teachErr").textContent = e.message; }
    finally { loader.stop(); b.disabled = false; b.textContent = "Katalogga qo'shish"; }
  };

  function setHl(sku) {
    hl = sku;
    document.querySelectorAll(".prow[data-sku]").forEach(el => el.classList.toggle("hl", !!sku && el.dataset.sku === sku));
    document.querySelectorAll("#donut circle").forEach(c => c.style.opacity = !sku || c.dataset.sku === sku ? 1 : .25);
    draw();
  }
}
