export default function mountPage() {
  const $ = id => document.getElementById(id);
  const orgSelect = $("organization");
  const dialog = $("drawer");
  const filters = ["q", "category-filter", "brand-filter", "image-filter"];
  const fields = ["name", "brand", "category", "barcode"];
  const pageSize = 24;
  let all = [], matches = [], files = [], previewUrls = [];
  let editing = null, page = 1, revision = 0, busy = false, opening = false, readOnly = false;
  const token = () => { try { return sessionStorage.getItem("rasta-admin") || ""; } catch { return ""; } };

  function scoped(url) {
    return readOnly && orgSelect.value ? url + "?organization=" + encodeURIComponent(orgSelect.value) : url;
  }

  async function api(url, opts = {}, retry = true) {
    const headers = token() ? { "X-Admin-Token": token() } : {};
    const response = await rastaFetch(scoped(url), { ...opts, headers });
    if (response.status === 401 && retry && !(await rastaSession).enabled) {
      const password = prompt("Admin paroli:");
      if (password === null) throw new Error("Parol kiritilmadi.");
      sessionStorage.setItem("rasta-admin", password);
      return api(url, opts, false);
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "So‘rov bajarilmadi.");
    return data;
  }

  function options(key, values) {
    const select = $(key + "-filter");
    const selected = select.value;
    select.replaceChildren(new Option("Barchasi", ""), new Option(key === "brand" ? "Brendsiz" : "Kategoriyasiz", "-"));
    for (const value of values) select.add(new Option(value, "=" + value));
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
  }

  function updateFilters() {
    for (const key of ["category", "brand"]) {
      const values = [...new Set(all.map(product => product[key]).filter(Boolean))].sort((a, b) => a.localeCompare(b, "uz"));
      options(key, values);
      $(key === "brand" ? "brands" : "categories").replaceChildren(...values.map(value => new Option(value)));
      if (key === "category") $("category-count").textContent = values.length;
    }
    $("product-count").textContent = all.length;
    $("barcode-count").textContent = all.filter(product => product.barcode).length;
    $("missing-count").textContent = all.filter(product => !product.images).length;
  }

  function render() {
    const [query, category, brand, images] = filters.map(id => $(id).value);
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const matchesFilter = (value, selected) => !selected || (value ? "=" + value : "-") === selected;
    matches = all.filter(product => {
      const haystack = [product.name, product.brand, product.category, product.sku_id, product.barcode].join(" ").toLowerCase();
      return terms.every(term => haystack.includes(term)) && matchesFilter(product.category, category)
        && matchesFilter(product.brand, brand) && (!images || (images === "available") === Boolean(product.images));
    });
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.min(page, pages);
    const start = (page - 1) * pageSize;
    $("grid").innerHTML = matches.slice(start, start + pageSize).map(p => `
      <article class="card pcard">
        <button class="catalog-product" type="button" data-sku="${esc(p.sku_id)}">
          <span class="img">${p.thumb ? `<img src="${esc(scoped(p.thumb))}" alt="" loading="lazy">` : icon("image")}</span>
          <span class="body">
            <strong class="nm">${esc(p.name)}</strong>
            <span class="catalog-brand">${esc([p.brand, p.category].filter(Boolean).join(" · ") || "Brend va kategoriya kiritilmagan")}</span>
            <span class="catalog-code">${esc(p.sku_id)}${p.barcode ? " · " + esc(p.barcode) : ""}</span>
            <span class="meta"><span class="chip${p.images ? "" : " catalog-no-photo"}">${p.images ? p.images + " ta rasm" : "Rasmsiz"}</span>${icon("arrow-right")}</span>
          </span>
        </button>
      </article>`).join("");
    $("empty").hidden = matches.length > 0;
    $("empty").textContent = all.length ? "Filtrlarga mos mahsulot topilmadi." : "Bu katalog hali bo‘sh.";
    $("result-count").textContent = pages > 1
      ? `${start + 1}–${Math.min(start + pageSize, matches.length)} / ${matches.length} ta mahsulot`
      : `${matches.length} ta mahsulot`;
    const filterCount = filters.slice(1).filter(id => $(id).value).length;
    $("filter-toggle").textContent = filterCount ? `Filtrlar (${filterCount})` : "Filtrlar";
    $("reset").hidden = !filters.some(id => $(id).value);
    $("export").disabled = !matches.length;
    $("bundleExport").disabled = !all.length;
    $("pagination").hidden = pages === 1;
    $("page-number").textContent = `${page} / ${pages}`;
    $("previous").disabled = page === 1;
    $("next").disabled = page === pages;
  }

  async function load() {
    const request = ++revision;
    $("catalog-content").hidden = true;
    $("export").disabled = true;
    $("bundleExport").disabled = true;
    const status = $("status");
    status.hidden = false;
    status.textContent = "Katalog yuklanmoqda…";
    if (readOnly && !orgSelect.value) {
      status.textContent = "Katalogini ko‘rish uchun korxonani tanlang.";
      return;
    }
    try {
      const { products } = await api("/api/products");
      if (request !== revision) return;
      all = products;
      updateFilters(); render();
      status.hidden = true;
      $("catalog-content").hidden = false;
    } catch (error) {
      if (request === revision) status.textContent = error.message;
    }
  }

  function resetFilters() {
    filters.forEach(id => { $(id).value = ""; });
    page = 1;
  }
  $("q").oninput = () => { page = 1; render(); };
  $("filter-toggle").onclick = event => {
    const button = event.currentTarget;
    button.setAttribute("aria-expanded", String(button.getAttribute("aria-expanded") === "false"));
  };
  filters.slice(1).forEach(id => { $(id).onchange = () => { page = 1; render(); }; });
  $("reset").onclick = () => { resetFilters(); render(); $("q").focus(); };
  $("previous").onclick = () => { page--; render(); $("q").scrollIntoView({ block: "start" }); };
  $("next").onclick = () => { page++; render(); $("q").scrollIntoView({ block: "start" }); };

  function csvCell(value) {
    const string = String(value ?? "");
    const safe = /^[\s\u0000-\u001f]*[=+\-@]|^[\t\r\n]/.test(string) ? "'" + string : string;
    return '"' + safe.replaceAll('"', '""') + '"';
  }
  $("export").onclick = () => {
    const rows = [["SKU", "Nomi", "Brend", "Kategoriya", "Shtrix-kod", "Namuna rasmlari"],
      ...matches.map(p => [p.sku_id, p.name, p.brand, p.category, p.barcode, p.images])];
    const blob = new Blob(["\ufeff" + rows.map(row => row.map(csvCell).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const download = document.createElement("a");
    download.href = url;
    download.download = "rasta-katalog.csv";
    download.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  function lock(active) {
    busy = active;
    for (const id of ["save", "cancel", "close", "delete-product", "files"]) $(id).disabled = active;
    for (const id of [...fields, "sku"]) $(id).readOnly = active || readOnly || (id === "sku" && !!editing);
    for (const button of dialog.querySelectorAll(".thumbs button")) button.disabled = active;
    $("save").textContent = active ? "Saqlanmoqda…" : "Saqlash";
  }

  function open(product = null) {
    editing = product;
    $("form").reset();
    setFiles([]);
    $("err").textContent = "";
    $("dTitle").textContent = readOnly ? "Mahsulot tafsilotlari" : editing ? "Mahsulotni tahrirlash" : "Yangi mahsulot";
    $("newLbl").textContent = editing ? "Rasm qo‘shish" : "Namuna rasmlari";
    for (const id of fields) $(id).value = editing?.[id] || "";
    $("sku").value = editing?.sku_id || "";
    $("curWrap").hidden = !editing;
    $("new-images").hidden = readOnly;
    $("tip").hidden = readOnly || Boolean(editing);
    $("save").hidden = readOnly;
    $("delete-product").hidden = readOnly || !editing;
    $("cancel").textContent = readOnly ? "Yopish" : "Bekor qilish";
    if (editing) renderCur();
    lock(false);
    if (!dialog.open) dialog.showModal();
  }
  $("addBtn").onclick = () => open();
  $("close").onclick = $("cancel").onclick = () => { if (!busy) dialog.close(); };
  dialog.addEventListener("cancel", event => { if (busy) event.preventDefault(); });
  dialog.addEventListener("close", () => { if (!dialog.open) setFiles([]); });
  $("grid").onclick = async event => {
    const button = event.target.closest("[data-sku]");
    if (!button || opening) return;
    opening = true;
    const request = revision;
    try {
      const product = await api("/api/products/" + encodeURIComponent(button.dataset.sku));
      if (request === revision && !dialog.open) open(product);
    } catch (error) { toast(error.message); }
    finally { opening = false; }
  };

  // Namuna rasmi + AI o'rganadigan qism (ramka). Ramka foizlarda, rasm o'z nisbatida
  // ko'rsatiladi — shuning uchun ramka aynan mahsulot ustiga tushadi.
  function shotTile(im) {
    const pct = value => (value * 100).toFixed(2) + "%";
    const [label, cls] = im.shelf ? ["Javondan", ""] : !im.located ? ["Ajratilmagan", "warn"]
      : !im.box ? ["Butun rasm", ""] : im.auto ? ["Avto", ""] : ["Qo‘lda", "manual"];
    const box = im.box ? `<span class="shot-box" style="left:${pct(im.box[0])};top:${pct(im.box[1])};width:${pct(im.box[2] - im.box[0])};height:${pct(im.box[3] - im.box[1])}"></span>` : "";
    const editable = !readOnly && !im.shelf;
    return `<div class="shot-tile"><a class="shot" href="${esc(scoped(im.url))}" ${editable
      ? `data-edit="${esc(im.name)}" title="Mahsulot chegarasini to‘g‘rilash"` : 'target="_blank" rel="noopener"'}>
      <img src="${esc(scoped(im.url))}" alt="${im.shelf ? "Javon namunasi" : "Mahsulot namunasi"}" loading="lazy">${box}</a>
      <span class="shot-badge ${cls}">${label}</span>`
      + (!readOnly && editing.images.length > 1 ? `<button type="button" data-n="${esc(im.name)}" aria-label="Rasmni o‘chirish">✕</button>` : "") + "</div>";
  }
  function renderCur() {
    $("curLbl").textContent = `Mavjud rasmlar · ${editing.images.length} ta`;
    $("cur").innerHTML = editing.images.map(shotTile).join("");
    for (const img of $("cur").querySelectorAll(".shot img")) {
      const fit = () => {
        if (!img.naturalWidth) return;
        img.parentElement.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
        img.parentElement.classList.toggle("wide", img.naturalWidth > img.naturalHeight);
      };
      if (img.complete) fit(); else img.addEventListener("load", fit, { once: true });
    }
    const photos = editing.images.filter(image => !image.shelf);
    $("shotHelp").hidden = readOnly || !photos.length;
    $("relocateTip").hidden = readOnly || !photos.some(image => !image.located);
  }
  async function refreshEditing() {
    editing = await api("/api/products/" + encodeURIComponent(editing.sku_id));
    renderCur();
  }
  $("relocate").onclick = async () => {
    if (!editing || busy) return;
    lock(true); $("err").textContent = "";
    const loader = window.rastaLoader($("form"), ["Mahsulot qidirilmoqda", "Kadrdan ajratilmoqda", "Qayta o‘rganilmoqda"]);
    try {
      await api("/api/products/" + encodeURIComponent(editing.sku_id) + "/reindex", { method: "POST" });
      await refreshEditing();
      toast("Mahsulot ajratildi — ramkalarni tekshiring");
      load();
    } catch (error) { $("err").textContent = error.message; }
    finally { loader.stop(); lock(false); }
  };

  // ---------- mahsulot chegarasi tahrirchisi ----------
  const boxDialog = $("boxEditor"), boxStage = $("boxStage"), boxRect = $("boxRect");
  const boxButtons = ["boxSave", "boxAuto", "boxFull", "boxCancel", "boxClose"];
  let boxTarget = null, boxValue = null, boxDrag = null, boxBusy = false;
  const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
  function showBox() {
    boxRect.hidden = !boxValue;
    $("boxState").textContent = boxValue ? "" : "Ramka yo‘q — butun rasm ishlatiladi. Yangi ramka chizish uchun rasm ustida torting.";
    if (!boxValue) return;
    const [x1, y1, x2, y2] = boxValue;
    Object.assign(boxRect.style, { left: x1 * 100 + "%", top: y1 * 100 + "%", width: (x2 - x1) * 100 + "%", height: (y2 - y1) * 100 + "%" });
  }
  function openBox(image) {
    boxTarget = image;
    boxValue = image.box ? [...image.box] : null;
    $("boxErr").textContent = "";
    $("boxImage").src = scoped(image.url);
    showBox();
    boxDialog.showModal();
  }
  const pointer = event => {
    const rect = boxStage.getBoundingClientRect();
    return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
  };
  boxStage.onpointerdown = event => {
    if (boxBusy || event.button > 0) return;
    const [x, y] = pointer(event), handle = event.target.dataset.h;
    if (handle && boxValue) {                 // burchak: qarama-qarshi burchak joyida qoladi
      boxDrag = { previous: [...boxValue], fx: handle.includes("w") ? boxValue[2] : boxValue[0],
                  fy: handle.includes("n") ? boxValue[3] : boxValue[1] };
    } else if (event.target === boxRect && boxValue) {
      boxDrag = { previous: [...boxValue], dx: x - boxValue[0], dy: y - boxValue[1],
                  w: boxValue[2] - boxValue[0], h: boxValue[3] - boxValue[1] };
    } else {
      boxDrag = { previous: boxValue && [...boxValue], fx: x, fy: y };
    }
    boxStage.setPointerCapture(event.pointerId);
    event.preventDefault();
  };
  boxStage.onpointermove = event => {
    if (!boxDrag) return;
    const [x, y] = pointer(event);
    if (boxDrag.w) {
      const nx = clamp(x - boxDrag.dx, 0, 1 - boxDrag.w), ny = clamp(y - boxDrag.dy, 0, 1 - boxDrag.h);
      boxValue = [nx, ny, nx + boxDrag.w, ny + boxDrag.h];
    } else {
      boxValue = [Math.min(boxDrag.fx, x), Math.min(boxDrag.fy, y), Math.max(boxDrag.fx, x), Math.max(boxDrag.fy, y)];
    }
    showBox();
  };
  boxStage.onpointerup = boxStage.onpointercancel = () => {
    if (!boxDrag) return;
    if (boxValue && (boxValue[2] - boxValue[0] < 0.04 || boxValue[3] - boxValue[1] < 0.04)) boxValue = boxDrag.previous;
    boxDrag = null;
    showBox();
  };
  async function saveBox(value, message) {
    if (boxBusy || !boxTarget) return;
    boxBusy = true; $("boxErr").textContent = "";
    for (const id of boxButtons) $(id).disabled = true;
    const loader = window.rastaLoader($("boxBody"), ["Chegara saqlanmoqda", "Belgilar ajratilmoqda", "Katalog yangilanmoqda"]);
    const body = new FormData();
    body.set("box", value === "auto" ? "auto" : JSON.stringify(value && value.map(number => +number.toFixed(4))));
    try {
      const saved = await api(`/api/products/${encodeURIComponent(editing.sku_id)}/images/${encodeURIComponent(boxTarget.name)}/box`, { method: "PUT", body });
      await refreshEditing();
      if (value === "auto") {                 // natijani ko'rsatib, oynani ochiq qoldiramiz
        boxTarget = editing.images.find(image => image.name === boxTarget.name) || boxTarget;
        boxValue = saved.box ? [...saved.box] : null;
        showBox();
        if (!saved.box) $("boxState").textContent = "Mahsulot avtomatik topilmadi — ramkani qo‘lda chizing.";
      } else boxDialog.close();
      toast(message);
      load();
    } catch (error) { $("boxErr").textContent = error.message; }
    finally {
      loader.stop(); boxBusy = false;
      for (const id of boxButtons) $(id).disabled = false;
    }
  }
  $("boxSave").onclick = () => saveBox(boxValue, "Chegara saqlandi — mahsulot qayta o‘rganildi");
  $("boxAuto").onclick = () => saveBox("auto", "Chegara avtomatik aniqlandi");
  $("boxFull").onclick = () => { if (!boxBusy) { boxValue = null; showBox(); } };
  $("boxClose").onclick = $("boxCancel").onclick = () => { if (!boxBusy) boxDialog.close(); };
  boxDialog.addEventListener("cancel", event => { if (boxBusy) event.preventDefault(); });

  $("cur").onclick = async event => {
    const shot = event.target.closest("a[data-edit]");
    if (shot) {
      event.preventDefault();
      const image = editing.images.find(item => item.name === shot.dataset.edit);
      if (image && !busy) openBox(image);
      return;
    }
    const button = event.target.closest("button[data-n]");
    if (!button || busy || !confirm("Rasm o‘chirilsinmi? Qolgan rasmlar qayta indekslanadi.")) return;
    lock(true); $("err").textContent = "";
    try {
      await api("/api/products/" + encodeURIComponent(editing.sku_id) + "/images/" + encodeURIComponent(button.dataset.n), { method: "DELETE" });
      editing.images = editing.images.filter(image => image.name !== button.dataset.n);
      renderCur(); toast("Rasm o‘chirildi"); load();
    } catch (error) { $("err").textContent = error.message; }
    finally { lock(false); }
  };
  $("delete-product").onclick = async () => {
    if (!editing || busy || !confirm(`“${editing.name}” va uning barcha katalog rasmlari o‘chirilsinmi?`)) return;
    lock(true); $("err").textContent = "";
    try {
      await api("/api/products/" + encodeURIComponent(editing.sku_id), { method: "DELETE" });
      dialog.close(); toast("Mahsulot o‘chirildi"); load();
    } catch (error) { $("err").textContent = error.message; }
    finally { lock(false); }
  };

  function setFiles(list) {
    previewUrls.forEach(url => URL.revokeObjectURL(url));
    files = list;
    previewUrls = files.map(file => URL.createObjectURL(file));
    $("thumbs").innerHTML = previewUrls.map((url, index) => `<div><img src="${url}" alt="">
      <button type="button" data-index="${index}" aria-label="Tanlangan rasmni olib tashlash">✕</button></div>`).join("");
  }
  $("thumbs").onclick = event => {
    const button = event.target.closest("[data-index]");
    if (button && !busy) setFiles(files.filter((_, index) => index !== Number(button.dataset.index)));
  };
  function addFiles(list) {
    if (busy || readOnly) return;
    const valid = [...list].filter(file => file.type.startsWith("image/") && file.size <= 20 * 1024 * 1024);
    $("err").textContent = valid.length === list.length ? "" : "Faqat 20 MB gacha bo‘lgan rasmlarni tanlang.";
    setFiles([...files, ...valid]);
  }
  $("files").onchange = event => { addFiles(event.target.files); event.target.value = ""; };
  const drop = $("drop");
  drop.ondragover = event => { event.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = event => { event.preventDefault(); drop.classList.remove("over"); addFiles(event.dataTransfer.files); };

  $("form").onsubmit = async event => {
    event.preventDefault();
    if (busy || readOnly) return;
    const values = Object.fromEntries(fields.map(id => [id, $(id).value.trim()]));
    if (!values.name || (!editing && !files.length)) {
      $("err").textContent = values.name ? "Kamida bitta rasm qo‘shing." : "Mahsulot nomini kiriting.";
      return;
    }
    lock(true); $("err").textContent = "";
    const loader = window.rastaLoader($("form"), editing
      ? ["Rasm o‘qilmoqda", "Belgilar ajratilmoqda", "Katalog yangilanmoqda"]
      : ["Rasm o‘qilmoqda", "Belgilar ajratilmoqda", "Variantlar yaratilmoqda", "Katalogga yozilmoqda"]);
    const body = new FormData();
    for (const [key, value] of Object.entries(values)) body.set(key, value);
    const savedFiles = new FormData();
    files.forEach(file => savedFiles.append("files", file));
    try {
      const added = files.length;
      let sku = editing?.sku_id;
      if (editing) {
        const url = "/api/products/" + encodeURIComponent(editing.sku_id);
        if (fields.some(id => values[id] !== (editing[id] || ""))) {
          Object.assign(editing, await api(url, { method: "PATCH", body }));
        }
        if (files.length) await api(url + "/images", { method: "POST", body: savedFiles });
      } else {
        body.set("sku_id", $("sku").value.trim());
        files.forEach(file => body.append("files", file));
        sku = (await api("/api/products", { method: "POST", body })).sku_id;
      }
      if (added) {
        // yangi rasmlar: AI nimani "mahsulot" deb ajratganini darhol ko'rsatamiz
        open(await api("/api/products/" + encodeURIComponent(sku)));
        toast("Saqlandi — yashil ramkalarni tekshiring");
      } else { dialog.close(); toast("Mahsulot saqlandi"); }
      load();
    } catch (error) {
      $("err").textContent = error.message;
      load();
    } finally { loader.stop(); lock(false); }
  };

  // Katalog to'plami: rasmlari bilan .zip — bir korxonada tayyorlab, boshqasiga ko'chirish
  $("bundleExport").onclick = () => { location.assign(scoped("/api/catalog/export")); };
  $("bundleImport").onclick = () => { if (!busy) $("bundleFile").click(); };
  $("bundleFile").onchange = async event => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file || busy) return;
    if (!confirm(`"${file.name}" faylidagi mahsulotlar katalogga qo‘shilsinmi? Katalogda bor mahsulotlar o‘zgarmaydi.`)) return;
    busy = true; $("bundleImport").disabled = true;
    const loader = window.rastaLoader($("catalog-content"), ["Fayl o‘qilmoqda", "Rasmlar tekshirilmoqda", "Belgilar ajratilmoqda", "Katalog yangilanmoqda"], { interval: 2500 });
    try {
      const body = new FormData(); body.append("file", file);
      const result = await api("/api/catalog/import", { method: "POST", body });
      const parts = [`${result.added.length} ta mahsulot qo‘shildi`];
      if (result.skipped.length) parts.push(`${result.skipped.length} tasi katalogda bor edi`);
      if (result.failed.length) parts.push(`${result.failed.length} tasi qo‘shilmadi`);
      toast(parts.join(", "));
      if (result.failed.length) alert("Qo‘shilmagan mahsulotlar:\n" + result.failed.map(p => `${p.name || p.sku_id}: ${p.reason}`).join("\n"));
    } catch (error) { toast(error.message); }
    finally { loader.stop(); busy = false; $("bundleImport").disabled = false; load(); }
  };

  $("reindexAll").onclick = async event => {
    if (busy || !confirm("Barcha mahsulotlar saqlangan rasmlaridan qayta o‘rganilsinmi? Fonli rasmlarda mahsulot avtomatik ajratib olinadi. Bu bir necha daqiqa olishi mumkin.")) return;
    const button = event.currentTarget;
    busy = true; button.disabled = true;
    const loader = window.rastaLoader($("catalog-content"), ["Rasmlar o‘qilmoqda", "Mahsulotlar ajratilmoqda", "Belgilar ajratilmoqda", "Katalog yangilanmoqda"], { interval: 2500 });
    try {
      const result = await api("/api/reindex", { method: "POST" });
      toast(`${result.reindexed.length} ta mahsulot qayta o‘rganildi`);
    } catch (error) { toast(error.message); }
    finally { loader.stop(); busy = false; button.disabled = false; load(); }
  };

  async function init() {
    const session = await rastaSession;
    if (session.enabled && !session.user) return;
    if (session.workspaces && !["org_admin", "super_admin"].includes(session.user?.role)) return;
    readOnly = Boolean(session.workspaces && session.user?.role === "super_admin");
    if (readOnly) {
      const { organizations } = await api("/api/organizations");
      orgSelect.replaceChildren(new Option("Korxonani tanlang", ""), ...organizations.map(org =>
        new Option(org.name + (org.active ? "" : " · To‘xtatilgan"), org.id)));
      const selected = new URLSearchParams(location.search).get("organization");
      if (organizations.some(org => String(org.id) === selected)) orgSelect.value = selected;
      $("organization-panel").hidden = false;
      $("sub").textContent = "Korxonalar katalogini ko‘rish va CSV yuklash";
      orgSelect.onchange = () => {
        const url = new URL(location.href);
        if (orgSelect.value) url.searchParams.set("organization", orgSelect.value);
        else url.searchParams.delete("organization");
        history.replaceState(null, "", url);
        resetFilters(); load();
      };
    } else {
      $("addBtn").hidden = false;
      $("reindexAll").hidden = false;
      $("bundleImport").hidden = false;
    }
    await load();
  }
  init().catch(error => { $("status").textContent = error.message; });
}
