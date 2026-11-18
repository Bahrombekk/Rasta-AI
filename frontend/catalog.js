(() => {
  const $ = id => document.getElementById(id);
  let all = [], files = [];
  const token = () => { try { return sessionStorage.getItem("rasta-admin") || ""; } catch (e) { return ""; } };

  async function api(url, opts = {}, retry = true) {
    opts.headers = Object.assign({}, opts.headers, token() ? { "X-Admin-Token": token() } : {});
    const r = await rastaFetch(url, opts);
    if (r.status === 401 && retry && !(await rastaSession).enabled) {
      const pw = prompt("Admin paroli:");
      if (pw === null) throw new Error("Parol kiritilmadi");
      try { sessionStorage.setItem("rasta-admin", pw); } catch (e) {}
      return api(url, opts, false);
    }
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || "Xatolik yuz berdi");
    return d;
  }

  // ---------- ro'yxat ----------
  function render() {
    const q = $("q").value.trim().toLowerCase();
    const ps = all.filter(p => !q || (p.name + " " + (p.brand || "")).toLowerCase().includes(q));
    const brands = new Set(all.map(p => p.brand).filter(Boolean));
    $("sub").textContent = all.length ? `${all.length} ta mahsulot · ${brands.size} ta brend` : "Tahlilda tanib olinadigan mahsulotlar.";
    $("empty").hidden = all.length > 0;
    $("grid").innerHTML = ps.map(p => `
      <article class="card pcard" data-sku="${esc(p.sku_id)}" tabindex="0" style="cursor:pointer" title="Tahrirlash">
        <div class="img">${p.thumb ? `<img src="${p.thumb}" alt="" loading="lazy">` : icon("image")}</div>
        <div class="body">
          <div class="nm" title="${esc(p.name)}">${esc(p.name)}</div>
          <div class="meta"><span class="chip">${esc(p.brand || "Brendsiz")}</span><span class="cnt">${p.images} ta rasm</span></div>
        </div>
        <button class="iconbtn del" data-sku="${esc(p.sku_id)}" data-name="${esc(p.name)}" title="O'chirish">${icon("trash")}</button>
      </article>`).join("") || (all.length ? `<p style="color:var(--muted)">"${esc(q)}" bo'yicha hech narsa topilmadi.</p>` : "");
    $("brands").innerHTML = [...brands].sort().map(b => `<option value="${esc(b)}">`).join("");
    $("grid").querySelectorAll(".pcard").forEach(c => {
      c.onclick = () => openEdit(c.dataset.sku);
      c.onkeydown = e => { if (e.key === "Enter") openEdit(c.dataset.sku); };
    });
    $("grid").querySelectorAll(".del").forEach(b => b.onclick = async ev => {
      ev.stopPropagation();
      if (!confirm(`"${b.dataset.name}" katalogdan o'chirilsinmi?`)) return;
      try { await api("/api/products/" + encodeURIComponent(b.dataset.sku), { method: "DELETE" }); toast("O'chirildi"); load(); }
      catch (e) { alert(e.message); }
    });
  }
  async function load() {
    try { all = (await api("/api/products")).products; render(); }
    catch (e) { $("grid").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
  }
  $("q").oninput = render;

  // ---------- qo'shish paneli ----------
  let editing = null;              // tahrirlanayotgan mahsulot (null — yangi)
  function show() { $("scrim").hidden = false; $("drawer").hidden = false; setTimeout(() => $("name").focus(), 50); }
  function open() {
    editing = null; $("dTitle").textContent = "Yangi mahsulot"; $("newLbl").textContent = "Rasmlar";
    $("curWrap").hidden = true; $("tip").hidden = false; show();
  }
  function close() { editing = null; $("scrim").hidden = true; $("drawer").hidden = true; $("form").reset(); setFiles([]); $("err").textContent = ""; }
  $("addBtn").onclick = open; $("addBtn2").onclick = open;

  // ---------- tahrirlash ----------
  function renderCur() {
    const ims = editing.images;
    $("curLbl").textContent = `Mavjud rasmlar · ${ims.length} ta`;
    $("cur").innerHTML = ims.map(im => `<div title="${im.shelf ? "Javondan o'rgatilgan" : "Katalog rasmi"}"><img src="${im.url}" alt="" loading="lazy">`
      + (ims.length > 1 ? `<button type="button" data-n="${esc(im.name)}" title="Rasmni o'chirish">✕</button>` : "") + `</div>`).join("");
    $("cur").querySelectorAll("button").forEach(b => b.onclick = async () => {
      if (!confirm("Bu rasm o'chirilsinmi? Mahsulot qolgan rasmlardan qayta o'rganiladi.")) return;
      b.disabled = true;
      try {
        await api(`/api/products/${encodeURIComponent(editing.sku_id)}/images/${encodeURIComponent(b.dataset.n)}`, { method: "DELETE" });
        editing.images = editing.images.filter(im => im.name !== b.dataset.n); renderCur(); toast("Rasm o'chirildi"); load();
      } catch (e) { $("err").textContent = e.message; b.disabled = false; }
    });
  }
  async function openEdit(sku) {
    try { editing = await api("/api/products/" + encodeURIComponent(sku)); }
    catch (e) { alert(e.message); return; }
    $("dTitle").textContent = "Mahsulotni tahrirlash"; $("newLbl").textContent = "Rasm qo'shish";
    $("name").value = editing.name; $("brand").value = editing.brand || "";
    $("curWrap").hidden = false; $("tip").hidden = true; renderCur(); show();
  }
  $("close").innerHTML = icon("x"); $("close").onclick = close; $("cancel").onclick = close; $("scrim").onclick = close;
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("drawer").hidden) close(); });

  function setFiles(list) {
    files = list;
    $("thumbs").innerHTML = files.map((f, i) => `<div><img src="${URL.createObjectURL(f)}" alt=""><button type="button" data-i="${i}" title="Olib tashlash">✕</button></div>`).join("");
    $("thumbs").querySelectorAll("button").forEach(b => b.onclick = () => setFiles(files.filter((_, i) => i !== +b.dataset.i)));
  }
  const addFiles = list => setFiles(files.concat([...list].filter(f => f.type.startsWith("image/"))));
  $("files").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
  const drop = $("drop");
  ["dragover", "dragenter"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", e => { e.preventDefault(); addFiles(e.dataTransfer.files); });

  $("form").onsubmit = async e => {
    e.preventDefault();
    if (!$("name").value.trim()) { $("err").textContent = "Mahsulot nomini kiriting."; return; }
    if (!editing && !files.length) { $("err").textContent = "Kamida bitta rasm qo'shing."; return; }
    if (editing) return saveEdit();
    const fd = new FormData();
    fd.append("name", $("name").value); fd.append("brand", $("brand").value);
    files.forEach(f => fd.append("files", f));
    const b = $("save"); b.disabled = true; b.innerHTML = '<span class="spin"></span> Saqlanmoqda…';
    $("err").textContent = "";
    try {
      await api("/api/products", { method: "POST", body: fd });
      const nm = $("name").value.trim(); close(); toast(`"${nm}" qo'shildi`); load();
    } catch (er) { $("err").textContent = er.message; }
    finally { b.disabled = false; b.textContent = "Saqlash"; }
  };

  async function saveEdit() {
    const b = $("save"); b.disabled = true; b.innerHTML = '<span class="spin"></span> Saqlanmoqda…';
    $("err").textContent = "";
    const sku = encodeURIComponent(editing.sku_id), nm = $("name").value.trim(), br = $("brand").value.trim();
    try {
      if (nm !== editing.name || br !== (editing.brand || "")) {
        const fd = new FormData(); fd.append("name", nm); fd.append("brand", br);
        await api(`/api/products/${sku}`, { method: "PATCH", body: fd });
      }
      if (files.length) {
        const fd = new FormData(); files.forEach(f => fd.append("files", f));
        await api(`/api/products/${sku}/images`, { method: "POST", body: fd });
      }
      const n = files.length; close(); toast(n ? `"${nm}" ga ${n} ta rasm qo'shildi` : "Saqlandi"); load();
    } catch (er) { $("err").textContent = er.message; }
    finally { b.disabled = false; b.textContent = "Saqlash"; }
  }

  load();
})();
