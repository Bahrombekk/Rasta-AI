(() => {
  const $ = id => document.getElementById(id);
  const body = $("field-body"), dialog = $("field-dialog"), form = $("field-form");
  const targetId = Number(new URLSearchParams(location.search).get("id"));
  const labels = {audit: "Fotoaudit", delivery: "Tovar topshirish", other: "Boshqa ish"};
  const statuses = {pending: "Kutilmoqda", done: "Bajarilgan", cancelled: "Bekor qilingan"};
  const number = new Intl.NumberFormat("uz-UZ");
  const count = value => number.format(value ?? 0);
  let manager = false, snapshot, stores, onSave;

  const fieldButton = (id, text, iconName, primary = false) => `<button class="btn${primary ? " primary" : ""}"
    type="button" id="${id}">${icon(iconName)}${text}</button>`;
  const link = (path, text, iconName, primary = false) => `<a class="btn${primary ? " primary" : ""}" href="${path}">${icon(iconName)}${text}</a>`;
  const note = (text, name = "note") => `<label class="field"><span>${text}</span><textarea class="input" name="${name}" maxlength="1000" rows="3"></textarea></label>`;
  const options = (rows, name = "name", key = "id") => rows.map(row => `<option value="${esc(row[key])}">${esc(row[name])}</option>`).join("");
  const panel = (title, content, actions = "") => `<section class="card work-section"><div class="panel-heading"><h2>${title}</h2>${actions}</div>${content}</section>`;
  const empty = (title, message, glyph = "visit") => `<div class="work-empty"><span class="metric-icon">${icon(glyph)}</span><strong>${title}</strong><p>${message}</p></div>`;
  const status = (text, state) => `<span class="work-tag ${state}">${text}</span>`;
  const summary = (label, value, glyph) => `<div class="field-stat"><span class="metric-icon">${icon(glyph)}</span><strong>${count(value)}</strong><small>${label}</small></div>`;

  function heading(title, description, actions = "") {
    $("field-title").textContent = title;
    $("field-subtitle").textContent = description;
    $("field-actions").innerHTML = actions;
    document.title = title + " · Rasta AI";
  }
  function error(message = "") {
    $("field-error").textContent = message;
    $("field-error").hidden = !message;
  }
  function open(title, fields, save, label = "Saqlash") {
    $("field-dialog-title").textContent = title;
    $("field-inputs").innerHTML = fields;
    $("field-dialog-error").textContent = "";
    $("field-save").textContent = label;
    $("field-save").disabled = false;
    onSave = save;
    dialog.showModal();
  }
  $("field-close").onclick = () => dialog.close();
  form.onsubmit = async event => {
    event.preventDefault();
    $("field-save").disabled = true;
    $("field-dialog-error").textContent = "";
    try {
      await onSave(new FormData(form));
      dialog.close();
      await load();
    } catch (e) {
      $("field-dialog-error").textContent = e.message;
    } finally {
      $("field-save").disabled = false;
    }
  };

  function visitRows(visits) {
    if (!visits.length) return empty("Tashriflar hali yo‘q", "Agent do‘konda tashrifni boshlaganda shu yerda ko‘rinadi.");
    return `<div class="field-list">${visits.map(visit => `<a class="field-row" href="/field?id=${visit.id}">
      <span class="metric-icon">${icon("store")}</span><span class="field-row-text"><strong>${esc(visit.store)}</strong>
      <small>${manager ? esc(visit.agent) + " · " : ""}${esc(rastaDate(visit.started))}</small>
      <small>${count(visit.audits)} audit · ${count(visit.units)} dona</small></span>
      ${visit.finished ? status("Yakunlangan","approved") : status("Ochiq","submitted")}${icon("arrow-right")}
    </a>`).join("")}</div>`;
  }

  function taskRows(tasks, visit = null) {
    if (!tasks.length) return empty("Topshiriqlar yo‘q", "Do‘kon bo‘yicha topshiriqlar shu yerda ko‘rinadi.", "report");
    return `<div class="field-list">${tasks.map(task => `<article class="field-row task-row">
      <span class="metric-icon">${icon(task.kind === "delivery" ? "box" : "report")}</span>
      <span class="field-row-text"><strong>${esc(task.title)}</strong><small>${esc(task.store)}${manager ? " · " + esc(task.agent) : ""}</small>
        <small>${labels[task.kind]}${task.due ? " · " + esc(task.due.split("-").reverse().join(".")) : ""}</small></span>
      ${status(statuses[task.status], task.status === "pending" ? "submitted" : task.status === "done" ? "approved" : "returned")}
      ${task.status === "pending" && manager ? `<button class="btn" type="button" data-cancel="${task.id}">Bekor qilish</button>` : ""}
      ${task.status === "pending" && visit?.store_id === task.store_id && !visit.finished
        ? `<button class="btn" type="button" data-complete="${task.id}">Bajarildi</button>` : ""}
    </article>`).join("")}</div>`;
  }

  function wireTasks(visit) {
    body.querySelectorAll("[data-cancel]").forEach(button => button.onclick = () => {
      open("Topshiriqni bekor qilish", "<p>Bu topshiriq agentning ochiq ishlari ro‘yxatidan olinadi.</p>",
        () => rastaJson(`/api/field/tasks/${button.dataset.cancel}/cancel`, "POST"), "Bekor qilish");
    });
    body.querySelectorAll("[data-complete]").forEach(button => button.onclick = () => {
      open("Topshiriqni yakunlash", note("Izoh (ixtiyoriy)"), values =>
        rastaJson(`/api/field/tasks/${button.dataset.complete}/complete`, "POST", {
          visit_id: visit.id, note: values.get("note")
        }));
    });
  }

  function agentHome() {
    const current = snapshot.visits.find(visit => !visit.finished);
    const pending = snapshot.tasks.filter(task => task.status === "pending" && stores.some(store => store.id === task.store_id));
    heading("Mening tashriflarim", "Do‘konni tanlang va bajarilgan ishlarni qayd eting.");
    body.innerHTML = `<div class="field-home">
      ${current ? `<section class="field-current"><span class="field-current-state">Ochiq tashrif</span>
       <h2>${esc(current.store)}</h2><p>Boshlandi · ${esc(rastaDate(current.started))}</p>
       ${link(`/field?id=${current.id}`, "Tashrifni davom ettirish", "arrow-right", true)}</section>` : ""}
      <div class="field-summary">${summary("Do‘konlar", stores.length, "store")}
       ${summary("Ochiq topshiriqlar", pending.length, "report")}</div>
      ${panel("Biriktirilgan do‘konlar", stores.length ? `<div class="field-stores">${stores.map(store => `<article class="field-store">
        <span class="metric-icon">${icon("store")}</span><div><h3>${esc(store.name)}</h3>
        ${store.address ? `<p>${esc(store.address)}</p>` : ""}</div>
        ${current?.store_id === store.id ? link(`/field?id=${current.id}`, "Ochish", "arrow-right") :
          `<button class="btn primary" type="button" data-start="${store.id}" ${current ? "disabled" : ""}>Tashrifni boshlash</button>`}</article>`).join("")}</div>`
        : empty("Do‘kon biriktirilmagan", "Korxona admini do‘kon biriktirgach, tashrifni boshlashingiz mumkin.", "store"))}
      ${panel("Ochiq topshiriqlar", taskRows(pending))}
      ${panel("So‘nggi tashriflar", visitRows(snapshot.visits.filter(visit => visit.finished)))}</div>`;
    body.querySelectorAll("[data-start]").forEach(button => button.onclick = async () => {
      button.disabled = true;
      try {
        const result = await rastaJson("/api/field/visits", "POST", {store_id: Number(button.dataset.start)});
        location.assign(`/field?id=${result.id}`);
      } catch (e) { error(e.message); button.disabled = false; }
    });
  }

  async function assign(agent) {
    const { store_ids } = await rastaJson(`/api/field/agents/${agent.id}/stores`);
    open(`${agent.username} · do‘konlar`, stores.length ? `<fieldset class="assignment-list"><legend>Biriktirilgan do‘konlar</legend>
      ${stores.map(store => `<label><input type="checkbox" name="stores" value="${store.id}" ${store_ids.includes(store.id) ? "checked" : ""}>
        <span>${esc(store.name)}<small>${esc(store.address)}</small></span></label>`).join("")}</fieldset>`
      : "<p>Avval do‘kon qo‘shing.</p>", values =>
      rastaJson(`/api/field/agents/${agent.id}/stores`, "PUT", {store_ids: values.getAll("stores").map(Number)}));
  }

  function createTask() {
    const active = snapshot.agents.filter(agent => agent.active && agent.stores > 0);
    if (!active.length) { error("Avval faol agentga do‘kon biriktiring."); return; }
    open("Topshiriq berish", `
      <label class="field"><span>Agent</span><select name="agent_id" class="input" required><option value="">Agentni tanlang</option>${options(active, "username")}</select></label>
      <label class="field"><span>Do‘kon</span><select name="store_id" class="input" required><option value="">Avval agentni tanlang</option></select></label>
      <label class="field"><span>Topshiriq</span><input name="title" class="input" required maxlength="160"></label>
      <label class="field"><span>Ish turi</span><select name="kind" class="input">${Object.entries(labels).map(([value,label]) => `<option value="${value}">${label}</option>`).join("")}</select></label>
      <label class="field"><span>Muddat (ixtiyoriy)</span><input type="date" name="due" class="input"></label>`, values =>
      rastaJson("/api/field/tasks", "POST", {agent_id: Number(values.get("agent_id")), store_id: Number(values.get("store_id")),
        title: values.get("title"), kind: values.get("kind"), due: values.get("due") || null}));
    const agent = form.elements.agent_id, store = form.elements.store_id;
    agent.onchange = async () => {
      const selected = agent.value;
      store.replaceChildren(new Option("Do‘konni tanlang", ""));
      if (!selected) return;
      try {
        const { store_ids } = await rastaJson(`/api/field/agents/${selected}/stores`);
        if (agent.value === selected) store.insertAdjacentHTML("beforeend", options(stores.filter(item => store_ids.includes(item.id))));
      } catch (e) { $("field-dialog-error").textContent = e.message; }
    };
  }

  function adminHome() {
    heading("Agent tashriflari", "Do‘kon biriktiring, topshiriq bering va tashriflarni kuzating.",
      fieldButton("new-task", "Topshiriq berish", "plus", true));
    body.innerHTML = `<div class="field-home">
      ${panel("Agentlar va do‘konlar", snapshot.agents.length ? `<div class="field-list">${snapshot.agents.map(agent => `<article class="field-row">
        <span class="metric-icon">${icon("user")}</span><span class="field-row-text"><strong>${esc(agent.username)}</strong>
        <small>${count(agent.stores)} do‘kon${agent.active ? "" : " · To‘xtatilgan"}</small></span>
        <button class="btn" type="button" data-assign="${agent.id}">Do‘kon biriktirish</button></article>`).join("")}</div>`
        : empty("Agentlar yo‘q", "Xodimlar sahifasida agent hisobini yarating.", "users"), link("/members", "Xodimlar", "arrow-right"))}
      ${panel("So‘nggi topshiriqlar", taskRows(snapshot.tasks))}
      ${panel("So‘nggi tashriflar", visitRows(snapshot.visits))}</div>`;
    $("new-task").onclick = createTask;
    body.querySelectorAll("[data-assign]").forEach(button => button.onclick = () =>
      assign(snapshot.agents.find(agent => agent.id === Number(button.dataset.assign))).catch(e => error(e.message)));
    wireTasks();
  }

  async function addDelivery(visit) {
    try {
      const { products } = await rastaJson(`/api/field/visits/${visit.id}/products`);
      if (!products.length) { error("Tovarni qayd etish uchun korxona admini katalogga SKU qo‘shishi kerak."); return; }
      const key = crypto.randomUUID();
      open("Tovar topshirish", `<label class="field"><span>Mahsulot (SKU)</span><select name="sku_id" class="input" required>
        <option value="">Mahsulotni tanlang</option>${options(products, "name", "sku_id")}</select></label>
        <label class="field"><span>Miqdor (dona)</span><input name="quantity" type="number" inputmode="numeric" required class="input" min="1" max="1000000"></label>`, values =>
        rastaJson(`/api/field/visits/${visit.id}/deliveries`, "POST", {
          request_key: key, sku_id: values.get("sku_id"), quantity: Number(values.get("quantity"))
        }));
    } catch (e) { error(e.message); }
  }

  async function visitDetail() {
    const visit = await rastaJson(`/api/field/visits/${targetId}`);
    const canWork = !manager && !visit.finished && stores.some(store => store.id === visit.store_id);
    heading(visit.store, `${visit.agent} · ${rastaDate(visit.started)}`, link("/field", "Tashriflar", "visit"));
    const visitTasks = snapshot.tasks.filter(task => task.store_id === visit.store_id && task.status === "pending");
    body.innerHTML = `<div class="field-home">
      <section class="field-current">
        <span class="field-current-state">${visit.finished ? "Yakunlangan tashrif" : "Ochiq tashrif"}</span>
        <h2>${esc(visit.store)}</h2><p>${visit.finished ? "Yakunlandi" : "Boshlandi"} · ${esc(rastaDate(visit.finished || visit.started))}</p>
        <div class="page-actions">${canWork ? link(`/?visit=${visit.id}`, "Fotoaudit", "camera", true) + fieldButton("delivery", "Tovar topshirish", "box") : ""}
        ${!manager && !visit.finished ? fieldButton("finish", "Tashrifni yakunlash", "check") : ""}</div>
      </section>
      <div class="field-summary">${summary("Fotoauditlar", visit.audits.length, "report")}${summary("Topshirilgan tovar", visit.deliveries.reduce((n,line)=>n+line.quantity,0), "box")}</div>
      ${visit.note ? panel("Tashrif izohi", `<p class="audit-note">${esc(visit.note)}</p>`) : ""}
      ${panel("Fotoauditlar", visit.audits.length ? `<div class="field-list">${visit.audits.map(audit => `<a class="field-row" href="/reports?id=${encodeURIComponent(audit.id)}">
        <span class="metric-icon">${icon("report")}</span><span class="field-row-text"><strong>Javon auditi</strong>
        <small>${count(audit.products)} mahsulot · ${count(audit.gaps)} bo‘sh joy</small></span>${icon("arrow-right")}</a>`).join("")}</div>`
        : empty("Audit yuborilmagan", "Yuborilgan fotoauditlar shu yerda ko‘rinadi.", "report"))}
      ${panel("Topshirilgan tovarlar", visit.deliveries.length ? `<div class="field-list">${visit.deliveries.map(line => `<div class="field-row">
        <span class="metric-icon">${icon("box")}</span><span class="field-row-text"><strong>${esc(line.product)}</strong><small>${esc(rastaDate(line.created))}</small></span>
        <strong>${count(line.quantity)} dona</strong></div>`).join("")}</div>`
        : empty("Tovar qayd etilmagan", "Topshirilgan mahsulotlar va miqdori shu yerda ko‘rinadi.", "box"))}
      ${canWork ? panel("Do‘kon topshiriqlari", taskRows(visitTasks, visit)) : ""}
      </div>`;
    if (canWork) $("delivery").onclick = () => addDelivery(visit);
    if ($("finish")) $("finish").onclick = () => open("Tashrifni yakunlash", note("Tashrif haqida izoh"), values =>
      rastaJson(`/api/field/visits/${visit.id}/finish`, "POST", {note: values.get("note")}), "Yakunlash");
    wireTasks(visit);
  }

  async function load() {
    error();
    body.setAttribute("aria-busy", "true");
    try {
      const [data, rows] = await Promise.all([rastaJson("/api/field"), rastaJson("/api/stores")]);
      snapshot = data;
      stores = rows.stores;
      if (targetId) await visitDetail();
      else if (manager) adminHome();
      else agentHome();
    } catch (e) { body.innerHTML = ""; error(e.message); }
    finally { body.setAttribute("aria-busy", "false"); }
  }
  rastaSession.then(session => {
    if (!session.user || !["org_admin", "operator"].includes(session.user.role)) return;
    manager = session.user.role === "org_admin";
    load();
  });
})();
