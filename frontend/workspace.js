(() => {
  const $ = id => document.getElementById(id);
  const body = $("work-body"), actions = $("work-actions"), errorBox = $("work-error");
  const editor = $("editor"), editorForm = $("editor-form"), editorError = $("editor-error");
  const path = location.pathname;
  const queries = new URLSearchParams(location.search);
  const statusNames = {draft: "Qoralama", submitted: "Tekshiruvda", approved: "Tasdiqlangan", returned: "Qaytarilgan"};
  const format = new Intl.NumberFormat("uz-UZ");
  const count = value => format.format(value ?? 0);
  const tag = (text, tone = "") => `<span class="work-tag ${tone}">${esc(text)}</span>`;
  const reportTag = status => tag(statusNames[status], status);
  const reportUrl = id => `/reports?id=${encodeURIComponent(id)}`;
  const empty = (title, message, glyph = "report") =>
    `<div class="work-empty"><span class="metric-icon">${icon(glyph)}</span><strong>${esc(title)}</strong><p>${esc(message)}</p></div>`;
  let session, editorSubmit;

  function heading(title, subtitle, buttons = "") {
    $("work-title").textContent = title;
    $("work-subtitle").textContent = subtitle;
    actions.innerHTML = buttons;
    document.title = title + " · Rasta AI";
  }
  function showError(error) {
    errorBox.textContent = error?.message || "";
    errorBox.hidden = !errorBox.textContent;
  }
  async function render(handler) {
    showError();
    body.setAttribute("aria-busy", "true");
    try { await handler(); }
    catch (error) { body.innerHTML = ""; showError(error); }
    finally { body.setAttribute("aria-busy", "false"); }
  }
  function openEditor(title, fields, submit, values = {}) {
    editorForm.reset();
    editorError.textContent = "";
    $("editor-title").textContent = title;
    $("editor-fields").innerHTML = fields;
    for (const [key, value] of Object.entries(values)) {
      const input = editorForm.elements.namedItem(key);
      if (!input) continue;
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value ?? "";
    }
    editorSubmit = submit;
    editor.showModal();
  }
  $("editor-close").onclick = () => editor.close();
  editorForm.onsubmit = async event => {
    event.preventDefault();
    const button = event.submitter || editorForm.querySelector('[type="submit"]');
    button.disabled = true;
    editorError.textContent = "";
    try {
      await editorSubmit(Object.fromEntries(new FormData(editorForm)));
      editor.close();
    } catch (error) { editorError.textContent = error.message; }
    finally { button.disabled = false; }
  };
  const field = (label, name, attributes) => `<label class="field"><span>${label}</span><input class="input" name="${name}" ${attributes}></label>`;
  const activeField = '<label class="check-field field"><input type="checkbox" name="active"><span>Faol</span></label>';
  const nameField = label => field(label, "name", 'required maxlength="100"');
  const usernameField = field("Login", "username", 'required minlength="3" maxlength="48" autocomplete="off"');
  const passwordField = (required = true) => field(required ? "Parol" : "Yangi parol (ixtiyoriy)", "password",
    `type="password" minlength="12" maxlength="128" autocomplete="new-password" ${required ? "required" : ""}`);
  const noteField = (label, name, required = false) => `<label class="field"><span>${label}</span><textarea name="${name}" class="input" rows="3" maxlength="1000" ${required ? "required" : ""}></textarea></label>`;
  const link = (href, label, primary = false, glyph = "") =>
    `<a class="btn${primary ? " primary" : ""}" href="${href}">${glyph ? icon(glyph) : ""}${label}</a>`;
  const button = (id, label, glyph = "plus", primary = true) =>
    `<button class="btn${primary ? " primary" : ""}" id="${id}" type="button">${icon(glyph)}${label}</button>`;
  const metric = (label, value, glyph, tone = "green") => `<article class="card work-metric">
    <div><span class="metric-icon ${tone}">${icon(glyph)}</span><span>${label}</span></div><strong>${value}</strong></article>`;
  const table = (labels, rows) => `<div class="work-table-wrap"><table class="work-table">
    <thead><tr>${labels.map(label => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const reportRows = rows => table(["Do‘kon", "Holat", "Mahsulot", "Bo‘sh joy", "Yuborilgan", ""],
    rows.map(row => `<tr><td><strong>${esc(row.store || "Do‘kon belgilanmagan")}</strong><small>${esc(row.author)}</small></td>
      <td>${reportTag(row.status)}</td><td>${count(row.products)}</td><td>${count(row.gaps)}</td>
      <td>${esc(rastaDate(row.submitted || row.created))}</td><td>${link(reportUrl(row.id), "Ochish")}</td></tr>`).join(""));

  async function dashboard(days = "30") {
    heading("Bosh panel", session.user.organization,
      '<label class="period-control"><span>Davr</span><select class="input" id="days"><option value="7">7 kun</option><option value="30">30 kun</option><option value="90">90 kun</option></select></label>'
      + link("/", "Yangi tahlil", true, "plus"));
    $("days").value = days;
    $("days").onchange = event => render(() => dashboard(event.target.value));
    const [stats, latest] = await Promise.all([rastaJson(`/api/overview?days=${days}`), rastaJson("/api/reports")]);
    if ($("days").value !== days) return;
    const recognition = stats.products ? Math.round(100 * stats.recognized / stats.products) + "%" : "—";
    const max = Math.max(1, ...stats.trend.map(item => item.count));
    const chart = stats.trend.map(item => {
      const date = item.day.split("-").slice(1).reverse().join(".");
      return `<div class="trend-col" title="${date} · ${count(item.count)} ta"><i style="--value:${100 * item.count / max}%"></i></div>`;
    }).join("");
    body.innerHTML = `<div class="work-metrics">
      ${metric("Kelgan auditlar", count(stats.reports), "report")}
      ${metric("Tekshiruv navbati", count(stats.pending), "info", "amber")}
      ${metric("Kuzatilgan do‘konlar", count(stats.stores), "store", "blue")}
      ${metric("SKU tanish", recognition, "scan")}</div>
      <div class="dashboard-grid">
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Audit faolligi</h2><p>Kunlik yuborilgan hisobotlar · Toshkent vaqti</p></div></div>
          ${stats.reports ? `<div class="trend-chart" role="img" aria-label="Davr ichida ${count(stats.reports)} ta audit yuborilgan">${chart}</div>
            <div class="trend-axis"><span>${stats.trend[0].day}</span><span>${stats.trend.at(-1).day}</span></div>`
            : empty("Faollik hali yo‘q", "Yuborilgan auditlar shu yerda paydo bo‘ladi.", "chart")}</section>
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Audit holati</h2><p>Tanlangan davr</p></div></div>
          ${[["submitted",stats.pending],["approved",stats.approved],["returned",stats.returned]].map(([status,value]) => `<div class="state-line">${reportTag(status)}<strong>${count(value)}</strong></div>`).join("")}
          <div class="state-line"><span>Topilgan bo‘sh joylar</span><strong>${count(stats.gaps)}</strong></div></section>
      </div>
      ${stats.reports ? `<div class="dashboard-grid">
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Do‘konlar kesimida</h2>${link("/stores","Do‘konlar")}</div>
        ${table(["Do‘kon","Audit","Bo‘sh joy"], stats.store_activity.map(row => `<tr><td>${esc(row.name)}</td><td>${count(row.reports)}</td><td>${count(row.gaps)}</td></tr>`).join(""))}</section>
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Jamoa faolligi</h2>${link("/members","Xodimlar")}</div>
        ${table(["Hisob","Audit","Tasdiqlangan"], stats.operator_activity.map(row => `<tr><td>${esc(row.username)}</td><td>${count(row.reports)}</td><td>${count(row.approved)}</td></tr>`).join(""))}</section>
      </div>` : ""}
      <section class="card work-section"><div class="panel-heading"><div><h2>So‘nggi hisobotlar</h2><p>Korxona jamoasidan yuborilgan auditlar</p></div>${link("/reports", "Barchasi")}</div>
        ${latest.reports.length ? reportRows(latest.reports.slice(0,5)) : empty("Hisobotlar hali yo‘q", "Jamoa birinchi auditni yuborgach, uni shu yerda ko‘rasiz.")}</section>`;
  }

  async function reports(filter = "", offset = 0) {
    const id = queries.get("id");
    if (id) { await reportDetails(id); return; }
    heading(session.user.role === "operator" ? "Mening hisobotlarim" : "Hisobotlar",
      "Auditlarni oching va tekshiruv holatini kuzating.", link("/", "Yangi tahlil", true, "plus"));
    const params = new URLSearchParams({offset: String(offset)});
    if (filter) params.set("status", filter);
    const data = await rastaJson("/api/reports?" + params);
    body.innerHTML = `<section class="card work-section">
      <div class="panel-heading"><h2>Auditlar ro‘yxati</h2><label class="period-control"><span>Holat</span>
      <select class="input" id="report-filter"><option value="">Yuborilgan barcha</option>${Object.entries(statusNames).map(([value,label]) =>
        `<option value="${value}">${label}</option>`).join("")}</select></label></div>
      ${data.reports.length ? reportRows(data.reports) : empty("Hisobotlar topilmadi", "Boshqa holatni tanlang yoki yangi tahlil yarating.")}
      ${data.has_more || offset ? '<div class="pager"><button class="btn" id="back">Oldingi</button><button class="btn" id="more">Keyingi</button></div>' : ""}</section>`;
    $("report-filter").value = filter;
    $("report-filter").onchange = event => render(() => reports(event.target.value, 0));
    if ($("more")) {
      $("more").disabled = !data.has_more;
      $("back").disabled = !offset;
      $("more").onclick = () => render(() => reports(filter, offset + 50));
      $("back").onclick = () => render(() => reports(filter, offset - 50));
    }
  }
  async function reportDetails(id) {
    const endpoint = "/api/reports/" + encodeURIComponent(id);
    const report = await rastaJson(endpoint);
    const result = report.result, summary = result.summary;
    heading(report.store || "Javon auditi", `${report.author} · ${rastaDate(report.submitted || report.created)}`,
      link("/reports", "Hisobotlar"));
    const showReview = session.user.role === "org_admin" && report.status === "submitted";
    const canSend = report.author === session.user.username && ["draft", "returned"].includes(report.status);
    body.innerHTML = `<div class="detail-grid">
      <section class="card work-section">
        <div class="panel-heading"><h2>Javon rasmi</h2><label class="check-field"><input id="marked" type="checkbox" checked> Belgilar</label></div>
        <div class="audit-image" id="audit-image"><img src="${endpoint}/photo" alt="Hisobotdagi javon rasmi"><div id="audit-overlay" aria-hidden="true"></div></div>
      </section>
      <div class="dashboard-grid detail-side">
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Natija</h2>${reportTag(report.status)}</div>
          <div class="state-line">Topilgan mahsulotlar <strong>${count(summary.num_products)}</strong></div>
          <div class="state-line">Tanilgan SKU <strong>${count(summary.num_known)}</strong></div>
          <div class="state-line">Bo‘sh joylar <strong>${count(summary.oos_gaps)}</strong></div>
          ${report.note ? `<p class="audit-note">${esc(report.note)}</p>` : ""}
          ${report.feedback ? `<p class="audit-note"><strong>Admin izohi</strong><br>${esc(report.feedback)}</p>` : ""}
          <div class="report-actions">
            ${showReview ? button("approve", "Tasdiqlash", "check") + button("return", "Qaytarish", "refresh", false) : ""}
            ${canSend ? button("resubmit", "Hisobotni yuborish", "upload") : ""}
          </div></section>
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Javondagi mahsulotlar</h2></div>
          ${summary.products.length ? summary.products.map(product => `<div class="state-line"><span>${esc(product.name)}</span><strong>${count(product.facings)} ta</strong></div>`).join("")
          : '<p class="work-loading">SKU aniqlanmadi.</p>'}</section>
        ${report.events.length ? `<section class="card dashboard-panel"><div class="panel-heading"><h2>Harakatlar tarixi</h2></div>
          <ol class="audit-events">${report.events.map(event => `<li>${reportTag(event.status)}<small>${esc(event.username)} · ${esc(rastaDate(event.created))}</small>
            ${event.note ? `<p>${esc(event.note)}</p>` : ""}</li>`).join("")}</ol></section>` : ""}
      </div></div>`;
    const [width, height] = result.size;
    const box = (coords, kind) => {
      const [x1,y1,x2,y2] = coords;
      return `<i class="audit-mark ${kind}" style="left:${100*x1/width}%;top:${100*y1/height}%;width:${100*(x2-x1)/width}%;height:${100*(y2-y1)/height}%"></i>`;
    };
    $("audit-overlay").innerHTML = result.detections.map(item => box(item.b, item.s ? "known" : "unknown")).join("")
      + result.gaps.map(item => box(item, "gap")).join("");
    $("marked").onchange = event => { $("audit-overlay").hidden = !event.target.checked; };
    if (showReview) {
      for (const [buttonId,status,title] of [["approve","approved","Hisobotni tasdiqlash"],["return","returned","Hisobotni qaytarish"]]) {
        $(buttonId).onclick = () => openEditor(title, noteField("Admin izohi", "feedback", status === "returned"), async values => {
          await rastaJson(endpoint + "/review", "POST", {status, feedback: values.feedback});
          await render(() => reportDetails(id));
        });
      }
    }
    if (canSend) $("resubmit").onclick = async () => {
      try {
        const { stores } = await rastaJson("/api/stores");
        if (!stores.length) throw new Error("Korxona admini avval do‘kon qo‘shishi kerak.");
        const options = stores.map(store => `<option value="${store.id}">${esc(store.name)}</option>`).join("");
        openEditor("Hisobotni yuborish", `<label class="field"><span>Do‘kon</span><select class="input" name="store_id" required><option value="">Do‘konni tanlang</option>${options}</select></label>`
          + noteField("Izoh", "note"), async values => {
          await rastaJson(endpoint + "/submit", "POST", {store_id: Number(values.store_id), note: values.note});
          await render(() => reportDetails(id));
        }, {store_id: report.store_id || "", note: report.note});
      } catch (error) { showError(error); }
    };
  }

  async function organizations() {
    const orgId = Number(queries.get("id"));
    const { organizations: rows } = await rastaJson("/api/organizations");
    const selected = rows.find(row => row.id === orgId);
    if (selected) { await organizationDetails(selected); return; }
    heading("Korxonalar", "Platformadagi korxonalar va ularning ish maydonlari.", button("add-org", "Korxona yaratish"));
    const totals = rows.reduce((acc,row) => ({active: acc.active + row.active, members: acc.members + row.members,
      reports: acc.reports + row.reports}), {active:0,members:0,reports:0});
    body.innerHTML = `<div class="work-metrics">${metric("Korxonalar", count(rows.length), "building")}
      ${metric("Faol korxonalar", count(totals.active), "check")}
      ${metric("Hisoblar", count(totals.members), "users","blue")}
      ${metric("Yuborilgan auditlar", count(totals.reports), "report","amber")}</div>
      <section class="card work-section"><div class="panel-heading"><h2>Korxonalar ro‘yxati</h2></div>${table(
        ["Korxona","Holat","Hisoblar","Do‘konlar","Auditlar",""], rows.map(row => `<tr><td><strong>${esc(row.name)}</strong></td>
          <td>${tag(row.active ? "Faol" : "To‘xtatilgan",row.active ? "approved" : "returned")}</td>
          <td>${count(row.members)}</td><td>${count(row.stores)}</td><td>${count(row.reports)}</td>
          <td>${link(`/organizations?id=${row.id}`,"Boshqarish")}</td></tr>`).join(""))}</section>`;
    $("add-org").onclick = () => openEditor("Korxona va birinchi admin", nameField("Korxona nomi") + usernameField + passwordField(), async values => {
      const org = await rastaJson("/api/organizations", "POST", values);
      location.assign(`/organizations?id=${org.id}`);
    });
  }
  async function organizationDetails(org) {
    heading(org.name, org.active ? "Faol ish maydoni" : "To‘xtatilgan korxona", link("/organizations", "Korxonalar")
      + button("edit-org", "Sozlamalar", "building", false) + button("add-member", "Admin yaratish"));
    $("edit-org").onclick = () => openEditor("Korxona sozlamalari", nameField("Korxona nomi") + activeField, async values => {
      await rastaJson(`/api/organizations/${org.id}`, "PATCH", {name:values.name, active: Boolean(values.active)});
      await render(organizations);
    }, org);
    await memberList(`/api/organizations/${org.id}/members`, "Korxona adminlari", "Admin", () => render(organizations), "org_admin");
  }
  async function members() {
    heading("Xodimlar", `${session.user.organization} jamoasining foydalanuvchi hisoblari.`, button("add-member", "Operator yaratish"));
    await memberList("/api/members", "Jamoa hisoblari", "Operator", () => render(members), "operator");
  }
  async function memberList(endpoint, title, label, refresh, editableRole) {
    const { users } = await rastaJson(endpoint);
    body.innerHTML = `<section class="card work-section"><div class="panel-heading"><h2>${title}</h2></div>${users.length
      ? table(["Login","Rol","Holat",""], users.map(user => `<tr><td><strong>${esc(user.username)}</strong></td>
        <td>${rastaRole(user.role)}</td><td>${tag(user.active ? "Faol" : "To‘xtatilgan", user.active ? "approved" : "returned")}</td>
        <td>${user.role === editableRole ? `<button class="btn" data-member="${user.id}">Tahrirlash</button>` : ""}</td></tr>`).join(""))
      : empty("Hisoblar hali yo‘q", "Yangi hisob yaratish uchun yuqoridagi tugmani bosing.", "users")}</section>`;
    $("add-member").onclick = () => openEditor(`${label} yaratish`, usernameField + passwordField(), async values => {
      await rastaJson(endpoint, "POST", values);
      await refresh();
    });
    body.querySelectorAll("[data-member]").forEach(button => button.onclick = () => {
      const user = users.find(item => item.id === Number(button.dataset.member));
      openEditor(user.username, activeField + passwordField(false), async values => {
        await rastaJson(`${endpoint}/${user.id}`, "PATCH", {active: Boolean(values.active), password: values.password || null});
        await refresh();
      }, user);
    });
  }
  async function stores() {
    heading("Do‘konlar", "Operator hisobot yuborishda shu ro‘yxatdan do‘kon tanlaydi.", button("add-store", "Do‘kon qo‘shish"));
    const { stores: rows } = await rastaJson("/api/stores");
    body.innerHTML = `<section class="card work-section"><div class="panel-heading"><h2>Do‘konlar ro‘yxati</h2></div>${rows.length
      ? table(["Nomi","Manzil",""], rows.map(store => `<tr><td><strong>${esc(store.name)}</strong></td><td>${esc(store.address || "—")}</td>
        <td><button class="btn" type="button" data-store="${store.id}">Tahrirlash</button></td></tr>`).join(""))
      : empty("Do‘konlar hali yo‘q", "Jamoa audit yuborishi uchun birinchi do‘konni qo‘shing.", "store")}</section>`;
    const fields = nameField("Do‘kon nomi") + field("Manzil", "address", 'maxlength="200"');
    $("add-store").onclick = () => openEditor("Yangi do‘kon", fields, async values => {
      await rastaJson("/api/stores", "POST", values);
      await render(stores);
    });
    body.querySelectorAll("[data-store]").forEach(button => button.onclick = () => {
      const store = rows.find(row => row.id === Number(button.dataset.store));
      openEditor("Do‘konni tahrirlash", fields, async values => {
        await rastaJson(`/api/stores/${store.id}`, "PATCH", values);
        await render(stores);
      }, store);
    });
  }
  rastaSession.then(value => {
    session = value;
    if (!value.workspaces || !value.user) {
      if (value.enabled !== undefined && !value.workspaces) location.replace("/");
      else if (value.unavailable) showError(new Error("Ish maydoni yuklanmadi. Server bilan bog‘lanishni tekshiring."));
      return;
    }
    const page = ({"/dashboard": dashboard, "/reports": reports, "/organizations": organizations,
      "/members": members, "/stores": stores})[path];
    if (page) render(page);
  });
})();
