export default function mountPage() {
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
  const reportFilterKeys = ["status", "store_id", "author_id", "from", "to"];
  function reportFilters(source = queries) {
    const params = new URLSearchParams();
    for (const key of reportFilterKeys) if (source.get(key)) params.set(key, source.get(key));
    return params;
  }
  function reportsUrl(params = new URLSearchParams()) {
    const query = params.toString();
    return query ? `/reports?${query}` : "/reports";
  }
  function reportUrl(id, filters = new URLSearchParams()) {
    const params = new URLSearchParams(filters);
    params.set("id", id);
    return reportsUrl(params);
  }
  const empty = (title, message, glyph = "report") =>
    `<div class="work-empty"><span class="metric-icon">${icon(glyph)}</span><strong>${esc(title)}</strong><p>${esc(message)}</p></div>`;
  let session, editorSubmit;

  function heading(title, subtitle, buttons = "", back = []) {
    $("work-title").textContent = title;
    $("work-subtitle").textContent = subtitle;
    actions.innerHTML = buttons;
    rastaBack("work-back", ...back);
    $("workspace").dataset.view = path.slice(1);
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
  const metric = (label, value, glyph, tone = "green", detail = "") => `<article class="card work-metric">
    <span class="metric-icon ${tone}">${icon(glyph)}</span><span class="metric-label">${label}</span>
    <strong>${value}</strong>${detail ? `<small class="metric-detail">${detail}</small>` : ""}</article>`;
  const periodLabel = '<label class="period-control"><span>Davr</span><select class="input" id="days"><option value="7">7 kun</option><option value="30">30 kun</option><option value="90">90 kun</option><option value="365">1 yil</option></select></label>';
  function bindPeriod(days, refresh) {
    $("days").value = days;
    $("days").onchange = event => {
      const next = event.target.value;
      const url = new URL(location.href);
      url.searchParams.set("days", next);
      history.replaceState(null, "", url);
      render(() => refresh(next));
    };
  }
  // Audit faolligi: kunlik ma'lumot kun / hafta / oy bo'yicha guruhlab ko'rsatiladi.
  const trendUnits = { day: ["Kun", "kuniga"], week: ["Hafta", "haftasiga"], month: ["Oy", "oyiga"] };
  const trendMetrics = { count: "Auditlar", gaps: "Bo‘sh joylar", rate: "SKU tanish" };
  let trendState = { trend: [], total: 0, unit: null, metric: "count" };
  const shortDate = day => day.split("-").slice(1).reverse().join(".");
  function trendBuckets(trend, unit) {
    const buckets = [];
    for (const item of trend) {
      const date = new Date(item.day + "T00:00:00Z");
      let key = item.day;
      if (unit === "month") key = item.day.slice(0, 7);                 // oy: YYYY-MM
      if (unit === "week") {                                            // hafta: dushanbadan
        date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 6) % 7);
        key = date.toISOString().slice(0, 10);
      }
      let bucket = buckets.at(-1);
      if (!bucket || bucket.key !== key) buckets.push(bucket = { key, first: item.day, last: item.day, count: 0, gaps: 0, products: 0, recognized: 0 });
      bucket.last = item.day;
      for (const name of ["count", "gaps", "products", "recognized"]) bucket[name] += item[name] || 0;
    }
    return buckets.map(bucket => ({ ...bucket,
      label: unit === "day" ? shortDate(bucket.first) : unit === "month" ? bucket.key.split("-").reverse().join(".")
        : `${shortDate(bucket.first)}–${shortDate(bucket.last)}`,
    }));
  }
  function trendBody() {
    const { trend, total, unit, metric } = trendState;
    if (!total) return empty("Faollik hali yo‘q", "Yuborilgan auditlar shu yerda paydo bo‘ladi.", "chart");
    const buckets = trendBuckets(trend, unit);
    const rate = metric === "rate";
    const value = bucket => rate ? (bucket.products ? 100 * bucket.recognized / bucket.products : null) : bucket[metric];
    const show = number => rate ? `${Math.round(number)}%` : count(number);
    const known = buckets.filter(bucket => value(bucket) != null);
    const max = rate ? 100 : Math.ceil(Math.max(1, ...known.map(value)) / 4) * 4;
    const peak = known.reduce((best, item) => value(item) > value(best) ? item : best, known[0]);
    const sum = name => buckets.reduce((all, bucket) => all + bucket[name], 0);
    let summary;
    if (rate) {
      const products = sum("products");
      summary = `<span>Davr bo‘yicha <strong>${products ? show(100 * sum("recognized") / products) : "—"}</strong></span>
        ${peak ? `<span>Eng yuqori: <strong>${show(value(peak))}</strong> (${peak.label})</span>` : ""}
        <span>Tanildi: <strong>${count(sum("recognized"))}</strong> / ${count(products)} mahsulot</span>`;
    } else {
      const whole = sum(metric), average = whole / buckets.length;
      summary = `<span>O‘rtacha <strong>${average < 10 ? average.toFixed(1) : count(Math.round(average))}</strong> ta ${trendUnits[unit][1]}</span>
        <span>Eng ko‘p: <strong>${count(value(peak))}</strong> ta (${peak.label})</span><span>Jami: <strong>${count(whole)}</strong></span>`;
    }
    return `<div class="chart-frame">
        <div class="chart-ticks" aria-hidden="true">${[max, max*0.75, max*0.5, max*0.25, 0].map(tick => `<span>${show(tick)}</span>`).join("")}</div>
        <div class="trend-chart" data-metric="${metric}" role="img" aria-label="${trendMetrics[metric]}: davr ichida ${count(total)} ta audit yuborilgan">${buckets.map(item => {
          const height = value(item);
          return `<div class="trend-col" title="${item.label} · ${height == null ? "audit yo‘q" : show(height)}"><i style="--value:${height == null ? 0 : 100 * height / max}%"></i></div>`;
        }).join("")}</div>
        <div class="trend-axis"><span>${buckets[0].label}</span><span>${buckets.at(-1).label}</span></div></div>
      <p class="trend-summary">${summary}</p>`;
  }
  const segmented = (label, name, options, current) => `<div class="segmented" role="group" aria-label="${label}">${options.map(([value, text]) =>
    `<button type="button" data-trend-${name}="${value}" aria-pressed="${value === current}">${text}</button>`).join("")}</div>`;
  function trendPanel(trend, total, action = "") {
    // davrga mos standart: qisqa davr — kunlik, 90 kun — haftalik, 1 yil — oylik
    const fallback = trend.length > 120 ? "month" : trend.length > 45 ? "week" : "day";
    const detailed = trend.length > 0 && trend[0].gaps !== undefined;       // bo'sh joy va tanish ham kunlar bo'yicha kelgan
    trendState = { trend, total, unit: trendState.unit && trendState.length === trend.length ? trendState.unit : fallback,
      metric: detailed ? trendState.metric || "count" : "count", length: trend.length };
    return `<section class="card dashboard-panel trend-panel"><div class="panel-heading"><div><h2>Audit faolligi</h2>
      <p>Yuborilgan hisobotlar · Toshkent vaqti</p></div>
      <div class="trend-tools">${detailed ? segmented("Ko‘rsatkich", "metric", Object.entries(trendMetrics), trendState.metric) : ""}
        ${segmented("Guruhlash", "unit", Object.entries(trendUnits).map(([unit, [label]]) => [unit, label]), trendState.unit)}${action}</div></div>
      <div class="trend-body">${trendBody()}</div></section>`;
  }
  body.addEventListener("click", event => {
    const button = event.target.closest("[data-trend-unit],[data-trend-metric]");
    if (!button) return;
    const name = button.dataset.trendUnit ? "unit" : "metric";
    trendState[name] = button.dataset.trendUnit || button.dataset.trendMetric;
    button.parentElement.querySelectorAll("button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    button.closest(".trend-panel").querySelector(".trend-body").innerHTML = trendBody();
  });
  // Oldingi davrga nisbatan o'zgarish: foiz (yoki foiz punkti). inverse — o'sish yomon (bo'sh joylar).
  function delta(current, previous, { inverse = false, points = false } = {}) {
    if (current == null || previous == null) return "";
    if (!points && !previous) return current ? `<span class="delta ${inverse ? "bad" : "good"}">yangi</span>` : "";
    const change = points ? current - previous : 100 * (current - previous) / previous;
    if (Math.abs(change) < 0.5) return '<span class="delta flat">o‘zgarishsiz</span>';
    return `<span class="delta ${(change > 0) !== inverse ? "good" : "bad"}">${change > 0 ? "▲" : "▼"} ${count(Math.round(Math.abs(change)))}${points ? " p." : "%"}</span>`;
  }
  const share = (part, whole) => whole ? Math.round(100 * part / whole) : null;
  const percent = value => value == null ? "—" : `${value}%`;
  const barList = (rows, row) => `<ol class="bar-list">${rows.map(row).join("")}</ol>`;
  const bar = (title, value, width, note) => `<li><div class="bar-head"><span>${title}</span><strong>${value}</strong></div>
    <div class="bar-track"><i style="--value:${Math.max(2, Math.min(100, width))}%"></i></div><small>${note}</small></li>`;
  const table = (labels, rows) => `<div class="work-table-wrap"><table class="work-table">
    <thead><tr>${labels.map(label => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const reportRows = (rows, filters) => {
    const desktop = table(["Do‘kon / xodim", "Holat", "Mahsulot", "Bo‘sh joy", "Vaqt", ""],
      rows.map(row => `<tr><td><div class="store-cell"><span class="store-icon">${icon("store")}</span>
        <span><strong>${esc(row.store || "Do‘kon belgilanmagan")}</strong><small>${esc(row.author)}</small></span></div></td>
        <td>${reportTag(row.status)}</td><td>${count(row.products)}</td><td>${count(row.gaps)}</td>
        <td>${esc(rastaDate(row.submitted || row.created))}</td><td>${link(reportUrl(row.id, filters), "Ochish", false, "arrow-right")}</td></tr>`).join(""));
    const mobile = rows.map(row => `<a class="mobile-audit" href="${reportUrl(row.id, filters)}">
      <div><strong>${esc(row.store || "Do‘kon belgilanmagan")}</strong><small>${esc(row.author)} · ${esc(rastaDate(row.submitted || row.created))}</small></div>
      ${reportTag(row.status)}
      <div class="mobile-audit-metrics"><span>${count(row.products)} mahsulot</span><span>${count(row.gaps)} bo‘sh joy</span>${icon("arrow-right")}</div>
    </a>`).join("");
    return `<div class="desktop-audits">${desktop}</div><div class="mobile-audits">${mobile}</div>`;
  };

  async function dashboard(days = requestedDays()) {
    heading("Bosh panel", session.user.organization, periodLabel + link("/", "Yangi tahlil", true, "plus"));
    bindPeriod(days, dashboard);
    const [stats, latest, field] = await Promise.all([
      rastaJson(`/api/overview?days=${days}`), rastaJson("/api/reports"), rastaJson(`/api/field/overview?days=${days}`)
    ]);
    if ($("days").value !== days) return;
    const before = stats.previous, shelf = stats.shelf, queue = stats.queue;
    const recognition = share(stats.recognized, stats.products);
    const range = {from: stats.trend[0].day, to: stats.trend.at(-1).day};
    const drill = filter => reportsUrl(new URLSearchParams({...range, ...filter}));
    const statuses = [["submitted",stats.pending],["approved",stats.approved],["returned",stats.returned]];
    const versus = (change, fallback = "") => change ? `${change} oldingi ${days} kunga nisbatan` : fallback;
    const perAudit = (value, audits) => audits ? value / audits : null;
    const gapsEach = perAudit(stats.gaps, stats.reports);
    const decided = stats.approved + stats.returned;
    const unresolved = stats.products - stats.recognized;        // SKU'si aniqlanmagan (katalogda yo'q yoki faqat brendi ma'lum)

    // E'tibor talab qiladigan holatlar: kechikkan navbat, qaytarilgan auditlar, auditsiz do'konlar...
    const alerts = [];
    if (queue.stale) alerts.push(["warn", "info", `${count(queue.stale)} ta audit 24 soatdan ortiq tekshiruv kutmoqda`, reportsUrl(new URLSearchParams({status: "submitted"}))]);
    if (stats.returned) alerts.push(["danger", "refresh", `${count(stats.returned)} ta audit qaytarilgan va hali qayta yuborilmagan`, drill({status: "returned"})]);
    if (field.backlog.overdue) alerts.push(["danger", "calendar", `${count(field.backlog.overdue)} ta topshiriqning muddati o‘tgan`, "/field"]);
    const silent = field.stores.filter(store => !store.audits);
    if (silent.length && field.stores.length) alerts.push(["warn", "store",
      `${count(silent.length)} ta do‘konda bu davrda audit bo‘lmagan: ${esc(silent.slice(0, 3).map(store => store.name).join(", "))}${silent.length > 3 ? " va boshqalar" : ""}`, "/stores"]);
    const worst = stats.store_activity.filter(row => row.gaps).sort((a, b) => b.gaps / b.reports - a.gaps / a.reports)[0];
    if (worst && worst.gaps / worst.reports >= 1) alerts.push(["warn", "empty",
      `«${esc(worst.name)}» do‘konida har auditda o‘rtacha ${(worst.gaps / worst.reports).toFixed(1)} ta bo‘sh joy`, drill({store_id: worst.id})]);
    if (recognition != null && recognition < 70 && unresolved) alerts.push(["info", "box",
      `Mahsulotlarning ${100 - recognition}% i tanilmadi — katalogni to‘ldirish aniqlikni oshiradi`, "/admin"]);

    body.innerHTML = `<div class="work-metrics">
      ${metric("Kelgan auditlar", count(stats.reports), "report", "green", versus(delta(stats.reports, before.reports), `Oxirgi ${days} kun`))}
      ${metric("Tekshiruv navbati", count(queue.open), "info", "amber", queue.stale ? `${count(queue.stale)} tasi 24 soatdan ortiq kutmoqda` : "Kechikkan audit yo‘q")}
      ${metric("Kuzatilgan do‘konlar", `${count(stats.stores)}<small> / ${count(field.stores.length)}</small>`, "store", "blue", versus(delta(stats.stores, before.stores)))}
      ${metric("SKU tanish", percent(recognition), "scan", "violet",
        (stats.products ? `${count(stats.recognized)} / ${count(stats.products)} mahsulot` : "") + (before.products ? ` · ${delta(recognition, share(before.recognized, before.products), {points: true})}` : ""))}
      ${metric("Topilgan bo‘sh joylar", count(stats.gaps), "empty", "amber",
        (gapsEach == null ? "" : `Auditga o‘rtacha ${gapsEach.toFixed(1)} ta`) + (before.reports ? ` · ${delta(stats.gaps, before.gaps, {inverse: true})}` : ""))}
      ${metric("Tasdiqlash ulushi", percent(share(stats.approved, decided)), "check", "green",
        decided ? `${count(stats.approved)} tasdiqlangan · ${count(stats.returned)} qaytarilgan` : "Hali ko‘rib chiqilmagan")}
      ${metric("Tekshiruv vaqti", reviewTime(stats.review_seconds), "clock", "blue", "Yuborilgandan qarorgacha, o‘rtacha")}
      ${metric("SKU aniqlanmagan", count(unresolved), "box", "violet",
        stats.products ? `Topilgan mahsulotlarning ${percent(share(unresolved, stats.products))} · <a class="table-drill" href="/admin">Katalogni to‘ldirish</a>` : "")}</div>
      <section class="card work-section operations-panel"><div class="panel-heading"><div><h2>Agentlar ishi</h2>
        <p>Tanlangan davrdagi tashrif va topshirishlar</p></div>${link("/field", "Tashriflar", false, "visit")}</div>
        <div class="operations-grid">
          <div><small>Tashriflar</small><strong>${count(field.totals.visits)}</strong><span>${count(field.totals.finished)} ta yakunlangan</span></div>
          <div><small>Qamrab olingan do‘konlar</small><strong>${count(field.visited_stores)} / ${count(field.stores.length)}</strong>
            <span>${percent(share(field.visited_stores, field.stores.length))} qamrov</span></div>
          <div><small>Topshirilgan tovar</small><strong>${count(field.totals.units)}</strong><span>dona</span></div>
          <div><small>Bajarilgan topshiriqlar</small><strong>${count(field.totals.tasks_done)}</strong><span>${count(field.backlog.pending)} ta kutmoqda · ${count(field.backlog.overdue)} ta muddati o‘tgan</span></div>
        </div></section>
      <div class="dashboard-grid">
        ${trendPanel(stats.trend, stats.reports, link(drill({}), "Auditlarni ko‘rish", false, "arrow-right"))}
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Audit holati</h2><p>Tanlangan davr</p></div></div>
          ${stats.reports ? `<div class="status-track" aria-hidden="true">${statuses.map(([status,value]) =>
            `<i class="${status}" style="flex:${value}"${value ? "" : " hidden"}></i>`).join("")}</div>` : ""}
          ${statuses.map(([status,value]) => `<a class="state-line drill-link" href="${drill({status})}">${reportTag(status)}<strong>${count(value)} ${icon("arrow-right")}</strong></a>`).join("")}
          <div class="state-line"><span>Jami ochiq navbat</span><strong>${count(queue.open)}</strong></div>
          <div class="state-line"><span>O‘rtacha tekshiruv vaqti</span><strong>${reviewTime(stats.review_seconds)}</strong></div></section>
      </div>
      ${stats.reports ? `<div class="dashboard-grid">
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Brendlar ulushi</h2>
          <p>Tanilgan mahsulotlar soni bo‘yicha · ${count(shelf.audits)} ta audit</p></div></div>
          ${shelf.brands.length ? barList(shelf.brands, row => bar(esc(row.brand), `${row.share_pct}%`, 100 * row.facings / shelf.brands[0].facings,
              `${count(row.facings)} ta mahsulot · ${count(row.audits)} ta auditda`))
            : empty("Brendlar hali tanilmadi", "Katalogga mahsulot qo‘shilgach, javondagi brendlar ulushi shu yerda ko‘rinadi.", "box")}</section>
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Javon holati</h2><p>Tanlangan davr auditlari bo‘yicha</p></div></div>
          <div class="state-line"><span>Javon to‘liqligi</span><strong>${shelf.fill_pct == null ? "—" : `${Math.round(shelf.fill_pct)}%`}</strong></div>
          <div class="state-line"><span>Taxminiy yetishmayotgan mahsulot</span><strong>${count(shelf.missing_facings)} dona</strong></div>
          <div class="state-line"><span>Tanilgan brendlar</span><strong>${count(shelf.brand_count)}</strong></div>
          <div class="state-line"><span>Tanilgan mahsulot turlari</span><strong>${count(shelf.product_count)}</strong></div>
          <div class="state-line"><span>SKU aniqlanmagan mahsulotlar</span><strong>${count(unresolved)}</strong></div>
          ${shelf.fill_pct == null ? "" : `<p class="panel-note">To‘liqlik zona belgilangan ${count(shelf.fill_audits)} ta auditda o‘lchangan.</p>`}</section>
      </div>
      <div class="dashboard-grid">
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Eng ko‘p uchragan mahsulotlar</h2>
          <p>Javondagi soni bo‘yicha</p></div>${link("/admin", "Katalog")}</div>
          ${shelf.products.length ? barList(shelf.products, row => bar(esc(row.name), `${count(row.facings)} ta`, 100 * row.facings / shelf.products[0].facings,
              `${row.brand ? esc(row.brand) + " · " : ""}${count(row.audits)} ta auditda`))
            : empty("Mahsulotlar hali tanilmadi", "Tanilgan mahsulotlar reytingi shu yerda ko‘rinadi.", "box")}</section>
        <section class="card dashboard-panel"><div class="panel-heading"><div><h2>E’tibor talab qiladi</h2><p>Hozir hal qilinishi kerak bo‘lgan holatlar</p></div></div>
          ${alerts.length ? `<div class="alert-list">${alerts.map(([tone, glyph, text, href]) =>
              `<a class="alert-line ${tone}" href="${href}">${icon(glyph)}<span>${text}</span>${icon("arrow-right")}</a>`).join("")}</div>`
            : empty("Hammasi joyida", "Kechikkan audit, muddati o‘tgan topshiriq yoki auditsiz do‘kon yo‘q.", "check")}</section>
      </div>
      <div class="dashboard-grid even">
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Do‘konlar kesimida</h2>${link("/stores","Do‘konlar")}</div>
        ${table(["Do‘kon","Audit","Bo‘sh joy","Tanish","So‘nggi audit"], stats.store_activity.map(row => `<tr><td><a class="table-drill" href="${drill({store_id:row.id})}">${esc(row.name)}</a></td>
          <td>${count(row.reports)}</td><td>${count(row.gaps)}</td><td>${percent(share(row.recognized, row.products))}</td><td>${esc(rastaDate(row.last_submission))}</td></tr>`).join(""))}</section>
        <section class="card dashboard-panel"><div class="panel-heading"><h2>Jamoa faolligi</h2>${link("/members","Xodimlar")}</div>
        ${table(["Hisob","Audit","Tasdiqlangan","Qaytarilgan","Tanish"], stats.operator_activity.map(row => `<tr><td><a class="table-drill" href="${drill({author_id:row.id})}">${esc(row.username)}</a></td>
          <td>${count(row.reports)}</td><td>${count(row.approved)}</td><td>${count(row.returned)}</td><td>${percent(share(row.recognized, row.products))}</td></tr>`).join(""))}</section>
      </div>` : ""}
      <section class="card work-section"><div class="panel-heading"><div><h2>So‘nggi hisobotlar</h2><p>Korxona jamoasidan yuborilgan auditlar</p></div>${link("/reports", "Barchasi")}</div>
        ${latest.reports.length ? reportRows(latest.reports.slice(0,5)) : empty("Hisobotlar hali yo‘q", "Jamoa birinchi auditni yuborgach, uni shu yerda ko‘rasiz.")}</section>`;
  }

  function requestedDays() {
    const days = new URLSearchParams(location.search).get("days");
    return ["7", "30", "90", "365"].includes(days) ? days : "30";
  }
  function reviewTime(seconds) {
    if (seconds == null) return "—";
    const minutes = Math.round(seconds / 60);
    if (minutes <= 0) return "1 daqiqadan kam";
    if (minutes < 60) return `${count(minutes)} daq`;
    const hours = Math.floor(minutes / 60);
    return hours < 24 ? `${count(hours)} soat ${count(minutes % 60)} daq` : `${count(Math.floor(hours / 24))} kun ${count(hours % 24)} soat`;
  }
  function platformAuditPanels(stats) {
    const audit = stats.audits;
    const recognized = audit.products ? `${Math.round(100 * audit.recognized / audit.products)}%` : "—";
    return `<div class="dashboard-grid">
      ${trendPanel(stats.trend, audit.total)}
      <section class="card dashboard-panel"><div class="panel-heading"><div><h2>Audit holati</h2><p>Tanlangan davr</p></div></div>
        ${[["submitted",audit.pending],["approved",audit.approved],["returned",audit.returned]].map(([status,value]) =>
          `<div class="state-line">${reportTag(status)}<strong>${count(value)}</strong></div>`).join("")}
        <div class="state-line"><span>O‘rtacha tekshiruv vaqti</span><strong>${reviewTime(audit.review_seconds)}</strong></div>
        <div class="state-line"><span>Jami ochiq navbat</span><strong>${count(audit.open_queue)}</strong></div>
        <div class="state-line"><span>SKU tanish</span><strong>${recognized}</strong></div>
        <div class="state-line"><span>Topilgan bo‘sh joylar</span><strong>${count(audit.gaps)}</strong></div>
      </section></div>`;
  }
  function platformEvents(events, scoped = false) {
    const labels = {
      organization_created: "Korxona yaratildi",
      organization_activated: "Korxona faollashtirildi",
      organization_suspended: "Korxona to‘xtatildi",
      organization_renamed: "Korxona nomi yangilandi",
      admin_created: "Admin yaratildi",
      admin_activated: "Admin faollashtirildi",
      admin_suspended: "Admin to‘xtatildi",
      admin_password_reset: "Admin paroli tiklandi",
    };
    return `<section class="card work-section platform-events"><div class="panel-heading">
        <div><h2>Boshqaruv tarixi</h2><p>So‘nggi 12 ta amal</p></div></div>
        ${events.length ? `<ol class="audit-events platform-timeline">${events.map(event => `<li>
          <span class="metric-icon">${icon(event.action.startsWith("admin_") ? "user" : "building")}</span>
          <div><strong>${esc(labels[event.action] || event.action)}</strong>
            <small>${scoped ? "" : `<a href="/organizations?id=${event.org_id}">${esc(event.organization)}</a> · `}
            ${esc(event.actor)}${event.subject ? ` · ${esc(event.subject)}` : ""}</small></div>
          <time>${esc(rastaDate(event.created))}</time>
        </li>`).join("")}</ol>`
        : empty("Amallar hali yo‘q", "Korxona va admin hisobidagi o‘zgarishlar shu yerda ko‘rinadi.", "report")}</section>`;
  }
  async function platform(days = requestedDays()) {
    heading("Platforma statistikasi", "Korxonalar va auditlar faolligi", periodLabel + link("/organizations?create=1", "Korxona yaratish", true, "plus"));
    bindPeriod(days, platform);
    const stats = await rastaJson(`/api/platform/overview?days=${days}`);
    if ($("days").value !== days) return;
    body.innerHTML = `<div class="work-metrics">
      ${metric("Korxonalar", count(stats.organizations.total), "building", "green", `${count(stats.organizations.active)} ta faol`)}
      ${metric("Korxona hisoblari", count(stats.members.total), "users", "blue", `${count(stats.members.active)} ta faol`)}
      ${metric("Ro‘yxatdagi do‘konlar", count(stats.stores), "store", "violet")}
      ${metric("Kelgan auditlar", count(stats.audits.total), "report", "amber", `${count(stats.audits.reporting_orgs)} ta korxonadan · ${days} kun`)}
      </div>${platformAuditPanels(stats)}
      <section class="card work-section platform-activity">
        <div class="panel-heading"><div><h2>Korxonalar faolligi</h2><p>Auditlar soni bo‘yicha eng faol 10 korxona</p></div>
          ${link("/organizations", "Barcha korxonalar")}</div>
        ${stats.activity.length ? table(["Korxona","Holat","Audit","Tekshiruvda","So‘nggi yuborilgan",""],
          stats.activity.map(org => `<tr><td><strong>${esc(org.name)}</strong>
            <small class="platform-mobile-summary">${count(org.audits)} ta audit · ${count(org.pending)} ta tekshiruvda
            ${org.active ? "" : " · To‘xtatilgan"}</small></td>
            <td>${tag(org.active ? "Faol" : "To‘xtatilgan",org.active ? "approved" : "returned")}</td>
            <td>${count(org.audits)}</td><td>${count(org.pending)}</td><td>${esc(rastaDate(org.last_submission))}</td>
            <td>${link(`/organizations?id=${org.id}&days=${days}`,"Boshqarish")}</td></tr>`).join(""))
          : empty("Korxonalardan audit kelmadi", "Tanlangan davrda yuborilgan auditlar shu yerda ko‘rinadi.", "building")}
      </section>${platformEvents(stats.events)}`;
  }

  async function reports() {
    const id = queries.get("id");
    if (id) { await reportDetails(id); return; }
    const manager = session.user.role === "org_admin";
    const filters = reportFilters();
    const offset = Number(queries.get("offset")) || 0;
    const canExport = manager && filters.get("status") !== "draft";
    heading(manager ? "Hisobotlar" : "Mening hisobotlarim",
      "Auditlarni oching va tekshiruv holatini kuzating.", link("/", "Yangi tahlil", true, "plus"));
    const params = new URLSearchParams(filters);
    params.set("offset", String(offset));
    const [data, {stores}, {users}] = await Promise.all([
      rastaJson("/api/reports?" + params), rastaJson("/api/stores"),
      manager ? rastaJson("/api/members") : Promise.resolve({users: []})
    ]);
    const select = (label, name, defaultLabel, items) => `<label class="field"><span>${label}</span>
      <select class="input" name="${name}"><option value="">${defaultLabel}</option>
      ${items.map(([value,text]) => `<option value="${esc(value)}">${esc(text)}</option>`).join("")}</select></label>`;
    const dateField = (label, name) => `<label class="field"><span>${label}</span><input class="input" name="${name}" type="date"></label>`;
    const paged = nextOffset => reportsUrl(new URLSearchParams({...Object.fromEntries(filters), offset: nextOffset}));
    body.innerHTML = `<section class="card work-section report-list">
      <div class="panel-heading"><div><h2>Auditlar ro‘yxati</h2><p>${count(data.total)} ta natija</p></div>
      <div class="page-actions">
        <button class="btn filter-toggle" id="filter-toggle" type="button" aria-controls="report-filters" aria-expanded="false">${icon("search")}
          Filtrlar${filters.size ? ` (${filters.size})` : ""}</button>
        ${canExport ? button("csv-export", "CSV yuklash", "download", false) : ""}</div></div>
      <form class="report-filters" id="report-filters" data-collapsed="true">
        ${select("Holat", "status", "Barcha yuborilgan", Object.entries(statusNames))}
        ${select("Do‘kon", "store_id", "Barcha do‘konlar", stores.map(store => [store.id, store.name]))}
        ${manager ? select("Yuborgan", "author_id", "Barcha xodimlar", users.map(user => [user.id, user.username])) : ""}
        ${dateField("Boshlanish", "from")}${dateField("Tugash", "to")}
        <div class="filter-actions"><button class="btn primary" type="submit">${icon("search")} Filtrlash</button>
          ${filters.size ? '<a class="btn ghost" href="/reports">Tozalash</a>' : ""}</div>
      </form>
      <div class="err" id="report-error" role="alert"></div>
      ${data.reports.length ? reportRows(data.reports, filters) : empty("Hisobotlar topilmadi", "Filtrlarni o‘zgartiring yoki yangi tahlil yarating.")}
      ${data.has_more || offset ? `<div class="pager">${offset ? link(paged(Math.max(0, offset-50)), "Oldingi") : ""}
        ${data.has_more ? link(paged(offset+50), "Keyingi") : ""}</div>` : ""}</section>`;
    const form = $("report-filters");
    $("filter-toggle").onclick = event => {
      const open = form.dataset.collapsed === "true";
      form.dataset.collapsed = String(!open);
      event.currentTarget.setAttribute("aria-expanded", String(open));
    };
    for (const [key,value] of filters) {
      const control = form.elements.namedItem(key);
      if (control) control.value = value;
    }
    form.onsubmit = event => {
      event.preventDefault();
      const values = reportFilters(new FormData(form));
      if (values.has("from") && values.has("to") && values.get("from") > values.get("to")) {
        $("report-error").textContent = "Boshlanish sanasi tugash sanasidan keyin bo‘lishi mumkin emas.";
        return;
      }
      location.assign(reportsUrl(values));
    };
    if (canExport) $("csv-export").onclick = () => exportReports(filters);
  }
  async function exportReports(filters) {
    const control = $("csv-export");
    control.disabled = true;
    $("report-error").textContent = "";
    try {
      const response = await rastaFetch("/api/reports/export.csv?" + filters);
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(typeof error.detail === "string" ? error.detail : "CSV yuklanmadi.");
      }
      const url = URL.createObjectURL(await response.blob());
      const anchor = Object.assign(document.createElement("a"), {href: url, download: "rasta-auditlar.csv"});
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      $("report-error").textContent = error.message;
    } finally { control.disabled = false; }
  }
  async function reportDetails(id) {
    const endpoint = "/api/reports/" + encodeURIComponent(id);
    const report = await rastaJson(endpoint);
    const result = report.result, summary = result.summary;
    heading(report.store || "Javon auditi", `${report.author} · ${rastaDate(report.submitted || report.created)}`,
      "", [reportsUrl(reportFilters()), "Hisobotlar"]);
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
        const legal = report.visit_id && report.store_id ? stores.filter(store => store.id === report.store_id) : stores;
        if (!legal.length) throw new Error("Bu do‘kon sizga biriktirilmagan.");
        const options = legal.map(store => `<option value="${store.id}">${esc(store.name)}</option>`).join("");
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
          <td><div class="page-actions">${link(`/admin?organization=${row.id}`,"Katalog")}
            ${link(`/organizations?id=${row.id}`,"Boshqarish")}</div></td></tr>`).join(""))}</section>`;
    $("add-org").onclick = () => openEditor("Korxona va birinchi admin", nameField("Korxona nomi") + usernameField + passwordField(), async values => {
      const org = await rastaJson("/api/organizations", "POST", values);
      location.assign(`/organizations?id=${org.id}`);
    });
    if (queries.get("create") === "1") {
      queries.delete("create");
      history.replaceState(null, "", queries.size ? "/organizations?" + queries : "/organizations");
      $("add-org").onclick();
    }
  }
  async function organizationDetails(org, days = requestedDays()) {
    heading(org.name, org.active ? "Faol ish maydoni" : "To‘xtatilgan korxona",
      link(`/admin?organization=${org.id}`, "Mahsulot katalogi", false, "box")
      + periodLabel + button("edit-org", "Sozlamalar", "building", false) + button("add-member", "Admin yaratish"),
      ["/organizations", "Korxonalar"]);
    bindPeriod(days, next => organizationDetails(org, next));
    $("edit-org").onclick = () => openEditor("Korxona sozlamalari", nameField("Korxona nomi") + activeField, async values => {
      await rastaJson(`/api/organizations/${org.id}`, "PATCH", {name:values.name, active: Boolean(values.active)});
      await render(organizations);
    }, org);
    const stats = await rastaJson(`/api/organizations/${org.id}/overview?days=${days}`);
    if ($("days").value !== days) return;
    body.innerHTML = `<div class="work-metrics">
      ${metric("Korxona hisoblari", count(stats.members.total), "users", "green", `${count(stats.members.active)} ta faol`)}
      ${metric("Operatorlar", count(stats.members.operators), "user", "blue", `${count(stats.members.admins)} ta admin`)}
      ${metric("Ro‘yxatdagi do‘konlar", count(stats.stores), "store", "violet")}
      ${metric("Kelgan auditlar", count(stats.audits.total), "report", "amber", `Oxirgi ${days} kun`)}
      </div>${platformAuditPanels(stats)}
      <div class="card organization-activity-note">${icon("calendar")}<span>So‘nggi audit</span>
        <strong>${esc(rastaDate(stats.audits.last_submission))}</strong></div>
      <div id="organization-admins"></div>${platformEvents(stats.events, true)}`;
    await memberList(`/api/organizations/${org.id}/members`, "Korxona adminlari", "Admin", () => render(organizations),
      "org_admin", $("organization-admins"));
  }
  async function members() {
    heading("Xodimlar", `${session.user.organization} jamoasining foydalanuvchi hisoblari.`,
      link("/field", "Do‘kon biriktirish", false, "visit") + button("add-member", "Agent yaratish"));
    await memberList("/api/members", "Jamoa hisoblari", "Agent", () => render(members), "operator");
  }
  async function memberList(endpoint, title, label, refresh, editableRole, target = body) {
    const { users } = await rastaJson(endpoint);
    target.innerHTML = `<section class="card work-section"><div class="panel-heading"><h2>${title}</h2></div>${users.length
      ? table(["Login","Rol","Holat",""], users.map(user => `<tr><td><strong>${esc(user.username)}</strong></td>
        <td>${rastaRole(user.role)}</td><td>${tag(user.active ? "Faol" : "To‘xtatilgan", user.active ? "approved" : "returned")}</td>
        <td>${user.role === editableRole ? `<button class="btn" data-member="${user.id}">Tahrirlash</button>` : ""}</td></tr>`).join(""))
      : empty("Hisoblar hali yo‘q", "Yangi hisob yaratish uchun yuqoridagi tugmani bosing.", "users")}</section>`;
    $("add-member").onclick = () => openEditor(`${label} yaratish`, usernameField + passwordField(), async values => {
      await rastaJson(endpoint, "POST", values);
      await refresh();
    });
    target.querySelectorAll("[data-member]").forEach(button => button.onclick = () => {
      const user = users.find(item => item.id === Number(button.dataset.member));
      openEditor(user.username, activeField + passwordField(false), async values => {
        await rastaJson(`${endpoint}/${user.id}`, "PATCH", {active: Boolean(values.active), password: values.password || null});
        await refresh();
      }, user);
    });
  }
  async function stores(days = requestedDays()) {
    const id = Number(queries.get("id"));
    const stats = await rastaJson(`/api/field/overview?days=${days}${id ? "&store_id=" + id : ""}`);
    const rows = stats.stores;
    const fields = nameField("Do‘kon nomi") + field("Manzil", "address", 'maxlength="200"');
    if (id) {
      const store = rows[0];
      heading(store.name, store.address || "Do‘kon statistikasi", periodLabel
        + button("edit-store", "Tahrirlash", "store", false), [`/stores?days=${days}`, "Do‘konlar"]);
      bindPeriod(days, stores);
      const reports = new URLSearchParams({store_id: store.id, from: stats.from, to: stats.to});
      body.innerHTML = `<div class="work-metrics">
        ${metric("Tashriflar", count(store.visits), "visit", "green", `${count(store.finished)} ta yakunlangan`)}
        ${metric("Fotoauditlar", count(store.audits), "report", "blue", `${count(store.gaps)} ta bo‘sh joy`)}
        ${metric("Topshirilgan tovar", count(store.units), "box", "violet", "dona")}
        ${metric("Faol agentlar", count(store.agents), "users", "amber", `${count(store.tasks_done)} ta topshiriq bajarilgan`)}
        </div>
        <div class="dashboard-grid">
          <section class="card dashboard-panel"><div class="panel-heading"><h2>So‘nggi tashriflar</h2>${link(reportsUrl(reports), "Fotoauditlar")}</div>
            ${stats.recent_visits.length ? stats.recent_visits.map(visit => `<a class="state-line drill-link" href="/field?id=${visit.id}">
              <span><strong>${esc(visit.agent)}</strong><br>${esc(rastaDate(visit.started))}</span>
              ${tag(visit.finished ? "Yakunlangan" : "Ochiq", visit.finished ? "approved" : "submitted")}</a>`).join("")
              : empty("Tashriflar hali yo‘q", "Tanlangan davrda qayd etilgan tashriflar shu yerda ko‘rinadi.", "visit")}
          </section>
          <section class="card dashboard-panel"><div class="panel-heading"><h2>Biriktirilgan agentlar</h2>${link("/field", "Biriktirish")}</div>
            ${stats.agents.length ? stats.agents.map(agent => `<div class="state-line"><strong>${esc(agent.username)}</strong>
              ${tag(agent.active ? "Faol" : "To‘xtatilgan", agent.active ? "approved" : "returned")}</div>`).join("")
              : empty("Agent biriktirilmagan", "Tashriflar sahifasida agentga do‘kon biriktiring.", "users")}
          </section>
        </div>
        <section class="card work-section"><div class="panel-heading"><h2>Topshirilgan mahsulotlar</h2></div>
          ${stats.delivery_items.length ? table(["Mahsulot","SKU","Dona"], stats.delivery_items.map(item => `<tr>
            <td>${esc(item.product)}</td><td>${esc(item.sku_id)}</td><td>${count(item.units)}</td></tr>`).join(""))
          : empty("Topshirishlar qayd etilmagan", "Agent qayd qilgan mahsulotlar shu yerda ko‘rinadi.", "box")}
        </section>`;
      $("edit-store").onclick = () => openEditor("Do‘konni tahrirlash", fields, async values => {
        await rastaJson(`/api/stores/${store.id}`, "PATCH", values);
        await render(() => stores(days));
      }, store);
      return;
    }
    heading("Do‘konlar", "Tashrif, fotoaudit va tovar topshirish natijalari.", periodLabel + button("add-store", "Do‘kon qo‘shish"));
    bindPeriod(days, stores);
    body.innerHTML = rows.length ? `<div class="store-grid">${rows.map(store => `<a class="card store-stat-card" href="/stores?id=${store.id}&days=${days}">
      <span class="metric-icon">${icon("store")}</span><div><h2>${esc(store.name)}</h2><small>${esc(store.address || "Manzil kiritilmagan")}</small></div>
      ${icon("arrow-right")}<div class="store-stat-values"><span><strong>${count(store.visits)}</strong>Tashrif</span>
      <span><strong>${count(store.audits)}</strong>Audit</span><span><strong>${count(store.units)}</strong>Tovar (dona)</span></div>
      <small>${count(store.agents)} ta faol agent</small></a>`).join("")}</div>`
      : `<section class="card work-section">${empty("Do‘konlar hali yo‘q", "Jamoa audit yuborishi uchun birinchi do‘konni qo‘shing.", "store")}</section>`;
    $("add-store").onclick = () => openEditor("Yangi do‘kon", fields, async values => {
      await rastaJson("/api/stores", "POST", values);
      await render(() => stores(days));
    });
  }
  rastaSession.then(value => {
    session = value;
    if (!value.workspaces || !value.user) {
      if (value.enabled !== undefined && !value.workspaces) location.replace("/");
      else if (value.unavailable) showError(new Error("Ish maydoni yuklanmadi. Server bilan bog‘lanishni tekshiring."));
      return;
    }
    const page = ({"/platform": platform, "/dashboard": dashboard, "/reports": reports, "/organizations": organizations,
      "/members": members, "/stores": stores})[path];
    if (page) render(page);
  });
}
