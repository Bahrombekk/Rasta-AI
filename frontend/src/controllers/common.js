// Umumiy: mavzu (och/qorong'i), ikonlar, kichik yordamchilar
(() => {
  const KEY = "rasta-theme";
  const get = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const set = v => { try { localStorage.setItem(KEY, v); } catch (e) {} };
  document.documentElement.dataset.theme = get() === "dark" ? "dark" : "light";

  const P = {
    logo: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 4v16"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
    poly: '<path d="M5 6 18 4l2 12-11 4z"/><circle cx="5" cy="6" r="1.5"/><circle cx="18" cy="4" r="1.5"/><circle cx="20" cy="16" r="1.5"/><circle cx="9" cy="20" r="1.5"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>',
    box: '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5M12 13v8"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    layers: '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    empty: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 12h18"/><path d="M8 8h3M8 16h3" stroke-dasharray="1 2"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6"/>',
    users: '<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a5 5 0 0 1 5-5h2a5 5 0 0 1 5 5v2m1-15a3 3 0 0 1 0 6m2 2a5 5 0 0 1 3 5v2"/>',
    user: '<circle cx="12" cy="8" r="3"/><path d="M5 21v-2a5 5 0 0 1 5-5h4a5 5 0 0 1 5 5v2"/>',
    exit: '<path d="M11 4H5v16h6m3-4 4-4-4-4m-4 4h8"/>',
    scan: '<path d="M8 4H4v4m12-4h4v4M4 16v4h4m12-4v4h-4M7 12h10"/>',
    "arrow-right": '<path d="M4 12h15m-6-6 6 6-6 6"/>',
    expand: '<path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/>',
    shrink: '<path d="M20 10h-6V4M4 14h6v6M14 10l7-7M3 21l7-7"/>',
    "arrow-left": '<path d="M20 12H5m6-6-6 6 6 6"/>',
    "eye-off": '<path d="M3 3l18 18M10.6 10.6a2 2 0 0 0 2.8 2.8M9.6 5.4a9.1 9.1 0 0 1 2.4-.4c5.5 0 9 7 9 7a17 17 0 0 1-2.5 3.4M6.2 6.7A17 17 0 0 0 3 12s3.5 7 9 7a8 8 0 0 0 4.1-1.2"/>',
    chart: '<path d="M4 4v16h16M8 16v-3m4 3V9m4 7V6"/>',
    report: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6m-6 4h6m-6 4h4"/>',
    download: '<path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5"/>',
    upload: '<path d="M12 15V3m-5 5 5-5 5 5M4 15v5h16v-5"/>',
    building: '<path d="M4 21V4h12v17m0-11h4v11M2 21h20M8 8h4m-4 4h4m-4 4h4"/>',
    store: '<path d="M4 10v10h16V10M3 10l2-6h14l2 6M3 10c0 3 4 3 4 0 0 3 5 3 5 0 0 3 5 3 5 0 0 3 4 3 4 0M10 20v-5h4v5"/>',
    camera: '<path d="M4 7h4l2-3h4l2 3h4a1 1 0 0 1 1 1v11H3V8a1 1 0 0 1 1-1Z"/><circle cx="12" cy="13" r="3"/>',
    calendar: '<rect x="4" y="6" width="16" height="14" rx="2"/><path d="M8 4v4m8-4v4M4 10h16"/>',
    visit: '<path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  };
  window.rastaIconPath = name => P[name] || "";
  window.icon = (n, cls = "i") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${window.rastaIconPath(n)}</svg>`;
  window.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // Ichki sahifadan ro'yxatga qaytish havolasi. Foydalanuvchi aynan shu ro'yxatdan kelgan bo'lsa,
  // brauzer tarixi bo'yicha qaytadi — filtr, sahifa raqami va aylantirish joyi saqlanadi.
  window.rastaBack = (id, href = "", label = "") => {
    const anchor = document.getElementById(id);
    if (!anchor) return;
    anchor.hidden = !href;
    if (!href) return;
    anchor.href = href;
    anchor.innerHTML = window.icon("arrow-left") + window.esc(label);
    anchor.onclick = event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.button) return;
      let from = null;
      try { from = new URL(document.referrer); } catch (e) {}
      if (from && from.origin === location.origin && from.pathname === new URL(href, location.href).pathname
          && history.length > 1) {
        event.preventDefault();
        history.back();
      }
    };
  };
  window.toast = msg => {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
    document.body.append(t); setTimeout(() => t.remove(), 2400);
  };

  // Skaner ko'rinishi uchun javon (mahsulotlar + aniqlash qutilari) yasaydi.
  const RL_PALETTE = ["#d9a84e", "#64b49d", "#c67088", "#8aa6d6", "#9bc46a", "#d98b5a", "#7fc4c0"];
  const rlShelf = () => {
    let rows = "";
    const R = 3;
    for (let r = 0; r < R; r++) {
      let cells = "";
      const n = 4 + Math.floor(Math.random() * 3);
      for (let c = 0; c < n; c++) {
        const bottle = Math.random() < 0.45;
        const col = RL_PALETTE[(r * 3 + c) % RL_PALETTE.length];
        const delay = ((r / R) * 1.7 + c * 0.05).toFixed(2);
        const w = (bottle ? 11 : 15) + Math.floor(Math.random() * 5);
        const h = (bottle ? 66 : 56) + Math.floor(Math.random() * 28);
        cells += `<span class="rl-pr${bottle ? " bottle" : ""}" style="width:${w}px;height:${h}%;`
          + `background:linear-gradient(160deg,${col},color-mix(in srgb,${col} 70%,#000))">`
          + `<span class="rl-det" style="animation-delay:${delay}s"></span>`
          + `<span class="rl-tick" style="animation-delay:${delay}s">✓</span></span>`;
      }
      rows += `<span class="rl-row">${cells}</span>`;
    }
    return `<span class="rl-grid"></span><span class="rl-shelf">${rows}</span><span class="rl-beam"></span>`;
  };
  const rlPack = () =>
    '<span class="rl-ring"></span><span class="rl-ring"></span><span class="rl-ring"></span>'
    + '<span class="rl-pack"><span class="rl-shine"></span>'
    + '<span class="rl-node" style="top:22%;left:8%;animation-delay:.2s"></span>'
    + '<span class="rl-node" style="top:50%;left:84%;animation-delay:.6s"></span>'
    + '<span class="rl-node" style="top:72%;left:14%;animation-delay:1s"></span>'
    + '<span class="rl-node" style="top:30%;left:78%;animation-delay:1.4s"></span></span>';

  // Bosqichли loading overlay. target — ustiga qo'yiladigan element (position:relative
  // bo'ladi). stages — matnlar ro'yxati; ular navbatma-navbat ko'rsatiladi (oxirgida
  // ushlanib turadi). opts.scan — javon skanerlash ko'rinishi (tahlil uchun), aks holda
  // katalog (mahsulot) ko'rinishi. Qaytaradi: { stop() }.
  window.rastaLoader = (target, stages, opts = {}) => {
    if (!target) return { stop() {} };
    const restore = getComputedStyle(target).position === "static";
    if (restore) target.style.position = "relative";
    const live = Boolean(opts.live);             // skaner haqiqiy rasm ustida yuradi, holat pastdagi panelda
    const ov = document.createElement("div");
    ov.className = "rasta-loader" + (opts.scan ? " scan" : "") + (live ? " live" : "")
      + (live && target.clientHeight < 430 ? " compact" : "");
    ov.setAttribute("role", "status");
    const status = '<div class="rl-status"><span class="rl-dot"></span><span class="rl-text"></span><span class="rl-time"></span></div>'
      + '<div class="rl-bar"><span class="rl-fill"></span></div>';
    if (live) {
      let probes = "";
      for (let k = 0; k < 16; k++) {
        const top = 6 + Math.random() * 86, left = 5 + Math.random() * 88;
        probes += `<span class="rl-probe" style="top:${top.toFixed(1)}%;left:${left.toFixed(1)}%;animation-delay:${(top / 100 * 1.95).toFixed(2)}s"></span>`;
      }
      ov.innerHTML = '<div class="rl-scan"><span class="rl-grid"></span>' + probes + '<span class="rl-beam"></span>'
        + '<span class="rl-corner tl"></span><span class="rl-corner tr"></span><span class="rl-corner bl"></span><span class="rl-corner br"></span></div>'
        + `<div class="rl-hud">${status}<ol class="rl-list">${stages.map(s => `<li class="rl-step"><i></i><span>${window.esc(s)}</span></li>`).join("")}</ol></div>`;
    } else {
      ov.innerHTML = `<div class="rl-visual">${opts.scan ? rlShelf() : rlPack()}</div>` + status
        + `<div class="rl-steps">${stages.map(s => `<span class="rl-step">${window.esc(s)}</span>`).join("")}</div>`;
    }
    target.appendChild(ov);
    // Skaner qatlami rasmning o'ziga (kanvasga) moslanadi: yon tomondagi bo'sh maydonda nur yurmaydi
    const frame = () => {
      const scan = ov.querySelector(".rl-scan");
      if (!scan || !opts.frame) return;
      const outer = target.getBoundingClientRect(), inner = opts.frame.getBoundingClientRect();
      Object.assign(scan.style, { left: inner.left - outer.left + "px", top: inner.top - outer.top + "px",
        width: inner.width + "px", height: inner.height + "px" });
    };
    frame();
    window.addEventListener("resize", frame);
    const text = ov.querySelector(".rl-text"), fill = ov.querySelector(".rl-fill"), clock = ov.querySelector(".rl-time");
    const chips = [...ov.querySelectorAll(".rl-step")];
    const started = performance.now();
    const mark = (i, width) => {
      text.textContent = stages[i] + "…";
      chips.forEach((c, k) => { c.classList.toggle("done", k < i); c.classList.toggle("now", k === i); });
      fill.style.width = width + "%";
    };
    let timer;
    if (opts.expected) {
      // Kutilgan vaqtga mos sur'at: bosqichlar o'z ulushiga qarab almashadi, chiziq 100% ga yetmay kutadi
      const weights = opts.weights || stages.map(() => 1), whole = weights.reduce((a, b) => a + b, 0);
      const edges = []; weights.reduce((sum, w) => { edges.push((sum + w) / whole); return sum + w; }, 0);
      const tick = () => {
        const elapsed = performance.now() - started;
        const done = 1 - Math.exp(-2.2 * elapsed / opts.expected);
        const i = Math.min(stages.length - 1, edges.findIndex(edge => done < edge * 0.92) >>> 0);
        mark(i, Math.max(4, Math.round(done * 94)));
        clock.textContent = `${Math.floor(elapsed / 1000)} s`;
      };
      tick();
      timer = setInterval(tick, 200);
    } else {
      let i = 0;
      const tick = () => { mark(i, Math.round((i + 1) / stages.length * 100)); i = Math.min(i + 1, stages.length - 1); };
      tick();
      timer = setInterval(tick, opts.interval || 800);
    }
    return {
      stop() {
        clearInterval(timer);
        window.removeEventListener("resize", frame);
        if (live) { chips.forEach(c => { c.classList.add("done"); c.classList.remove("now"); }); fill.style.width = "100%"; }
        ov.classList.add("rl-out");
        setTimeout(() => { ov.remove(); if (restore) target.style.position = ""; }, live ? 260 : 180);
        return performance.now() - started;
      }
    };
  };

  const loginUrl = () => "/login?next=" + encodeURIComponent(location.pathname + location.search);
  window.rastaSession = fetch("/api/session", { cache: "no-store" })
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .catch(() => ({ unavailable: true, user: null }));
  window.rastaHome = user => ({super_admin: "/platform", org_admin: "/dashboard", operator: "/field"})[user?.role] || "/";
  window.rastaRole = role => ({super_admin: "Super admin", org_admin: "Korxona admini", operator: "Agent"})[role] || "Admin";
  const dateFormat = new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
    hourCycle: "h23", timeZone: "Asia/Tashkent"
  });
  window.rastaDate = seconds => {
    if (!seconds) return "—";
    const { day, month, year, hour, minute } = Object.fromEntries(
      dateFormat.formatToParts(new Date(seconds * 1000)).map(part => [part.type, part.value]));
    return `${day}.${month}.${year}, ${hour}:${minute}`;
  };
  window.rastaFetch = async (url, options = {}) => {
    const headers = new Headers(options.headers);
    if (options.method && options.method.toUpperCase() !== "GET") headers.set("X-Rasta-Request", "1");
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401 && url !== "/api/session/login" && (await rastaSession).enabled) {
      location.replace(loginUrl());
    }
    return response;
  };
  window.rastaJson = async (url, method = "GET", body) => {
    const options = { method };
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    let response;
    try { response = await rastaFetch(url, options); }
    catch { throw new Error("Server bilan bog‘lanib bo‘lmadi."); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "So‘rov bajarilmadi.");
    return data;
  };


  window.rastaSetTheme = theme => {
    document.documentElement.dataset.theme = theme;
    set(theme);
    document.dispatchEvent(new Event("themechange"));
  };
})();
