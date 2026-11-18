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
    "eye-off": '<path d="M3 3l18 18M10.6 10.6a2 2 0 0 0 2.8 2.8M9.6 5.4a9.1 9.1 0 0 1 2.4-.4c5.5 0 9 7 9 7a17 17 0 0 1-2.5 3.4M6.2 6.7A17 17 0 0 0 3 12s3.5 7 9 7a8 8 0 0 0 4.1-1.2"/>',
    chart: '<path d="M4 4v16h16M8 16v-3m4 3V9m4 7V6"/>',
    report: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6m-6 4h6m-6 4h4"/>',
    building: '<path d="M4 21V4h12v17m0-11h4v11M2 21h20M8 8h4m-4 4h4m-4 4h4"/>',
    store: '<path d="M4 10v10h16V10M3 10l2-6h14l2 6M3 10c0 3 4 3 4 0 0 3 5 3 5 0 0 3 5 3 5 0 0 3 4 3 4 0M10 20v-5h4v5"/>',
  };
  window.icon = (n, cls = "i") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[n] || ""}</svg>`;
  window.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  window.toast = msg => {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
    document.body.append(t); setTimeout(() => t.remove(), 2400);
  };

  const loginUrl = () => "/login?next=" + encodeURIComponent(location.pathname);
  window.rastaSession = fetch("/api/session", { cache: "no-store" })
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .catch(() => ({ unavailable: true, user: null }));
  window.rastaHome = user => ({super_admin: "/organizations", org_admin: "/dashboard"})[user?.role] || "/";
  window.rastaRole = role => ({super_admin: "Super admin", org_admin: "Korxona admini", operator: "Operator"})[role] || "Admin";
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

  function shell(el) {
    const page = el.dataset.page || (location.pathname.slice(1) || "analysis");
    const links = [
      ["analysis", "/", "scan", "Javon tahlili"],
      ["catalog", "/admin", "box", "Katalog"],
      ["team", "/team", "users", "Jamoa"],
      ["organizations", "/organizations", "building", "Korxonalar"],
      ["dashboard", "/dashboard", "chart", "Bosh panel"],
      ["reports", "/reports", "report", "Hisobotlar"],
      ["stores", "/stores", "store", "Do‘konlar"],
      ["members", "/members", "users", "Xodimlar"],
    ];
    el.className = "shell";
    el.innerHTML = `
      <a class="logo" href="/"><span class="mark">${icon("scan")}</span><span>Rasta AI</span></a>
      <div class="shell-context" hidden><span class="shell-context-icon">${icon("building")}</span><span></span></div>
      <nav class="shell-links" aria-label="Asosiy menyu"></nav>
      <div class="shell-account">
        <a class="profile" href="/account" aria-label="Mening hisobim" hidden>${icon("user")}<span><strong></strong><small></small></span></a>
        <div class="shell-tools">
          <button class="iconbtn" type="button" id="theme" aria-label="Rang mavzusini almashtirish"></button>
          <button class="iconbtn" type="button" id="logout" aria-label="Hisobdan chiqish" title="Chiqish" hidden>${icon("exit")}</button>
        </div>
      </div>`;
    const profile = el.querySelector(".profile"), logout = el.querySelector("#logout");
    logout.onclick = async () => {
      logout.disabled = true;
      try {
        const response = await rastaFetch("/api/session/logout", { method: "POST" });
        if (!response.ok) throw new Error();
        location.replace("/login");
      } catch {
        logout.disabled = false;
        toast("Hisobdan chiqishda xato yuz berdi.");
      }
    };
    rastaSession.then(session => {
      if (session.enabled && !session.user) { location.replace(loginUrl()); return; }
      const role = session.user?.role;
      const menu = session.workspaces
        ? ({super_admin: ["organizations"], org_admin: ["dashboard","analysis","reports","catalog","stores","members"],
            operator: ["analysis","reports"]})[role] || []
        : ["analysis", "catalog", ...(role === "super_admin" ? ["team"] : [])];
      if (session.workspaces && page !== "account" && !menu.includes(page)) {
        location.replace(rastaHome(session.user));
        return;
      }
      if (!session.workspaces && page === "team" && role !== "super_admin") { location.replace("/"); return; }
      const nav = el.querySelector(".shell-links");
      nav.innerHTML = menu.map(key => links.find(link => link[0] === key)).map(([key, href, glyph, label]) =>
        `<a href="${href}" ${key === page ? 'aria-current="page"' : ''}>${icon(glyph)}<span>${label}</span></a>`).join("");
      if (session.workspaces) {
        el.classList.add("workspace-shell");
        el.dataset.role = role;
        el.querySelector(".logo").href = rastaHome(session.user);
        const context = el.querySelector(".shell-context");
        context.querySelector("span:last-child").textContent = session.user.organization || "Platforma boshqaruvi";
        context.hidden = false;
      }
      if (!session.user) return;
      profile.querySelector("strong").textContent = session.user.username;
      profile.querySelector("small").textContent = rastaRole(role);
      profile.hidden = false;
      logout.hidden = false;
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const header = document.querySelector("[data-shell]");
    if (header) shell(header);
    document.querySelectorAll("[data-icon]").forEach(el => { el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon)); });
    const btn = document.getElementById("theme");
    if (!btn) return;
    const paint = () => { btn.innerHTML = icon(document.documentElement.dataset.theme === "dark" ? "sun" : "moon"); };
    paint();
    btn.onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next; set(next); paint();
      document.dispatchEvent(new Event("themechange"));
    };
  });
})();
