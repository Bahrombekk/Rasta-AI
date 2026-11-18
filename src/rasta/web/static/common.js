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
  };
  window.icon = (n, cls = "i") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[n] || ""}</svg>`;
  window.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  window.toast = msg => {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg;
    document.body.append(t); setTimeout(() => t.remove(), 2400);
  };

  document.addEventListener("DOMContentLoaded", () => {
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
