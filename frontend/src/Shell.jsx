import { useRef, useState } from "react";
import { Icon, ThemeButton } from "./shared.jsx";

const links = {
  analysis: ["/", "scan", "Javon tahlili"],
  catalog: ["/admin", "box", "Katalog"],
  team: ["/team", "users", "Jamoa"],
  organizations: ["/organizations", "building", "Korxonalar"],
  platform: ["/platform", "chart", "Platforma statistikasi"],
  dashboard: ["/dashboard", "chart", "Bosh panel"],
  field: ["/field", "visit", "Tashriflar"],
  reports: ["/reports", "report", "Hisobotlar"],
  stores: ["/stores", "store", "Do‘konlar"],
  members: ["/members", "users", "Xodimlar"],
};
const workspaceMenus = {
  super_admin: ["platform", "organizations", "catalog"],
  org_admin: ["dashboard", "field", "analysis", "reports", "catalog", "stores", "members"],
  operator: ["field", "analysis", "reports"],
};
const companyPrimary = ["dashboard", "field", "stores"];

export function menuFor(session) {
  if (session.workspaces) return workspaceMenus[session.user?.role] || [];
  return session.user?.role === "super_admin" ? ["analysis", "catalog", "team"] : ["analysis", "catalog"];
}

export default function Shell({ page, session }) {
  const [loggingOut, setLoggingOut] = useState(false);
  const menuDialog = useRef(null);
  const { user, workspaces } = session;
  const role = user?.role;
  const menu = menuFor(session);
  const mobile = role === "org_admin" ? companyPrimary : menu;
  const extra = menu.filter(key => !mobile.includes(key));
  const organization = user?.organization || "Platforma boshqaruvi";
  const navLink = key => {
    const [url, icon, label] = links[key];
    return <a key={key} href={url} aria-current={key === page ? "page" : undefined}><Icon name={icon} /><span>{label}</span></a>;
  };
  async function logout() {
    setLoggingOut(true);
    try {
      await window.rastaJson("/api/session/logout", "POST");
      location.replace("/login");
    } catch {
      setLoggingOut(false);
      window.toast("Hisobdan chiqishda xato yuz berdi.");
    }
  }
  return <header data-shell data-role={role} className={workspaces ? "shell workspace-shell" : "shell"}>
    <a className="logo" href={workspaces ? window.rastaHome(user) : "/"}>
      <span className="mark"><Icon name="scan" /></span>
      <span>Rasta<span className="shell-brand-ai"> AI</span>{workspaces && <small className="mobile-organization">{organization}</small>}</span>
    </a>
    {workspaces && <div className="shell-context"><span className="shell-context-icon"><Icon name="building" /></span><span>{organization}</span></div>}
    <nav className="shell-links" aria-label="Asosiy menyu">{menu.map(navLink)}</nav>
    <div className="shell-account">
      {user && <a className="profile" href="/account" aria-label="Mening hisobim"><Icon name="user" />
        <span><strong>{user.username}</strong><small>{window.rastaRole(role)}</small></span></a>}
      <div className="shell-tools"><ThemeButton />
        {user && <button className="iconbtn" type="button" id="logout" aria-label="Hisobdan chiqish" title="Chiqish"
          disabled={loggingOut} onClick={logout}><Icon name="exit" /></button>}
      </div>
    </div>
    {workspaces && <nav className="mobile-nav" aria-label="Mobil menyu">
      {mobile.map(navLink)}
      {extra.length > 0 && <button type="button" id="mobile-more" aria-haspopup="dialog"
        className={extra.includes(page) ? "current" : undefined} onClick={() => menuDialog.current.showModal()}>
        <Icon name="menu" /><span>Menyu</span></button>}
    </nav>}
    {workspaces && extra.length > 0 && <dialog className="mobile-menu" aria-label="Qo‘shimcha sahifalar" ref={menuDialog}>
      <div className="panel-heading"><h2>Sahifalar</h2>
        <button type="button" className="iconbtn" aria-label="Yopish" onClick={() => menuDialog.current.close()}><Icon name="x" /></button></div>
      <nav aria-label="Qo‘shimcha sahifalar">{extra.map(navLink)}</nav>
    </dialog>}
  </header>;
}
