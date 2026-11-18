import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles/app.css";
import "./styles/workspace.css";
import "./styles/catalog.css";
import "./styles/login.css";
import "./controllers/common.js";
import Shell, { menuFor } from "./Shell.jsx";
import { useSession } from "./shared.jsx";
import Login from "./pages/Login.jsx";
import Analysis from "./pages/Analysis.jsx";
import Catalog from "./pages/Catalog.jsx";
import Workspace from "./pages/Workspace.jsx";
import Field from "./pages/Field.jsx";
import Account from "./pages/Account.jsx";
import Team from "./pages/Team.jsx";

const workspaceController = () => import("./controllers/workspace.js");
const routes = {
  "/": ["analysis", "Javon tahlili", Analysis, () => import("./controllers/analysis.js")],
  "/login": ["login", "Kirish", Login],
  "/admin": ["catalog", "Katalog", Catalog, () => import("./controllers/catalog.js")],
  "/account": ["account", "Mening hisobim", Account, () => import("./controllers/account.js")],
  "/team": ["team", "Jamoa", Team, () => import("./controllers/team.js")],
  "/field": ["field", "Tashriflar", Field, () => import("./controllers/field.js")],
  "/platform": ["platform", "Platforma statistikasi", Workspace, workspaceController],
  "/organizations": ["organizations", "Korxonalar", Workspace, workspaceController],
  "/dashboard": ["dashboard", "Bosh panel", Workspace, workspaceController],
  "/reports": ["reports", "Hisobotlar", Workspace, workspaceController],
  "/members": ["members", "Xodimlar", Workspace, workspaceController],
  "/stores": ["stores", "Do‘konlar", Workspace, workspaceController],
};
const path = location.pathname;
const route = Object.hasOwn(routes, path) ? routes[path] : null;

function PageContent({ component: Component, controller }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let active = true;
    controller().then(module => { if (active) module.default(); })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [controller]);
  return failed ? <main className="page"><p role="alert">Sahifani yuklab bo‘lmadi. Sahifani yangilang.</p></main> : <Component />;
}

function App() {
  const session = useSession();
  const [page, title, Component, controller] = route || [];
  let redirect = null;
  if (session && page && page !== "login") {
    if (session.enabled && !session.user) {
      redirect = "/login?next=" + encodeURIComponent(path + location.search);
    } else if ((session.workspaces && page !== "account" && !menuFor(session).includes(page))
      || (page === "team" && session.user?.role !== "super_admin")) {
      redirect = window.rastaHome(session.user);
    }
  }
  useEffect(() => { document.title = (title || "Sahifa topilmadi") + " · Rasta AI"; }, [title]);
  useEffect(() => { if (redirect) location.replace(redirect); }, [redirect]);
  if (!route) return <main className="page"><h1>Sahifa topilmadi</h1><a href="/">Bosh sahifaga qaytish</a></main>;
  if (page === "login") return <Login session={session} routes={routes} />;
  if (!session || redirect) return <main className="page" aria-busy="true"><p role="status">Yuklanmoqda…</p></main>;
  return <><Shell page={page} session={session} /><PageContent component={Component} controller={controller} /></>;
}

createRoot(document.getElementById("root")).render(<App />);
