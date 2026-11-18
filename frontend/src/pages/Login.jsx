import { useEffect, useState } from "react";
import { Icon, ThemeButton } from "../shared.jsx";

function target(session, routes) {
  const next = new URLSearchParams(location.search).get("next");
  if (next?.startsWith("/")) {
    try {
      const url = new URL(next, location.origin);
      if (url.origin === location.origin && Object.hasOwn(routes, url.pathname) && url.pathname !== "/login") {
        return url.pathname + url.search;
      }
    } catch { /* Use the role's entry page. */ }
  }
  return session.workspaces ? window.rastaHome(session.user) : "/";
}

export default function Login({ session, routes }) {
  const [reveal, setReveal] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    document.body.classList.add("login-page");
    return () => document.body.classList.remove("login-page");
  }, []);
  useEffect(() => {
    if (session?.user || session?.enabled === false) location.replace(target(session, routes));
  }, [session, routes]);
  async function submit(e) {
    e.preventDefault();
    setPending(true);
    setError("");
    try {
      const response = await window.rastaJson("/api/session/login", "POST", Object.fromEntries(new FormData(e.currentTarget)));
      location.replace(target(response, routes));
    } catch (failure) {
      setError(failure.message);
      setPending(false);
    }
  }
  const message = error || (session?.unavailable ? "Serverga ulanib bo‘lmadi. Keyinroq urinib ko‘ring." : "");
  return <main className="login-layout">
    <section className="login-main">
      <div className="login-brand">
        <a className="logo" href="/"><span className="mark"><Icon name="scan" /></span><span>Rasta<span className="brand-ai"> AI</span></span></a>
        <ThemeButton />
      </div>
      <div className="login-form" aria-labelledby="login-title">
        <h1 id="login-title">Hisobga kirish</h1>
        <p>Jamoangiz bergan login va parolni kiriting.</p>
        <form id="login-form" method="post" onSubmit={submit}>
          <div className="login-field">
            <label htmlFor="login-name">Login</label>
            <input className="input" id="login-name" name="username" autoComplete="username" autoCapitalize="none" spellCheck={false}
              placeholder="Loginingiz" required maxLength={64} aria-describedby="error" />
          </div>
          <div className="login-field">
            <label htmlFor="login-password">Parol</label>
            <div className="password-input">
              <input className="input" id="login-password" name="password" type={reveal ? "text" : "password"}
                autoComplete="current-password" placeholder="Parolingiz" required maxLength={128} aria-describedby="error" />
              <button className="iconbtn" id="toggle-password" type="button"
                aria-label={reveal ? "Parolni yashirish" : "Parolni ko‘rsatish"} aria-pressed={reveal} onClick={() => setReveal(value => !value)}>
                <Icon name={reveal ? "eye-off" : "eye"} />
              </button>
            </div>
          </div>
          <div className="login-error" id="error" role="alert" hidden={!message}>{message}</div>
          <button className="btn primary lg block login-submit" type="submit" disabled={pending || !session}>
            <span id="submit-text">{pending ? "Kirilmoqda…" : "Kirish"}</span><span className="submit-icon"><Icon name="arrow-right" /></span>
          </button>
        </form>
      </div>
    </section>
    <aside className="login-visual" aria-label="Rasta AI haqida">
      <img className="login-photo" src="/static/login-shelf.webp" alt="" fetchPriority="high" />
      <div className="login-caption">
        <p className="login-visual-title">Javondagi holatni<br /> bir qarashda ko‘ring.</p>
        <p>Mahsulotlar, javondagi ulush va bo‘sh joylar tahlili.</p>
      </div>
    </aside>
  </main>;
}
