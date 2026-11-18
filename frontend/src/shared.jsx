import { useEffect, useState } from "react";

export function Icon({ name }) {
  return <svg className="i" viewBox="0 0 24 24" aria-hidden="true"
    dangerouslySetInnerHTML={{ __html: window.rastaIconPath(name) }} />;
}

export function ThemeButton() {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme);
  return <button className="iconbtn" type="button" id="theme" aria-label="Rang mavzusini almashtirish"
    onClick={() => {
      const next = theme === "dark" ? "light" : "dark";
      window.rastaSetTheme(next);
      setTheme(next);
    }}><Icon name={theme === "dark" ? "sun" : "moon"} /></button>;
}

export function useSession() {
  const [session, setSession] = useState(null);
  useEffect(() => {
    let mounted = true;
    window.rastaSession.then(value => { if (mounted) setSession(value); });
    return () => { mounted = false; };
  }, []);
  return session;
}
