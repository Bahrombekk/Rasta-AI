export default function Account() {
  return (
    <>
      <main className="page workspace-page">
          <div className="page-head"><div><h1>Mening hisobim</h1><p id="account-name"></p></div></div>
          <section className="card card-pad account-card">
            <h2>Parolni almashtirish</h2>
            <form id="password-form" method="post">
              <label className="field"><span>Joriy parol</span><input className="input" name="current" type="password" autoComplete="current-password" required maxLength="128" /></label>
              <label className="field"><span>Yangi parol</span><input className="input" name="password" type="password" autoComplete="new-password" required minLength="12" maxLength="128" /><small>Kamida 12 ta belgi.</small></label>
              <div className="err" id="account-error" role="alert"></div>
              <button type="submit" className="btn primary" disabled>Parolni almashtirish</button>
            </form>
          </section>
        </main>
    </>
  );
}
