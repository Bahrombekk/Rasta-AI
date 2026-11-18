import { Icon } from "../shared.jsx";

export default function Team() {
  return (
    <>
      <main className="page">
          <div className="page-head">
            <div><h1>Jamoa</h1><p>Admin hisoblarini yarating va boshqaring.</p></div>
            <button className="btn primary" id="add-admin" type="button" disabled><Icon name="plus" />Admin qo‘shish</button>
          </div>
          <section className="card team-card" aria-label="Jamoa hisoblari">
            <div id="team-list" className="team-list"></div>
            <div className="empty" id="team-status" role="status">Hisoblar yuklanmoqda…</div>
          </section>
          <dialog id="admin-dialog" className="edit-dialog">
            <form id="admin-form">
              <div className="dialog-head">
                <h2 id="dialog-title">Yangi admin</h2>
                <button className="iconbtn" id="dialog-close" type="button" aria-label="Yopish"><Icon name="x" /></button>
              </div>
              <div className="dialog-body">
                <label className="field"><span>Login</span><input className="input" name="username" required minLength="3" maxLength="48" autoComplete="off" /></label>
                <label className="field"><span id="password-label">Parol</span>
                  <input className="input" name="password" type="password" minLength="12" maxLength="128" autoComplete="new-password" />
                  <small id="password-hint">Kamida 12 ta belgi.</small>
                </label>
                <label className="field check-field" id="active-field" hidden><input type="checkbox" name="active" /><span>Hisob faol</span></label>
                <div className="err" id="dialog-error" role="alert"></div>
              </div>
              <div className="dialog-actions"><button className="btn primary" type="submit">Saqlash</button></div>
            </form>
          </dialog>
        </main>
    </>
  );
}
