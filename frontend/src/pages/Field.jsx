import { Icon } from "../shared.jsx";

export default function Field() {
  return (
    <>
      <main className="page workspace-page" data-view="field" id="field">
          <div className="work-heading"><div><h1 id="field-title">Tashriflar</h1><p id="field-subtitle"></p></div>
            <div className="page-actions" id="field-actions"></div></div>
          <div className="work-error" id="field-error" role="alert" hidden></div>
          <div id="field-body" aria-busy="true"><p className="work-loading" role="status">Yuklanmoqda…</p></div>
          <dialog className="edit-dialog" id="field-dialog" aria-labelledby="field-dialog-title">
            <form id="field-form">
              <div className="dialog-head"><h2 id="field-dialog-title"></h2><button className="iconbtn" type="button" aria-label="Yopish" id="field-close"><Icon name="x" /></button></div>
              <div className="dialog-body"><div id="field-inputs"></div><div className="err" id="field-dialog-error" role="alert"></div></div>
              <div className="dialog-actions"><button className="btn primary" type="submit" id="field-save">Saqlash</button></div>
            </form>
          </dialog>
        </main>
    </>
  );
}
