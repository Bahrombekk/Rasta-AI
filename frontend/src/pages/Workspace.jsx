import { Icon } from "../shared.jsx";

export default function Workspace() {
  return (
    <>
      <main className="page workspace-page" id="workspace">
          <a className="back-link" id="work-back" hidden></a>
          <div className="work-heading">
            <div><h1 id="work-title">Ish maydoni</h1><p id="work-subtitle"></p></div>
            <div className="page-actions" id="work-actions"></div>
          </div>
          <div id="work-error" className="work-error" role="alert" hidden></div>
          <div id="work-body" aria-busy="true"><p className="work-loading" role="status">Yuklanmoqda…</p></div>
          <dialog className="edit-dialog" id="editor">
            <form id="editor-form">
              <div className="dialog-head"><h2 id="editor-title"></h2><button className="iconbtn" type="button" id="editor-close" aria-label="Yopish"><Icon name="x" /></button></div>
              <div className="dialog-body"><div id="editor-fields"></div><div id="editor-error" className="err" role="alert"></div></div>
              <div className="dialog-actions"><button className="btn primary" type="submit">Saqlash</button></div>
            </form>
          </dialog>
        </main>
    </>
  );
}
