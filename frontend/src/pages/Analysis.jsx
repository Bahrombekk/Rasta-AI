import { Icon } from "../shared.jsx";

export default function Analysis() {
  return (
    <>
      <main className="page workspace-page" data-view="analysis">
        <div className="work-heading">
          <div><h1>Javon tahlili</h1><p>Javon rasmini yuklang, natijani tekshiring va hisobotni yuboring.</p></div>
          <div className="page-actions"><a className="btn" href="/field" id="visitLink" hidden><Icon name="visit" />Tashrif</a>
            <a className="btn" href="/reports" id="historyLink" hidden><Icon name="report" />Hisobotlar</a></div>
        </div>
        <section className="card card-pad" id="visitPrompt" hidden>
          <h2 className="section-title">Do‘konda tashrifni boshlang</h2>
          <p id="visitPromptText">Biriktirilgan do‘konni tanlagach, shu tashrifga fotoaudit qo‘sha olasiz.</p>
          <a href="/field" className="btn primary"><Icon name="visit" />Tashriflar</a>
        </section>
        <ol className="analysis-flow" aria-label="Javon tahlili bosqichlari">
          <li aria-current="step" data-phase="photo"><span className="flow-icon"><Icon name="image" /></span>Rasm tanlash</li>
          <li data-phase="result"><span className="flow-icon"><Icon name="scan" /></span>Natijani ko‘rish</li>
          <li data-phase="submit" id="submitPhase" hidden><span className="flow-icon"><Icon name="report" /></span>Auditni yuborish</li>
        </ol>
        <section className="analysis-entry" id="start">
          <div className="card upload-panel">
            <div className="panel-heading"><span className="metric-icon"><Icon name="upload" /></span><div><h2>Javon rasmi</h2><p>Javon yoki muzlatgichning bitta suratini tanlang.</p></div></div>
            <div className="drop" id="drop">
              <div className="ico"><Icon name="upload" /></div>
              <strong className="drop-title"><span className="drop-desktop">Rasmni shu yerga tashlang</span><span className="drop-touch">Rasmni yuklang</span></strong>
              <p>JPG, PNG yoki WebP · 20 MB gacha</p>
              <div className="upload-actions">
                <button className="btn primary lg" id="choose" type="button"><Icon name="image" />Fayl tanlash</button>
                <button className="btn lg camera-action" id="cameraBtn" type="button"><Icon name="camera" />Suratga olish</button>
              </div>
            </div>
          </div>
          <aside className="card capture-guide">
            <img src="/static/login-shelf.webp" alt="" width="300" height="150" />
            <div><h2>Suratga olish bo‘yicha tavsiyalar</h2><ul><li>Javonni old tomondan suratga oling.</li><li>Yorug‘lik aksini kamaytiring.</li><li>Bir nechta javon bo‘lsa, keraklisini zona bilan belgilang.</li></ul></div>
          </aside>
        </section>
        <div className="err" id="fileErr" role="alert" hidden></div>
        <input id="file" type="file" accept="image/jpeg,image/png,image/webp" />
        <input id="cameraFile" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" />


        <div className="work" id="work" hidden>
          <section className="card viewer">
            <div className="viewer-heading">
              <span className="metric-icon"><Icon name="image" /></span>
              <div><strong id="photoName"></strong><small id="photoInfo"></small></div>
            </div>
            <div className="toolbar">
              <button className="btn" id="zoneBtn"><Icon name="poly" />Zona belgilash</button>
              <button className="btn ghost" id="zoneClear" hidden><Icon name="x" />Zonani olib tashlash</button>
              <span className="hint" id="zoneHint"></span>
              <span className="sp"></span>
              <button className="btn ghost on" id="boxesBtn" hidden><Icon name="eye" />Belgilar</button>
              <button className="btn ghost" id="newBtn"><Icon name="refresh" />Boshqa rasm</button>
            </div>
            <div className="stage"><canvas id="cv" aria-label="Javon rasmi va tahlil belgilari"></canvas></div>
            <ul className="gap-legend" id="gapLegend" aria-label="Bo‘sh joy belgilari" hidden></ul>
          </section>

          <aside className="side">

            <section className="card card-pad" id="pre">
              <h2 className="section-title">Rasm tayyor</h2>
              <p className="analysis-help">Rasmda bir nechta javon bo‘lsa, keraklisini <strong>Zona belgilash</strong> orqali tanlang.</p>
              <button className="btn primary lg block" id="go">Tahlil qilish</button>
              <div className="err" id="err" role="alert"></div>
            </section>


            <div id="res" hidden style={{"display": "contents"}}>
              <section className="card card-pad result-overview">
                <h2 className="section-title">Tahlil natijasi</h2>
                <div className="kpis" id="kpis"></div>
              </section>
              <section className="card card-pad" id="sendPanel" hidden>
                <h2 className="section-title">Auditni yuborish</h2>
                <form id="sendForm">
                  <label className="field"><span>Do‘kon</span><select className="input" name="store" required></select></label>
                  <label className="field"><span>Izoh (ixtiyoriy)</span><textarea className="input" name="note" rows="2" maxLength="1000"></textarea></label>
                  <button className="btn primary block" id="sendBtn" type="submit">Hisobotni yuborish</button>
                  <div className="err" id="sendError" role="alert"></div>
                </form>
                <p id="sentMessage" hidden role="status">Hisobot yuborildi. <a href="/reports">Hisobotlarni ko‘rish</a></p>
              </section>
              <section className="card card-pad" id="shareCard">
                <p className="section-title">Javondagi ulush</p>
                <div className="share">
                  <svg className="donut" id="donut" viewbox="0 0 120 120"></svg>
                  <div className="plist" id="legend"></div>
                </div>
              </section>
              <section className="card card-pad">
                <p className="section-title" id="plTitle">Mahsulotlar</p>
                <div className="plist" id="plist"></div>
              </section>
              <section className="card card-pad" id="teach" hidden>
                <p className="section-title" id="teachTitle">Tanlangan</p>
                <label className="field"><span>Bu qaysi mahsulot?</span>
                  <select className="input" id="teachSku"></select></label>
                <div id="teachNew" hidden>
                  <label className="field"><span>Nomi</span><input className="input" id="teachName" placeholder="Masalan: Черноголовка Мохито 1.5L" /></label>
                  <label className="field"><span>Brend</span><input className="input" id="teachBrand" placeholder="Masalan: Черноголовка" /></label>
                </div>
                <div style={{"display": "flex", "gap": "8px"}}>
                  <button className="btn ghost" id="teachCancel">Bekor qilish</button>
                  <button className="btn primary" id="teachGo" style={{"flex": "1"}}>Katalogga qo'shish</button>
                </div>
                <div className="err" id="teachErr"></div>
              </section>
              <div id="notes" style={{"display": "flex", "flexDirection": "column", "gap": "10px"}}></div>
              <div className="result-actions">
                <button className="btn" id="exportJson">JSON yuklash</button>
                <button className="btn" id="exportImage">Rasmni saqlash</button>
                <button className="btn ghost" id="again"><Icon name="refresh" />Qayta tahlil qilish</button>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </>
  );
}
