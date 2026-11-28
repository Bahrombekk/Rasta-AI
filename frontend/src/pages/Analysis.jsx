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
          <section className="card result-bar" id="resultBar" hidden>
            <div className="result-head">
              <div><h2>Tahlil natijasi</h2><p id="resultMeta"></p></div>
              <div className="result-head-actions">
                <button className="btn ghost" id="again" type="button"><Icon name="refresh" />Qayta tahlil</button>
                <button className="btn primary" id="sendJump" type="button" hidden><Icon name="report" />Auditni yuborish</button>
              </div>
            </div>
            <div className="kpis" id="kpis"></div>
          </section>
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
              <button className="btn ghost" id="wideBtn" type="button" aria-pressed="false" hidden><Icon name="expand" /><span>Kattalashtirish</span></button>
              <button className="btn ghost" id="newBtn"><Icon name="refresh" />Boshqa rasm</button>
            </div>
            <div className="stage"><canvas id="cv" aria-label="Javon rasmi va tahlil belgilari"></canvas></div>
            <ul className="gap-legend" id="gapLegend" aria-label="Bo‘sh joy belgilari" hidden></ul>
          </section>

          <aside className="side">

            <section className="card card-pad scope-card" id="scopeCard" hidden>
              <div className="scope-head">
                <div><p className="section-title">Qidiruv doirasi</p><p id="scopeSummary"></p></div>
                <button className="btn" id="scopeToggle" type="button" aria-expanded="false" aria-controls="scopeBody">Tanlash</button>
              </div>
              <div id="scopeBody" hidden>
                <p className="analysis-help">Javonda aynan qaysi mahsulotlar borligini bilsangiz, faqat shularni belgilang: tizim ular orasidan izlaydi va boshqa mahsulotlar bilan adashmaydi.</p>
                <input className="input" id="scopeSearch" type="search" placeholder="Mahsulot yoki brend qidirish" autoComplete="off" />
                <div className="scope-actions">
                  <button className="btn ghost" id="scopeAll" type="button">Hammasi</button>
                  <button className="btn ghost" id="scopeNone" type="button">Hech biri</button>
                </div>
                <div className="scope-list" id="scopeList"></div>
              </div>
            </section>

            <section className="card card-pad" id="pre">
              <h2 className="section-title">Rasm tayyor</h2>
              <p className="analysis-help">Rasmda bir nechta javon bo‘lsa, keraklisini <strong>Zona belgilash</strong> orqali tanlang.</p>
              <button className="btn primary lg block" id="go">Tahlil qilish</button>
              <div className="err" id="err" role="alert"></div>
            </section>


            <div id="res" className="result-panel" hidden>
              <div className="result-tabs" role="tablist" aria-label="Natija bo‘limlari">
                <button type="button" role="tab" data-tab="overview" aria-selected="true"><Icon name="chart" />Umumiy</button>
                <button type="button" role="tab" data-tab="unknown" aria-selected="false"><Icon name="box" />Katalogda yo‘q<b id="unknownBadge" hidden></b></button>
                <button type="button" role="tab" data-tab="send" id="sendTab" aria-selected="false" hidden><Icon name="report" />Yuborish</button>
              </div>
              <section className="card card-pad why-card" id="why" hidden>
                <p className="section-title">Tanlangan mahsulot</p>
                <p id="whyText"></p>
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
              <div className="tab-panel" data-panel="overview" role="tabpanel">
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
                <div className="note-list" id="notes"></div>
              </div>
              <div className="tab-panel" data-panel="unknown" role="tabpanel" hidden>
                <section className="card card-pad" id="groupsCard" hidden>
                  <p className="section-title" id="groupsTitle">Katalogda yo‘q</p>
                  <p className="analysis-help" id="groupsHelp"></p>
                  <div className="group-grid" id="groups"></div>
                </section>
                <section className="card card-pad" id="flavourCard" hidden>
                  <p className="section-title" id="flavourTitle">Brend ta‘mlari</p>
                  <p className="analysis-help" id="flavourHelp"></p>
                  <div id="flavours"></div>
                </section>
                <section className="card card-pad tab-empty" id="unknownEmpty" hidden>
                  <span className="metric-icon"><Icon name="check" /></span>
                  <strong>Guruhlar yo‘q</strong>
                  <p>Katalogda yo‘q, bir-biriga o‘xshash mahsulotlar topilmadi. Yakka mahsulotni rasmda bosib tanlash mumkin.</p>
                </section>
              </div>
              <div className="tab-panel" data-panel="send" role="tabpanel" hidden>
                <section className="card card-pad" id="sendPanel" hidden>
                  <h2 className="section-title">Auditni yuborish</h2>
                  <form id="sendForm">
                    <label className="field"><span>Do‘kon</span><select className="input" name="store" required></select></label>
                    <label className="field"><span>Izoh (ixtiyoriy)</span><textarea className="input" name="note" rows="3" maxLength="1000"></textarea></label>
                    <button className="btn primary lg block" id="sendBtn" type="submit">Hisobotni yuborish</button>
                    <div className="err" id="sendError" role="alert"></div>
                  </form>
                  <p id="sentMessage" hidden role="status">Hisobot yuborildi. <a href="/reports">Hisobotlarni ko‘rish</a></p>
                </section>
              </div>
              <div className="result-actions">
                <button className="btn" id="exportJson"><Icon name="download" />JSON yuklash</button>
                <button className="btn" id="exportImage"><Icon name="image" />Rasmni saqlash</button>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </>
  );
}
