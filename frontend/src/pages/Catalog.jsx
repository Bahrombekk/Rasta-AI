import { Icon } from "../shared.jsx";

export default function Catalog() {
  return (
    <>
      <main className="page workspace-page catalog-page">
        <div className="work-heading">
          <div>
            <h1>Mahsulotlar katalogi</h1>
            <p id="sub">Mahsulotlar va AI uchun namuna rasmlari</p>
          </div>
          <div className="page-actions">
            <button className="btn ghost" id="reindexAll" type="button" hidden><Icon name="refresh" />Qayta o‘rganish</button>
            <button className="btn ghost" id="export" disabled><Icon name="download" />CSV yuklash</button>
            <button className="btn ghost" id="bundleExport" type="button" disabled title="Katalogni rasmlari bilan bitta faylga (.zip) saqlash — boshqa korxonaga ko‘chirish uchun"><Icon name="download" />Katalog fayli</button>
            <button className="btn ghost" id="bundleImport" type="button" hidden title="Tayyor katalog faylini (.zip) shu katalogga qo‘shish"><Icon name="upload" />Katalog faylini yuklash</button>
            <input type="file" id="bundleFile" accept=".zip,application/zip" hidden />
            <button className="btn primary" type="button" id="addBtn" hidden><Icon name="plus" />Mahsulot qo'shish</button>
          </div>
        </div>

        <section className="card catalog-organization" id="organization-panel" hidden>
          <label className="field"><span>Korxona</span><select className="input" id="organization"></select></label>
        </section>
        <section id="catalog-content" hidden>
          <div className="catalog-metrics" aria-label="Katalog ko‘rsatkichlari">
            <div className="card"><span>Mahsulotlar</span><strong id="product-count">0</strong></div>
            <div className="card"><span>Kategoriyalar</span><strong id="category-count">0</strong></div>
            <div className="card"><span>Shtrix-kodli</span><strong id="barcode-count">0</strong></div>
            <div className="card"><span>Rasmsiz</span><strong id="missing-count">0</strong></div>
          </div>
          <div className="card catalog-filters">
            <label className="field catalog-query"><span>Qidiruv</span><span className="search"><span><Icon name="search" /></span>
              <input type="search" className="input" id="q" placeholder="Nomi, SKU yoki shtrix-kod" /></span></label>
            <button className="btn ghost catalog-filter-toggle" id="filter-toggle" type="button" aria-controls="extra-filters" aria-expanded="false">Filtrlar</button>
            <div className="catalog-extra-filters" id="extra-filters">
              <label className="field"><span>Kategoriya</span><select className="input" id="category-filter"></select></label>
              <label className="field"><span>Brend</span><select className="input" id="brand-filter"></select></label>
              <label className="field"><span>Namuna rasmlari</span><select className="input" id="image-filter">
                <option value="">Barchasi</option><option value="missing">Rasmsiz</option><option value="available">Rasmi bor</option>
              </select></label>
            </div>
          </div>
          <div className="catalog-results"><p id="result-count" aria-live="polite"></p><button className="btn ghost" id="reset" hidden>Filtrlarni tozalash</button></div>
          <div className="grid" id="grid"></div>
          <div className="catalog-empty card" id="empty" hidden></div>
          <nav className="catalog-pagination" id="pagination" aria-label="Katalog sahifalari" hidden>
            <button className="btn ghost" id="previous">Oldingi</button><span id="page-number"></span><button className="btn ghost" id="next">Keyingi</button>
          </nav>
        </section>
        <p id="status" className="catalog-status" role="status">Katalog yuklanmoqda…</p>
      </main>

      <dialog className="catalog-dialog" id="drawer" aria-labelledby="dTitle">
        <div className="hd"><h2 id="dTitle">Yangi mahsulot</h2><button className="iconbtn" id="close" type="button" aria-label="Yopish"><Icon name="x" /></button></div>
        <form className="bd" id="form">
          <label className="field"><span>Nomi</span>
            <input className="input" id="name" required placeholder="Masalan: Coca-Cola 1.5L" autoComplete="off" /></label>
          <label className="field"><span>Brend</span>
            <input className="input" id="brand" placeholder="Masalan: Coca-Cola" list="brands" autoComplete="off" /></label>
          <datalist id="brands"></datalist>
          <label className="field"><span>Kategoriya</span>
            <input className="input" id="category" list="categories" maxLength="80" placeholder="Masalan: Ichimliklar" autoComplete="off" /></label>
          <datalist id="categories"></datalist>
          <div className="catalog-fields">
            <label className="field"><span>SKU</span>
              <input className="input" id="sku" maxLength="48" placeholder="Avtomatik" autoComplete="off" /></label>
            <label className="field"><span>Shtrix-kod</span>
              <input className="input" id="barcode" inputMode="numeric" pattern={"[0-9]{4,32}"} maxLength="32" placeholder="4–32 ta raqam" autoComplete="off" /></label>
          </div>
          <div className="field" id="curWrap" hidden><span id="curLbl">Mavjud rasmlar</span>
            <p className="shot-help" id="shotHelp">Yashil ramka — AI o‘rganadigan qism. Ramka mahsulotni noto‘g‘ri qamrasa, rasmni bosib to‘g‘rilang.</p>
            <div className="tip shot-warn" id="relocateTip" hidden>
              <span>Bu rasmlarda mahsulot hali fondan ajratilmagan — AI butun kadrni (fon bilan) o‘rgangan.</span>
              <button className="btn" id="relocate" type="button">Mahsulotni ajratish</button>
            </div>
            <div className="thumbs" id="cur"></div>
          </div>
          <div className="field" id="new-images"><span id="newLbl">Rasmlar</span>
            <label className="drop compact" id="drop" htmlFor="files">
              <div className="ico"><Icon name="upload" /></div>
              <div style={{"fontWeight": "600", "fontSize": "14px"}}>Rasmlarni tanlang yoki tashlang</div>
              <div style={{"color": "var(--muted)", "fontSize": "13px", "marginTop": "2px"}}>Bir nechta rasm tanlash mumkin</div>
            </label>
            <input id="files" type="file" accept="image/*" multiple />
            <div className="thumbs" id="thumbs"></div>
          </div>
          <div className="tip" id="tip">Mahsulotni <strong>old tomonidan</strong>, javonda turganidek suratga oling — yon yoki orqa tomon rasmlari noto‘g‘ri tanishga olib keladi. Har bir ta’m yoki hajm uchun alohida SKU yarating. Eng aniq namuna — tahlil natijasida javondagi mahsulot ramkasini bosib qo‘shilgani.</div>
          <div className="err" id="err" role="status"></div>
        </form>
        <div className="ft">
          <button className="btn ghost catalog-danger" id="delete-product" type="button" hidden>O‘chirish</button>
          <button className="btn ghost" id="cancel" type="button">Bekor qilish</button>
          <button className="btn primary" id="save" type="submit" form="form">Saqlash</button>
        </div>
      </dialog>

      <dialog className="catalog-dialog box-dialog" id="boxEditor" aria-labelledby="boxTitle">
        <div className="hd"><h2 id="boxTitle">Mahsulot chegarasi</h2><button className="iconbtn" id="boxClose" type="button" aria-label="Yopish"><Icon name="x" /></button></div>
        <div className="bd" id="boxBody">
          <p className="shot-help">AI faqat ramka ichidagi qismni o‘rganadi. Ramka mahsulotni to‘liq va ortiqcha fonsiz qamrasin: burchaklaridan torting yoki rasm ustida yangisini chizing.</p>
          <div className="box-stage" id="boxStage">
            <img id="boxImage" alt="Namuna rasmi" draggable="false" />
            <div className="box-rect" id="boxRect" hidden><i data-h="nw"></i><i data-h="ne"></i><i data-h="sw"></i><i data-h="se"></i></div>
          </div>
          <p className="box-state" id="boxState" role="status"></p>
          <div className="err" id="boxErr" role="alert"></div>
        </div>
        <div className="ft">
          <button className="btn ghost" id="boxAuto" type="button">Avtomatik</button>
          <button className="btn ghost" id="boxFull" type="button">Butun rasm</button>
          <span style={{"flex": "1"}}></span>
          <button className="btn ghost" id="boxCancel" type="button">Bekor qilish</button>
          <button className="btn primary" id="boxSave" type="button">Saqlash</button>
        </div>
      </dialog>
    </>
  );
}
