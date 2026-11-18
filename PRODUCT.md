# RASTA · Korxona javon auditi

## Bozor tahlili

- Trax daladagi xodimdan javon fotosini olib, mahsulot, facings va javondagi ulushni hisoblaydi. RASTA’da mobil oqimning markazi rasm yuklash, natijani ko‘rib chiqish va auditni yuborish bo‘ladi. [Trax Image Recognition](https://traxretail.com/solutions/trax-image-recognition/) · [Trax field workflow](https://traxretail.com/blog/trax-quarterly-whats-new-in-product-innovation/)
- Repsly menejerga jamoa faolligi, do‘konlar va fotosuratlar bo‘yicha drill-down beradi. RASTA’da admin bosh sahifasida davr bo‘yicha ko‘rsatkichlar va tekshiruv navbati bo‘ladi. [Repsly Insights Dashboards](https://www.repsly.com/product/insights-dashboards)
- AWS SaaS Lens korxona kontekstini foydalanuvchi identifikatsiyasiga bog‘lashni va har bir resursga qo‘llashni tavsiflaydi. RASTA’da sessiyadan olingan korxona hisobot so‘rovlari, SKU vektorlari va media ruxsatini belgilaydi. [AWS tenant isolation](https://docs.aws.amazon.com/whitepapers/latest/saas-architecture-fundamentals/tenant-isolation.html)

## Amalga oshirish rejasi

1. **Platforma.** Super admin korxona va uning birinchi adminini yaratadi, korxonani faollashtiradi yoki to‘xtatadi. Har bir korxona alohida katalog va ma’lumot chegarasiga ega.
2. **Korxona ish maydoni.** Admin katalog, do‘konlar, xodimlar va hisobotlarni boshqaradi. Bosh panelda kelgan auditlar, ko‘rib chiqish navbati va javon ko‘rsatkichlari mavjud.
3. **Daladagi ish.** Operator rasm yuklaydi, kerakli zonani belgilaydi, AI natijasini ko‘radi, do‘kon va izohni kiritadi va korxonasiga yuboradi. Operator o‘z hisobotlari va tekshiruv holatini ko‘radi.
4. **Ko‘rib chiqish.** Korxona admini foto va natijani ochadi, tasdiqlaydi yoki izoh bilan qaytaradi. Qaytarilgan hisobot qayta yuborilishi mumkin.
5. **Tekshiruv.** Rollar va korxonalararo chegaralar avtomatik testlanadi. Lokal Docker’da haqiqiy GPU, mobil va desktop oqimlari tekshiriladi.

## Joylashtirish chegarasi

Hozirgi implementatsiya bitta backend nusxasi, SQLite va lokal Qdrant bilan ishlaydi. Bir serverdagi pilot uchun uchlik Docker orqali yuritiladi. Keyingi server bosqichida bir nechta nusxa talab qilinsa, boshqariladigan SQL, obyekt ombori, ishlar navbati, yuklama sinovi va tiklash sinovi alohida bajariladi.
