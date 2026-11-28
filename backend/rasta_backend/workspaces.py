"""Organization membership and durable shelf audit records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
from uuid import uuid4

from .accounts import new_password, username_key
from .database import INTEGRITY_ERRORS
from .field import FieldOperations

TASHKENT = timezone(timedelta(hours=5))
SHELF_REPORTS = 1000          # javon kesimi (brend, mahsulot) uchun o'qiladigan so'nggi auditlar soni


def period(days):
    end = datetime.now(TASHKENT).date()
    start = end - timedelta(days=days - 1)
    start_time = int(datetime.combine(start, datetime.min.time(), TASHKENT).timestamp())
    return start, (start_time, start_time + days * 86400)


def daily_counts(start, days, counts):
    dates = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    return [{"day": date, "count": counts.get(date, 0)} for date in dates]


def shelf_insights(summaries, top=8):
    """Auditlar xulosalaridan javon kesimi: brendlar ulushi, eng ko'p uchragan mahsulotlar, to'liqlik."""
    brands, products, fill = {}, {}, []
    unknown = missing = 0
    for summary in summaries:
        if summary.get("fill_pct") is not None:
            fill.append(float(summary["fill_pct"]))
        unknown += int(summary.get("num_unknown") or 0)
        missing += int(summary.get("oos_facings") or 0)
        for brand in summary.get("brands") or []:
            row = brands.setdefault(brand["brand"], {"brand": brand["brand"], "facings": 0, "audits": 0})
            row["facings"] += int(brand.get("facings") or 0)
            row["audits"] += 1
        for product in summary.get("products") or []:
            row = products.setdefault(product["sku_id"], {
                "sku_id": product["sku_id"], "name": product.get("name") or product["sku_id"],
                "brand": product.get("brand"), "facings": 0, "audits": 0})
            row["facings"] += int(product.get("facings") or 0)
            row["audits"] += 1
    branded = sum(row["facings"] for row in brands.values())
    ranked = sorted(brands.values(), key=lambda row: (-row["facings"], row["brand"]))
    for row in ranked:
        row["share_pct"] = round(100.0 * row["facings"] / branded, 1) if branded else 0.0
    return {
        "audits": len(summaries), "unknown": unknown, "missing_facings": missing,
        "fill_pct": round(sum(fill) / len(fill), 1) if fill else None, "fill_audits": len(fill),
        "brand_count": len(ranked), "product_count": len(products),
        "brands": ranked[:top],
        "products": sorted(products.values(), key=lambda row: (-row["facings"], row["name"]))[:top],
    }


class Workspaces:
    def __init__(self, accounts, *, seed=True):
        self.accounts = accounts
        self.db = accounts._db
        database = accounts.database
        database.migrate("002_workspaces")
        database.migrate("003_field", field_visit_column=True)
        database.migrate("004_query_indexes")
        if database.postgres:
            database.migrate("005_tenant_constraints")
        self.field = FieldOperations(self.db)
        if seed:
            self.seed_default_organization()

    def seed_default_organization(self):
        with self.db() as db:
            db.write_lock()
            inserted = db.execute(
                "INSERT INTO organizations VALUES(1,'Asosiy korxona',1,?) ON CONFLICT(id) DO NOTHING",
                (int(time.time()),),
            )
            if inserted.rowcount:
                db.advance_identity("organizations")
            db.execute("""
                INSERT INTO members(user_id,org_id,role)
                SELECT id,1,'org_admin' FROM users WHERE role='admin'
                ON CONFLICT(user_id) DO NOTHING
            """)

    def identity(self, user):
        if not user or user["role"] == "super_admin":
            return user
        with self.db() as db:
            row = db.execute("""
                SELECT m.role, o.id, o.name, o.active FROM members m
                JOIN organizations o ON o.id=m.org_id WHERE m.user_id=?
            """, (user["id"],)).fetchone()
        if not row or not row["active"]:
            return None
        return {**user, "role": row["role"], "org_id": row["id"], "organization": row["name"]}

    @staticmethod
    def _new_user(db, org_id, username, hashed, role):
        user_id = db.execute("INSERT INTO users(username,password,role,created) VALUES(?,?,'admin',?) RETURNING id",
                             (username, hashed, int(time.time()))).fetchone()["id"]
        db.execute("INSERT INTO members VALUES(?,?,?)", (user_id, org_id, role))
        return {"id": user_id, "username": username, "role": role, "active": True}

    @staticmethod
    def _platform_event(db, actor_id, org_id, action, subject):
        if actor_id is not None:
            db.execute("INSERT INTO platform_events(org_id,actor_id,action,subject,created) VALUES(?,?,?,?,?)",
                       (org_id, actor_id, action, subject, int(time.time())))

    def create_org(self, name, username, password, actor_id=None):
        name = name.strip()
        if not name:
            raise ValueError("Korxona nomini kiriting.")
        username, hashed = username_key(username), new_password(password)
        with self.db() as db:
            db.write_lock()
            try:
                cursor = db.execute("INSERT INTO organizations(name,created) VALUES(?,?) RETURNING id", (name, int(time.time())))
                org_id = cursor.fetchone()["id"]
                admin = self._new_user(db, org_id, username, hashed, "org_admin")
                self._platform_event(db, actor_id, org_id, "organization_created", username)
            except INTEGRITY_ERRORS as exc:
                raise ValueError("Korxona nomi yoki admin logini band.") from exc
        return {"id": org_id, "name": name, "active": True, "admin": admin}

    def organizations(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("""
                SELECT o.id, o.name, o.active,
                (SELECT COUNT(*) FROM members WHERE org_id=o.id) AS members,
                (SELECT COUNT(*) FROM stores WHERE org_id=o.id) AS stores,
                (SELECT COUNT(*) FROM reports WHERE org_id=o.id AND submitted IS NOT NULL) AS reports
                FROM organizations o ORDER BY o.id
            """)]

    def organization_exists(self, org_id):
        with self.db() as db:
            return db.execute("SELECT 1 FROM organizations WHERE id=?", (org_id,)).fetchone() is not None

    def update_org(self, org_id, name, active, actor_id=None):
        name = name.strip()
        if not name:
            raise ValueError("Korxona nomini kiriting.")
        with self.db() as db:
            db.write_lock()
            previous = db.execute("SELECT name,active FROM organizations WHERE id=?", (org_id,)).fetchone()
            if previous is None:
                return False
            try:
                db.execute("UPDATE organizations SET name=?, active=? WHERE id=?", (name, int(active), org_id))
            except INTEGRITY_ERRORS as exc:
                raise ValueError("Korxona nomi band.") from exc
            if not active:
                db.execute("DELETE FROM sessions WHERE user_id IN (SELECT user_id FROM members WHERE org_id=?)",
                           (org_id,))
            if previous["active"] != active:
                action = "organization_activated" if active else "organization_suspended"
                self._platform_event(db, actor_id, org_id, action, "")
            if previous["name"] != name:
                self._platform_event(db, actor_id, org_id, "organization_renamed", previous["name"])
            return True

    def members(self, org_id):
        with self.db() as db:
            return [dict(row) for row in db.execute("""
                SELECT u.id, u.username, u.active, m.role FROM members m
                JOIN users u ON u.id=m.user_id WHERE m.org_id=? ORDER BY m.role,u.username
            """, (org_id,))]

    def create_member(self, org_id, username, password, role, actor_id=None):
        username, hashed = username_key(username), new_password(password)
        with self.db() as db:
            try:
                user = self._new_user(db, org_id, username, hashed, role)
                self._platform_event(db, actor_id, org_id, "admin_created", username)
                return user
            except INTEGRITY_ERRORS as exc:
                raise ValueError("Login band yoki korxona topilmadi.") from exc

    def update_member(self, org_id, target, role, active, password, actor_id=None):
        hashed = new_password(password) if password is not None else None
        with self.db() as db:
            db.write_lock()
            previous = db.execute("""
                SELECT u.username,u.active FROM users u JOIN members m ON m.user_id=u.id
                WHERE u.id=? AND m.org_id=? AND m.role=?
            """, (target, org_id, role)).fetchone()
            if previous is None:
                return False
            db.execute("""
                UPDATE users SET active=?, password=COALESCE(?,password)
                WHERE id=?
            """, (int(active), hashed, target))
            if not active or hashed:
                db.execute("DELETE FROM sessions WHERE user_id=?", (target,))
            if previous["active"] != active:
                action = "admin_activated" if active else "admin_suspended"
                self._platform_event(db, actor_id, org_id, action, previous["username"])
            if hashed:
                self._platform_event(db, actor_id, org_id, "admin_password_reset", previous["username"])
            return True

    def stores(self, org_id):
        with self.db() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,name,address FROM stores WHERE org_id=? ORDER BY name", (org_id,))]

    def create_store(self, org_id, name, address):
        name = name.strip()
        if not name:
            raise ValueError("Do‘kon nomini kiriting.")
        with self.db() as db:
            try:
                cursor = db.execute("INSERT INTO stores(org_id,name,address) VALUES(?,?,?) RETURNING id", (org_id, name, address.strip()))
            except INTEGRITY_ERRORS as exc:
                raise ValueError("Bu nomdagi do‘kon mavjud.") from exc
            return {"id": cursor.fetchone()["id"], "name": name, "address": address.strip()}

    def update_store(self, org_id, store_id, name, address):
        name = name.strip()
        if not name:
            raise ValueError("Do‘kon nomini kiriting.")
        with self.db() as db:
            try:
                cursor = db.execute("UPDATE stores SET name=?,address=? WHERE org_id=? AND id=?",
                                    (name, address.strip(), org_id, store_id))
            except INTEGRITY_ERRORS as exc:
                raise ValueError("Bu nomdagi do‘kon mavjud.") from exc
            return bool(cursor.rowcount)

    def create_draft(self, user, data, photo, visit_id=None):
        now = int(time.time())
        report_id = uuid4().hex
        summary = data["summary"]
        with self.db() as db:
            db.write_lock()
            if visit_id is not None and self.field.own_visit(user, visit_id, db) is None:
                raise ValueError("Tashrif ochiq va do‘kon sizga biriktirilgan bo‘lishi kerak.")
            db.execute("DELETE FROM reports WHERE status='draft' AND created<?", (now - 86400,))
            db.execute("""
                DELETE FROM reports WHERE id IN (SELECT id FROM reports
                WHERE org_id=? AND author_id=? AND status='draft' ORDER BY created DESC,id DESC
                LIMIT 9223372036854775807 OFFSET 19)
            """, (user["org_id"], user["id"]))
            db.execute("""
                INSERT INTO reports(id,org_id,author_id,status,result,photo,created,products,recognized,gaps,visit_id)
                VALUES(?,?,?,'draft',?,?,?,?,?,?,?)
            """, (report_id, user["org_id"], user["id"], json.dumps(data, ensure_ascii=False), photo, now,
                  summary["num_products"], summary["num_known"], summary["oos_gaps"], visit_id))
        return report_id

    def submit(self, user, report_id, store_id, note):
        now = int(time.time())
        note = note.strip()
        with self.db() as db:
            db.write_lock()
            if user["role"] == "operator":
                existing = db.execute("""
                    SELECT r.status,r.store_id,r.visit_id,v.store_id AS visit_store,v.finished
                    FROM reports r LEFT JOIN field_visits v ON v.id=r.visit_id
                    WHERE r.id=? AND r.org_id=? AND r.author_id=?
                """, (report_id, user["org_id"], user["id"])).fetchone()
                if existing is None:
                    return None
                if existing["status"] == "submitted" and existing["store_id"] == store_id:
                    return "submitted"
                if not self.field.assigned(db, user, store_id):
                    raise ValueError("Do‘kon sizga biriktirilmagan.")
                if existing["visit_id"] is not None and (
                        existing["visit_store"] != store_id or
                        (existing["status"] == "draft" and existing["finished"] is not None)):
                    raise ValueError("Auditni shu do‘konning ochiq tashrifi orqali yuboring.")
            changed = db.execute("""
                UPDATE reports SET status='submitted',store_id=?,note=?,submitted=?,
                reviewed=NULL,reviewer_id=NULL,feedback=''
                WHERE id=? AND org_id=? AND author_id=? AND status IN ('draft','returned')
                AND (status!='draft' OR created>=?)
                AND EXISTS(SELECT 1 FROM stores WHERE id=? AND org_id=reports.org_id)
            """, (store_id, note, now, report_id, user["org_id"], user["id"], now - 86400, store_id)).rowcount
            if not changed:
                row = db.execute("SELECT status FROM reports WHERE id=? AND org_id=? AND author_id=?",
                                 (report_id, user["org_id"], user["id"])).fetchone()
                if not row:
                    return None
                if row["status"] == "submitted":
                    return "submitted"
                raise ValueError("Do‘kon yoki hisobot holatini tekshiring. Qoralama 24 soat saqlanadi.")
            db.execute("INSERT INTO report_events(report_id,user_id,status,note,created) VALUES(?,?,'submitted',?,?)",
                       (report_id, user["id"], note, now))
        return "submitted"

    @staticmethod
    def _visibility(user):
        if user["role"] == "org_admin":
            return "r.org_id=? AND (r.status!='draft' OR r.author_id=?)", (user["org_id"], user["id"])
        return "r.org_id=? AND r.author_id=?", (user["org_id"], user["id"])

    def _report_filters(self, user, filters):
        where, args = self._visibility(user)
        where += " AND r.status=?" if filters.status else " AND r.status!='draft'"
        if filters.status:
            args += (filters.status,)
        if filters.store_id is not None:
            where += " AND r.store_id=?"
            args += (filters.store_id,)
        if filters.author_id is not None:
            where += " AND r.author_id=?"
            args += (filters.author_id,)
        if filters.start is not None:
            where += " AND COALESCE(r.submitted,r.created)>=?"
            args += (int(datetime.combine(filters.start, datetime.min.time(), TASHKENT).timestamp()),)
        if filters.end is not None:
            where += " AND COALESCE(r.submitted,r.created)<?"
            args += (int(datetime.combine(filters.end, datetime.min.time(), TASHKENT).timestamp()) + 86400,)
        return where, args

    def list_reports(self, user, filters, offset=0, *, limit=50, count_total=True):
        where, args = self._report_filters(user, filters)
        with self.db() as db:
            if count_total:
                db.read_snapshot()
                total = db.execute(f"SELECT COUNT(*) AS total FROM reports r WHERE {where}", args).fetchone()["total"]
            else:
                total = None
            rows = [dict(row) for row in db.execute(f"""
                SELECT r.id,r.status,r.created,r.submitted,r.products,r.recognized,r.gaps,
                s.name AS store,u.username AS author FROM reports r
                LEFT JOIN stores s ON s.id=r.store_id JOIN users u ON u.id=r.author_id
                WHERE {where} ORDER BY COALESCE(r.submitted,r.created) DESC,r.created DESC,r.id DESC LIMIT ? OFFSET ?
            """, (*args, limit, offset))]
            return rows, total

    def report(self, user, report_id, *, photo=False):
        where, args = self._visibility(user)
        columns = "r.photo" if photo else """r.id,r.status,r.created,r.submitted,r.reviewed,r.note,r.feedback,
                r.result,r.store_id,r.visit_id,s.name AS store,a.username AS author,v.username AS reviewer"""
        with self.db() as db:
            row = db.execute(f"""
                SELECT {columns} FROM reports r LEFT JOIN stores s ON s.id=r.store_id
                JOIN users a ON a.id=r.author_id LEFT JOIN users v ON v.id=r.reviewer_id
                WHERE r.id=? AND {where}
            """, (report_id, *args)).fetchone()
            if row is None:
                return None
            if photo:
                return row["photo"]
            result = dict(row)
            result["events"] = [dict(event) for event in db.execute("""
                SELECT e.status,e.note,e.created,u.username FROM report_events e
                JOIN users u ON u.id=e.user_id WHERE e.report_id=? ORDER BY e.id DESC LIMIT 100
            """, (report_id,))]
        result["result"] = json.loads(result["result"])
        return result

    def review(self, user, report_id, status, feedback):
        now = int(time.time())
        feedback = feedback.strip()
        with self.db() as db:
            reviewed = db.execute("""
                UPDATE reports SET status=?, feedback=?, reviewed=?,reviewer_id=?
                WHERE id=? AND org_id=? AND status='submitted'
            """, (status, feedback, now, user["id"], report_id, user["org_id"])).rowcount
            if reviewed:
                db.execute("INSERT INTO report_events(report_id,user_id,status,note,created) VALUES(?,?,?,?,?)",
                           (report_id, user["id"], status, feedback, now))
            return bool(reviewed)

    def overview(self, user, days):
        visible, identity = self._visibility(user)
        start, bounds = period(days)
        where = visible + " AND r.status!='draft' AND r.submitted>=? AND r.submitted<?"
        args = identity + bounds
        earlier = identity + (bounds[0] - days * 86400, bounds[0])       # shu uzunlikdagi oldingi davr
        aggregate = f"""
            SELECT COUNT(*) AS reports, COUNT(CASE WHEN status='submitted' THEN 1 END) AS pending,
            COUNT(CASE WHEN status='approved' THEN 1 END) AS approved, COUNT(CASE WHEN status='returned' THEN 1 END) AS returned,
            COUNT(DISTINCT store_id) AS stores, COALESCE(SUM(products),0) AS products,
            COALESCE(SUM(recognized),0) AS recognized, COALESCE(SUM(gaps),0) AS gaps,
            AVG(CASE WHEN reviewed>=submitted THEN CAST(reviewed-submitted AS DOUBLE PRECISION) END) AS review_seconds
            FROM reports r WHERE {where}
        """
        with self.db() as db:
            db.read_snapshot()
            totals = dict(db.execute(aggregate, args).fetchone())
            totals["previous"] = dict(db.execute(aggregate, earlier).fetchone())
            totals["queue"] = dict(db.execute(f"""
                SELECT COUNT(*) AS open, COUNT(CASE WHEN r.submitted<? THEN 1 END) AS stale FROM reports r
                WHERE {visible} AND r.status='submitted'
            """, (int(time.time()) - 86400, *identity)).fetchone())
            daily = {str(row["day"]): dict(row) for row in db.execute(f"""
                SELECT {db.submission_day} AS day, COUNT(*) AS count, SUM(r.gaps) AS gaps,
                SUM(r.products) AS products, SUM(r.recognized) AS recognized FROM reports r
                WHERE {where} GROUP BY day
            """, args)}
            totals["store_activity"] = [dict(row) for row in db.execute(f"""
                SELECT s.id,s.name,COUNT(*) AS reports,SUM(r.gaps) AS gaps,SUM(r.products) AS products,
                SUM(r.recognized) AS recognized,MAX(r.submitted) AS last_submission FROM reports r
                JOIN stores s ON s.id=r.store_id WHERE {where}
                GROUP BY s.id,s.name ORDER BY reports DESC,s.name LIMIT 8
            """, args)]
            totals["operator_activity"] = [dict(row) for row in db.execute(f"""
                SELECT u.id,u.username,COUNT(*) AS reports,COUNT(CASE WHEN r.status='approved' THEN 1 END) AS approved,
                COUNT(CASE WHEN r.status='returned' THEN 1 END) AS returned,SUM(r.gaps) AS gaps,
                SUM(r.products) AS products,SUM(r.recognized) AS recognized,MAX(r.submitted) AS last_submission FROM reports r
                JOIN users u ON u.id=r.author_id WHERE {where}
                GROUP BY u.id,u.username ORDER BY reports DESC,u.username LIMIT 8
            """, args)]
        counts = {day: row["count"] for day, row in daily.items()}
        totals["trend"] = [{**item, **{key: int(daily.get(item["day"], {}).get(key) or 0)
                                       for key in ("gaps", "products", "recognized")}}
                           for item in daily_counts(start, days, counts)]
        totals["shelf"] = shelf_insights(self._summaries(where, args))
        return totals

    def _summaries(self, where, args):
        """Davrdagi so'nggi auditlarning xulosalari. Xulosa bazaning o'zida ajratib olinadi; baza buni bajara
        olmasa (JSON funksiyasi yo'q) — butun natija o'qilib, shu yerda ajratiladi."""
        tail = f"FROM reports r WHERE {where} ORDER BY r.submitted DESC LIMIT ?"
        try:
            with self.db() as db:
                rows = [row["summary"] for row in db.execute(
                    f"SELECT {db.report_summary} AS summary {tail}", (*args, SHELF_REPORTS))]
            return [json.loads(text) for text in rows if text]
        except Exception:  # noqa: BLE001
            with self.db() as db:
                rows = [row["result"] for row in db.execute(f"SELECT r.result {tail}", (*args, SHELF_REPORTS // 5))]
            return [json.loads(text).get("summary") or {} for text in rows]

    def platform_overview(self, days, org_id=None):
        start, bounds = period(days)
        org_where = " WHERE o.id=?" if org_id is not None else ""
        org_params = (org_id,) if org_id is not None else ()
        report_where = " AND r.org_id=?" if org_id is not None else ""
        with self.db() as db:
            db.read_snapshot()
            organization = None
            if org_id is not None:
                row = db.execute("SELECT id,name,active FROM organizations WHERE id=?", (org_id,)).fetchone()
                if row is None:
                    return None
                organization = dict(row)
            organizations = dict(db.execute(f"""
                SELECT COUNT(*) AS total, COUNT(CASE WHEN o.active=1 THEN 1 END) AS active
                FROM organizations o{org_where}
            """, org_params).fetchone())
            members = dict(db.execute(f"""
                SELECT COUNT(*) AS total,
                    COUNT(CASE WHEN u.active=1 AND o.active=1 THEN 1 END) AS active,
                    COUNT(CASE WHEN m.role='org_admin' THEN 1 END) AS admins,
                    COUNT(CASE WHEN m.role='operator' THEN 1 END) AS operators
                FROM members m JOIN organizations o ON o.id=m.org_id JOIN users u ON u.id=m.user_id{org_where}
            """, org_params).fetchone())
            stores = db.execute(f"""
                SELECT COUNT(*) AS total FROM stores s JOIN organizations o ON o.id=s.org_id{org_where}
            """, org_params).fetchone()["total"]
            audits = dict(db.execute(f"""
                SELECT COUNT(*) AS total, COUNT(DISTINCT r.org_id) AS reporting_orgs,
                    COUNT(CASE WHEN r.status='submitted' THEN 1 END) AS pending,
                    COUNT(CASE WHEN r.status='approved' THEN 1 END) AS approved,
                    COUNT(CASE WHEN r.status='returned' THEN 1 END) AS returned,
                    COALESCE(SUM(r.products),0) AS products, COALESCE(SUM(r.recognized),0) AS recognized,
                    COALESCE(SUM(r.gaps),0) AS gaps,
                    AVG(CASE WHEN r.reviewed>=r.submitted THEN CAST(r.reviewed-r.submitted AS DOUBLE PRECISION) END) AS review_seconds
                FROM reports r WHERE r.submitted>=? AND r.submitted<?{report_where}
            """, (*bounds, *org_params)).fetchone())
            latest = dict(db.execute(f"""
                SELECT MAX(r.submitted) AS last_submission,
                    COUNT(CASE WHEN r.status='submitted' THEN 1 END) AS open_queue
                FROM reports r WHERE r.submitted IS NOT NULL{report_where}
            """, org_params).fetchone())
            trend = {str(row["day"]): row["count"] for row in db.execute(f"""
                SELECT {db.submission_day} AS day, COUNT(*) AS count
                FROM reports r WHERE r.submitted>=? AND r.submitted<?{report_where} GROUP BY day
            """, (*bounds, *org_params))}
            activity = [] if organization else [dict(row) for row in db.execute("""
                SELECT o.id,o.name,o.active,COUNT(*) AS audits,
                    COUNT(CASE WHEN r.status='submitted' THEN 1 END) AS pending,
                    MAX(r.submitted) AS last_submission
                FROM reports r JOIN organizations o ON o.id=r.org_id
                WHERE r.submitted>=? AND r.submitted<?
                GROUP BY o.id ORDER BY audits DESC,last_submission DESC,o.id LIMIT 10
            """, bounds)]
            events = [dict(row) for row in db.execute(f"""
                SELECT e.action,e.subject,e.created,o.id AS org_id,o.name AS organization,u.username AS actor
                FROM platform_events e JOIN organizations o ON o.id=e.org_id
                JOIN users u ON u.id=e.actor_id{org_where} ORDER BY e.id DESC LIMIT 12
            """, org_params)]
        return {"organization": organization, "organizations": organizations, "members": members,
                "stores": stores, "audits": {**audits, **latest}, "activity": activity, "events": events,
                "trend": daily_counts(start, days, trend)}
