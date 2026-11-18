"""Organization membership and durable shelf audit records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
import time
from uuid import uuid4

from .accounts import new_password, username_key

TASHKENT = timezone(timedelta(hours=5))


def period(days):
    end = datetime.now(TASHKENT).date()
    start = end - timedelta(days=days - 1)
    start_time = int(datetime.combine(start, datetime.min.time(), TASHKENT).timestamp())
    return start, (start_time, start_time + days * 86400)


def daily_counts(start, days, counts):
    dates = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    return [{"day": date, "count": counts.get(date, 0)} for date in dates]


class Workspaces:
    def __init__(self, accounts):
        self.accounts = accounts
        self.db = accounts._db
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    active INTEGER NOT NULL DEFAULT 1, created INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS members (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id),
                    org_id INTEGER NOT NULL REFERENCES organizations(id),
                    role TEXT NOT NULL CHECK(role IN ('org_admin','operator'))
                );
                CREATE INDEX IF NOT EXISTS members_org ON members(org_id);
                CREATE TABLE IF NOT EXISTS stores (
                    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
                    name TEXT NOT NULL COLLATE NOCASE, address TEXT NOT NULL,
                    UNIQUE(org_id, name)
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
                    author_id INTEGER NOT NULL REFERENCES users(id),
                    store_id INTEGER REFERENCES stores(id), note TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN ('draft','submitted','approved','returned')),
                    result TEXT NOT NULL, photo BLOB NOT NULL, created INTEGER NOT NULL,
                    submitted INTEGER, reviewed INTEGER, reviewer_id INTEGER REFERENCES users(id),
                    feedback TEXT NOT NULL DEFAULT '',
                    products INTEGER NOT NULL, recognized INTEGER NOT NULL, gaps INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reports_org_time ON reports(org_id, submitted DESC);
                CREATE INDEX IF NOT EXISTS reports_platform_time ON reports(submitted DESC)
                    WHERE submitted IS NOT NULL;
                CREATE INDEX IF NOT EXISTS reports_author_time ON reports(org_id, author_id, created DESC);
                CREATE INDEX IF NOT EXISTS reports_org_display_time
                    ON reports(org_id, COALESCE(submitted,created) DESC);
                CREATE TABLE IF NOT EXISTS report_events (
                    id INTEGER PRIMARY KEY, report_id TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id), status TEXT NOT NULL,
                    note TEXT NOT NULL, created INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS report_events_report ON report_events(report_id,id);
                CREATE TABLE IF NOT EXISTS platform_events (
                    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
                    actor_id INTEGER NOT NULL REFERENCES users(id), action TEXT NOT NULL,
                    subject TEXT NOT NULL, created INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS platform_events_org ON platform_events(org_id,id);
            """)
            db.execute("INSERT OR IGNORE INTO organizations VALUES(1,'Asosiy korxona',1,?)", (int(time.time()),))
            db.execute("""
                INSERT OR IGNORE INTO members(user_id,org_id,role)
                SELECT id,1,'org_admin' FROM users WHERE role='admin'
                AND id NOT IN (SELECT user_id FROM members)
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
        cursor = db.execute("INSERT INTO users(username,password,role,created) VALUES(?,?,'admin',?)",
                            (username, hashed, int(time.time())))
        db.execute("INSERT INTO members VALUES(?,?,?)", (cursor.lastrowid, org_id, role))
        return {"id": cursor.lastrowid, "username": username, "role": role, "active": True}

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
            try:
                cursor = db.execute("INSERT INTO organizations(name,created) VALUES(?,?)", (name, int(time.time())))
                org_id = cursor.lastrowid
                admin = self._new_user(db, org_id, username, hashed, "org_admin")
                self._platform_event(db, actor_id, org_id, "organization_created", username)
            except sqlite3.IntegrityError as exc:
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

    def update_org(self, org_id, name, active, actor_id=None):
        name = name.strip()
        if not name:
            raise ValueError("Korxona nomini kiriting.")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT name,active FROM organizations WHERE id=?", (org_id,)).fetchone()
            if previous is None:
                return False
            try:
                db.execute("UPDATE organizations SET name=?, active=? WHERE id=?", (name, active, org_id))
            except sqlite3.IntegrityError as exc:
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
            except sqlite3.IntegrityError as exc:
                raise ValueError("Login band yoki korxona topilmadi.") from exc

    def update_member(self, org_id, target, role, active, password, actor_id=None):
        hashed = new_password(password) if password is not None else None
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("""
                SELECT u.username,u.active FROM users u JOIN members m ON m.user_id=u.id
                WHERE u.id=? AND m.org_id=? AND m.role=?
            """, (target, org_id, role)).fetchone()
            if previous is None:
                return False
            db.execute("""
                UPDATE users SET active=?, password=COALESCE(?,password)
                WHERE id=?
            """, (active, hashed, target))
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
                cursor = db.execute("INSERT INTO stores(org_id,name,address) VALUES(?,?,?)", (org_id, name, address.strip()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Bu nomdagi do‘kon mavjud.") from exc
            return {"id": cursor.lastrowid, "name": name, "address": address.strip()}

    def update_store(self, org_id, store_id, name, address):
        name = name.strip()
        if not name:
            raise ValueError("Do‘kon nomini kiriting.")
        with self.db() as db:
            try:
                cursor = db.execute("UPDATE stores SET name=?,address=? WHERE org_id=? AND id=?",
                                    (name, address.strip(), org_id, store_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Bu nomdagi do‘kon mavjud.") from exc
            return bool(cursor.rowcount)

    def create_draft(self, user, data, photo):
        now = int(time.time())
        report_id = uuid4().hex
        summary = data["summary"]
        with self.db() as db:
            db.execute("DELETE FROM reports WHERE status='draft' AND created<?", (now - 86400,))
            db.execute("""
                DELETE FROM reports WHERE id IN (SELECT id FROM reports
                WHERE org_id=? AND author_id=? AND status='draft' ORDER BY created DESC,rowid DESC LIMIT -1 OFFSET 19)
            """, (user["org_id"], user["id"]))
            db.execute("""
                INSERT INTO reports(id,org_id,author_id,status,result,photo,created,products,recognized,gaps)
                VALUES(?,?,?,'draft',?,?,?,?,?,?)
            """, (report_id, user["org_id"], user["id"], json.dumps(data, ensure_ascii=False), photo, now,
                  summary["num_products"], summary["num_known"], summary["oos_gaps"]))
        return report_id

    def submit(self, user, report_id, store_id, note):
        now = int(time.time())
        note = note.strip()
        with self.db() as db:
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
                db.execute("BEGIN")
                total = db.execute(f"SELECT COUNT(*) FROM reports r WHERE {where}", args).fetchone()[0]
            else:
                total = None
            rows = [dict(row) for row in db.execute(f"""
                SELECT r.id,r.status,r.created,r.submitted,r.products,r.recognized,r.gaps,
                s.name AS store,u.username AS author FROM reports r
                LEFT JOIN stores s ON s.id=r.store_id JOIN users u ON u.id=r.author_id
                WHERE {where} ORDER BY COALESCE(r.submitted,r.created) DESC,r.rowid DESC LIMIT ? OFFSET ?
            """, (*args, limit, offset))]
            return rows, total

    def report(self, user, report_id, *, photo=False):
        where, args = self._visibility(user)
        columns = "r.photo" if photo else """r.id,r.status,r.created,r.submitted,r.reviewed,r.note,r.feedback,
                   r.result,r.store_id,s.name AS store,a.username AS author,v.username AS reviewer"""
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
        where, args = self._visibility(user)
        start, bounds = period(days)
        where += " AND r.status!='draft' AND r.submitted>=? AND r.submitted<?"
        args += bounds
        with self.db() as db:
            totals = dict(db.execute(f"""
                SELECT COUNT(*) AS reports, COALESCE(SUM(status='submitted'),0) AS pending,
                COALESCE(SUM(status='approved'),0) AS approved, COALESCE(SUM(status='returned'),0) AS returned,
                COUNT(DISTINCT store_id) AS stores, COALESCE(SUM(products),0) AS products,
                COALESCE(SUM(recognized),0) AS recognized, COALESCE(SUM(gaps),0) AS gaps
                FROM reports r WHERE {where}
            """, args).fetchone())
            trend = {row["day"]: row["count"] for row in db.execute(f"""
                SELECT date(r.submitted,'unixepoch','+05:00') AS day, COUNT(*) AS count FROM reports r
                WHERE {where} GROUP BY day
            """, args)}
            totals["store_activity"] = [dict(row) for row in db.execute(f"""
                SELECT s.id,s.name,COUNT(*) AS reports,SUM(r.gaps) AS gaps FROM reports r
                JOIN stores s ON s.id=r.store_id WHERE {where}
                GROUP BY s.id ORDER BY reports DESC,s.name LIMIT 5
            """, args)]
            totals["operator_activity"] = [dict(row) for row in db.execute(f"""
                SELECT u.id,u.username,COUNT(*) AS reports,SUM(r.status='approved') AS approved FROM reports r
                JOIN users u ON u.id=r.author_id WHERE {where}
                GROUP BY u.id ORDER BY reports DESC,u.username LIMIT 5
            """, args)]
        totals["trend"] = daily_counts(start, days, trend)
        return totals

    def platform_overview(self, days, org_id=None):
        start, bounds = period(days)
        org_where = " WHERE o.id=?" if org_id is not None else ""
        org_params = (org_id,) if org_id is not None else ()
        report_where = " AND r.org_id=?" if org_id is not None else ""
        with self.db() as db:
            db.execute("BEGIN")
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
                SELECT COUNT(*) FROM stores s JOIN organizations o ON o.id=s.org_id{org_where}
            """, org_params).fetchone()[0]
            audits = dict(db.execute(f"""
                SELECT COUNT(*) AS total, COUNT(DISTINCT r.org_id) AS reporting_orgs,
                    COUNT(CASE WHEN r.status='submitted' THEN 1 END) AS pending,
                    COUNT(CASE WHEN r.status='approved' THEN 1 END) AS approved,
                    COUNT(CASE WHEN r.status='returned' THEN 1 END) AS returned,
                    COALESCE(SUM(r.products),0) AS products, COALESCE(SUM(r.recognized),0) AS recognized,
                    COALESCE(SUM(r.gaps),0) AS gaps,
                    AVG(CASE WHEN r.reviewed>=r.submitted THEN r.reviewed-r.submitted END) AS review_seconds
                FROM reports r WHERE r.submitted>=? AND r.submitted<?{report_where}
            """, (*bounds, *org_params)).fetchone())
            latest = dict(db.execute(f"""
                SELECT MAX(r.submitted) AS last_submission,
                    COUNT(CASE WHEN r.status='submitted' THEN 1 END) AS open_queue
                FROM reports r WHERE r.submitted IS NOT NULL{report_where}
            """, org_params).fetchone())
            trend = {row["day"]: row["count"] for row in db.execute(f"""
                SELECT date(r.submitted,'unixepoch','+05:00') AS day, COUNT(*) AS count
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
