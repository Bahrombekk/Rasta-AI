"""Store assignments, field visits, delivery records and agent tasks."""
from datetime import datetime, timedelta, timezone
import time

LOCAL_TZ = timezone(timedelta(hours=5))


class FieldOperations:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def assigned(conn, user, store_id):
        return conn.execute("""
            SELECT 1 FROM store_assignments WHERE org_id=? AND agent_id=? AND store_id=?
        """, (user["org_id"], user["id"], store_id)).fetchone() is not None

    @staticmethod
    def _agent(conn, org_id, agent_id, *, active=False):
        return conn.execute("""
            SELECT 1 FROM members m JOIN users u ON u.id=m.user_id
            WHERE m.org_id=? AND m.user_id=? AND m.role='operator' AND (?=0 OR u.active=1)
        """, (org_id, agent_id, int(active))).fetchone() is not None

    def scoped_stores(self, user):
        with self.db() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT s.id,s.name,s.address FROM stores s JOIN store_assignments a ON a.store_id=s.id AND a.org_id=s.org_id
                WHERE a.org_id=? AND a.agent_id=? ORDER BY s.name
            """, (user["org_id"], user["id"]))]

    def assignments(self, org_id, agent_id):
        with self.db() as conn:
            if not self._agent(conn, org_id, agent_id):
                return None
            return [row["store_id"] for row in conn.execute("""
                SELECT store_id FROM store_assignments WHERE org_id=? AND agent_id=? ORDER BY store_id
            """, (org_id, agent_id))]

    def replace_assignments(self, org_id, agent_id, store_ids):
        store_ids = sorted(set(store_ids))
        with self.db() as conn:
            conn.write_lock()
            if not self._agent(conn, org_id, agent_id):
                return None
            available = {row["id"] for row in conn.execute("SELECT id FROM stores WHERE org_id=?", (org_id,))}
            if not set(store_ids) <= available:
                raise ValueError("Do‘kon shu korxonaga tegishli bo‘lishi kerak.")
            conn.execute("DELETE FROM store_assignments WHERE org_id=? AND agent_id=?", (org_id, agent_id))
            conn.executemany("INSERT INTO store_assignments VALUES(?,?,?)",
                             [(org_id, agent_id, store_id) for store_id in store_ids])
        return store_ids

    def own_visit(self, user, visit_id, conn=None, *, for_work=True):
        if conn is None:
            with self.db() as db:
                return self.own_visit(user, visit_id, db, for_work=for_work)
        row = conn.execute("""
            SELECT * FROM field_visits WHERE id=? AND org_id=? AND agent_id=? AND finished IS NULL
        """, (visit_id, user["org_id"], user["id"])).fetchone()
        if row is None or (for_work and not self.assigned(conn, user, row["store_id"])):
            return None
        return row

    def start_visit(self, user, store_id):
        with self.db() as conn:
            conn.write_lock()
            if not self.assigned(conn, user, store_id):
                raise ValueError("Do‘kon sizga biriktirilmagan.")
            opened = conn.execute("""
                SELECT id,store_id FROM field_visits WHERE org_id=? AND agent_id=? AND finished IS NULL
            """, (user["org_id"], user["id"])).fetchone()
            if opened:
                if opened["store_id"] != store_id:
                    raise ValueError("Avval ochiq tashrifni yakunlang.")
                return opened["id"]
            return conn.execute("INSERT INTO field_visits(org_id,store_id,agent_id,started) VALUES(?,?,?,?) RETURNING id",
                                (user["org_id"], store_id, user["id"], int(time.time()))).fetchone()["id"]

    def finish_visit(self, user, visit_id, note):
        with self.db() as conn:
            conn.write_lock()
            if self.own_visit(user, visit_id, conn, for_work=False) is None:
                return False
            work = conn.execute("""
                SELECT (EXISTS(SELECT 1 FROM field_deliveries WHERE visit_id=?)
                     OR EXISTS(SELECT 1 FROM reports WHERE visit_id=? AND submitted IS NOT NULL)
                     OR EXISTS(SELECT 1 FROM field_tasks WHERE visit_id=? AND status='done')) AS has_work
            """, (visit_id, visit_id, visit_id)).fetchone()["has_work"]
            if not note.strip() and not work:
                raise ValueError("Tashrifni yakunlash uchun bajarilgan ishni yoki qisqa izohni kiriting.")
            conn.execute("UPDATE field_visits SET finished=?,note=? WHERE id=?", (int(time.time()), note.strip(), visit_id))
            return True

    def record_delivery(self, user, visit_id, key, sku_id, product, quantity):
        with self.db() as conn:
            conn.write_lock()
            previous = conn.execute("""
                SELECT d.id,d.sku_id,d.quantity FROM field_deliveries d JOIN field_visits v ON v.id=d.visit_id
                WHERE d.visit_id=? AND d.request_key=? AND v.org_id=? AND v.agent_id=?
            """, (visit_id, key, user["org_id"], user["id"])).fetchone()
            if previous:
                if previous["sku_id"] != sku_id or previous["quantity"] != quantity:
                    raise ValueError("Bu topshirish yozuvi allaqachon mavjud.")
                return previous["id"]
            if self.own_visit(user, visit_id, conn) is None:
                return None
            if product is None:
                raise ValueError("Mahsulot katalogda topilmadi.")
            return conn.execute("""
                INSERT INTO field_deliveries(visit_id,request_key,sku_id,product,quantity,created) VALUES(?,?,?,?,?,?) RETURNING id
            """, (visit_id, key, sku_id, product, quantity, int(time.time()))).fetchone()["id"]

    def create_task(self, org_id, agent_id, store_id, title, kind, due):
        title = title.strip()
        if not title:
            raise ValueError("Topshiriq nomini kiriting.")
        with self.db() as conn:
            conn.write_lock()
            if not self._agent(conn, org_id, agent_id, active=True) or not self.assigned(
                    conn, {"org_id": org_id, "id": agent_id}, store_id):
                raise ValueError("Topshiriq faol agentga va unga biriktirilgan do‘konga beriladi.")
            return conn.execute("""
                INSERT INTO field_tasks(org_id,store_id,agent_id,title,kind,due) VALUES(?,?,?,?,?,?) RETURNING id
            """, (org_id, store_id, agent_id, title, kind, due)).fetchone()["id"]

    def finish_task(self, user, task_id, visit_id, note):
        with self.db() as conn:
            conn.write_lock()
            visit = self.own_visit(user, visit_id, conn)
            if visit is None:
                return False
            task = conn.execute("""
                SELECT kind FROM field_tasks WHERE id=? AND org_id=? AND agent_id=? AND store_id=? AND status='pending'
            """, (task_id, user["org_id"], user["id"], visit["store_id"])).fetchone()
            if task is None:
                return False
            if task["kind"] == "audit" and not conn.execute(
                    "SELECT 1 FROM reports WHERE visit_id=? AND submitted IS NOT NULL", (visit_id,)).fetchone():
                raise ValueError("Avval shu tashrifning fotoauditini yuboring.")
            if task["kind"] == "delivery" and not conn.execute(
                    "SELECT 1 FROM field_deliveries WHERE visit_id=?", (visit_id,)).fetchone():
                raise ValueError("Avval shu tashrifda topshirilgan tovarni qayd eting.")
            conn.execute("UPDATE field_tasks SET status='done',visit_id=?,note=?,completed=? WHERE id=?",
                         (visit_id, note.strip(), int(time.time()), task_id))
            return True

    def cancel_task(self, org_id, task_id):
        with self.db() as conn:
            return bool(conn.execute("UPDATE field_tasks SET status='cancelled' WHERE id=? AND org_id=? AND status='pending'",
                                     (task_id, org_id)).rowcount)

    def snapshot(self, user):
        is_admin = user["role"] == "org_admin"
        args = (user["org_id"],) if is_admin else (user["org_id"], user["id"])
        visit_scope = "v.org_id=?" if is_admin else "v.org_id=? AND v.agent_id=?"
        task_scope = "t.org_id=?" if is_admin else """t.org_id=? AND t.agent_id=? AND
            (t.status!='pending' OR EXISTS(SELECT 1 FROM store_assignments a
                WHERE a.org_id=t.org_id AND a.agent_id=t.agent_id AND a.store_id=t.store_id))"""
        with self.db() as conn:
            visits = [dict(row) for row in conn.execute(f"""
                SELECT v.id,v.store_id,v.started,v.finished,v.note,s.name AS store,u.username AS agent,
                    (SELECT COUNT(*) FROM reports r WHERE r.visit_id=v.id AND r.submitted IS NOT NULL) AS audits,
                    (SELECT COALESCE(SUM(quantity),0) FROM field_deliveries WHERE visit_id=v.id) AS units
                FROM field_visits v JOIN stores s ON s.id=v.store_id JOIN users u ON u.id=v.agent_id
                WHERE {visit_scope} ORDER BY v.finished IS NOT NULL, v.started DESC,v.id DESC LIMIT 30
            """, args)]
            tasks = [dict(row) for row in conn.execute(f"""
                SELECT t.id,t.store_id,t.agent_id,t.title,t.kind,t.due,t.status,t.note,t.completed,
                    s.name AS store,u.username AS agent
                FROM field_tasks t JOIN stores s ON s.id=t.store_id JOIN users u ON u.id=t.agent_id
                WHERE {task_scope} ORDER BY t.status!='pending',t.due IS NULL,t.due,t.id DESC LIMIT 100
            """, args)]
            agents = [dict(row) for row in conn.execute("""
                SELECT u.id,u.username,u.active,COUNT(a.store_id) AS stores FROM members m
                JOIN users u ON u.id=m.user_id LEFT JOIN store_assignments a ON a.agent_id=m.user_id AND a.org_id=m.org_id
                WHERE m.org_id=? AND m.role='operator' GROUP BY u.id ORDER BY u.username
            """, (user["org_id"],))] if is_admin else []
        return {"visits": visits, "tasks": tasks, "agents": agents}

    def visit(self, user, visit_id):
        with self.db() as conn:
            row = conn.execute("""
                SELECT v.*,s.name AS store,u.username AS agent FROM field_visits v
                JOIN stores s ON s.id=v.store_id JOIN users u ON u.id=v.agent_id
                WHERE v.id=? AND v.org_id=? AND (? OR v.agent_id=?)
            """, (visit_id, user["org_id"], user["role"] == "org_admin", user["id"])).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["deliveries"] = [dict(r) for r in conn.execute("""
                SELECT id,product,sku_id,quantity,created FROM field_deliveries WHERE visit_id=? ORDER BY id DESC
            """, (visit_id,))]
            result["audits"] = [dict(r) for r in conn.execute("""
                SELECT id,status,products,recognized,gaps FROM reports WHERE visit_id=? AND submitted IS NOT NULL ORDER BY submitted DESC
            """, (visit_id,))]
            return result

    def overview(self, org_id, days, store_id=None):
        today = datetime.now(LOCAL_TZ).date()
        start = today - timedelta(days=days - 1)
        stamp = int(datetime.combine(start, datetime.min.time(), LOCAL_TZ).timestamp())
        window = (org_id, stamp, stamp + days * 86400)
        with self.db() as conn:
            rows = [dict(row) for row in conn.execute("""
                WITH visit_counts AS (
                    SELECT store_id,COUNT(*) AS visits,COUNT(finished) AS finished,MAX(started) AS last_visit
                    FROM field_visits WHERE org_id=? AND started>=? AND started<? GROUP BY store_id
                ), delivery_counts AS (
                    SELECT v.store_id,SUM(d.quantity) AS units
                    FROM field_deliveries d JOIN field_visits v ON v.id=d.visit_id
                    WHERE v.org_id=? AND d.created>=? AND d.created<? GROUP BY v.store_id
                ), report_counts AS (
                    SELECT r.store_id,COUNT(*) AS audits,SUM(r.products) AS products,
                        SUM(r.recognized) AS recognized,SUM(r.gaps) AS gaps
                    FROM reports r WHERE r.org_id=? AND r.submitted>=? AND r.submitted<? GROUP BY r.store_id
                ), task_counts AS (
                    SELECT store_id,COUNT(*) AS tasks_done FROM field_tasks
                    WHERE org_id=? AND completed>=? AND completed<? GROUP BY store_id
                )
                SELECT s.id,s.name,s.address, COALESCE(v.visits,0) AS visits,COALESCE(v.finished,0) AS finished,
                    COALESCE(d.units,0) AS units,COALESCE(r.audits,0) AS audits,COALESCE(r.products,0) AS products,
                    COALESCE(r.recognized,0) AS recognized,COALESCE(r.gaps,0) AS gaps,
                    COALESCE(t.tasks_done,0) AS tasks_done,v.last_visit,
                    (SELECT COUNT(*) FROM store_assignments a JOIN users u ON u.id=a.agent_id
                     WHERE a.org_id=s.org_id AND a.store_id=s.id AND u.active=1) AS agents
                FROM stores s LEFT JOIN visit_counts v ON v.store_id=s.id LEFT JOIN delivery_counts d ON d.store_id=s.id
                LEFT JOIN report_counts r ON r.store_id=s.id LEFT JOIN task_counts t ON t.store_id=s.id
                WHERE s.org_id=? AND (CAST(? AS BIGINT) IS NULL OR s.id=?) ORDER BY visits DESC,audits DESC,s.name
            """, (*window, *window, *window, *window, org_id, store_id, store_id))]
            agents = [dict(row) for row in conn.execute("""
                SELECT u.id,u.username,u.active FROM store_assignments a JOIN users u ON u.id=a.agent_id
                WHERE a.org_id=? AND a.store_id=? ORDER BY u.username
            """, (org_id, store_id))] if store_id is not None else []
            delivery_items = [dict(row) for row in conn.execute("""
                SELECT d.sku_id,d.product,SUM(d.quantity) AS units FROM field_deliveries d
                JOIN field_visits v ON v.id=d.visit_id
                WHERE v.org_id=? AND d.created>=? AND d.created<? AND v.store_id=?
                GROUP BY d.sku_id,d.product ORDER BY units DESC,d.product LIMIT 10
            """, (*window, store_id))] if store_id is not None else []
            recent_visits = [dict(row) for row in conn.execute("""
                SELECT v.id,u.username AS agent,v.started,v.finished FROM field_visits v JOIN users u ON u.id=v.agent_id
                WHERE v.org_id=? AND v.started>=? AND v.started<? AND v.store_id=? ORDER BY v.started DESC,v.id DESC LIMIT 8
            """, (*window, store_id))] if store_id is not None else []
            backlog = dict(conn.execute("""
                SELECT COUNT(*) AS pending,COUNT(CASE WHEN due<? THEN 1 END) AS overdue
                FROM field_tasks WHERE org_id=? AND status='pending' AND (CAST(? AS BIGINT) IS NULL OR store_id=?)
            """, (today.isoformat(), org_id, store_id, store_id)).fetchone())
        keys = ("visits", "finished", "units", "audits", "products", "recognized", "gaps", "tasks_done")
        totals = {key: sum(row[key] for row in rows) for key in keys}
        return {"totals": totals, "stores": rows, "visited_stores": sum(row["visits"] > 0 for row in rows),
                "agents": agents, "delivery_items": delivery_items, "recent_visits": recent_visits,
                "backlog": backlog, "from": start.isoformat(), "to": today.isoformat()}
