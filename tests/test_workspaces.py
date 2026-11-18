import csv
from datetime import datetime, timezone
import io
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from test_accounts import HEADERS, INITIAL, login
from test_web import FakePipe, jpg, web
from rasta.config import Config
from rasta_backend.accounts import Accounts
from rasta_backend.database import Database
from rasta_backend.workspaces import Workspaces


@pytest.fixture(params=["sqlite", "postgres"])
def workspace_database(request, tmp_path):
    target = request.getfixturevalue("postgres_url") if request.param == "postgres" else tmp_path / "accounts.sqlite3"
    database = Database(target)
    try:
        yield database
    finally:
        database.close()


def test_organization_boundaries_and_shelf_review(tmp_path, monkeypatch, workspace_database):
    accounts = Accounts(workspace_database)
    accounts.create("root-admin", INITIAL, "super_admin")
    accounts.create("legacy-admin", INITIAL)
    spaces = Workspaces(accounts)
    fake = FakePipe()
    monkeypatch.setattr(web.State, "pipe", fake)
    monkeypatch.setattr(web.State, "images_dir", tmp_path / "images")
    app = web.create_app(Config({"catalog": {"path": ":memory:"}}), accounts=accounts, workspaces=spaces)
    image = {"image": ("s.jpg", jpg(), "image/jpeg")}
    packshot = [("files", ("s.jpg", jpg(), "image/jpeg"))]
    with (TestClient(app) as root, TestClient(app) as admin, TestClient(app) as other,
          TestClient(app) as operator):
        assert login(root, "root-admin").status_code == 200
        migrated = login(admin, "legacy-admin").json()["user"]
        assert (migrated["role"], migrated["org_id"]) == ("org_admin", 1)
        response = root.post("/api/organizations", headers=HEADERS, json={
            "name": "Another Company", "username": "foreign-admin", "password": INITIAL,
        })
        assert response.status_code == 201
        org_id = response.json()["id"]
        assert login(other, "foreign-admin").json()["user"]["org_id"] == org_id
        assert admin.get("/api/organizations").status_code == 403
        assert admin.get("/api/team").status_code == 404
        assert root.post("/api/analyze", headers=HEADERS, files=image).status_code == 403

        for client, brand in [(admin, "A"), (other, "B")]:
            assert client.post("/api/products", headers=HEADERS, data={
                "name": "Cola", "brand": brand, "category": "Ichimliklar", "barcode": "001234567890",
            }, files=packshot).status_code == 200
        assert admin.get("/api/products/cola").json()["brand"] == "A"
        assert other.get("/api/products/cola").json()["brand"] == "B"
        assert (tmp_path / "images" / "cola" / "000.jpg").exists()
        assert (tmp_path / "images" / ".organizations" / str(org_id) / "cola" / "000.jpg").exists()
        assert admin.get("/media/cola/000.jpg").status_code == 200
        scope = f"?organization={org_id}"
        assert root.get("/api/products").status_code == 400
        assert root.get("/api/products?organization=99999").status_code == 404
        assert root.get("/api/products" + scope).json()["products"][0]["brand"] == "B"
        assert root.get("/api/products/cola?organization=1").json()["brand"] == "A"
        assert root.get("/media/cola/000.jpg" + scope).status_code == 200
        assert root.delete("/api/products/cola" + scope, headers=HEADERS).status_code == 403
        assert admin.get("/api/products" + scope).status_code == 403
        assert admin.get("/media/cola/000.jpg" + scope).status_code == 403

        response = admin.post("/api/members", headers=HEADERS, json={"username": "field-user", "password": INITIAL})
        assert response.status_code == 201
        member_id = response.json()["id"]
        assert login(operator, "field-user").json()["user"]["role"] == "operator"
        assert operator.get("/api/products").status_code == 403
        assert operator.post("/api/products", headers=HEADERS, data={"name": "Test"}, files=packshot).status_code == 403
        assert other.patch(f"/api/members/{member_id}", headers=HEADERS, json={"active": False}).status_code == 404
        assert operator.post("/api/stores", headers=HEADERS, json={"name": "Blocked"}).status_code == 403

        store = admin.post("/api/stores", headers=HEADERS, json={"name": "Tashkent"}).json()["id"]
        foreign_store = other.post("/api/stores", headers=HEADERS, json={"name": "Foreign"}).json()["id"]
        assert other.patch(f"/api/stores/{store}", headers=HEADERS, json={"name": "Blocked"}).status_code == 404
        assert admin.patch(f"/api/stores/{store}", headers=HEADERS, json={"name": "Tashkent Shop"}).status_code == 200
        assert operator.get("/api/stores").json()["stores"] == []
        assert operator.post("/api/field/visits", headers=HEADERS, json={"store_id": store}).status_code == 400
        assert operator.post("/api/analyze", headers=HEADERS, files=image).status_code == 400
        assignment = f"/api/field/agents/{member_id}/stores"
        assert other.put(assignment, headers=HEADERS, json={"store_ids": [foreign_store]}).status_code == 404
        assert admin.put(assignment, headers=HEADERS, json={"store_ids": [foreign_store]}).status_code == 400
        assert admin.put(assignment, headers=HEADERS, json={"store_ids": [store, store]}).json()["store_ids"] == [store]
        assert [row["id"] for row in operator.get("/api/stores").json()["stores"]] == [store]
        visit_id = operator.post("/api/field/visits", headers=HEADERS, json={"store_id": store}).json()["id"]
        visit_products = operator.get(f"/api/field/visits/{visit_id}/products").json()["products"]
        assert [(product["sku_id"], product["barcode"]) for product in visit_products] == [("cola", "001234567890")]
        visit_url = f"/api/field/visits/{visit_id}"
        assert operator.post("/api/field/visits", headers=HEADERS, json={"store_id": store}).json()["id"] == visit_id
        assert other.get(visit_url).status_code == 404
        assert operator.post(visit_url + "/finish", headers=HEADERS, json={}).status_code == 400
        assert operator.get(visit_url + "/products").json()["products"][0]["sku_id"] == "cola"
        task = admin.post("/api/field/tasks", headers=HEADERS, json={
            "store_id": store, "agent_id": member_id, "title": "Check shelf", "kind": "audit",
        })
        assert task.status_code == 201
        task_url = f"/api/field/tasks/{task.json()['id']}/complete"
        assert operator.post(task_url, headers=HEADERS, json={"visit_id": visit_id}).status_code == 400
        delivery = {"request_key": str(uuid4()), "sku_id": "cola", "quantity": 6}
        recorded = operator.post(visit_url + "/deliveries", headers=HEADERS, json=delivery)
        assert recorded.status_code == 201
        assert operator.post(visit_url + "/deliveries", headers=HEADERS, json=delivery).json() == recorded.json()
        assert operator.post(visit_url + "/deliveries", headers=HEADERS, json={**delivery, "quantity": 7}).status_code == 400
        response = operator.post("/api/analyze", headers=HEADERS, data={"visit_id": visit_id}, files=image)
        assert response.status_code == 200
        assert response.json()["visit_store_id"] == store
        report = "/api/reports/" + response.json()["report_id"]
        assert admin.get(report).status_code == 404
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": foreign_store}).status_code == 400
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert operator.post(task_url, headers=HEADERS, json={"visit_id": visit_id}).status_code == 200
        assert operator.post(visit_url + "/finish", headers=HEADERS, json={}).status_code == 200
        assert operator.post("/api/analyze", headers=HEADERS, data={"visit_id": visit_id}, files=image).status_code == 404
        field = admin.get("/api/field/overview", params={"store_id": store, "days": "7"}).json()
        assert (field["totals"]["finished"], field["totals"]["units"], field["totals"]["tasks_done"]) == (1, 6, 1)
        assert field["totals"]["audits"] == field["visited_stores"] == 1
        assert field["delivery_items"][0]["units"] == 6
        assert other.get("/api/field/overview", params={"store_id": store}).status_code == 404
        assert other.get(report).status_code == 404
        assert other.get(report + "/photo").status_code == 404
        assert admin.get(report + "/photo").headers["content-type"] == "image/jpeg"
        assert operator.post(report + "/review", headers=HEADERS, json={"status": "approved"}).status_code == 403
        assert admin.post(report + "/review", headers=HEADERS, json={
            "status": "returned", "feedback": "Try again",
        }).status_code == 200
        assert operator.get(report).json()["feedback"] == "Try again"
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert admin.post(report + "/review", headers=HEADERS, json={"status": "approved"}).status_code == 200
        assert [event["status"] for event in admin.get(report).json()["events"]] == [
            "approved", "submitted", "returned", "submitted",
        ]
        assert admin.get("/api/overview?days=7").json()["approved"] == 1
        assert other.get("/api/overview?days=7").json()["reports"] == 0
        platform = root.get("/api/platform/overview?days=7").json()
        assert platform["organizations"] == {"total": 2, "active": 2}
        assert platform["members"] == {"total": 3, "active": 3, "admins": 2, "operators": 1}
        assert platform["stores"] == 2
        assert platform["audits"]["approved"] == 1 and platform["audits"]["open_queue"] == 0
        assert platform["audits"]["products"] == 2
        assert len(platform["trend"]) == 7 and sum(item["count"] for item in platform["trend"]) == 1
        assert [row["id"] for row in platform["activity"]] == [1]
        assert platform["events"][0]["action"] == "organization_created"
        assert platform["events"][0]["actor"] == "root-admin"
        assert INITIAL not in str(platform)
        status = root.get(f"/api/organizations/{org_id}/overview?days=30").json()
        assert status["organization"]["name"] == "Another Company"
        assert status["audits"]["total"] == 0 and status["members"]["admins"] == 1
        assert status["audits"]["review_seconds"] is None
        assert admin.get("/api/platform/overview").status_code == 403
        assert operator.get(f"/api/organizations/{org_id}/overview").status_code == 403
        assert root.get("/api/organizations/999999/overview").status_code == 404
        report_id = report.rsplit("/", 1)[1]
        before_midnight_utc = int(datetime(2025, 12, 31, 19, tzinfo=timezone.utc).timestamp())
        with spaces.db() as db:
            db.execute("UPDATE reports SET submitted=? WHERE id=?", (before_midnight_utc, report_id))
        params = {"store_id": store, "author_id": member_id, "status": "approved",
                  "from": "2026-01-01", "to": "2026-01-01"}
        filtered = admin.get("/api/reports", params=params).json()
        assert filtered["total"] == 1 and filtered["reports"][0]["id"] == report_id
        assert other.get("/api/reports", params=params).json()["total"] == 0
        assert admin.get("/api/reports", params={**params, "store_id": foreign_store}).json()["total"] == 0
        assert admin.get("/api/reports", params={**params, "to": "2025-12-31"}).status_code == 400
        assert admin.get("/api/reports?to=2025-12-31").json()["total"] == 0
        assert admin.get("/api/reports", params={**params, "offset": 50}).json() == {
            "total": 1, "reports": [], "has_more": False,
        }
        assert admin.patch(f"/api/stores/{store}", headers=HEADERS,
                           json={"name": "=Spreadsheet text"}).status_code == 200
        response = admin.get("/api/reports/export.csv", params=params)
        assert response.status_code == 200 and response.content.startswith(b"\xef\xbb\xbf")
        csv_rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
        assert csv_rows[1][:5] == [report_id, "'=Spreadsheet text", "field-user", "Tasdiqlangan", "01.01.2026 00:00"]
        assert len(list(csv.reader(io.StringIO(other.get("/api/reports/export.csv", params=params).text)))) == 1
        assert operator.get("/api/reports/export.csv", params=params).status_code == 403
        assert admin.get("/api/reports/export.csv?status=draft").status_code == 400
        assert root.patch(f"/api/organizations/{org_id}", headers=HEADERS,
                          json={"name": "Another Company", "active": False}).status_code == 200
        assert other.get("/api/products").status_code == 401
        inactive = root.get(f"/api/organizations/{org_id}/overview").json()
        assert inactive["members"]["total"] == 1 and inactive["members"]["active"] == 0
        assert inactive["events"][0]["action"] == "organization_suspended"
